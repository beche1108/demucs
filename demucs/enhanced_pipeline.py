from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import av
import numpy as np
import soundfile as sf
import torch

from .api import Separator
from .audio import convert_audio

logger = logging.getLogger(__name__)

DEMUCS_SAMPLE_RATE = 44100


def _extract_audio_av(input_path: str, start_s: float, duration_s: float,
                      sample_rate: int = DEMUCS_SAMPLE_RATE, channels: int = 2) -> torch.Tensor:
    """Extract an audio segment from a media file via PyAV (ffmpeg binding).

    Seeks to nearest keyframe before start_s, decodes forward, then trims
    to the exact [start_s, start_s + duration_s] window by sample offset.
    """
    layout = "stereo" if channels == 2 else "mono"

    with av.open(str(input_path)) as container:
        audio_stream = next(iter(container.streams.audio), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found: {input_path}")

        # Seek backward to nearest keyframe at or before start_s.
        seek_pts = int(start_s / audio_stream.time_base)
        container.seek(seek_pts, stream=audio_stream, backward=True)

        resampler = av.audio.resampler.AudioResampler(
            format="fltp",  # planar float32 → to_ndarray() returns (channels, samples)
            layout=layout,
            rate=sample_rate,
        )

        end_s = start_s + duration_s
        chunks = []
        landed_time = None

        done = False
        for packet in container.demux(audio_stream):
            if done:
                break
            for frame in packet.decode():
                if frame.pts is not None:
                    t = float(frame.pts * audio_stream.time_base)
                    if landed_time is None:
                        landed_time = t
                    if t > end_s + 0.5:
                        done = True
                        break
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())

        # Flush resampler.
        for flushed in resampler.resample(None):
            chunks.append(flushed.to_ndarray())

    if not chunks:
        raise ValueError(
            f"No audio decoded for [{start_s:.3f}, {start_s + duration_s:.3f}]: {input_path}"
        )

    audio = np.concatenate(chunks, axis=1)  # (channels, samples)

    # Trim: we decoded from landed_time (<= start_s), skip the leading samples.
    skip = int((start_s - (landed_time or start_s)) * sample_rate)
    skip = max(0, skip)
    target = int(duration_s * sample_rate)
    audio = audio[:, skip : skip + target]

    return torch.from_numpy(audio.copy())


@dataclass
class EnhancedPipelineConfig:
    demucs_model: str = "htdemucs_ft"
    use_gpu: bool = False
    labels_to_separate: tuple = ("speech_with_music", "singing", "uncertain")
    labels_to_passthrough: tuple = ("speech",)
    labels_to_skip: tuple = ("silence", "music")
    output_sample_rate: int = 16000
    save_accompaniment: bool = True
    shifts: int = 1
    overlap: float = 0.25
    segment: Optional[int] = None
    jobs: int = 0
    progress: bool = True


class DemucsProcessor:
    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        if config.use_gpu and torch.cuda.is_available():
            device = "cuda"
        elif config.use_gpu:
            logger.warning("use_gpu=True but CUDA is unavailable. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cpu"
        self.device = device
        self.separator = Separator(
            model=config.demucs_model,
            device=device,
            shifts=config.shifts,
            overlap=config.overlap,
            split=True,
            segment=config.segment,
            jobs=config.jobs,
            progress=config.progress,
        )

    def separate(self, wav: torch.Tensor) -> dict:
        """Separate a (channels, samples) tensor at DEMUCS_SAMPLE_RATE.

        Returns dict with 'vocals' and 'accompaniment' as numpy float32 arrays
        resampled to output_sample_rate mono.
        """
        _, stems = self.separator.separate_tensor(wav, DEMUCS_SAMPLE_RATE)

        vocals = stems["vocals"]
        accompaniment = stems["drums"] + stems["bass"] + stems["other"]

        out_sr = self.config.output_sample_rate
        vocals_mono = convert_audio(vocals, DEMUCS_SAMPLE_RATE, out_sr, 1)
        accompaniment_mono = convert_audio(accompaniment, DEMUCS_SAMPLE_RATE, out_sr, 1)

        return {
            "vocals": vocals_mono.squeeze().cpu().numpy().astype(np.float32),
            "accompaniment": accompaniment_mono.squeeze().cpu().numpy().astype(np.float32),
            "sample_rate": out_sr,
        }


class EnhancedPipeline:
    def __init__(self, processor: DemucsProcessor, config: EnhancedPipelineConfig):
        self.processor = processor
        self.config = config

    @classmethod
    def from_config(cls, config: Optional[EnhancedPipelineConfig] = None) -> EnhancedPipeline:
        if config is None:
            config = EnhancedPipelineConfig()
        processor = DemucsProcessor(config)
        return cls(processor, config)

    def process(self, segments_json_path: str, output_dir: str,
                input_path: str | None = None) -> dict:
        """Process FireRedVAD segments using Demucs on the original media file.

        Args:
            segments_json_path: Path to FireRedVAD segments.json.
            output_dir: Directory to save enhanced outputs.
            input_path: Override for the source media file. If None, uses
                        'input_path' from segments.json.
        """
        segments_json_path = Path(segments_json_path)
        output_dir = Path(output_dir)
        segments_dir = output_dir / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        with open(segments_json_path, encoding="utf-8") as f:
            source = json.load(f)

        if input_path is None:
            input_path = source.get("input_path")
        if not input_path or not Path(input_path).exists():
            raise FileNotFoundError(
                f"Source media not found: {input_path}. "
                "Pass --input_path to specify the correct location."
            )

        logger.info("Source media: %s", input_path)

        config = self.config
        result_segments = []

        for seg in source["timeline"]:
            seg_id = seg["segment_id"]
            label = seg["label"]
            start_s = seg["start"]
            end_s = seg["end"]
            duration_s = seg["duration"]

            seg_entry = {
                "segment_id": seg_id,
                "original_label": label,
                "start": start_s,
                "end": end_s,
                "duration": duration_s,
            }

            if label in config.labels_to_skip:
                seg_entry["enhanced_action"] = "skipped"
                result_segments.append(seg_entry)
                continue

            if label in config.labels_to_passthrough:
                # Extract from original at output_sample_rate mono, save directly.
                wav = _extract_audio_av(
                    input_path, start_s, duration_s,
                    sample_rate=config.output_sample_rate, channels=1,
                )
                vocals_np = wav.squeeze().numpy().astype(np.float32)
                dest = segments_dir / f"{seg_id:04d}_vocals.wav"
                sf.write(str(dest), vocals_np, config.output_sample_rate)
                seg_entry["vocals_path"] = str(dest.relative_to(output_dir))
                seg_entry["vocals_rms"] = float(np.sqrt(np.mean(vocals_np ** 2)))
                seg_entry["enhanced_action"] = "passthrough"
                result_segments.append(seg_entry)
                logger.debug("Segment %d (%s): passthrough", seg_id, label)
                continue

            if label in config.labels_to_separate:
                logger.info(
                    "Segment %d (%s): separating [%.3f - %.3f]",
                    seg_id, label, start_s, end_s,
                )
                # Extract from original at 44100Hz stereo for Demucs.
                wav = _extract_audio_av(
                    input_path, start_s, duration_s,
                    sample_rate=DEMUCS_SAMPLE_RATE, channels=2,
                )
                result = self.processor.separate(wav)

                vocals_path = segments_dir / f"{seg_id:04d}_vocals.wav"
                sf.write(str(vocals_path), result["vocals"], result["sample_rate"])
                seg_entry["vocals_path"] = str(vocals_path.relative_to(output_dir))
                seg_entry["vocals_rms"] = float(np.sqrt(np.mean(result["vocals"] ** 2)))

                if config.save_accompaniment:
                    acc_path = segments_dir / f"{seg_id:04d}_accompaniment.wav"
                    sf.write(str(acc_path), result["accompaniment"], result["sample_rate"])
                    seg_entry["accompaniment_path"] = str(acc_path.relative_to(output_dir))
                    seg_entry["accompaniment_rms"] = float(
                        np.sqrt(np.mean(result["accompaniment"] ** 2))
                    )

                seg_entry["enhanced_action"] = "separated"
                result_segments.append(seg_entry)
                continue

            logger.warning("Segment %d: unhandled label '%s', skipping", seg_id, label)
            seg_entry["enhanced_action"] = "skipped"
            result_segments.append(seg_entry)

        manifest = {
            "source_manifest": str(segments_json_path),
            "input_path": str(input_path),
            "demucs_model": config.demucs_model,
            "config": dataclasses.asdict(config),
            "segments": result_segments,
        }

        stem = segments_json_path.stem.split(".")[0]
        manifest_path = output_dir / f"{stem}.enhanced.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info("Enhanced manifest saved to %s", manifest_path)
        return manifest
