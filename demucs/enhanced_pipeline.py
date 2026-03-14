from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import json
import logging
import os
from collections import Counter
from collections.abc import Callable
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
DEFAULT_OUTPUT_SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION = 0.05  # seconds; segments shorter than this are skipped
RESUME_RUNTIME_CONFIG_FIELDS = {"jobs", "progress", "resume", "use_gpu"}


class _PrefixLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        prefix = self.extra.get("prefix", "")
        if prefix:
            msg = f"{prefix}{msg}"
        return msg, kwargs


class SegmentSkipError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class PendingWrite:
    future: concurrent.futures.Future
    path_field: str
    relative_path: str
    extra_fields: dict = field(default_factory=dict)


@dataclass
class PendingSegmentWrite:
    entry: dict
    success_action: str
    writes: list[PendingWrite] = field(default_factory=list)


@dataclass
class SeparationGroup:
    group_id: int
    start_index: int
    end_index: int
    member_indices: list[int]
    bridge_indices: list[int]
    start_s: float
    end_s: float
    extract_start_s: float
    extract_end_s: float


def _as_label_tuple(value) -> tuple:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _to_mono_audio_tensor(wav: torch.Tensor, context: str) -> torch.Tensor:
    audio = wav.detach()
    if audio.ndim == 2:
        if audio.shape[0] != 1:
            raise ValueError(f"{context} must be mono, got shape {tuple(audio.shape)}")
        audio = audio[0]
    elif audio.ndim != 1:
        raise ValueError(f"{context} must be 1D or (1, N), got shape {tuple(audio.shape)}")
    return audio.contiguous()


def _to_mono_numpy(wav: torch.Tensor, context: str) -> np.ndarray:
    audio = _to_mono_audio_tensor(wav, context).cpu().numpy()
    return np.ascontiguousarray(audio)


def _compute_rms(audio: np.ndarray, context: str) -> float:
    if audio.size == 0:
        raise ValueError(f"{context} is empty")
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    if not np.isfinite(rms):
        raise ValueError(f"{context} RMS is not finite")
    return rms


def _segment_duration(seg: dict) -> float:
    return max(0.0, float(seg["end"]) - float(seg["start"]))


def _manifest_relative_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _iter_manifest_path_candidates(path_value: str) -> list[Path]:
    candidates = [Path(path_value)]
    if "\\" in path_value:
        candidates.append(Path(path_value.replace("\\", "/")))
    if os.sep == "\\" and "/" in path_value:
        candidates.append(Path(path_value.replace("/", "\\")))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _resolve_manifest_reference(base_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None

    fallback = None
    for candidate in _iter_manifest_path_candidates(path_value):
        current = candidate if candidate.is_absolute() else (base_dir / candidate)
        resolved = current.resolve()
        if fallback is None:
            fallback = resolved
        if resolved.exists():
            return resolved
    return fallback


def _normalize_jsonish(value):
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonish(inner) for inner in value]
    return value


def _normalized_resume_config(config: dict | None) -> dict:
    normalized = _normalize_jsonish(config or {})
    for field_name in RESUME_RUNTIME_CONFIG_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def _same_time(left: object, right: object, *, tol: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def _segment_entry_matches_timeline(segment_entry: dict, timeline_segment: dict) -> bool:
    try:
        segment_id = int(segment_entry.get("segment_id"))
        timeline_segment_id = int(timeline_segment["segment_id"])
    except (KeyError, TypeError, ValueError):
        return False

    return (
        segment_id == timeline_segment_id
        and segment_entry.get("original_label") == timeline_segment.get("label")
        and _same_time(segment_entry.get("start"), timeline_segment.get("start"))
        and _same_time(segment_entry.get("end"), timeline_segment.get("end"))
    )


def _resolve_output_sample_rate(audio_stream, configured_sample_rate: Optional[int]) -> int:
    if configured_sample_rate is not None:
        rate = int(configured_sample_rate)
        if rate <= 0:
            raise ValueError("output_sample_rate must be a positive integer")
        return rate

    for candidate in (
        getattr(audio_stream, "rate", None),
        getattr(getattr(audio_stream, "codec_context", None), "sample_rate", None),
    ):
        if candidate:
            rate = int(candidate)
            if rate > 0:
                return rate

    logger.warning(
        "Could not determine input audio sample rate. Falling back to %d Hz.",
        DEFAULT_OUTPUT_SAMPLE_RATE,
    )
    return DEFAULT_OUTPUT_SAMPLE_RATE


def _slice_audio_window(
    audio: torch.Tensor | np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    window_start_s: float,
    context: str,
) -> torch.Tensor | np.ndarray:
    start_idx = max(0, int(round((start_s - window_start_s) * sample_rate)))
    end_idx = min(audio.shape[-1], int(round((end_s - window_start_s) * sample_rate)))
    if end_idx <= start_idx:
        raise SegmentSkipError(
            "empty_after_group_slice",
            f"{context} has no samples for [{start_s:.3f}, {end_s:.3f}]",
        )
    sliced = audio[..., start_idx:end_idx]
    sample_count = int(sliced.shape[-1])
    min_samples = max(1, int(sample_rate * MIN_SEGMENT_DURATION))
    if sample_count < min_samples:
        raise SegmentSkipError(
            "too_short_after_group_slice",
            f"{context} has only {sample_count} samples for [{start_s:.3f}, {end_s:.3f}]",
        )
    if isinstance(sliced, torch.Tensor):
        return sliced.contiguous()
    return np.ascontiguousarray(sliced)


def _extract_audio_av(
    container,
    audio_stream,
    start_s: float,
    duration_s: float,
    sample_rate: int = DEMUCS_SAMPLE_RATE,
    channels: int = 2,
) -> torch.Tensor:
    """Extract an audio segment from an already-open container via PyAV."""
    layout = "stereo" if channels == 2 else "mono"

    # AudioResampler cannot be safely reused across independently seeked
    # segment extractions once it has been flushed with ``resample(None)``.
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout=layout, rate=sample_rate,
    )

    if audio_stream.time_base is None:
        container.seek(0, stream=audio_stream)
        for packet in container.demux(audio_stream):
            for _ in packet.decode():
                break
            break
        if audio_stream.time_base is None:
            raise ValueError("Cannot determine time_base for audio stream")

    seek_pts = int(start_s / audio_stream.time_base)
    container.seek(seek_pts, stream=audio_stream, backward=True)

    end_s = start_s + duration_s
    chunks: list[np.ndarray] = []
    landed_time: float | None = None

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

    for flushed in resampler.resample(None):
        chunks.append(flushed.to_ndarray())

    if not chunks:
        raise SegmentSkipError(
            "no_audio_decoded",
            f"No audio decoded for [{start_s:.3f}, {start_s + duration_s:.3f}]",
        )

    audio = np.concatenate(chunks, axis=1)

    skip = int((start_s - (landed_time or start_s)) * sample_rate)
    skip = max(0, skip)
    target = int(duration_s * sample_rate)
    audio = audio[:, skip : skip + target]

    actual_samples = audio.shape[1]
    if actual_samples == 0:
        raise SegmentSkipError(
            "empty_after_trim",
            f"No samples remain after trim for [{start_s:.3f}, {end_s:.3f}]",
        )

    min_samples = max(1, int(sample_rate * MIN_SEGMENT_DURATION))
    if actual_samples < min_samples:
        raise SegmentSkipError(
            "too_short_after_trim",
            f"Only {actual_samples} samples remain after trim for [{start_s:.3f}, {end_s:.3f}]",
        )

    if target > 0 and actual_samples < target * 0.9:
        logger.warning(
            "Short audio: expected %d samples, got %d (%.1f%%) at [%.3f, %.3f]",
            target, actual_samples, actual_samples / target * 100,
            start_s, start_s + duration_s,
        )

    return torch.from_numpy(audio.copy())


@dataclass
class EnhancedPipelineConfig:
    demucs_model: str = "htdemucs_ft"
    stem_mode: str = "two"
    use_gpu: bool = False
    resume: bool = False
    labels_to_separate: tuple = ("speech_with_music", "singing", "uncertain")
    labels_to_passthrough: tuple = ("speech",)
    labels_to_skip: tuple = ("silence", "music")
    merge_adjacent_separation: bool = True
    merge_gap_s: float = 0.2
    merge_max_duration_s: float = 20.0
    merge_min_segments: int = 2
    merge_context_s: float = 0.25
    merge_bridge_labels: tuple = ("silence", "music")
    output_sample_rate: Optional[int] = None
    save_accompaniment: bool = True
    shifts: int = 5
    overlap: float = 0.8
    segment: Optional[int] = None
    jobs: int = 0
    progress: bool = True

    def __post_init__(self):
        self.labels_to_separate = _as_label_tuple(self.labels_to_separate)
        self.labels_to_passthrough = _as_label_tuple(self.labels_to_passthrough)
        self.labels_to_skip = _as_label_tuple(self.labels_to_skip)
        self.merge_bridge_labels = _as_label_tuple(self.merge_bridge_labels)
        self.stem_mode = str(self.stem_mode).lower()
        if self.stem_mode not in {"two", "four"}:
            raise ValueError("stem_mode must be either 'two' or 'four'")
        self.merge_gap_s = max(0.0, float(self.merge_gap_s))
        self.merge_max_duration_s = max(MIN_SEGMENT_DURATION, float(self.merge_max_duration_s))
        self.merge_min_segments = max(2, int(self.merge_min_segments))
        self.merge_context_s = max(0.0, float(self.merge_context_s))
        if self.output_sample_rate is not None:
            self.output_sample_rate = int(self.output_sample_rate)
            if self.output_sample_rate <= 0:
                raise ValueError("output_sample_rate must be positive when provided")


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
        if device != "cpu":
            self.separator._model.to(device)

    def separate(self, wav: torch.Tensor, output_sample_rate: int) -> dict:
        wav = wav.to(self.device)

        _, stems = self.separator.separate_tensor(wav, DEMUCS_SAMPLE_RATE)

        if "vocals" not in stems:
            raise ValueError(
                f"Model '{self.config.demucs_model}' does not provide a 'vocals' stem. "
                f"Available stems: {sorted(stems)}"
            )
        processed_stems: dict[str, torch.Tensor] = {}
        vocals_mono = convert_audio(stems["vocals"], DEMUCS_SAMPLE_RATE, output_sample_rate, 1)
        processed_stems["vocals"] = _to_mono_audio_tensor(vocals_mono, "vocals output")

        non_vocal_names = [name for name in stems if name != "vocals"]
        if self.config.stem_mode == "two":
            if not non_vocal_names:
                raise ValueError(
                    f"Model '{self.config.demucs_model}' has no non-vocal stems to build accompaniment. "
                    f"Available stems: {sorted(stems)}"
                )
            accompaniment = stems[non_vocal_names[0]]
            for name in non_vocal_names[1:]:
                accompaniment = accompaniment + stems[name]
            accompaniment_mono = convert_audio(
                accompaniment,
                DEMUCS_SAMPLE_RATE,
                output_sample_rate,
                1,
            )
            processed_stems["accompaniment"] = _to_mono_audio_tensor(
                accompaniment_mono,
                "accompaniment output",
            )
        else:
            for name in non_vocal_names:
                stem_mono = convert_audio(stems[name], DEMUCS_SAMPLE_RATE, output_sample_rate, 1)
                processed_stems[name] = _to_mono_audio_tensor(stem_mono, f"{name} output")

        return {
            "stems": processed_stems,
            "sample_rate": output_sample_rate,
            "stem_mode": self.config.stem_mode,
        }


class EnhancedPipeline:
    def __init__(self, processor: DemucsProcessor, config: EnhancedPipelineConfig):
        self.processor = processor
        self.config = config
        self.logger = _PrefixLoggerAdapter(logger, {"prefix": ""})

    @classmethod
    def from_config(cls, config: Optional[EnhancedPipelineConfig] = None) -> EnhancedPipeline:
        if config is None:
            config = EnhancedPipelineConfig()
        processor = DemucsProcessor(config)
        return cls(processor, config)

    def _required_resume_path_fields(self, seg_entry: dict) -> list[str] | None:
        action = seg_entry.get("enhanced_action")
        if action not in {"separated", "passthrough"}:
            return None

        required_fields = ["vocals_path"]
        if action == "separated" and self.config.save_accompaniment:
            if self.config.stem_mode == "two":
                required_fields.append("accompaniment_path")
            else:
                non_vocal_fields = sorted(
                    field_name
                    for field_name, value in seg_entry.items()
                    if field_name.endswith("_path")
                    and field_name != "vocals_path"
                    and value
                )
                if not non_vocal_fields:
                    return None
                required_fields.extend(non_vocal_fields)
        return required_fields

    def _is_resume_segment_entry_reusable(
        self,
        seg_entry: dict,
        timeline_segment: dict,
        output_dir: Path,
    ) -> bool:
        if not _segment_entry_matches_timeline(seg_entry, timeline_segment):
            return False

        action = seg_entry.get("enhanced_action")
        if action == "error":
            return False
        if action == "skipped":
            return True

        required_fields = self._required_resume_path_fields(seg_entry)
        if required_fields is None:
            return False

        for field_name in required_fields:
            resolved = _resolve_manifest_reference(output_dir, seg_entry.get(field_name))
            if resolved is None or not resolved.exists():
                return False
        return True

    def _load_resume_state(
        self,
        *,
        manifest_path: Path,
        timeline: list[dict],
        output_dir: Path,
        resolved_output_sample_rate: int,
    ) -> tuple[dict[int, dict], dict[int, dict], int]:
        if not self.config.resume or not manifest_path.exists():
            return {}, {}, 0

        try:
            with open(manifest_path, encoding="utf-8") as fin:
                existing_manifest = json.load(fin)
        except Exception as exc:
            self.logger.warning(
                "Failed to load existing Demucs manifest %s for resume. Reprocessing from scratch: %s",
                manifest_path,
                exc,
            )
            return {}, {}, 0

        if not isinstance(existing_manifest, dict):
            self.logger.warning(
                "Existing Demucs manifest %s is not a JSON object. Reprocessing from scratch.",
                manifest_path,
            )
            return {}, {}, 0

        existing_config = _normalized_resume_config(existing_manifest.get("config"))
        current_config = _normalized_resume_config(dataclasses.asdict(self.config))
        if existing_config != current_config:
            self.logger.warning(
                "Existing Demucs manifest %s uses different settings. Reprocessing from scratch.",
                manifest_path,
            )
            return {}, {}, 0

        try:
            existing_output_sample_rate = int(existing_manifest.get("output_sample_rate"))
        except (TypeError, ValueError):
            self.logger.warning(
                "Existing Demucs manifest %s is missing a valid output_sample_rate. Reprocessing from scratch.",
                manifest_path,
            )
            return {}, {}, 0
        if existing_output_sample_rate != resolved_output_sample_rate:
            self.logger.warning(
                "Existing Demucs manifest %s uses output_sample_rate=%s, but the current run resolved %s. Reprocessing from scratch.",
                manifest_path,
                existing_output_sample_rate,
                resolved_output_sample_rate,
            )
            return {}, {}, 0

        existing_segments = existing_manifest.get("segments")
        if not isinstance(existing_segments, list):
            self.logger.warning(
                "Existing Demucs manifest %s is missing a valid segments list. Reprocessing from scratch.",
                manifest_path,
            )
            return {}, {}, 0

        timeline_by_id = {
            int(segment["segment_id"]): segment
            for segment in timeline
        }
        reusable_segments: dict[int, dict] = {}
        ignored_segments = 0
        for seg_entry in existing_segments:
            if not isinstance(seg_entry, dict):
                ignored_segments += 1
                continue
            try:
                segment_id = int(seg_entry.get("segment_id"))
            except (TypeError, ValueError):
                ignored_segments += 1
                continue
            timeline_segment = timeline_by_id.get(segment_id)
            if timeline_segment is None:
                ignored_segments += 1
                continue
            if not self._is_resume_segment_entry_reusable(seg_entry, timeline_segment, output_dir):
                ignored_segments += 1
                continue
            reusable_segments[segment_id] = copy.deepcopy(seg_entry)

        reusable_merge_groups: dict[int, dict] = {}
        for merge_group in existing_manifest.get("merge_groups", []):
            if not isinstance(merge_group, dict):
                continue
            try:
                group_id = int(merge_group["group_id"])
            except (KeyError, TypeError, ValueError):
                continue
            reusable_merge_groups[group_id] = copy.deepcopy(merge_group)

        reused_group_ids = set()
        for seg_entry in reusable_segments.values():
            try:
                group_id = int(seg_entry["merge_group_id"])
            except (KeyError, TypeError, ValueError):
                continue
            reused_group_ids.add(group_id)

        self.logger.info(
            "Resume scan: reusable Demucs segments=%d/%d from %s",
            len(reusable_segments),
            len(timeline),
            manifest_path,
        )
        if ignored_segments:
            self.logger.info(
                "Resume scan ignored %d existing segment entrie(s) due to errors, missing files, or metadata mismatch.",
                ignored_segments,
            )

        next_group_id = max(reused_group_ids, default=-1) + 1
        return reusable_segments, reusable_merge_groups, next_group_id

    def _make_segment_entry(self, seg: dict) -> dict:
        return {
            "segment_id": seg["segment_id"],
            "original_label": seg["label"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": _segment_duration(seg),
        }

    def _apply_merge_metadata(
        self,
        seg_entry: dict,
        group: SeparationGroup,
        member_index: Optional[int] = None,
        merge_bridge: bool = False,
        merge_fallback: bool = False,
    ) -> None:
        seg_entry["merge_group_id"] = group.group_id
        seg_entry["merge_group_size"] = len(group.member_indices)
        seg_entry["merge_group_start"] = group.start_s
        seg_entry["merge_group_end"] = group.end_s
        seg_entry["merge_extract_start"] = group.extract_start_s
        seg_entry["merge_extract_end"] = group.extract_end_s
        if member_index is not None:
            seg_entry["merge_member_index"] = member_index
        if merge_bridge:
            seg_entry["merge_bridge"] = True
        if merge_fallback:
            seg_entry["merge_group_fallback"] = True

    def _queue_passthrough_segment(
        self,
        seg_entry: dict,
        seg_id: int,
        vocals_np: np.ndarray,
        sample_rate: int,
        output_dir: Path,
        segments_dir: Path,
        write_executor,
        pending_writes: list[PendingSegmentWrite],
    ) -> None:
        dest = segments_dir / f"{seg_id:04d}_vocals.wav"
        pending_writes.append(
            PendingSegmentWrite(
                entry=seg_entry,
                success_action="passthrough",
                writes=[
                    PendingWrite(
                        future=write_executor.submit(
                            sf.write,
                            str(dest),
                            vocals_np,
                            sample_rate,
                        ),
                        path_field="vocals_path",
                        relative_path=_manifest_relative_path(dest, output_dir),
                        extra_fields={
                            "vocals_rms": _compute_rms(vocals_np, "passthrough audio"),
                        },
                    )
                ],
            )
        )

    def _queue_separated_segment(
        self,
        seg_entry: dict,
        seg_id: int,
        stems_tensors: dict[str, torch.Tensor],
        sample_rate: int,
        output_dir: Path,
        segments_dir: Path,
        write_executor,
        pending_writes: list[PendingSegmentWrite],
    ) -> None:
        vocals_tensor = stems_tensors["vocals"]
        vocals_np = _to_mono_numpy(vocals_tensor, "vocals output")
        vocals_path = segments_dir / f"{seg_id:04d}_vocals.wav"
        segment_writes = [
            PendingWrite(
                future=write_executor.submit(
                    sf.write,
                    str(vocals_path),
                    vocals_np,
                    sample_rate,
                ),
                path_field="vocals_path",
                relative_path=_manifest_relative_path(vocals_path, output_dir),
                extra_fields={
                    "vocals_rms": _compute_rms(vocals_np, "vocals output"),
                },
            )
        ]
        if self.config.save_accompaniment:
            for stem_name, stem_audio in stems_tensors.items():
                if stem_name == "vocals":
                    continue
                stem_np = _to_mono_numpy(stem_audio, f"{stem_name} output")
                stem_path = segments_dir / f"{seg_id:04d}_{stem_name}.wav"
                segment_writes.append(
                    PendingWrite(
                        future=write_executor.submit(
                            sf.write,
                            str(stem_path),
                            stem_np,
                            sample_rate,
                        ),
                        path_field=f"{stem_name}_path",
                        relative_path=_manifest_relative_path(stem_path, output_dir),
                        extra_fields={
                            f"{stem_name}_rms": _compute_rms(
                                stem_np,
                                f"{stem_name} output",
                            )
                        },
                    )
                )
        pending_writes.append(
            PendingSegmentWrite(
                entry=seg_entry,
                success_action="separated",
                writes=segment_writes,
            )
        )

    def _process_single_segment(
        self,
        seg: dict,
        container,
        audio_stream,
        output_dir: Path,
        segments_dir: Path,
        output_sample_rate: int,
        write_executor,
        pending_writes: list[PendingSegmentWrite],
        result_segments: list[dict],
        merge_group: Optional[SeparationGroup] = None,
        merge_member_index: Optional[int] = None,
        merge_bridge: bool = False,
        merge_fallback: bool = False,
    ) -> None:
        config = self.config
        seg_id = seg["segment_id"]
        label = seg["label"]
        start_s = seg["start"]
        end_s = seg["end"]
        duration_s = _segment_duration(seg)

        seg_entry = self._make_segment_entry(seg)
        if merge_group is not None:
            self._apply_merge_metadata(
                seg_entry,
                merge_group,
                member_index=merge_member_index,
                merge_bridge=merge_bridge,
                merge_fallback=merge_fallback,
            )

        if duration_s < MIN_SEGMENT_DURATION:
            seg_entry["enhanced_action"] = "skipped"
            seg_entry["skip_reason"] = "too_short"
            result_segments.append(seg_entry)
            return

        if label in config.labels_to_skip:
            seg_entry["enhanced_action"] = "skipped"
            result_segments.append(seg_entry)
            return

        try:
            if label in config.labels_to_passthrough:
                wav = _extract_audio_av(
                    container,
                    audio_stream,
                    start_s,
                    duration_s,
                    sample_rate=output_sample_rate,
                    channels=1,
                )
                vocals_np = _to_mono_numpy(wav, "passthrough audio")
                self._queue_passthrough_segment(
                    seg_entry,
                    seg_id,
                    vocals_np,
                    output_sample_rate,
                    output_dir,
                    segments_dir,
                    write_executor,
                    pending_writes,
                )
                self.logger.debug("Segment %d (%s): passthrough", seg_id, label)
            elif label in config.labels_to_separate:
                self.logger.info(
                    "Segment %d (%s): separating [%.3f - %.3f]",
                    seg_id, label, start_s, end_s,
                )
                wav = _extract_audio_av(
                    container,
                    audio_stream,
                    start_s,
                    duration_s,
                    sample_rate=DEMUCS_SAMPLE_RATE,
                    channels=2,
                )
                result = self.processor.separate(wav, output_sample_rate)
                self._queue_separated_segment(
                    seg_entry,
                    seg_id,
                    result["stems"],
                    result["sample_rate"],
                    output_dir,
                    segments_dir,
                    write_executor,
                    pending_writes,
                )
            else:
                self.logger.warning("Segment %d: unhandled label '%s', skipping", seg_id, label)
                seg_entry["enhanced_action"] = "skipped"
        except SegmentSkipError as e:
            self.logger.info("Segment %d skipped: %s", seg_id, e)
            seg_entry["enhanced_action"] = "skipped"
            seg_entry["skip_reason"] = e.reason
        except Exception as e:
            self.logger.warning("Segment %d failed (%s): %s", seg_id, type(e).__name__, e)
            seg_entry["enhanced_action"] = "error"
            seg_entry["error"] = str(e)

        result_segments.append(seg_entry)

    def _build_separation_group(
        self,
        timeline: list[dict],
        start_index: int,
        group_id: int,
    ) -> Optional[SeparationGroup]:
        config = self.config
        if not config.merge_adjacent_separation:
            return None
        if timeline[start_index]["label"] not in config.labels_to_separate:
            return None

        member_indices = [start_index]
        bridge_indices: list[int] = []
        pending_bridge_indices: list[int] = []
        first_start = float(timeline[start_index]["start"])
        span_end = float(timeline[start_index]["end"])
        pending_bridge_span = 0.0
        cursor = start_index + 1

        while cursor < len(timeline):
            seg = timeline[cursor]
            label = seg["label"]
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            gap = max(0.0, seg_start - span_end)

            if label in config.merge_bridge_labels:
                candidate_bridge_span = pending_bridge_span + gap + _segment_duration(seg)
                if candidate_bridge_span <= config.merge_gap_s:
                    pending_bridge_indices.append(cursor)
                    pending_bridge_span = candidate_bridge_span
                    span_end = seg_end
                    cursor += 1
                    continue
                break

            if label in config.labels_to_separate:
                candidate_gap = pending_bridge_span + gap
                candidate_duration = seg_end - first_start
                if (
                    candidate_gap <= config.merge_gap_s
                    and candidate_duration <= config.merge_max_duration_s
                ):
                    bridge_indices.extend(pending_bridge_indices)
                    pending_bridge_indices = []
                    pending_bridge_span = 0.0
                    member_indices.append(cursor)
                    span_end = seg_end
                    cursor += 1
                    continue

            break

        if len(member_indices) < config.merge_min_segments:
            return None

        start_s = float(timeline[member_indices[0]]["start"])
        end_s = float(timeline[member_indices[-1]]["end"])
        extract_start_s = max(0.0, start_s - config.merge_context_s)
        extract_end_s = end_s + config.merge_context_s
        return SeparationGroup(
            group_id=group_id,
            start_index=start_index,
            end_index=member_indices[-1],
            member_indices=member_indices,
            bridge_indices=bridge_indices,
            start_s=start_s,
            end_s=end_s,
            extract_start_s=extract_start_s,
            extract_end_s=extract_end_s,
        )

    def _fallback_group_to_individual(
        self,
        group: SeparationGroup,
        timeline: list[dict],
        container,
        audio_stream,
        output_dir: Path,
        segments_dir: Path,
        output_sample_rate: int,
        write_executor,
        pending_writes: list[PendingSegmentWrite],
        result_segments: list[dict],
    ) -> None:
        member_positions = {idx: pos for pos, idx in enumerate(group.member_indices)}
        bridge_index_set = set(group.bridge_indices)
        for idx in range(group.start_index, group.end_index + 1):
            self._process_single_segment(
                timeline[idx],
                container,
                audio_stream,
                output_dir,
                segments_dir,
                output_sample_rate,
                write_executor,
                pending_writes,
                result_segments,
                merge_group=group,
                merge_member_index=member_positions.get(idx),
                merge_bridge=idx in bridge_index_set,
                merge_fallback=True,
            )

    def _process_group(
        self,
        group: SeparationGroup,
        timeline: list[dict],
        container,
        audio_stream,
        output_dir: Path,
        segments_dir: Path,
        output_sample_rate: int,
        write_executor,
        pending_writes: list[PendingSegmentWrite],
        result_segments: list[dict],
        merge_groups: list[dict],
    ) -> None:
        config = self.config
        merge_record = {
            "group_id": group.group_id,
            "status": "pending",
            "start": group.start_s,
            "end": group.end_s,
            "extract_start": group.extract_start_s,
            "extract_end": group.extract_end_s,
            "member_segment_ids": [
                timeline[idx]["segment_id"] for idx in group.member_indices
            ],
            "bridge_segment_ids": [
                timeline[idx]["segment_id"] for idx in group.bridge_indices
            ],
        }

        self.logger.info(
            "Merge group %d: separating %d segment(s) [%.3f - %.3f]",
            group.group_id,
            len(group.member_indices),
            group.start_s,
            group.end_s,
        )

        try:
            wav = _extract_audio_av(
                container,
                audio_stream,
                group.extract_start_s,
                group.extract_end_s - group.extract_start_s,
                sample_rate=DEMUCS_SAMPLE_RATE,
                channels=2,
            )
            result = self.processor.separate(wav, output_sample_rate)
        except Exception as e:
            merge_record["status"] = "fallback_individual"
            merge_record["error"] = str(e)
            merge_groups.append(merge_record)
            self.logger.warning(
                "Merge group %d failed (%s), falling back to per-segment processing: %s",
                group.group_id,
                type(e).__name__,
                e,
            )
            self._fallback_group_to_individual(
                group,
                timeline,
                container,
                audio_stream,
                output_dir,
                segments_dir,
                output_sample_rate,
                write_executor,
                pending_writes,
                result_segments,
            )
            return

        merge_record["status"] = "grouped"
        merge_groups.append(merge_record)
        member_positions = {idx: pos for pos, idx in enumerate(group.member_indices)}
        bridge_index_set = set(group.bridge_indices)

        for idx in range(group.start_index, group.end_index + 1):
            seg = timeline[idx]
            seg_entry = self._make_segment_entry(seg)
            if idx in member_positions:
                self._apply_merge_metadata(
                    seg_entry,
                    group,
                    member_index=member_positions[idx],
                )
                try:
                    stem_slices = {}
                    for stem_name, stem_audio in result["stems"].items():
                        stem_slices[stem_name] = _slice_audio_window(
                            stem_audio,
                            result["sample_rate"],
                            seg["start"],
                            seg["end"],
                            group.extract_start_s,
                            f"group {group.group_id} {stem_name}",
                        )
                    self._queue_separated_segment(
                        seg_entry,
                        seg["segment_id"],
                        stem_slices,
                        result["sample_rate"],
                        output_dir,
                        segments_dir,
                        write_executor,
                        pending_writes,
                    )
                except SegmentSkipError as e:
                    self.logger.info(
                        "Segment %d skipped after merged separation: %s",
                        seg["segment_id"],
                        e,
                    )
                    seg_entry["enhanced_action"] = "skipped"
                    seg_entry["skip_reason"] = e.reason
                except Exception as e:
                    self.logger.warning(
                        "Segment %d failed after merged separation: %s",
                        seg["segment_id"],
                        e,
                    )
                    seg_entry["enhanced_action"] = "error"
                    seg_entry["error"] = str(e)
            else:
                self._apply_merge_metadata(
                    seg_entry,
                    group,
                    merge_bridge=idx in bridge_index_set,
                )
                seg_entry["enhanced_action"] = "skipped"
                seg_entry["skip_reason"] = "merge_bridge"

            result_segments.append(seg_entry)

    def process(self, segments_json_path: str, output_dir: str,
                input_path: str | None = None,
                progress_callback: Callable[..., None] | None = None) -> dict:
        """Process FireRedVAD segments using Demucs on the original media file."""
        def report_progress(event_type: str, **payload) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event_type, **payload)
            except Exception:
                return

        segments_json_path = Path(segments_json_path)
        output_dir = Path(output_dir)
        segments_dir = output_dir / "segments"
        manifest_path = output_dir / f"{segments_json_path.stem}.enhanced.json"
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

        self.logger.extra["prefix"] = f"[{Path(str(input_path)).name}] "
        self.logger.info("Source media: %s", input_path)

        config = self.config
        timeline = source["timeline"]
        result_segments: list[dict] = []
        merge_groups: list[dict] = []
        reusable_segments: dict[int, dict] = {}
        reusable_merge_groups: dict[int, dict] = {}
        reused_merge_group_ids: set[int] = set()
        resumed_segments = 0
        processed_segments = 0
        pending_error_segments = 0

        container = av.open(str(input_path))
        audio_stream = next(iter(container.streams.audio), None)
        if audio_stream is None:
            container.close()
            raise ValueError(f"No audio stream found: {input_path}")
        resolved_output_sample_rate = _resolve_output_sample_rate(
            audio_stream,
            config.output_sample_rate,
        )
        reusable_segments, reusable_merge_groups, next_group_id = self._load_resume_state(
            manifest_path=manifest_path,
            timeline=timeline,
            output_dir=output_dir,
            resolved_output_sample_rate=resolved_output_sample_rate,
        )
        report_progress(
            "demucs_started",
            total_segments=len(timeline),
        )

        write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        pending_writes: list[PendingSegmentWrite] = []

        try:
            index = 0
            while index < len(timeline):
                group = self._build_separation_group(timeline, index, next_group_id)
                if group is not None:
                    group_segment_ids = [
                        int(timeline[group_index]["segment_id"])
                        for group_index in range(group.start_index, group.end_index + 1)
                    ]
                    if all(segment_id in reusable_segments for segment_id in group_segment_ids):
                        group_id = reusable_segments[group_segment_ids[0]].get("merge_group_id")
                        try:
                            resolved_group_id = int(group_id)
                        except (TypeError, ValueError):
                            resolved_group_id = None
                        if (
                            resolved_group_id is not None
                            and resolved_group_id in reusable_merge_groups
                            and resolved_group_id not in reused_merge_group_ids
                        ):
                            merge_groups.append(copy.deepcopy(reusable_merge_groups[resolved_group_id]))
                            reused_merge_group_ids.add(resolved_group_id)
                        for segment_id in group_segment_ids:
                            result_segments.append(copy.deepcopy(reusable_segments[segment_id]))
                            resumed_segments += 1
                        if progress_callback is None:
                            self.logger.info(
                                "Resume reused merge group %s with %d segment(s) [%.3f - %.3f]",
                                resolved_group_id if resolved_group_id is not None else "?",
                                len(group_segment_ids),
                                group.start_s,
                                group.end_s,
                            )
                        else:
                            self.logger.debug(
                                "Resume reused merge group %s with %d segment(s) [%.3f - %.3f]",
                                resolved_group_id if resolved_group_id is not None else "?",
                                len(group_segment_ids),
                                group.start_s,
                                group.end_s,
                            )
                        report_progress(
                            "demucs_progress",
                            total_segments=len(timeline),
                            completed_segments=len(result_segments),
                            resumed_segments=resumed_segments,
                            processed_segments=processed_segments,
                            error_segments=pending_error_segments,
                        )
                        index = group.end_index + 1
                        continue
                    before_len = len(result_segments)
                    self._process_group(
                        group,
                        timeline,
                        container,
                        audio_stream,
                        output_dir,
                        segments_dir,
                        resolved_output_sample_rate,
                        write_executor,
                        pending_writes,
                        result_segments,
                        merge_groups,
                    )
                    new_entries = result_segments[before_len:]
                    processed_segments += len(new_entries)
                    pending_error_segments += sum(
                        1 for entry in new_entries if entry.get("enhanced_action") == "error"
                    )
                    report_progress(
                        "demucs_progress",
                        total_segments=len(timeline),
                        completed_segments=len(result_segments),
                        resumed_segments=resumed_segments,
                        processed_segments=processed_segments,
                        error_segments=pending_error_segments,
                    )
                    next_group_id += 1
                    index = group.end_index + 1
                    continue

                segment_id = int(timeline[index]["segment_id"])
                if segment_id in reusable_segments:
                    resumed_entry = copy.deepcopy(reusable_segments[segment_id])
                    result_segments.append(resumed_entry)
                    resumed_segments += 1
                    if progress_callback is None:
                        self.logger.info(
                            "Resume reused segment %d (%s)",
                            segment_id,
                            resumed_entry.get("enhanced_action"),
                        )
                    else:
                        self.logger.debug(
                            "Resume reused segment %d (%s)",
                            segment_id,
                            resumed_entry.get("enhanced_action"),
                        )
                    report_progress(
                        "demucs_progress",
                        total_segments=len(timeline),
                        completed_segments=len(result_segments),
                        resumed_segments=resumed_segments,
                        processed_segments=processed_segments,
                        error_segments=pending_error_segments,
                    )
                    index += 1
                    continue

                before_len = len(result_segments)
                self._process_single_segment(
                    timeline[index],
                    container,
                    audio_stream,
                    output_dir,
                    segments_dir,
                    resolved_output_sample_rate,
                    write_executor,
                    pending_writes,
                    result_segments,
                )
                new_entries = result_segments[before_len:]
                processed_segments += len(new_entries)
                pending_error_segments += sum(
                    1 for entry in new_entries if entry.get("enhanced_action") == "error"
                )
                report_progress(
                    "demucs_progress",
                    total_segments=len(timeline),
                    completed_segments=len(result_segments),
                    resumed_segments=resumed_segments,
                    processed_segments=processed_segments,
                    error_segments=pending_error_segments,
                )
                index += 1

        finally:
            for pending in pending_writes:
                errors = []
                for write in pending.writes:
                    try:
                        write.future.result()
                    except Exception as e:
                        errors.append(f"{write.path_field}: {e}")
                    else:
                        pending.entry[write.path_field] = write.relative_path
                        pending.entry.update(write.extra_fields)
                if errors:
                    self.logger.warning(
                        "Segment %s WAV write failed: %s",
                        pending.entry["segment_id"],
                        "; ".join(errors),
                    )
                    pending.entry["enhanced_action"] = "error"
                    pending.entry["error"] = "; ".join(errors)
                else:
                    pending.entry["enhanced_action"] = pending.success_action
            write_executor.shutdown(wait=True)
            container.close()
            counts = Counter(seg.get("enhanced_action", "unknown") for seg in result_segments)
            self.logger.info(
                "Processing complete: %d newly processed, %d resumed, %d total. Actions: %s",
                processed_segments,
                resumed_segments,
                len(result_segments),
                ", ".join(f"{action}={count}" for action, count in sorted(counts.items())),
            )
            report_progress(
                "demucs_summary",
                total_segments=len(result_segments),
                resumed_segments=resumed_segments,
                processed_segments=processed_segments,
                error_segments=counts.get("error", 0),
                counts=dict(counts),
            )

            manifest = {
                "source_manifest": str(segments_json_path),
                "input_path": str(input_path),
                "demucs_model": config.demucs_model,
                "stem_mode": config.stem_mode,
                "output_sample_rate": resolved_output_sample_rate,
                "config": dataclasses.asdict(config),
                "counts": dict(counts),
                "resume": {
                    "enabled": bool(config.resume),
                    "reused_segments": resumed_segments,
                    "processed_segments": max(0, len(result_segments) - resumed_segments),
                },
                "merge_groups": merge_groups,
                "segments": result_segments,
                "manifest_path": str(manifest_path),
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            self.logger.info("Enhanced manifest saved to %s", manifest_path)

        return manifest
