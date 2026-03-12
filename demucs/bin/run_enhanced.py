#!/usr/bin/env python3
import argparse
import logging

from demucs.enhanced_pipeline import EnhancedPipeline, EnhancedPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("demucs.bin.run_enhanced")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply Demucs source separation to FireRedVAD segments."
    )
    parser.add_argument(
        "--segments_json",
        required=True,
        help="Path to the FireRedVAD segments.json file.",
    )
    parser.add_argument(
        "--input_path",
        default=None,
        help="Source media file (video/audio). Overrides input_path in segments.json.",
    )
    parser.add_argument(
        "--output_dir",
        default="out/enhanced",
        help="Output directory (default: out/enhanced).",
    )
    parser.add_argument(
        "--model",
        default="htdemucs_ft",
        help="Demucs model name (default: htdemucs_ft).",
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=0,
        help="Use GPU for inference (0=CPU, 1=GPU). Default: 0.",
    )
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=16000,
        help="Output sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for prediction averaging (default: 1).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between splits (default: 0.25).",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Segment length in seconds. None uses model default.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel jobs (default: 0).",
    )
    parser.add_argument(
        "--no_accompaniment",
        action="store_true",
        help="Skip saving accompaniment stems.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--labels_to_separate",
        nargs="+",
        default=["speech_with_music", "singing", "uncertain"],
        help=(
            "Labels to apply Demucs separation to "
            "(default: speech_with_music singing uncertain)."
        ),
    )
    parser.add_argument(
        "--labels_to_passthrough",
        nargs="+",
        default=["speech"],
        help="Labels to copy as-is without separation (default: speech).",
    )
    parser.add_argument(
        "--labels_to_skip",
        nargs="+",
        default=["silence", "music"],
        help="Labels to skip entirely (default: silence music).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = EnhancedPipelineConfig(
        demucs_model=args.model,
        use_gpu=bool(args.use_gpu),
        output_sample_rate=args.output_sample_rate,
        save_accompaniment=not args.no_accompaniment,
        shifts=args.shifts,
        overlap=args.overlap,
        segment=args.segment,
        jobs=args.jobs,
        progress=not args.no_progress,
        labels_to_separate=tuple(args.labels_to_separate),
        labels_to_passthrough=tuple(args.labels_to_passthrough),
        labels_to_skip=tuple(args.labels_to_skip),
    )

    logger.info("Building pipeline with model=%s use_gpu=%s", config.demucs_model, config.use_gpu)
    pipeline = EnhancedPipeline.from_config(config)

    logger.info(
        "Processing segments_json=%s  output_dir=%s",
        args.segments_json,
        args.output_dir,
    )
    result = pipeline.process(args.segments_json, args.output_dir, input_path=args.input_path)

    # Log result summary.
    from collections import Counter
    counts = Counter(s["enhanced_action"] for s in result["segments"])
    total = len(result["segments"])
    logger.info(
        "Done. %d segment(s) total — separated: %d, passthrough: %d, skipped: %d",
        total,
        counts.get("separated", 0),
        counts.get("passthrough", 0),
        counts.get("skipped", 0),
    )


if __name__ == "__main__":
    main()
