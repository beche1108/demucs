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
        "--stem_mode",
        choices=["two", "four"],
        default="two",
        help="Stem output mode: `two` saves vocals + accompaniment, `four` saves individual non-vocal stems (default: two).",
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
        default=0,
        help="Output sample rate in Hz. Use 0 to match the input media sample rate (default: 0).",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=5,
        help="Number of random shifts for prediction averaging (default: 5).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help="Overlap between splits (default: 0.8).",
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
        help="Skip saving non-vocal stem outputs.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--no_merge_adjacent",
        action="store_true",
        help="Disable adjacent separation-segment merging.",
    )
    parser.add_argument(
        "--merge_gap_s",
        type=float,
        default=0.2,
        help="Maximum bridge gap duration to merge across (default: 0.2).",
    )
    parser.add_argument(
        "--merge_max_duration_s",
        type=float,
        default=20.0,
        help="Maximum merged separation window length in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--merge_min_segments",
        type=int,
        default=2,
        help="Minimum number of separation segments required to form a merge group.",
    )
    parser.add_argument(
        "--merge_context_s",
        type=float,
        default=0.25,
        help="Context padding to include around each merged group (default: 0.25).",
    )
    parser.add_argument(
        "--merge_bridge_labels",
        nargs="+",
        default=["silence", "music"],
        help="Labels allowed between merged separation segments (default: silence music).",
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
        stem_mode=args.stem_mode,
        use_gpu=bool(args.use_gpu),
        output_sample_rate=None if args.output_sample_rate == 0 else args.output_sample_rate,
        save_accompaniment=not args.no_accompaniment,
        shifts=args.shifts,
        overlap=args.overlap,
        segment=args.segment,
        jobs=args.jobs,
        progress=not args.no_progress,
        merge_adjacent_separation=not args.no_merge_adjacent,
        merge_gap_s=args.merge_gap_s,
        merge_max_duration_s=args.merge_max_duration_s,
        merge_min_segments=args.merge_min_segments,
        merge_context_s=args.merge_context_s,
        merge_bridge_labels=tuple(args.merge_bridge_labels),
        labels_to_separate=tuple(args.labels_to_separate),
        labels_to_passthrough=tuple(args.labels_to_passthrough),
        labels_to_skip=tuple(args.labels_to_skip),
    )

    logger.info(
        "Building pipeline with model=%s stem_mode=%s use_gpu=%s",
        config.demucs_model,
        config.stem_mode,
        config.use_gpu,
    )
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
    merge_counts = Counter(g["status"] for g in result.get("merge_groups", []))
    total = len(result["segments"])
    logger.info(
        "Done. %d segment(s) total - separated: %d, passthrough: %d, skipped: %d, errors: %d",
        total,
        counts.get("separated", 0),
        counts.get("passthrough", 0),
        counts.get("skipped", 0),
        counts.get("error", 0),
    )
    logger.info(
        "Merge groups: %d total - grouped: %d, fallback: %d",
        len(result.get("merge_groups", [])),
        merge_counts.get("grouped", 0),
        merge_counts.get("fallback_individual", 0),
    )
    if counts.get("error", 0) > 0:
        logger.warning("Some segments failed. Check the manifest for details.")


if __name__ == "__main__":
    main()
