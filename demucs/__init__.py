# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "4.1.0a3"


def enhance_segments(segments_json_path, output_dir="out/enhanced", model="htdemucs_ft",
                     use_gpu=False, input_path=None):
    """Apply Demucs source separation to FireRedVAD pipeline segments.

    Args:
        segments_json_path: Path to FireRedVAD segments.json file.
        output_dir: Directory to save enhanced outputs.
        model: Demucs model name (default: htdemucs_ft).
        use_gpu: Use GPU for inference (default: False).
        input_path: Override source media file path.

    Returns:
        Enhanced manifest dict.
    """
    from demucs.enhanced_pipeline import EnhancedPipeline, EnhancedPipelineConfig

    config = EnhancedPipelineConfig(demucs_model=model, use_gpu=use_gpu)
    pipeline = EnhancedPipeline.from_config(config)
    return pipeline.process(segments_json_path, output_dir, input_path=input_path)
