import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import torch # Import torch for feature tensors

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent)) # sys.path already points to parent, remove .parent twice

from src.config import AppConfig
from src.parser import parse_dev_txt, DialogueRecord, UtteranceRecord
from src.context_index import build_context_index
from src.video_utils import sample_frames # Not directly called here, but good to import if needed for extractors
from src.extractors.base import FeatureExtractor # For type hinting
from src.extractors.clip_extractor import CLIPVisualExtractor # Concrete extractor for instantiation

def run_pipeline(config: AppConfig):
    """
    Orchestrates the feature extraction pipeline.
    This function outlines the full flow using module interfaces.
    """
    print(f"--- Starting Feature Extraction Pipeline ---")
    print(f"Loading configuration from: {config.paths.dev_txt}")

    # 1. Parse dev.txt
    print(f"Parsing dev.txt from: {config.paths.dev_txt}...")
    dialogue_records: List[DialogueRecord] = parse_dev_txt(config.paths.dev_txt)
    print(f"Successfully parsed {len(dialogue_records)} dialogues.")

    # 2. Build context index for non-speaker video lookup
    print(f"Building context index for non-speaker feature inference...")
    context_index: Dict[Tuple[int, str], List[Path]] = build_context_index(
        dialogue_records, config.paths.video_dir
    )
    print(f"Context index built with {len(context_index)} entries.")

    # 3. Initialize Feature Extractor
    # Based on config.extractor.active_type, instantiate the correct extractor.
    # Currently, only CLIPVisualExtractor is implemented as a skeleton.
    extractor: FeatureExtractor
    if config.extractor.active_type == "visual_clip":
        print(f"Initializing CLIPVisualExtractor...")
        assert config.extractor.visual_clip_config is not None, "Visual CLIP config is missing."
        extractor = CLIPVisualExtractor(config.extractor.visual_clip_config)
    else:
        raise ValueError(f"Unsupported active extractor type: {config.extractor.active_type}")
    print(f"Extractor initialized with output dimension: {extractor.output_dim}")

    # Ensure output feature directory exists
    config.paths.feat_out.mkdir(parents=True, exist_ok=True)

    # 4. Iterate through utterances, extract features for speakers and listeners, then save
    print(f"Starting feature extraction and saving...")
    total_utterances_processed = 0
    total_features_saved = 0

    for dialogue in dialogue_records:
        for utterance in dialogue.utterances:
            total_utterances_processed += 1
            all_persons_in_utterance = [utterance.speaker] + utterance.listeners
            utterance_features: List[torch.Tensor] = []

            # Placeholder for storing video paths for the current utterance
            # This is needed to get the speaker's video
            dialogue_folder_name = f"C_{utterance.dialogue_id}"
            speaker_video_path = config.paths.video_dir / dialogue_folder_name / f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.mp4"

            # 4.1. Extract speaker features (index 0)
            # This is a full implementation call to the extractor.
            assert speaker_video_path.exists(), f"Speaker video not found: {speaker_video_path}"
            speaker_feature = extractor.extract(speaker_video_path)
            utterance_features.append(speaker_feature)

            # 4.2. Extract listener features (index 1 to N-1)
            for listener in utterance.listeners:
                listener_feature: torch.Tensor
                if config.non_speaker.strategy == "context_video":
                    # Placeholder for getting context videos for listener
                    # In actual implementation, we'd use context_index to find
                    # videos where 'listener' was a speaker in this dialogue.
                    # For now, we'll use a zero vector as a placeholder.
                    # TODO: Implement context_video logic in detail
                    listener_feature = torch.zeros(1, extractor.output_dim, device=extractor.config.device)
                elif config.non_speaker.fallback == "zero":
                    listener_feature = torch.zeros(1, extractor.output_dim, device=extractor.config.device)
                else:
                    raise ValueError(f"Unsupported non_speaker strategy/fallback: {config.non_speaker.strategy}/{config.non_speaker.fallback}")
                utterance_features.append(listener_feature)

            # 4.3. Combine all person features for this utterance
            combined_features = torch.cat(utterance_features, dim=0) # Shape: [N, target_dim]
            assert combined_features.shape[0] == len(all_persons_in_utterance), \
                f"Feature count mismatch for dialogue {dialogue.dialogue_id}, utterance {utterance.utterance_idx}: " \
                f"Expected {len(all_persons_in_utterance)} features, got {combined_features.shape[0]}"
            assert combined_features.shape[1] == extractor.output_dim, \
                f"Feature dimension mismatch: Expected {extractor.output_dim}, got {combined_features.shape[1]}"

            # 4.4. Define output path and save features
            output_dialogue_folder = config.paths.feat_out / dialogue_folder_name
            output_dialogue_folder.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dialogue_folder / f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.pt"

            if config.pipeline.skip_existing and output_file_path.exists():
                # print(f"Skipping existing feature file: {output_file_path}")
                pass
            else:
                torch.save(combined_features, output_file_path)
                total_features_saved += 1
                # Optional: Load back and verify (as per CLAUDE.md)
                # loaded_features = torch.load(output_file_path)
                # assert torch.equal(combined_features, loaded_features), "Feature save/load verification failed!"


    print(f"--- Pipeline Finished ---")
    print(f"Total utterances processed: {total_utterances_processed}")
    print(f"Total feature files saved: {total_features_saved}")
    print(f"Feature output directory: {config.paths.feat_out}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Dialogue Feature Extraction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        app_config = AppConfig.from_yaml(args.config)
        print(f"Configuration loaded successfully from {args.config}")
        # print(app_config) # Keep this commented for cleaner output
    except AssertionError as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

    # Run the pipeline
    run_pipeline(app_config)


if __name__ == "__main__":
    main()
