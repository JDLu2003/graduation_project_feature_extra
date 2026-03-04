import argparse
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.config import AppConfig
from src.parser import parse_dev_txt, DialogueRecord

def run_pipeline(config: AppConfig):
    """
    Orchestrates the feature extraction pipeline.
    """
    print(f"--- Starting Feature Extraction Pipeline ---")
    print(f"Loading configuration from: {config.paths.dev_txt}")

    # 1. Parse dev.txt
    print(f"Parsing dev.txt from: {config.paths.dev_txt}...")
    dialogue_records: list[DialogueRecord] = parse_dev_txt(config.paths.dev_txt)
    print(f"Successfully parsed {len(dialogue_records)} dialogues.")

    # TODO: Further steps will be added here
    # 2. Build context index (for non-speaker video lookup)
    # 3. Initialize Feature Extractor (CLIP, projection, etc.)
    # 4. Iterate through utterances, extract features for speakers and listeners
    # 5. Save features

    print(f"--- Pipeline Finished ---")


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
        print(app_config)
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
