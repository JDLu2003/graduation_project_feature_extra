import argparse
from pathlib import Path
import sys

# Add parent directory to path to import src modules (if running directly from project root)
# This might not be strictly necessary if installed as a package, but helpful for development.
# The original logic of appending parent directory twice is incorrect and has been fixed.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AppConfig
from src.parser import parse_dev_txt
from src.extractors.visual_clip.strategy import VisualCLIPStrategy # Import the concrete strategy
from src.pipeline import run_pipeline # Import the pipeline orchestrator

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
    except AssertionError as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

    # 1. Parse dev.txt to get dialogue records
    print(f"Parsing dev.txt from: {app_config.paths.dev_txt}...")
    dialogue_records = parse_dev_txt(app_config.paths.dev_txt)
    print(f"Successfully parsed {len(dialogue_records)} dialogues.")

    # 2. Instantiate the feature extraction strategy
    if app_config.extractor.active_type == "visual_clip":
        print(f"Initializing VisualCLIPStrategy...")
        assert app_config.extractor.visual_clip_config is not None, "Visual CLIP config is missing."
        extractor = VisualCLIPStrategy(app_config.extractor.visual_clip_config, app_config.non_speaker)
    else:
        raise ValueError(f"Unsupported active extractor type: {app_config.extractor.active_type}")
    print(f"Extractor initialized with output dimension: {extractor.output_dim}")

    # Ensure output feature directory exists
    app_config.paths.feat_out.mkdir(parents=True, exist_ok=True)

    # 3. Run the pipeline
    run_pipeline(app_config, extractor, dialogue_records)

if __name__ == "__main__":
    main()

