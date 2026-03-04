from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.parser import DialogueRecord, UtteranceRecord

def build_context_index(
    dialogue_records: List[DialogueRecord], video_base_dir: Path
) -> Dict[Tuple[int, str], List[Path]]:
    """
    Builds an index that maps (dialogue_id, person_name) to a list of video paths
    where that person appears as a speaker within that specific dialogue.

    This index is used to infer non-speaker features from available speaker videos
    within the same dialogue context.

    Args:
        dialogue_records: A list of parsed DialogueRecord objects.
        video_base_dir: The base directory where video files are located.

    Returns:
        A dictionary where keys are (dialogue_id, person_name) tuples and values
        are lists of Path objects pointing to the video files where the person
        is the speaker in that dialogue.
    """
    context_index: Dict[Tuple[int, str], List[Path]] = defaultdict(list)

    for dialogue in dialogue_records:
        for utterance in dialogue.utterances:
            speaker_name = utterance.speaker.name
            dialogue_id = utterance.dialogue_id

            # Construct the video path for the current utterance's speaker
            dialogue_folder_name = f"C_{dialogue_id}"
            video_file_name = f"C_{dialogue_id}_U_{utterance.utterance_idx}.mp4"
            video_path = video_base_dir / dialogue_folder_name / video_file_name

            # Add this video path to the index for the current speaker in this dialogue
            # Only add if the video file actually exists
            if video_path.exists():
                context_index[(dialogue_id, speaker_name)].append(video_path)

    return dict(context_index)

# Example usage (for testing purposes, will be removed later or moved to main.py)
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig
    from src.parser import parse_dev_txt

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        dev_txt_path = app_config.paths.dev_txt
        video_base_dir = app_config.paths.video_dir

        print(f"Attempting to parse dev.txt from: {dev_txt_path}")
        dialogue_records = parse_dev_txt(dev_txt_path)
        print(f"Successfully parsed {len(dialogue_records)} dialogues.")

        print(f"\nBuilding context index using video base directory: {video_base_dir}")
        context_idx = build_context_index(dialogue_records, video_base_dir)
        print(f"Context index built. Found {len(context_idx)} unique (dialogue_id, person_name) entries.")

        # Print a few examples from the context index
        print("\n--- Sample Context Index Entries ---")
        count = 0
        for (dialogue_id, person_name), video_paths in context_idx.items():
            if count >= 5: # Print first 5 entries
                break
            print(f"  ({dialogue_id}, '{person_name}'): {len(video_paths)} videos, e.g., {video_paths[0].name if video_paths else 'N/A'}")
            count += 1

        # Test case: person not speaking in any video
        if (18, "SomeNonSpeaker") not in context_idx:
            print("\nSuccessfully confirmed 'SomeNonSpeaker' not in context index for Dialogue 18 (as expected).")

        print("\nContext index builder test successful!")

    except AssertionError as e:
        print(f"Test failed due to assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")
        sys.exit(1)