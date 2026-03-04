from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Use relative import for inter-package modules
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
