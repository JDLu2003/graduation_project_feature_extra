from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

@dataclass(frozen=True)
class PersonEntry:
    """Represents a person (speaker or listener) in an utterance."""
    name: str
    emotion: str
    emotion_score: int

@dataclass(frozen=True)
class UtteranceRecord:
    """Represents a single utterance (sentence) in a dialogue."""
    dialogue_id: int
    utterance_idx: int
    text_content: str
    speaker: PersonEntry
    listeners: List[PersonEntry] = field(default_factory=list)

@dataclass(frozen=True)
class DialogueRecord:
    """Represents a full dialogue, containing its ID, total utterances, and a list of utterances."""
    dialogue_id: int
    total_utterances: int
    utterances: List[UtteranceRecord] = field(default_factory=list)

def parse_dev_txt(file_path: Path) -> List[DialogueRecord]:
    """
    Parses the dev.txt file and returns a list of DialogueRecord objects.

    Args:
        file_path: The path to the dev.txt file.

    Returns:
        A list of DialogueRecord objects.

    Raises:
        AssertionError: If the file format does not strictly adhere to the specification.
    """
    assert file_path.exists(), f"dev.txt file not found: {file_path}"

    dialogues: List[DialogueRecord] = []
    current_dialogue: DialogueRecord | None = None
    expected_utterances_count: int = 0
    actual_utterances_count: int = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts_pipe = line.split(' | ')
            parts_space = line.split(' ')

            is_dialogue_header = False
            if len(parts_space) == 2 and parts_space[0].isdigit() and parts_space[1].isdigit():
                is_dialogue_header = True

            if is_dialogue_header:
                # Dialogue Header: "dialogue_id total_utterances"
                if current_dialogue:
                    # Assert that previous dialogue had the expected number of utterances
                    assert actual_utterances_count == expected_utterances_count, \
                        f"Dialogue {current_dialogue.dialogue_id} (line {line_num - actual_utterances_count}-{line_num-1}): " \
                        f"Expected {expected_utterances_count} utterances, but found {actual_utterances_count}. Last line: {line}"
                    dialogues.append(current_dialogue)

                dialogue_id = int(parts_space[0])
                total_utterances = int(parts_space[1])

                current_dialogue = DialogueRecord(dialogue_id=dialogue_id, total_utterances=total_utterances)
                expected_utterances_count = total_utterances
                actual_utterances_count = 0
            else:
                # Utterance Line: "utterance_idx | text | speaker_name | speaker_emotion | listener1_name | listener1_emotion | ..."
                assert current_dialogue is not None, \
                    f"Line {line_num}: Utterance found before any dialogue header: {line}"
                assert len(parts_pipe) >= 4, \
                    f"Line {line_num}: Malformed utterance line (too few fields): {line}"

                utterance_idx = int(parts_pipe[0])
                text_content = parts_pipe[1]
                speaker_name = parts_pipe[2]
                speaker_emotion_str = parts_pipe[3]

                # Parse speaker emotion
                assert '(' in speaker_emotion_str and ')' in speaker_emotion_str, \
                    f"Line {line_num}: Malformed speaker emotion format: {speaker_emotion_str}"
                speaker_emotion_parts = speaker_emotion_str.split('(')
                speaker_emotion = speaker_emotion_parts[0].strip()
                speaker_emotion_score_str = speaker_emotion_parts[1].replace(')', '').strip()
                speaker_emotion_score = int(speaker_emotion_score_str) if speaker_emotion_score_str.isdigit() else -1 # -1 for null

                speaker = PersonEntry(name=speaker_name, emotion=speaker_emotion, emotion_score=speaker_emotion_score)

                listeners: List[PersonEntry] = []
                for i in range(4, len(parts_pipe), 2):
                    assert i + 1 < len(parts_pipe), \
                        f"Line {line_num}: Malformed listener entry (missing emotion for {parts_pipe[i]}): {line}"
                    listener_name = parts_pipe[i]
                    listener_emotion_str = parts_pipe[i+1]

                    # Parse listener emotion
                    # Handle cases like "neutral ()" where there's no score
                    if '(' in listener_emotion_str and ')' in listener_emotion_str:
                        listener_emotion_parts = listener_emotion_str.split('(')
                        listener_emotion = listener_emotion_parts[0].strip()
                        listener_emotion_score_str = listener_emotion_parts[1].replace(')', '').strip()
                        listener_emotion_score = int(listener_emotion_score_str) if listener_emotion_score_str.isdigit() else -1
                    else:
                        listener_emotion = listener_emotion_str.strip()
                        listener_emotion_score = -1 # Default to -1 if no score found

                    listeners.append(PersonEntry(name=listener_name, emotion=listener_emotion, emotion_score=listener_emotion_score))

                # Assert utterance index matches expected
                assert utterance_idx == actual_utterances_count + 1, \
                    f"Line {line_num}: Expected utterance index {actual_utterances_count + 1}, but got {utterance_idx}. Line: {line}"

                current_dialogue.utterances.append(UtteranceRecord(
                    dialogue_id=current_dialogue.dialogue_id,
                    utterance_idx=utterance_idx,
                    text_content=text_content,
                    speaker=speaker,
                    listeners=listeners
                ))
                actual_utterances_count += 1


    # After loop, add the last dialogue if it exists
    if current_dialogue:
        assert actual_utterances_count == expected_utterances_count, \
            f"Dialogue {current_dialogue.dialogue_id}: Expected {expected_utterances_count} utterances, " \
            f"but found {actual_utterances_count}."
        dialogues.append(current_dialogue)

    return dialogues

# Example usage (for testing purposes)
if __name__ == "__main__":
    from pathlib import Path
    import sys
    # Add parent directory to path to import config module
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig

    # Assume config.yaml is in the parent directory
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        dev_txt_path = app_config.paths.dev_txt
        video_base_dir = app_config.paths.video_dir # Correctly define video_base_dir

        print(f"Attempting to parse dev.txt from: {dev_txt_path}")

        dialogue_records = parse_dev_txt(dev_txt_path)
        print(f"Parsing successful! Found {len(dialogue_records)} dialogues.")

        total_utterances = 0
        video_files_found = 0
        video_files_missing = 0
        missing_video_details: List[str] = []

        for dialogue in dialogue_records:
            for utterance in dialogue.utterances:
                total_utterances += 1
                # Construct video file path based on CLAUDE.md specification
                # Format: Video_en_dev/C_{对话编号}/C_{对话编号}_U_{句子编号}.mp4
                # Note: video_base_dir points to `../data_set/Viedeo_en_dev/Video_en_dev`
                dialogue_folder_name = f"C_{utterance.dialogue_id}"
                video_file_name = f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.mp4"

                video_path = video_base_dir / dialogue_folder_name / video_file_name

                if video_path.exists():
                    video_files_found += 1
                else:
                    video_files_missing += 1
                    missing_video_details.append(f"Dialogue {utterance.dialogue_id}, Utterance {utterance.utterance_idx}: {video_path}")

        print(f"\n--- Video File Verification Summary ---")
        print(f"Total utterances processed: {total_utterances}")
        print(f"Video files found: {video_files_found}")
        print(f"Video files missing: {video_files_missing}")

        if video_files_missing > 0:
            print("\n!!! WARNING: Some video files are missing. !!!")
            for detail in missing_video_details:
                print(f"  - {detail}")
            print("\nPlease ensure all video files exist at the specified paths in config.yaml.")
            sys.exit(1)
        else:
            print("\nAll required video files found. Parser validation successful!")

    except AssertionError as e:
        print(f"Parsing failed due to assertion error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file specified in config.yaml ({dev_txt_path}) or a video file was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    except AssertionError as e:
        print(f"Parsing failed due to assertion error: {e}")
    except FileNotFoundError:
        print(f"Error: The file specified in config.yaml ({dev_txt_path}) was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
