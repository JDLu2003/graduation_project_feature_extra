import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# 正则：匹配 "emotion_label (cause_idx_or_null)" 格式，容忍首尾空白与内部多余空格
# 括号内的值代表诱发该情绪的句子编号（utterance_idx），或 "null" 表示无明确诱因
_EMOTION_PATTERN = re.compile(r"^\s*(\w+)\s*\(\s*([^)]*?)\s*\)\s*$")


@dataclass(frozen=True)
class PersonEntry:
    """Represents a person (speaker or listener) in an utterance."""
    name: str
    emotion: str
    emotion_cause_idxs: List[int] = field(default_factory=list)  # 诱发该情绪的句子编号列表（utterance_idx），空列表表示 null (无明确诱因)


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


def _parse_emotion(emotion_str: str, line_num: int) -> tuple[str, List[int]]:
    """
    使用正则表达式解析情绪字段，如 "neutral (null)" 或 "joy (1)" 或 "anger (5, 6)"。
    括号内的值可以是：
    - "null" 或空：无明确诱因（返回空列表）
    - 单个整数：诱发该情绪的句子编号
    - 逗号分隔的整数列表：多个诱发句子编号

    Args:
        emotion_str: 原始情绪字符串，可能含多余空白或异常字符。
        line_num: 文件行号，用于错误信息定位。

    Returns:
        (emotion_label, emotion_cause_idxs) 元组。emotion_cause_idxs 为空列表表示 null 或无效值。

    Raises:
        ValueError: 若字符串完全无法匹配预期格式。
    """
    match = _EMOTION_PATTERN.match(emotion_str)
    if not match:
        raise ValueError(
            f"Line {line_num}: Malformed emotion string, cannot parse: '{emotion_str}'"
        )
    emotion_label = match.group(1).strip()
    cause_str = match.group(2).strip()

    emotion_cause_idxs: List[int] = []

    if cause_str == "null" or cause_str == "":
        # 无明确诱因，返回空列表
        emotion_cause_idxs = []
    else:
        # 尝试解析逗号分隔的索引列表
        parts = [p.strip() for p in cause_str.split(",")]
        valid_indices = []
        all_valid = True
        
        for part in parts:
            if part.isdigit():
                valid_indices.append(int(part))
            else:
                all_valid = False
                break
        
        if all_valid and valid_indices:
            emotion_cause_idxs = valid_indices
        else:
            # 无法识别的格式，打印警告并返回空列表
            print(f"[parser] Warning: Line {line_num}: Unrecognized emotion cause format '{cause_str}', defaulting to empty list.")
            emotion_cause_idxs = []

    return emotion_label, emotion_cause_idxs


def parse_dev_txt(file_path: Path) -> List[DialogueRecord]:
    """
    解析 dev.txt 文件，返回 DialogueRecord 列表。

    Args:
        file_path: dev.txt 文件路径。

    Returns:
        解析后的 DialogueRecord 列表。

    Raises:
        FileNotFoundError: 文件不存在时。
        ValueError: 文件格式不符合规范时（字段缺失、顺序错误等）。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"dev.txt file not found: {file_path}")

    dialogues: List[DialogueRecord] = []
    current_dialogue: Optional[DialogueRecord] = None
    expected_utterances_count: int = 0
    actual_utterances_count: int = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts_space = line.split(" ")
            is_dialogue_header = (
                len(parts_space) == 2
                and parts_space[0].isdigit()
                and parts_space[1].isdigit()
            )

            if is_dialogue_header:
                # --- 对话头：结算上一段对话 ---
                if current_dialogue is not None:
                    if actual_utterances_count != expected_utterances_count:
                        raise ValueError(
                            f"Line {line_num}: Dialogue {current_dialogue.dialogue_id} ended with "
                            f"{actual_utterances_count} utterances, expected {expected_utterances_count}."
                        )
                    
                    # 防御性断言：验证所有 emotion_cause_idx 的合法性
                    for utterance in current_dialogue.utterances:
                        # 验证说话人的情绪诱因索引
                        for cause_idx in utterance.speaker.emotion_cause_idxs:
                            if cause_idx != -1 and not (1 <= cause_idx <= expected_utterances_count):
                                raise ValueError(
                                    f"Dialogue {current_dialogue.dialogue_id}, Utterance {utterance.utterance_idx}: "
                                    f"Speaker '{utterance.speaker.name}' has invalid emotion_cause_idx "
                                    f"{cause_idx} (must be in range [1, {expected_utterances_count}] or -1)."
                                )
                        # 验证听众的情绪诱因索引
                        for listener in utterance.listeners:
                            for cause_idx in listener.emotion_cause_idxs:
                                if cause_idx != -1 and not (1 <= cause_idx <= expected_utterances_count):
                                    raise ValueError(
                                        f"Dialogue {current_dialogue.dialogue_id}, Utterance {utterance.utterance_idx}: "
                                        f"Listener '{listener.name}' has invalid emotion_cause_idx "
                                        f"{cause_idx} (must be in range [1, {expected_utterances_count}] or -1)."
                                    )
                    
                    dialogues.append(current_dialogue)

                dialogue_id = int(parts_space[0])
                total_utterances = int(parts_space[1])
                current_dialogue = DialogueRecord(dialogue_id=dialogue_id, total_utterances=total_utterances)
                expected_utterances_count = total_utterances
                actual_utterances_count = 0

            else:
                # --- 语句行 ---
                if current_dialogue is None:
                    raise ValueError(
                        f"Line {line_num}: Utterance line encountered before any dialogue header: '{line}'"
                    )

                parts_pipe = line.split(" | ")
                if len(parts_pipe) < 4:
                    raise ValueError(
                        f"Line {line_num}: Malformed utterance line (expected ≥4 pipe-separated fields): '{line}'"
                    )

                # 字段提取
                try:
                    utterance_idx = int(parts_pipe[0].strip())
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: Utterance index is not an integer: '{parts_pipe[0]}'"
                    )
                text_content = parts_pipe[1].strip()
                speaker_name = parts_pipe[2].strip()
                speaker_emotion_str = parts_pipe[3].strip()

                # 解析说话人情绪
                speaker_emotion, speaker_emotion_cause_idxs = _parse_emotion(speaker_emotion_str, line_num)
                speaker = PersonEntry(
                    name=speaker_name,
                    emotion=speaker_emotion,
                    emotion_cause_idxs=speaker_emotion_cause_idxs,
                )

                # 解析听众（每两个字段为一组：名字 + 情绪）
                listeners: List[PersonEntry] = []
                i = 4
                while i < len(parts_pipe):
                    if i + 1 >= len(parts_pipe):
                        raise ValueError(
                            f"Line {line_num}: Incomplete listener entry — name without emotion "
                            f"for '{parts_pipe[i]}': '{line}'"
                        )
                    listener_name = parts_pipe[i].strip()
                    listener_emotion_str = parts_pipe[i + 1].strip()
                    listener_emotion, listener_emotion_cause_idxs = _parse_emotion(listener_emotion_str, line_num)
                    listeners.append(
                        PersonEntry(
                            name=listener_name,
                            emotion=listener_emotion,
                            emotion_cause_idxs=listener_emotion_cause_idxs,
                        )
                    )
                    i += 2

                # 验证语句索引连续性（内部逻辑一致性，保留 assert）
                assert utterance_idx == actual_utterances_count + 1, (
                    f"Line {line_num}: Expected utterance index {actual_utterances_count + 1}, "
                    f"got {utterance_idx}."
                )

                current_dialogue.utterances.append(
                    UtteranceRecord(
                        dialogue_id=current_dialogue.dialogue_id,
                        utterance_idx=utterance_idx,
                        text_content=text_content,
                        speaker=speaker,
                        listeners=listeners,
                    )
                )
                actual_utterances_count += 1

    # 结算最后一段对话
    if current_dialogue is not None:
        if actual_utterances_count != expected_utterances_count:
            raise ValueError(
                f"Dialogue {current_dialogue.dialogue_id}: ended with {actual_utterances_count} "
                f"utterances, expected {expected_utterances_count}."
            )
        
        # 防御性断言：验证所有 emotion_cause_idx 的合法性
        for utterance in current_dialogue.utterances:
            # 验证说话人的情绪诱因索引
            for cause_idx in utterance.speaker.emotion_cause_idxs:
                if cause_idx != -1 and not (1 <= cause_idx <= expected_utterances_count):
                    raise ValueError(
                        f"Dialogue {current_dialogue.dialogue_id}, Utterance {utterance.utterance_idx}: "
                        f"Speaker '{utterance.speaker.name}' has invalid emotion_cause_idx "
                        f"{cause_idx} (must be in range [1, {expected_utterances_count}] or -1)."
                    )
            # 验证听众的情绪诱因索引
            for listener in utterance.listeners:
                for cause_idx in listener.emotion_cause_idxs:
                    if cause_idx != -1 and not (1 <= cause_idx <= expected_utterances_count):
                        raise ValueError(
                            f"Dialogue {current_dialogue.dialogue_id}, Utterance {utterance.utterance_idx}: "
                            f"Listener '{listener.name}' has invalid emotion_cause_idx "
                            f"{cause_idx} (must be in range [1, {expected_utterances_count}] or -1)."
                        )
        
        dialogues.append(current_dialogue)

    return dialogues


# Example usage (for testing purposes)
if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig

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
        print(f"Parsing successful! Found {len(dialogue_records)} dialogues.")

        total_utterances = 0
        video_files_found = 0
        video_files_missing = 0
        missing_video_details: List[str] = []

        for dialogue in dialogue_records:
            for utterance in dialogue.utterances:
                total_utterances += 1
                dialogue_folder_name = f"C_{utterance.dialogue_id}"
                video_file_name = f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.mp4"
                video_path = video_base_dir / dialogue_folder_name / video_file_name
                if video_path.exists():
                    video_files_found += 1
                else:
                    video_files_missing += 1
                    missing_video_details.append(
                        f"Dialogue {utterance.dialogue_id}, Utterance {utterance.utterance_idx}: {video_path}"
                    )

        print("\n--- Video File Verification Summary ---")
        print(f"Total utterances processed: {total_utterances}")
        print(f"Video files found: {video_files_found}")
        print(f"Video files missing: {video_files_missing}")

        if video_files_missing > 0:
            print("\n!!! WARNING: Some video files are missing. !!!")
            for detail in missing_video_details:
                print(f"  - {detail}")
            sys.exit(1)
        else:
            print("\nAll required video files found. Parser validation successful!")

    except (FileNotFoundError, ValueError) as e:
        print(f"Parsing failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
