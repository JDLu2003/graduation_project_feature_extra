from pathlib import Path
from typing import List
import cv2
import numpy as np


def sample_frames(video_path: Path, num_frames: int, strategy: str = "uniform") -> List[np.ndarray]:
    """
    Samples frames from a video file using sequential read (no random seeking).

    Sequential cap.read() is used throughout to avoid the severe performance penalty
    of cap.set(CAP_PROP_POS_FRAMES, i), which forces a full keyframe-to-target decode
    for every random access.  Instead, we compute the target frame indices upfront and
    collect only those frames as we stream through the video in one pass.

    Args:
        video_path: Path to the video file.
        num_frames: The desired number of frames to sample.
        strategy: Sampling strategy — "uniform", "middle", or "first".

    Returns:
        A list of numpy arrays (H, W, C) in BGR format for the sampled frames.
        Returns an empty list if the video cannot be opened or has no frames.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If strategy is not one of the supported values.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    valid_strategies = {"uniform", "middle", "first"}
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid sampling strategy '{strategy}'. Must be one of {valid_strategies}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[sample_frames] Warning: Could not open video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[sample_frames] Warning: Video file has no frames: {video_path}")
        cap.release()
        return []

    # --- Compute target indices (no I/O yet) ---
    indices: np.ndarray
    if strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    elif strategy == "middle":
        if num_frames >= total_frames:
            indices = np.arange(total_frames, dtype=int)
        else:
            mid = total_frames // 2
            half = num_frames // 2
            start = max(0, mid - half)
            end = min(total_frames, start + num_frames)
            if end - start < num_frames:
                start = max(0, end - num_frames)
            indices = np.arange(start, end, dtype=int)
    else:  # "first"
        indices = np.arange(min(num_frames, total_frames), dtype=int)

    # Use a set for O(1) membership test during sequential scan
    target_set: set = set(indices.tolist())

    # --- Single sequential pass through the video ---
    frames: List[np.ndarray] = []
    current_frame_idx: int = 0

    while current_frame_idx <= int(indices[-1]) if len(indices) > 0 else False:
        ret, frame = cap.read()
        if not ret:
            print(f"[sample_frames] Warning: Stream ended unexpectedly at frame {current_frame_idx} in {video_path}")
            break
        if current_frame_idx in target_set:
            frames.append(frame)
        current_frame_idx += 1

    cap.release()
    return frames


# Example usage (for testing purposes)
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
        video_base_dir = app_config.paths.video_dir
        frame_sampling_config = app_config.extractor.visual_clip_config.frame_sampling

        dialogue_records = parse_dev_txt(app_config.paths.dev_txt)
        if not dialogue_records:
            print("No dialogues parsed, cannot test video sampling.")
            sys.exit(1)

        first_utterance = dialogue_records[0].utterances[0]
        dialogue_folder_name = f"C_{first_utterance.dialogue_id}"
        video_file_name = f"C_{first_utterance.dialogue_id}_U_{first_utterance.utterance_idx}.mp4"
        test_video_path = video_base_dir / dialogue_folder_name / video_file_name

        print("--- Testing Frame Sampling ---")
        print(f"Video path: {test_video_path}")
        print(f"Strategy: {frame_sampling_config.strategy}, Num Frames: {frame_sampling_config.num_frames}")

        sampled_frames = sample_frames(
            test_video_path,
            frame_sampling_config.num_frames,
            frame_sampling_config.strategy,
        )

        if sampled_frames:
            print(f"Successfully sampled {len(sampled_frames)} frames.")
            print(f"First frame shape: {sampled_frames[0].shape}, dtype: {sampled_frames[0].dtype}")
            print("Frame sampling test successful!")
        else:
            print("Failed to sample any frames.")
            sys.exit(1)

    except (FileNotFoundError, ValueError) as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
