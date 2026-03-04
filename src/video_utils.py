from pathlib import Path
from typing import List
import cv2
import numpy as np

def sample_frames(video_path: Path, num_frames: int, strategy: str = "uniform") -> List[np.ndarray]:
    """
    Samples frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: The desired number of frames to sample.
        strategy: Sampling strategy - "uniform", "middle", or "first".

    Returns:
        A list of numpy arrays, where each array is a sampled frame (H, W, C).
        Returns an empty list if video cannot be opened or has no frames.

    Raises:
        AssertionError: If video_path does not exist or strategy is invalid.
    """
    assert video_path.exists(), f"Video file not found: {video_path}"
    assert strategy in ["uniform", "middle", "first"], f"Invalid sampling strategy: {strategy}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video file has no frames: {video_path}")
        cap.release()
        return []

    frames: List[np.ndarray] = []

    if strategy == "uniform":
        # Sample frames uniformly across the video
        # Ensure we don't sample beyond total_frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {i} from {video_path}")
    elif strategy == "middle":
        # Sample num_frames around the middle of the video
        # Prioritize sampling within valid frame range
        if num_frames >= total_frames:
            # If requesting more frames than available, sample all frames
            indices = np.arange(total_frames)
        else:
            mid_frame = total_frames // 2
            half_num_frames = num_frames // 2
            start_index = max(0, mid_frame - half_num_frames)
            end_index = min(total_frames, start_index + num_frames)
            # Adjust start_index if we hit the end
            if end_index - start_index < num_frames:
                start_index = max(0, end_index - num_frames)
            indices = np.arange(start_index, end_index)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {i} from {video_path}")
    elif strategy == "first":
        # Sample the first num_frames frames
        for i in range(min(num_frames, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {i} from {video_path}")

    cap.release()
    return frames

# Example usage (for testing purposes)
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig, FrameSamplingConfig
    from src.parser import parse_dev_txt # Import parse_dev_txt for getting video paths

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        video_base_dir = app_config.paths.video_dir
        frame_sampling_config = app_config.extractor.visual_clip_config.frame_sampling

        # Use parse_dev_txt to get a real video path from the dataset
        dialogue_records = parse_dev_txt(app_config.paths.dev_txt)
        if not dialogue_records:
            print("No dialogues parsed, cannot test video sampling.")
            sys.exit(1)

        first_utterance = dialogue_records[0].utterances[0]
        dialogue_folder_name = f"C_{first_utterance.dialogue_id}"
        video_file_name = f"C_{first_utterance.dialogue_id}_U_{first_utterance.utterance_idx}.mp4"
        test_video_path = video_base_dir / dialogue_folder_name / video_file_name

        print(f"--- Testing Frame Sampling ---")
        print(f"Video path: {test_video_path}")
        print(f"Strategy: {frame_sampling_config.strategy}, Num Frames: {frame_sampling_config.num_frames}")

        sampled_frames = sample_frames(
            test_video_path,
            frame_sampling_config.num_frames,
            frame_sampling_config.strategy
        )

        if sampled_frames:
            print(f"Successfully sampled {len(sampled_frames)} frames.")
            print(f"First frame shape: {sampled_frames[0].shape}, dtype: {sampled_frames[0].dtype}")
            # Further assertions could go here, e.g., checking content if specific frames are expected
            print("Frame sampling test successful!")
        else:
            print("Failed to sample any frames.")
            sys.exit(1)

    except AssertionError as e:
        print(f"Test failed due to assertion error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: A required file (config.yaml or video) was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)