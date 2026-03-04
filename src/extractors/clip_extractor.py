import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Any

from src.config import VisualClipConfig, FrameSamplingConfig
from src.extractors.base import FeatureExtractor
from src.extractors.projection import LinearProjection
from src.video_utils import sample_frames

class CLIPVisualExtractor(FeatureExtractor):
    """
    Feature extractor using OpenAI's CLIP model for visual features.
    It samples frames from a video, processes them with CLIP, aggregates the features,
    applies a linear projection, and performs L2 normalization.
    """
    def __init__(self, config: VisualClipConfig):
        self.config = config
        self._output_dim = config.target_dim

        # Load CLIP model
        self.model, self.preprocess = clip.load(self.config.model_name, device=self.config.device)
        self.model.eval() # Set model to evaluation mode

        # Initialize linear projection layer
        self.projection_layer = LinearProjection(self.config.clip_output_dim, self.config.target_dim).to(self.config.device)

        # Transforms for sampled frames (PIL Image -> CLIP preprocess -> Tensor)
        # CLIP's preprocess already includes resizing, cropping, normalization to tensor
        self.transforms = self.preprocess


    @property
    def output_dim(self) -> int:
        return self._output_dim

    def extract(self, video_path: Path) -> torch.Tensor:
        """
        Extracts features from a video file using CLIP.

        Args:
            video_path: Path to the video file.

        Returns:
            A torch.Tensor of shape [1, target_dim] representing the aggregated
            and projected CLIP features for the video, L2-normalized.
        """
        assert video_path.exists(), f"Video file not found for extraction: {video_path}"

        # 1. Sample frames from video
        sampled_frames_np = sample_frames(
            video_path,
            num_frames=self.config.frame_sampling.num_frames,
            strategy=self.config.frame_sampling.strategy
        )

        if not sampled_frames_np:
            # If no frames sampled (e.g., empty video), return a zero vector
            return torch.zeros(1, self.output_dim, device=self.config.device)

        # 2. Preprocess frames for CLIP and convert to tensor
        # CLIP expects PIL Image, then applies its own transforms
        processed_frames_tensors: List[torch.Tensor] = []
        for frame_np in sampled_frames_np:
            # OpenCV reads as BGR, PIL expects RGB
            frame_rgb = Image.fromarray(frame_np[:, :, ::-1])
            processed_frames_tensors.append(self.transforms(frame_rgb))

        # Stack into a single batch tensor
        frame_batch = torch.stack(processed_frames_tensors).to(self.config.device)

        # 3. Extract features using CLIP
        with torch.no_grad():
            clip_features = self.model.encode_image(frame_batch) # Shape: [num_sampled_frames, clip_output_dim]

        # 4. Aggregate features
        if self.config.frame_sampling.aggregation == "mean":
            aggregated_features = clip_features.mean(dim=0, keepdim=True) # Shape: [1, clip_output_dim]
        elif self.config.frame_sampling.aggregation == "max":
            aggregated_features = clip_features.max(dim=0, keepdim=True).values # Shape: [1, clip_output_dim]
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.config.frame_sampling.aggregation}")

        # 5. Apply linear projection
        projected_features = self.projection_layer(aggregated_features) # Shape: [1, target_dim]

        # 6. L2 Normalize
        l2_normalized_features = F.normalize(projected_features, p=2, dim=-1)

        return l2_normalized_features

# Example usage (for testing purposes)
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig, VisualClipConfig
    from src.parser import parse_dev_txt # For getting a video path

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        # Ensure visual_clip_config exists and is active
        assert app_config.extractor.active_type == "visual_clip", "Visual CLIP extractor not active in config."
        clip_config = app_config.extractor.visual_clip_config
        assert clip_config is not None, "Visual CLIP configuration missing."

        print(f"--- Testing CLIPVisualExtractor ---")
        print(f"CLIP Model: {clip_config.model_name}, Device: {clip_config.device}")
        print(f"Output Dim: {clip_config.target_dim}, Frame Sampling Strategy: {clip_config.frame_sampling.strategy}")

        # Instantiate the extractor
        extractor = CLIPVisualExtractor(clip_config)

        # Get a real video path from the dataset for testing
        dialogue_records = parse_dev_txt(app_config.paths.dev_txt)
        if not dialogue_records:
            print("No dialogues parsed, cannot test CLIPVisualExtractor.")
            sys.exit(1)

        first_utterance = dialogue_records[0].utterances[0]
        dialogue_folder_name = f"C_{first_utterance.dialogue_id}"
        video_file_name = f"C_{first_utterance.dialogue_id}_U_{first_utterance.utterance_idx}.mp4"
        test_video_path = app_config.paths.video_dir / dialogue_folder_name / video_file_name

        assert test_video_path.exists(), f"Test video file not found: {test_video_path}"

        # Extract features
        print(f"\nExtracting features from: {test_video_path}")
        features = extractor.extract(test_video_path)

        # Assertions
        assert isinstance(features, torch.Tensor), "Extracted features are not a torch.Tensor."
        assert features.shape == (1, extractor.output_dim), \
            f"Expected feature shape (1, {extractor.output_dim}), but got {features.shape}"
        # Check if L2 normalized (approx)
        assert torch.isclose(features.norm(p=2, dim=-1), torch.tensor(1.0, device=clip_config.device)).all(), \
            "Extracted features are not L2 normalized."

        print(f"Features extracted successfully! Shape: {features.shape}")
        print("CLIPVisualExtractor test successful!")

    except AssertionError as e:
        print(f"Test failed due to assertion error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: A required file (config.yaml or video) was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")
        sys.exit(1)