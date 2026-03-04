import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import List

from src.config import VisualClipConfig
from src.video_utils import sample_frames

class LinearProjection(nn.Module):
    """
    A simple linear projection layer to transform feature dimensions.
    Used to project CLIP's output dimension (e.g., 512) to the target dimension (e.g., 1024).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the linear projection.

        Args:
            x: Input tensor with shape (..., input_dim).

        Returns:
            Output tensor with shape (..., output_dim).
        """
        return self.projection(x)


class CLIPEncoder:
    """
    Encapsulates the core CLIP model and its associated logic for encoding visual features
    from video frames. This class handles frame sampling, CLIP feature extraction,
    aggregation, linear projection, and L2 normalization.
    """
    def __init__(self, config: VisualClipConfig):
        self.config = config
        self._output_dim = config.target_dim

        # Load CLIP model
        self.model, self.preprocess = clip.load(self.config.model_name, device=self.config.device)
        self.model.eval() # Set model to evaluation mode

        # Initialize linear projection layer
        self.projection_layer = LinearProjection(self.config.clip_output_dim, self.config.target_dim).to(self.config.device)
        print(f"[CLIPEncoder] Loaded model '{self.config.model_name}' on {self.config.device}. "
              f"CLIP output dim: {self.config.clip_output_dim}, target dim: {self.config.target_dim}")

        # Transforms for sampled frames (PIL Image -> CLIP preprocess)
        self.transforms = self.preprocess

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode_video_frames(self, video_path: Path) -> torch.Tensor:
        """
        Samples frames from a video, extracts CLIP features, aggregates,
        projects, and L2 normalizes them.

        Args:
            video_path: Path to the video file.

        Returns:
            A torch.Tensor of shape [1, target_dim] representing the aggregated
            and projected CLIP features for the video, L2-normalized.
        """
        assert video_path.exists(), f"Video file not found for encoding: {video_path}"

        # 1. Sample frames from video
        sampled_frames_np = sample_frames(
            video_path,
            num_frames=self.config.frame_sampling.num_frames,
            strategy=self.config.frame_sampling.strategy
        )

        if not sampled_frames_np:
            print(f"[CLIPEncoder] Warning: No frames sampled from '{video_path}', returning zero vector.")
            return torch.zeros(1, self.output_dim, device=self.config.device)

        # 2. Preprocess frames for CLIP and convert to tensor
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
