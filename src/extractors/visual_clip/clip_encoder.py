import clip
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import List

from src.config import VisualClipConfig
from src.video_utils import sample_frames


class CLIPEncoder:
    """
    Encapsulates the core CLIP model and its associated logic for encoding visual features
    from video frames. This class handles frame sampling, CLIP feature extraction,
    aggregation, L2 normalization, and zero-padding to the target dimension.
    """

    def __init__(self, config: VisualClipConfig) -> None:
        self.config = config
        self._output_dim = config.target_dim

        # Load CLIP model
        self.model, self.preprocess = clip.load(
            self.config.model_name, device=self.config.device)
        self.model.eval()  # Set model to evaluation mode

        # Note: LinearProjection is intentionally NOT used here.
        # A randomly-initialized linear layer would destroy the pretrained CLIP feature
        # distribution.  Instead, we zero-pad the 512-dim CLIP output to the required
        # 1024-dim target by appending 512 zeros, preserving all pretrained information.
        self.transforms = self.preprocess

        pad_size = self.config.target_dim - self.config.clip_output_dim
        if pad_size < 0:
            raise ValueError(
                f"target_dim ({self.config.target_dim}) must be >= clip_output_dim "
                f"({self.config.clip_output_dim})"
            )
        self._pad_size = pad_size

        print(
            f"[CLIPEncoder] Loaded model '{self.config.model_name}' on {self.config.device}. "
            f"CLIP output dim: {self.config.clip_output_dim}, "
            f"target dim: {self.config.target_dim} (zero-padded by {self._pad_size})."
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode_video_frames(self, video_path: Path) -> torch.Tensor:
        """
        Samples frames from a video, extracts CLIP features, aggregates,
        L2 normalizes, then zero-pads to target_dim.

        Args:
            video_path: Path to the video file.

        Returns:
            A torch.Tensor of shape [1, target_dim] on CPU, L2-normalized on the
            clip_output_dim sub-space and zero-padded to target_dim.

        Raises:
            FileNotFoundError: If video_path does not exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found for encoding: {video_path}")

        # 1. Sample frames from video
        sampled_frames_np = sample_frames(
            video_path,
            num_frames=self.config.frame_sampling.num_frames,
            strategy=self.config.frame_sampling.strategy,
        )

        if not sampled_frames_np:
            print(f"[CLIPEncoder] Warning: No frames sampled from '{video_path}', returning zero vector.")
            return torch.zeros(1, self.output_dim)

        # 2. Preprocess frames for CLIP and convert to tensor
        processed_frames_tensors: List[torch.Tensor] = []
        for frame_np in sampled_frames_np:
            # OpenCV reads as BGR; PIL expects RGB
            frame_rgb = Image.fromarray(frame_np[:, :, ::-1])
            processed_frames_tensors.append(self.transforms(frame_rgb))

        # Stack into a single batch tensor
        frame_batch = torch.stack(
            processed_frames_tensors).to(self.config.device)

        # 3. Extract features using CLIP
        with torch.no_grad():
            # [num_sampled_frames, clip_output_dim]
            clip_features = self.model.encode_image(frame_batch)

        # 4. Aggregate features across sampled frames
        if self.config.frame_sampling.aggregation == "mean":
            aggregated_features = clip_features.mean(
                dim=0, keepdim=True)  # [1, clip_output_dim]
        elif self.config.frame_sampling.aggregation == "max":
            aggregated_features = clip_features.max(
                dim=0, keepdim=True).values  # [1, clip_output_dim]
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.config.frame_sampling.aggregation}")

        # 5. L2 normalize BEFORE zero-padding.
        #    Normalizing after padding would dilute the norm with the padded zeros and
        #    break the unit-norm property of the meaningful sub-space.
        l2_normalized: torch.Tensor = F.normalize(
            aggregated_features, p=2, dim=-1)  # [1, clip_output_dim]

        # 6. Zero-pad from clip_output_dim to target_dim.
        #    This preserves the pretrained CLIP feature distribution intact.
        #    The padded zeros simply occupy the remaining dimensions without interference.
        padded_features: torch.Tensor = F.pad(
            l2_normalized, (0, self._pad_size))  # [1, target_dim]

        # Internal consistency check (not external input — kept as assert)
        assert padded_features.shape == (1, self.output_dim), (
            f"Unexpected output shape: {padded_features.shape}, expected (1, {self.output_dim})"
        )

        # Move to CPU before returning to prevent CUDA memory accumulation in the pipeline
        return padded_features.cpu().detach()
