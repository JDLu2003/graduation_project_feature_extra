from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
import torch

class FeatureExtractor(ABC):
    """
    Abstract Base Class for all feature extractors.
    Ensures that any new extractor implements the core 'extract' method
    and provides its output dimension.
    """
    @abstractmethod
    def extract(self, video_path: Path) -> torch.Tensor:
        """
        Extracts features from a given video path.

        Args:
            video_path: The path to the video file.

        Returns:
            A torch.Tensor representing the extracted features (e.g., [1, DIMS]).
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns the output dimension of the features produced by this extractor.
        """
        pass

# No direct execution for ABC
