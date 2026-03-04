from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import torch

# Assuming DialogueRecord is defined elsewhere, or will be defined.
# For now, using Any to avoid import cycles.
from typing import Any # Placeholder for DialogueRecord


class FeatureExtractor(ABC):
    @abstractmethod
    def prepare(self, dialogue_records: List[Any], video_base_dir: Path) -> None:
        """
        Prepares the extractor for processing, e.g., building context indexes.
        """
        pass

    @abstractmethod
    def extract_speaker(self, video_path: Path) -> torch.Tensor:
        """
        Extracts features for a speaker from a given video path.

        Args:
            video_path: The path to the video file.

        Returns:
            A torch.Tensor representing the extracted features for the speaker (e.g., [1, DIMS]).
        """
        pass

    @abstractmethod
    def extract_non_speaker(self, person_name: str, dialogue_id: int) -> torch.Tensor:
        """
        Extracts features for a non-speaker.
        This might involve looking up pre-indexed features, generating, or using fallback.

        Args:
            person_name: The name of the non-speaker.
            dialogue_id: The ID of the dialogue context.

        Returns:
            A torch.Tensor representing the extracted features for the non-speaker (e.g., [1, DIMS]).
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns the output dimension of the features produced by this extractor.
        """
        pass
