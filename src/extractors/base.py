"""
FeatureExtractor 基类定义。

该文件定义了特征提取器的这一抽象基类。所有具体的特征提取策略（如 VisualCLIPStrategy, FaceSceneFRStrategy）
都必须继承此类并实现其抽象方法。

主要接口：
- prepare: 预处理步骤，通常用于构建索引或加载必要的上下文。
- extract_speaker: 提取说话人的特征。
- extract_non_speaker: 提取非说话人的特征。
- output_dim: 返回提取特征的维度。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from src.parser import DialogueRecord


class FeatureExtractor(ABC):
    @abstractmethod
    def prepare(self, dialogue_records: List['DialogueRecord'], video_base_dir: Path) -> None:
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
