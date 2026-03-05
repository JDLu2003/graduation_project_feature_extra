from pathlib import Path
from typing import List
import torch

from src.config import AppConfig
from src.extractors.base import FeatureExtractor
from src.saver import save_features
from src.parser import DialogueRecord, UtteranceRecord


def process_utterance(
    utterance: UtteranceRecord,
    extractor: FeatureExtractor,
    video_base_dir: Path,
    feat_out_dir: Path,
    skip_existing: bool,
) -> None:
    """
    处理单条语句：提取说话人及所有非说话人特征，合并，然后保存。

    Args:
        utterance: 待处理的语句记录。
        extractor: 特征提取器实例。
        video_base_dir: 视频文件所在的根目录。
        feat_out_dir: 特征输出的根目录。
        skip_existing: 如果对应的特征文件已存在，则跳过处理。
    """
    print(
        f"  Processing utterance C_{utterance.dialogue_id}_U_{utterance.utterance_idx} "
        f"(speaker: {utterance.speaker.name}, listeners: {len(utterance.listeners)})"
    )
    dialogue_folder_name = f"C_{utterance.dialogue_id}"
    video_file_name = f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.mp4"
    speaker_video_path = video_base_dir / dialogue_folder_name / video_file_name

    output_path = feat_out_dir / dialogue_folder_name / f"C_{utterance.dialogue_id}_U_{utterance.utterance_idx}.pt"

    all_person_features: List[torch.Tensor] = []

    # 1. 提取说话人特征，强制迁移至 CPU 防止 CUDA OOM
    speaker_features = extractor.extract_speaker(speaker_video_path).cpu().detach()
    assert speaker_features.shape == (1, extractor.output_dim), (
        f"Speaker feature shape mismatch: Expected (1, {extractor.output_dim}), got {speaker_features.shape}"
    )
    all_person_features.append(speaker_features)

    # 2. 提取非说话人特征，同样强制迁移至 CPU
    for listener in utterance.listeners:
        non_speaker_features = extractor.extract_non_speaker(listener.name, utterance.dialogue_id).cpu().detach()
        assert non_speaker_features.shape == (1, extractor.output_dim), (
            f"Non-speaker feature shape mismatch: Expected (1, {extractor.output_dim}), got {non_speaker_features.shape}"
        )
        all_person_features.append(non_speaker_features)

    # 3. 合并所有人物特征（所有张量已在 CPU 上，cat 不会增加显存占用）
    final_features = torch.cat(all_person_features, dim=0)

    # 防御性编程：验证最终 Tensor 内部维度一致性（内部逻辑校验，保留 assert）
    expected_num_people = len(utterance.listeners) + 1
    assert final_features.shape == (expected_num_people, extractor.output_dim), (
        f"Final feature shape mismatch: Expected ({expected_num_people}, {extractor.output_dim}), "
        f"got {final_features.shape}"
    )

    # 4. 保存特征
    save_features(final_features, output_path, skip_existing=skip_existing)
    print(f"Saved features for dialogue {utterance.dialogue_id}, utterance {utterance.utterance_idx} to {output_path}")


def run_pipeline(config: AppConfig, extractor: FeatureExtractor, dialogue_records: List[DialogueRecord]) -> None:
    """
    遍历所有对话和语句，调用 process_utterance 进行特征提取和保存。

    Args:
        config: 应用配置。
        extractor: 特征提取器实例。
        dialogue_records: 解析后的对话记录列表。
    """
    print(f"Starting feature extraction pipeline with {len(dialogue_records)} dialogues...")
    extractor.prepare(dialogue_records, config.paths.video_dir)

    for dialogue in dialogue_records:
        print(f"Processing Dialogue C_{dialogue.dialogue_id} with {len(dialogue.utterances)} utterances.")
        for utterance in dialogue.utterances:
            process_utterance(
                utterance,
                extractor,
                config.paths.video_dir,
                config.paths.feat_out,
                config.pipeline.skip_existing,
            )
    print("Feature extraction pipeline finished.")
