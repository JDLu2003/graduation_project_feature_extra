"""
该文件实现一个面向“语句级多人特征提取”的策略类，目标是在不破坏现有主流程接口的前提下，
将每个人物的最终特征统一组织为 1024 维，其中前 512 维是人物身份相关特征，后 512 维是
该语句视频片段对应的环境/场景特征。

整体策略采用“单次视频解析 + 多人物复用”的方式以保证效率：
1) 每次进入 extract_speaker(video_path) 时，只对该视频做一次抽帧、检测、识别与场景编码。
2) 抽到的人脸使用已训练好的 facenet-fr 模型做身份判别，同时保留其 512 维 embedding。
3) 同一视频内同一身份若出现多次，采用平均聚合作为该身份的人物特征。
4) 环境特征由 CLIP 对视频帧做全局编码后聚合得到 512 维，语句中所有人物共享该环境向量。
5) 当前语句的中间结果会缓存在策略对象中，随后 extract_non_speaker(...) 直接复用，避免重复推理。

对失败情况的处理遵循保守策略：
- 未检测到目标身份的人脸，人物 512 维置零。
- 视频无法采样帧或场景编码失败，环境 512 维置零。
- 整体输出始终保证 [1, 1024]，以满足 pipeline 与 saver 的既有约束。

该实现保持与 FeatureExtractor 接口一致，不修改 pipeline 的调用方式，
可通过 config.yaml 切换 active_type 为 face_scene_fr 使用。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

from src.config import FaceSceneFRConfig, NonSpeakerConfig
from src.extractors.base import FeatureExtractor
from src.parser import DialogueRecord
from src.video_utils import sample_frames


class FaceSceneFRStrategy(FeatureExtractor):
    def __init__(self, config: FaceSceneFRConfig, non_speaker_config: NonSpeakerConfig) -> None:
        self.config = config
        self.non_speaker_config = non_speaker_config
        self.scene_device = torch.device(config.device)
        if config.device == "mps":
            # MPS 在 MTCNN 内部插值/池化上存在已知兼容性问题；人脸分支回退 CPU 保证稳定性。
            self.face_device = torch.device("cpu")
            print("[FaceSceneFRStrategy] MPS detected: face branch fallback to CPU for stability.")
        else:
            self.face_device = torch.device(config.device)
        self._output_dim = 1024

        print("[FaceSceneFRStrategy] Loading facenet-fr checkpoint...")
        ckpt = torch.load(config.face_checkpoint, map_location="cpu")
        self.idx_to_label: List[str] = list(ckpt["idx_to_label"])
        self._label_norm_to_raw = {self._norm_name(x): x for x in self.idx_to_label}
        self._known_label_norm_set = set(self._label_norm_to_raw.keys())
        self._other_norm = self._norm_name(config.other_label_name)
        if self._other_norm not in self._known_label_norm_set:
            print(
                f"[FaceSceneFRStrategy] warning: other_label_name='{config.other_label_name}' "
                "not found in checkpoint labels, unknown-person fallback will degrade to ZERO."
            )

        pretrained_name = ckpt.get("config", {}).get("pretrained", "vggface2")
        dropout = float(ckpt.get("config", {}).get("dropout", 0.2))

        self.backbone = InceptionResnetV1(pretrained=pretrained_name, classify=False).to(self.face_device).eval()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, len(self.idx_to_label))).to(self.face_device).eval()

        state_dict = ckpt["model_state_dict"]
        backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        classifier_state = {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")}
        if not backbone_state or not classifier_state:
            raise ValueError("Invalid facenet-fr checkpoint format: missing backbone/classifier weights")
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.classifier.load_state_dict(classifier_state, strict=True)

        print("[FaceSceneFRStrategy] Loading MTCNN for face detection...")
        self.mtcnn = MTCNN(
            image_size=config.mtcnn_image_size,
            margin=config.mtcnn_margin,
            min_face_size=config.mtcnn_min_face_size,
            thresholds=config.mtcnn_thresholds,
            keep_all=config.mtcnn_keep_all,
            device=self.face_device,
        )

        print(f"[FaceSceneFRStrategy] Loading CLIP scene encoder: {config.clip_model_name}...")
        self.clip_model, self.clip_preprocess = clip.load(config.clip_model_name, device=self.scene_device)
        self.clip_model.eval()

        self._speaker_name_by_video: Dict[str, str] = {}
        self._current_person_feature: Dict[str, torch.Tensor] = {}
        self._current_env_feature: torch.Tensor = torch.zeros(1, 512)
        self._current_utterance_tag: str = "unknown"

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def prepare(self, dialogue_records: List[DialogueRecord], video_base_dir: Path) -> None:
        self._speaker_name_by_video.clear()
        for dialogue in dialogue_records:
            for utt in dialogue.utterances:
                vp = (video_base_dir / f"C_{utt.dialogue_id}" / f"C_{utt.dialogue_id}_U_{utt.utterance_idx}.mp4").resolve()
                self._speaker_name_by_video[str(vp)] = utt.speaker.name
        print(f"[FaceSceneFRStrategy] Prepared speaker index with {len(self._speaker_name_by_video)} entries.")

    def extract_speaker(self, video_path: Path) -> torch.Tensor:
        self._refresh_cache(video_path)
        speaker_name = self._speaker_name_by_video.get(str(video_path.resolve()), "")
        person_feature, status = self._resolve_person_feature(speaker_name)
        print(
            f"[face_scene_fr][{self._current_utterance_tag}] person='{speaker_name}' role=speaker "
            f"feature_status={status}"
        )
        return self._compose(person_feature, self._current_env_feature)

    def extract_non_speaker(self, person_name: str, dialogue_id: int) -> torch.Tensor:
        person_feature, status = self._resolve_person_feature(person_name)
        print(
            f"[face_scene_fr][{self._current_utterance_tag}] person='{person_name}' role=listener "
            f"feature_status={status}"
        )
        return self._compose(person_feature, self._current_env_feature)

    def _refresh_cache(self, video_path: Path) -> None:
        self._current_utterance_tag = video_path.stem
        frames = sample_frames(
            video_path,
            num_frames=self.config.frame_sampling.num_frames,
            strategy=self.config.frame_sampling.strategy,
        )
        self._current_env_feature = self._extract_env_feature(frames)
        self._current_person_feature = self._extract_person_features(frames)

    def _extract_env_feature(self, frames: List) -> torch.Tensor:
        if not frames:
            return torch.zeros(1, 512)
        batch = []
        for frame_bgr in frames:
            frame_rgb = Image.fromarray(frame_bgr[:, :, ::-1])
            batch.append(self.clip_preprocess(frame_rgb))
        x = torch.stack(batch).to(self.scene_device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(x)
        if self.config.frame_sampling.aggregation == "max":
            agg = feat.max(dim=0, keepdim=True).values
        else:
            agg = feat.mean(dim=0, keepdim=True)
        agg = F.normalize(agg, p=2, dim=-1)
        return agg.cpu().detach()

    def _extract_person_features(self, frames: List) -> Dict[str, torch.Tensor]:
        bucket: Dict[str, List[torch.Tensor]] = {}
        with torch.no_grad():
            for frame_bgr in frames:
                pil_img = Image.fromarray(frame_bgr[:, :, ::-1])
                boxes, probs = self.mtcnn.detect(pil_img)
                if boxes is None or probs is None:
                    continue
                for box, prob in zip(boxes, probs):
                    if prob is None or float(prob) < self.config.min_detection_confidence:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(pil_img.width, x2)
                    y2 = min(pil_img.height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = pil_img.crop((x1, y1, x2, y2))
                    face_tensor = self._prepare_face_tensor(crop)
                    emb = self.backbone(face_tensor.unsqueeze(0).to(self.face_device))
                    logits = self.classifier(emb)
                    cls_prob = torch.softmax(logits, dim=1)
                    conf, idx = cls_prob.max(dim=1)
                    if (
                        self.config.classification_strategy == "top1_with_threshold"
                        and float(conf.item()) < self.config.min_classification_confidence
                    ):
                        continue
                    pred_label = self.idx_to_label[int(idx.item())]
                    norm_label = self._norm_name(pred_label)
                    bucket.setdefault(norm_label, []).append(emb.cpu())

        out: Dict[str, torch.Tensor] = {}
        for norm_label, vecs in bucket.items():
            if not vecs:
                continue
            mean_vec = torch.mean(torch.cat(vecs, dim=0), dim=0, keepdim=True)
            out[norm_label] = F.normalize(mean_vec, p=2, dim=-1).cpu().detach()
        return out

    def _resolve_person_feature(self, person_name: str) -> tuple[torch.Tensor, str]:
        """
        解析某个人名在当前语句下的 512 维人物特征。
        返回 (feature, status)，status:
        - FOUND: 该名字直接命中预测标签桶
        - OTHER_MEAN: 该名字不在主标签中，按策略回退到 other 聚合特征
        - ZERO: 未命中且无可用回退
        """
        norm_name = self._norm_name(person_name)
        if norm_name in self._current_person_feature:
            return self._current_person_feature[norm_name], "FOUND"

        # 非主角/未知人名策略：可选择回退到 other 聚合特征，或保持零向量。
        if norm_name not in self._known_label_norm_set:
            if self.config.unknown_person_strategy == "other_mean":
                other_feat = self._current_person_feature.get(self._other_norm)
                if other_feat is not None:
                    return other_feat, "OTHER_MEAN"

        return self._zero_person(), "ZERO"

    @staticmethod
    def _norm_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _zero_person() -> torch.Tensor:
        return torch.zeros(1, 512)

    def _compose(self, person_512: torch.Tensor, env_512: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([person_512.cpu(), env_512.cpu()], dim=1)
        assert feat.shape == (1, self.output_dim), f"Expected (1, {self.output_dim}), got {feat.shape}"
        return feat

    def _prepare_face_tensor(self, crop: Image.Image) -> torch.Tensor:
        """
        将检测后的裁剪人脸直接转换为 FaceNet 输入，避免对 crop 再次运行 MTCNN。
        这样可规避空框异常并减少额外计算开销。
        """
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.config.mtcnn_image_size, self.config.mtcnn_image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # 与 facenet-pytorch 的 fixed_image_standardization 一致
        x = (x - 0.5) / 0.5
        return x
