#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

IMPORT_ERROR: Optional[Exception] = None
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from PIL import Image
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover - 仅用于缺依赖时给出友好提示
    IMPORT_ERROR = exc
    class _TorchStub:
        @staticmethod
        def inference_mode():
            def decorator(func):
                return func
            return decorator
    torch = _TorchStub()  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import DialogueRecord, PersonEntry, UtteranceRecord, parse_dev_txt  # noqa: E402
try:
    from src.video_utils import sample_frames  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - 仅用于缺依赖时给出友好提示
    if IMPORT_ERROR is None:
        IMPORT_ERROR = exc

DEFAULT_TXT_PATH = Path("/data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt")
DEFAULT_REFERENCE_ROOT = Path("/data16T_2/sunshengzhe/reference_face_zh")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "zh_reference_coverage" / "artifacts"
DEFAULT_DEVICE = "cuda" if IMPORT_ERROR is None and torch.cuda.is_available() else "cpu"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_REFERENCE_FILES = {"identity_bank.pkl", "_placeholder.jpg"}


@dataclass(frozen=True)
class FaceRecord:
    face_id: int
    dialogue_id: int
    utterance_idx: int
    video_path: str
    frame_order: int
    box_x1: int
    box_y1: int
    box_x2: int
    box_y2: int
    detect_confidence: float
    participant_names: str
    speaker_name: str
    listener_names: str
    crop_path: str


@dataclass(frozen=True)
class ClusterAssignment:
    cluster_id: int
    size: int
    assigned_name: str
    best_name: str
    best_score: float
    second_score: float
    score_margin: float
    is_known: bool
    top_candidates: str


@dataclass(frozen=True)
class ReferenceIdentityEmbedding:
    name: str
    image_count: int
    prototype: np.ndarray


def normalize_name(name: str) -> str:
    return name.strip()


def safe_fs_name(name: str) -> str:
    cleaned = normalize_name(name)
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", cleaned)
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_video_dir(txt_path: Path, explicit_video_dir: Optional[Path]) -> Path:
    if explicit_video_dir is not None:
        return explicit_video_dir.resolve()
    split_name = txt_path.stem
    candidate = (txt_path.parent / f"Video_{split_name}").resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"无法根据 txt 路径推断视频目录: {candidate}，请显式传入 --video-dir"
        )
    return candidate


def iter_utterances(dialogues: Iterable[DialogueRecord], max_dialogues: Optional[int]) -> list[UtteranceRecord]:
    selected_dialogues = list(dialogues)
    if max_dialogues is not None:
        selected_dialogues = selected_dialogues[:max_dialogues]
    utterances: list[UtteranceRecord] = []
    for dialogue in selected_dialogues:
        utterances.extend(dialogue.utterances)
    return utterances


def build_video_path(video_dir: Path, utt: UtteranceRecord) -> Path:
    return (video_dir / f"C_{utt.dialogue_id}" / f"C_{utt.dialogue_id}_U_{utt.utterance_idx}.mp4").resolve()


def to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[:, :, ::-1])


def detect_faces_batched(
    mtcnn: MTCNN,
    pil_images: list[Image.Image],
    min_confidence: float,
    max_faces_per_frame: int,
) -> list[list[tuple[tuple[int, int, int, int], float, Image.Image]]]:
    if not pil_images:
        return []
    boxes_batch, probs_batch = mtcnn.detect(pil_images)
    if len(pil_images) == 1 and not isinstance(boxes_batch, list):
        boxes_batch = [boxes_batch]
        probs_batch = [probs_batch]

    results: list[list[tuple[tuple[int, int, int, int], float, Image.Image]]] = []
    for pil_img, boxes, probs in zip(pil_images, boxes_batch, probs_batch):
        frame_faces: list[tuple[tuple[int, int, int, int], float, Image.Image]] = []
        if boxes is None or probs is None:
            results.append(frame_faces)
            continue
        order = np.argsort(-np.asarray(probs, dtype=np.float32))
        for idx in order[:max_faces_per_frame]:
            prob = float(probs[idx])
            if prob < min_confidence:
                continue
            box = boxes[idx]
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pil_img.width, x2)
            y2 = min(pil_img.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            frame_faces.append(((x1, y1, x2, y2), prob, crop))
        results.append(frame_faces)
    return results


def preprocess_face_crop(crop: Image.Image, image_size: int) -> torch.Tensor:
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    tensor = (tensor - 127.5) / 128.0
    return tensor.squeeze(0)


@torch.inference_mode()
def embed_crops(
    embedder: InceptionResnetV1,
    device: torch.device,
    crops: list[Image.Image],
    image_size: int,
    batch_size: int,
    use_fp16: bool,
) -> np.ndarray:
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    batches: list[np.ndarray] = []
    autocast_enabled = use_fp16 and device.type == "cuda"
    for start in range(0, len(crops), batch_size):
        chunk = crops[start:start + batch_size]
        tensor = torch.stack([preprocess_face_crop(crop, image_size) for crop in chunk], dim=0).to(device)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            emb = embedder(tensor)
        emb = F.normalize(emb, p=2, dim=1)
        batches.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def load_reference_images(reference_root: Path) -> dict[str, list[Path]]:
    if not reference_root.exists():
        raise FileNotFoundError(f"reference root not found: {reference_root}")
    items: dict[str, list[Path]] = {}
    for child in sorted(reference_root.iterdir(), key=lambda p: p.name):
        if child.name in IGNORED_REFERENCE_FILES or not child.is_dir():
            continue
        images = sorted(p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        items[normalize_name(child.name)] = images
    return items


def build_reference_bank(
    embedder: InceptionResnetV1,
    device: torch.device,
    reference_root: Path,
    image_size: int,
    batch_size: int,
    use_fp16: bool,
) -> tuple[list[ReferenceIdentityEmbedding], dict[str, int]]:
    image_paths_by_name = load_reference_images(reference_root)
    rows: list[ReferenceIdentityEmbedding] = []
    ref_count_by_name: dict[str, int] = {}
    for name, paths in tqdm(image_paths_by_name.items(), desc="[pipeline] reference embeddings"):
        ref_count_by_name[name] = len(paths)
        if not paths:
            continue
        crops: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as img:
                crops.append(img.convert("RGB").copy())
        embs = embed_crops(embedder, device, crops, image_size=image_size, batch_size=batch_size, use_fp16=use_fp16)
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        rows.append(ReferenceIdentityEmbedding(name=name, image_count=len(paths), prototype=proto.astype(np.float32)))
    rows.sort(key=lambda x: x.name)
    return rows, ref_count_by_name


def online_cluster_embeddings(embeddings: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 512), dtype=np.float32)

    assignments = np.full((len(embeddings),), -1, dtype=np.int32)
    sums: list[np.ndarray] = []
    counts: list[int] = []
    centroids: list[np.ndarray] = []

    for idx, emb in enumerate(tqdm(embeddings, desc="[pipeline] online clustering")):
        if not centroids:
            assignments[idx] = 0
            sums.append(emb.copy())
            counts.append(1)
            centroids.append(emb.copy())
            continue

        centroid_mat = np.stack(centroids, axis=0)
        sims = centroid_mat @ emb
        best_cluster = int(np.argmax(sims))
        best_score = float(sims[best_cluster])
        if best_score >= threshold:
            assignments[idx] = best_cluster
            sums[best_cluster] = sums[best_cluster] + emb
            counts[best_cluster] += 1
            updated = sums[best_cluster] / (np.linalg.norm(sums[best_cluster]) + 1e-12)
            centroids[best_cluster] = updated.astype(np.float32)
        else:
            new_cluster = len(centroids)
            assignments[idx] = new_cluster
            sums.append(emb.copy())
            counts.append(1)
            centroids.append(emb.copy())

    centroid_arr = np.stack(centroids, axis=0).astype(np.float32)
    return assignments, centroid_arr


def assign_clusters_to_reference(
    centroids: np.ndarray,
    reference_bank: list[ReferenceIdentityEmbedding],
    match_threshold: float,
    match_margin: float,
) -> list[ClusterAssignment]:
    if len(centroids) == 0:
        return []
    if not reference_bank:
        return [
            ClusterAssignment(
                cluster_id=i,
                size=0,
                assigned_name="unknown",
                best_name="unknown",
                best_score=0.0,
                second_score=0.0,
                score_margin=0.0,
                is_known=False,
                top_candidates="",
            )
            for i in range(len(centroids))
        ]

    ref_names = [row.name for row in reference_bank]
    ref_mat = np.stack([row.prototype for row in reference_bank], axis=0)
    sims = centroids @ ref_mat.T
    out: list[ClusterAssignment] = []
    for cluster_id in range(len(centroids)):
        row = sims[cluster_id]
        order = np.argsort(-row)
        best_idx = int(order[0])
        best_score = float(row[best_idx])
        second_score = float(row[order[1]]) if len(order) > 1 else -1.0
        margin = best_score - second_score
        is_known = best_score >= match_threshold and margin >= match_margin
        topk_indices = order[: min(3, len(order))]
        top_candidates = "|".join(f"{ref_names[int(i)]}:{float(row[int(i)]):.4f}" for i in topk_indices)
        out.append(
            ClusterAssignment(
                cluster_id=cluster_id,
                size=0,
                assigned_name=ref_names[best_idx] if is_known else "unknown",
                best_name=ref_names[best_idx],
                best_score=best_score,
                second_score=second_score,
                score_margin=margin,
                is_known=is_known,
                top_candidates=top_candidates,
            )
        )
    return out


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def participant_names(utt: UtteranceRecord) -> list[str]:
    return [normalize_name(utt.speaker.name)] + [normalize_name(x.name) for x in utt.listeners]


def evaluate_predictions(
    utterances: list[UtteranceRecord],
    face_records: list[FaceRecord],
    assignments: np.ndarray,
    cluster_matches: list[ClusterAssignment],
    centroids: np.ndarray,
    embeddings: np.ndarray,
    role_names: list[str],
    ref_count_by_name: dict[str, int],
    face_verify_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    face_pred_names: dict[int, list[str]] = defaultdict(list)
    cluster_map = {row.cluster_id: row for row in cluster_matches}
    cluster_sizes = Counter(assignments.tolist())
    utterance_face_counter = Counter((face.dialogue_id, face.utterance_idx) for face in face_records)
    cluster_rows: list[dict[str, object]] = []
    for row in cluster_matches:
        member_indices = np.where(assignments == row.cluster_id)[0]
        centroid = centroids[row.cluster_id] if len(centroids) > row.cluster_id else np.zeros((512,), dtype=np.float32)
        member_sims = embeddings[member_indices] @ centroid if len(member_indices) > 0 else np.zeros((0,), dtype=np.float32)
        consistent_count = int(np.sum(member_sims >= face_verify_threshold))
        consistent_ratio = float(consistent_count / len(member_indices)) if len(member_indices) > 0 else 0.0
        cluster_rows.append(
            {
                "cluster_id": row.cluster_id,
                "size": cluster_sizes.get(row.cluster_id, 0),
                "assigned_name": row.assigned_name,
                "best_name": row.best_name,
                "best_score": row.best_score,
                "second_score": row.second_score,
                "score_margin": row.score_margin,
                "is_known": row.is_known,
                "reference_image_count": ref_count_by_name.get(row.assigned_name, 0),
                "top_candidates": row.top_candidates,
                "mean_member_similarity_to_centroid": float(member_sims.mean()) if len(member_sims) > 0 else 0.0,
                "min_member_similarity_to_centroid": float(member_sims.min()) if len(member_sims) > 0 else 0.0,
                "max_member_similarity_to_centroid": float(member_sims.max()) if len(member_sims) > 0 else 0.0,
                "consistent_face_count": consistent_count,
                "consistent_face_ratio": consistent_ratio,
                "face_verify_threshold": face_verify_threshold,
            }
        )

    for face, cluster_id in zip(face_records, assignments.tolist()):
        assigned = cluster_map[cluster_id].assigned_name
        if assigned != "unknown":
            face_pred_names[(face.dialogue_id, face.utterance_idx)].append(assigned)

    utterance_rows: list[dict[str, object]] = []
    matched_roles: set[str] = set()
    total_listener_slots = 0
    matched_listener_slots = 0
    with_face_count = 0
    with_known_match_count = 0
    speaker_match_count = 0
    participant_match_count = 0

    role_counter = Counter(role_names)

    for utt in utterances:
        key = (utt.dialogue_id, utt.utterance_idx)
        predicted = sorted(set(face_pred_names.get(key, [])))
        predicted_set = set(predicted)
        participants = participant_names(utt)
        participant_set = {name for name in participants if name}
        speaker_name = normalize_name(utt.speaker.name)
        listeners = [normalize_name(x.name) for x in utt.listeners if normalize_name(x.name)]
        total_listener_slots += len(listeners)
        face_count = utterance_face_counter.get((utt.dialogue_id, utt.utterance_idx), 0)
        if face_count > 0:
            with_face_count += 1
        if predicted:
            with_known_match_count += 1
        speaker_hit = speaker_name in predicted_set
        listener_hits = sorted(set(predicted_set.intersection(listeners)))
        participant_hit = bool(predicted_set.intersection(participant_set))
        if speaker_hit:
            speaker_match_count += 1
            matched_roles.add(speaker_name)
        if participant_hit:
            participant_match_count += 1
            matched_roles.update(predicted_set.intersection(participant_set))
        matched_listener_slots += len(listener_hits)
        utterance_rows.append(
            {
                "dialogue_id": utt.dialogue_id,
                "utterance_idx": utt.utterance_idx,
                "speaker_name": speaker_name,
                "listener_names": "|".join(listeners),
                "predicted_names": "|".join(predicted),
                "num_detected_faces": face_count,
                "speaker_hit": speaker_hit,
                "listener_hit_count": len(listener_hits),
                "participant_hit": participant_hit,
            }
        )

    summary = {
        "utterances_total": len(utterances),
        "utterances_with_face": with_face_count,
        "utterances_with_face_ratio": with_face_count / len(utterances) if utterances else 0.0,
        "utterances_with_known_match": with_known_match_count,
        "utterances_with_known_match_ratio": with_known_match_count / len(utterances) if utterances else 0.0,
        "speaker_hit_utterances": speaker_match_count,
        "speaker_hit_ratio": speaker_match_count / len(utterances) if utterances else 0.0,
        "participant_hit_utterances": participant_match_count,
        "participant_hit_ratio": participant_match_count / len(utterances) if utterances else 0.0,
        "matched_listener_slots": matched_listener_slots,
        "total_listener_slots": total_listener_slots,
        "listener_slot_hit_ratio": matched_listener_slots / total_listener_slots if total_listener_slots else 0.0,
        "matched_roles": len(matched_roles),
        "total_roles": len(set(role_names)),
        "matched_role_ratio": len(matched_roles) / len(set(role_names)) if role_names else 0.0,
        "role_counter_top20": role_counter.most_common(20),
        "cluster_count": len(cluster_matches),
        "known_cluster_count": sum(1 for row in cluster_matches if row.is_known),
        "artifacts": {
            "extracted_faces_dir": "extracted_faces",
            "cluster_gallery_dir": "cluster_gallery",
            "utterance_predictions_csv": "utterance_predictions.csv",
            "cluster_matches_csv": "cluster_matches.csv",
        },
    }
    return summary, utterance_rows, cluster_rows


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    pipeline_summary: dict[str, object],
    reference_bank: list[ReferenceIdentityEmbedding],
) -> None:
    ensure_dir(path.parent)
    lines = [
        "# FaceNet 聚类匹配评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- txt 路径：`{Path(args.txt_path).resolve()}`",
        f"- video 目录：`{Path(args.video_dir).resolve()}`",
        f"- reference 根目录：`{Path(args.reference_root).resolve()}`",
        f"- device：`{args.device}`",
        f"- detect_batch_size：`{args.detect_batch_size}`",
        f"- embed_batch_size：`{args.embed_batch_size}`",
        f"- num_frames：`{args.num_frames}`",
        f"- cluster_threshold：`{args.cluster_threshold}`",
        f"- reference_match_threshold：`{args.reference_match_threshold}`",
        f"- reference_match_margin：`{args.reference_match_margin}`",
        f"- face_verify_threshold：`{args.face_verify_threshold}`",
        "",
        "## 技术路线",
        "",
        "1. 使用 MTCNN 对每条语句视频的采样帧做人脸检测，不区分说话人/听话人，保留所有高置信人脸。",
        "2. 使用预训练 FaceNet (`InceptionResnetV1`, `vggface2`) 批量提取 512 维 embedding。",
        "3. 用余弦相似度阈值做在线聚类，将视觉上相近的人脸聚成 cluster。",
        "4. 对每个 cluster 计算 centroid，再与 reference 人脸库的 prototype 做 1:N 检索匹配。",
        "5. 以 reference 阈值 + top1/top2 margin 做拒识，最后统计 utterance / speaker / participant 级别的覆盖情况。",
        "",
        "## 关键指标",
        "",
        f"- utterance 有脸覆盖率：`{pipeline_summary['utterances_with_face_ratio']:.2%}`",
        f"- utterance 有已知角色匹配率：`{pipeline_summary['utterances_with_known_match_ratio']:.2%}`",
        f"- speaker 命中率：`{pipeline_summary['speaker_hit_ratio']:.2%}`",
        f"- participant 命中率：`{pipeline_summary['participant_hit_ratio']:.2%}`",
        f"- listener 槽位命中率：`{pipeline_summary['listener_slot_hit_ratio']:.2%}`",
        f"- 角色命中覆盖率：`{pipeline_summary['matched_role_ratio']:.2%}`",
        f"- cluster 总数：`{pipeline_summary['cluster_count']}`，其中已知 cluster：`{pipeline_summary['known_cluster_count']}`",
        f"- reference 身份数：`{len(reference_bank)}`",
        "",
        "## 建议调参方向",
        "",
        "1. 如果 GPU 余量足够，可以优先增大 `embed_batch_size`，它对吞吐提升最明显。",
        "2. 如果检测阶段变慢或显存不足，先减小 `detect_batch_size`，检测比 embedding 更容易抖动。",
        "3. 如果同一角色被拆成多个 cluster，适当降低 `cluster_threshold`。",
        "4. 如果误认过多，优先提高 `reference_match_threshold` 或 `reference_match_margin`。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cluster_gallery(
    output_dir: Path,
    face_records: list[FaceRecord],
    assignments: np.ndarray,
    cluster_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    gallery_root = output_dir / "cluster_gallery"
    ensure_dir(gallery_root)
    rows_by_cluster = {int(row["cluster_id"]): row for row in cluster_rows}
    faces_by_cluster: dict[int, list[FaceRecord]] = defaultdict(list)
    for face, cluster_id in zip(face_records, assignments.tolist()):
        faces_by_cluster[int(cluster_id)].append(face)

    for cluster_id, members in tqdm(sorted(faces_by_cluster.items()), desc="[pipeline] write cluster gallery"):
        row = rows_by_cluster.get(cluster_id, {})
        assigned_name = str(row.get("assigned_name", "unknown"))
        folder_name = f"cluster_{cluster_id:05d}__{safe_fs_name(assigned_name)}"
        cluster_dir = gallery_root / folder_name
        images_dir = cluster_dir / "images"
        ensure_dir(images_dir)

        for index, face in enumerate(members, start=1):
            src = Path(face.crop_path)
            ext = src.suffix.lower() or ".jpg"
            dst = images_dir / f"{index:05d}_{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)

        explanation_lines = [
            f"cluster_id: {cluster_id}",
            f"cluster_size: {row.get('size', len(members))}",
            f"assigned_name: {assigned_name}",
            f"best_name_before_reject: {row.get('best_name', assigned_name)}",
            f"is_known: {row.get('is_known', False)}",
            f"best_score: {row.get('best_score', 0.0):.4f}",
            f"second_score: {row.get('second_score', 0.0):.4f}",
            f"score_margin: {row.get('score_margin', 0.0):.4f}",
            f"top_candidates: {row.get('top_candidates', '')}",
            "",
            "一致性统计：",
            f"- face_verify_threshold: {row.get('face_verify_threshold', 0.0):.4f}",
            f"- consistent_face_count: {row.get('consistent_face_count', 0)}",
            f"- consistent_face_ratio: {row.get('consistent_face_ratio', 0.0):.4f}",
            f"- mean_member_similarity_to_centroid: {row.get('mean_member_similarity_to_centroid', 0.0):.4f}",
            f"- min_member_similarity_to_centroid: {row.get('min_member_similarity_to_centroid', 0.0):.4f}",
            f"- max_member_similarity_to_centroid: {row.get('max_member_similarity_to_centroid', 0.0):.4f}",
            "",
            "关键超参数：",
            f"- num_frames: {args.num_frames}",
            f"- min_detect_confidence: {args.min_detect_confidence}",
            f"- max_faces_per_frame: {args.max_faces_per_frame}",
            f"- detect_batch_size: {args.detect_batch_size}",
            f"- embed_batch_size: {args.embed_batch_size}",
            f"- cluster_threshold: {args.cluster_threshold}",
            f"- reference_match_threshold: {args.reference_match_threshold}",
            f"- reference_match_margin: {args.reference_match_margin}",
            f"- face_verify_threshold: {args.face_verify_threshold}",
            "",
            "成员样本：",
        ]
        for face in members[:50]:
            explanation_lines.append(
                f"- face_id={face.face_id}, dialogue={face.dialogue_id}, utterance={face.utterance_idx}, "
                f"frame={face.frame_order}, speaker={face.speaker_name}, listeners={face.listener_names}, "
                f"crop={face.crop_path}"
            )
        (cluster_dir / "说明.txt").write_text("\n".join(explanation_lines), encoding="utf-8")


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, embeddings)


def flush_embedding_buffer(
    embedder: InceptionResnetV1,
    device: torch.device,
    pending_crops: list[Image.Image],
    embedding_chunks: list[np.ndarray],
    image_size: int,
    batch_size: int,
    use_fp16: bool,
) -> None:
    if not pending_crops:
        return
    chunk = embed_crops(
        embedder,
        device,
        pending_crops,
        image_size=image_size,
        batch_size=batch_size,
        use_fp16=use_fp16,
    )
    embedding_chunks.append(chunk)
    pending_crops.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="MTCNN + 预训练 FaceNet + 聚类 + reference 检索评估脚本。")
    parser.add_argument("--txt-path", type=Path, default=DEFAULT_TXT_PATH, help="中文数据集 txt 路径。")
    parser.add_argument("--video-dir", type=Path, default=None, help="视频目录；留空时按 txt 路径推断。")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="reference 根目录。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录。")
    parser.add_argument("--run-name", type=str, default="pipeline_latest", help="输出子目录名。")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="cpu / cuda / cuda:0。")
    parser.add_argument("--num-frames", type=int, default=6, help="每条语句采样多少帧。")
    parser.add_argument("--frame-strategy", type=str, default="uniform", choices=["uniform", "middle", "first"], help="视频采样策略。")
    parser.add_argument("--image-size", type=int, default=160, help="FaceNet 输入尺寸。")
    parser.add_argument("--mtcnn-margin", type=int, default=12, help="MTCNN margin。")
    parser.add_argument("--mtcnn-min-face-size", type=int, default=24, help="MTCNN 最小脸尺寸。")
    parser.add_argument("--mtcnn-thresholds", type=float, nargs=3, default=[0.6, 0.7, 0.7], help="MTCNN 三阶段阈值。")
    parser.add_argument("--min-detect-confidence", type=float, default=0.95, help="人脸检测最小置信度。")
    parser.add_argument("--max-faces-per-frame", type=int, default=8, help="每帧最多保留多少张人脸。")
    parser.add_argument("--detect-batch-size", type=int, default=8, help="MTCNN 批量检测帧数。")
    parser.add_argument("--embed-batch-size", type=int, default=256, help="FaceNet embedding 批大小。RTX A6000 上建议先从 256 起。")
    parser.add_argument("--reference-embed-batch-size", type=int, default=256, help="reference embedding 批大小。")
    parser.add_argument("--cluster-threshold", type=float, default=0.72, help="在线聚类余弦阈值。")
    parser.add_argument("--reference-match-threshold", type=float, default=0.80, help="cluster 到 reference 的最小相似度。")
    parser.add_argument("--reference-match-margin", type=float, default=0.03, help="top1 与 top2 的最小间隔。")
    parser.add_argument("--face-verify-threshold", type=float, default=0.75, help="两张脸是否视为同人的一致性阈值，用于 cluster 内部一致性评估。")
    parser.add_argument("--max-dialogues", type=int, default=None, help="只跑前 N 个 dialogue，用于 smoke。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录。")
    parser.add_argument("--use-fp16", action="store_true", help="在 CUDA 上对 embedding 阶段启用 autocast。")
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(
            f"缺少依赖，无法运行流水线脚本: {IMPORT_ERROR}。"
            "请在目标环境中安装 numpy / torch / pillow / opencv / facenet_pytorch。"
        )

    txt_path = args.txt_path.resolve()
    video_dir = infer_video_dir(txt_path, args.video_dir)
    reference_root = args.reference_root.resolve()
    output_dir = (args.output_root / args.run_name).resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在：{output_dir}；如需复用请传 --overwrite")
    ensure_dir(output_dir)

    device = torch.device(args.device)
    dialogues = parse_dev_txt(txt_path)
    utterances = iter_utterances(dialogues, max_dialogues=args.max_dialogues)
    role_names = [normalize_name(utt.speaker.name) for utt in utterances]
    role_names.extend(normalize_name(listener.name) for utt in utterances for listener in utt.listeners)

    mtcnn = MTCNN(
        image_size=args.image_size,
        margin=args.mtcnn_margin,
        min_face_size=args.mtcnn_min_face_size,
        thresholds=args.mtcnn_thresholds,
        keep_all=True,
        device=device,
    )
    embedder = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

    reference_bank, ref_count_by_name = build_reference_bank(
        embedder,
        device,
        reference_root=reference_root,
        image_size=args.image_size,
        batch_size=args.reference_embed_batch_size,
        use_fp16=args.use_fp16,
    )
    write_json(
        output_dir / "reference_bank_summary.json",
        [
            {"name": row.name, "image_count": row.image_count}
            for row in reference_bank
        ],
    )
    if reference_bank:
        save_embeddings(output_dir / "reference_prototypes.npy", np.stack([row.prototype for row in reference_bank], axis=0))
    else:
        save_embeddings(output_dir / "reference_prototypes.npy", np.zeros((0, 512), dtype=np.float32))

    face_records: list[FaceRecord] = []
    pending_crops: list[Image.Image] = []
    embedding_chunks: list[np.ndarray] = []
    face_id = 0
    flush_trigger = max(args.embed_batch_size * 4, 512)
    extracted_root = output_dir / "extracted_faces"
    ensure_dir(extracted_root)

    for utt in tqdm(utterances, desc="[pipeline] utterances"):
        video_path = build_video_path(video_dir, utt)
        if not video_path.exists():
            continue
        frames = sample_frames(video_path, num_frames=args.num_frames, strategy=args.frame_strategy)
        pil_frames = [to_pil(frame) for frame in frames]
        for start in range(0, len(pil_frames), args.detect_batch_size):
            batch_frames = pil_frames[start:start + args.detect_batch_size]
            detected = detect_faces_batched(
                mtcnn,
                batch_frames,
                min_confidence=args.min_detect_confidence,
                max_faces_per_frame=args.max_faces_per_frame,
            )
            for local_idx, faces in enumerate(detected):
                frame_order = start + local_idx
                for box, conf, crop in faces:
                    utterance_face_dir = extracted_root / f"C_{utt.dialogue_id:04d}" / f"U_{utt.utterance_idx:04d}"
                    ensure_dir(utterance_face_dir)
                    crop_filename = (
                        f"face_{face_id:07d}_f{frame_order:02d}_"
                        f"{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg"
                    )
                    crop_path = utterance_face_dir / crop_filename
                    crop.convert("RGB").save(crop_path, format="JPEG", quality=95)
                    face_records.append(
                        FaceRecord(
                            face_id=face_id,
                            dialogue_id=utt.dialogue_id,
                            utterance_idx=utt.utterance_idx,
                            video_path=str(video_path),
                            frame_order=frame_order,
                            box_x1=box[0],
                            box_y1=box[1],
                            box_x2=box[2],
                            box_y2=box[3],
                            detect_confidence=conf,
                            participant_names="|".join(participant_names(utt)),
                            speaker_name=normalize_name(utt.speaker.name),
                            listener_names="|".join(normalize_name(x.name) for x in utt.listeners),
                            crop_path=str(crop_path),
                        )
                    )
                    pending_crops.append(crop)
                    face_id += 1
                    if len(pending_crops) >= flush_trigger:
                        flush_embedding_buffer(
                            embedder,
                            device,
                            pending_crops,
                            embedding_chunks,
                            image_size=args.image_size,
                            batch_size=args.embed_batch_size,
                            use_fp16=args.use_fp16,
                        )

    flush_embedding_buffer(
        embedder,
        device,
        pending_crops,
        embedding_chunks,
        image_size=args.image_size,
        batch_size=args.embed_batch_size,
        use_fp16=args.use_fp16,
    )
    embeddings = np.concatenate(embedding_chunks, axis=0) if embedding_chunks else np.zeros((0, 512), dtype=np.float32)
    save_embeddings(output_dir / "query_embeddings.npy", embeddings)
    write_csv(output_dir / "face_records.csv", [asdict(row) for row in face_records])

    cluster_assignments, centroids = online_cluster_embeddings(embeddings, threshold=args.cluster_threshold)
    save_embeddings(output_dir / "cluster_centroids.npy", centroids)
    cluster_matches = assign_clusters_to_reference(
        centroids,
        reference_bank=reference_bank,
        match_threshold=args.reference_match_threshold,
        match_margin=args.reference_match_margin,
    )
    summary, utterance_rows, cluster_rows = evaluate_predictions(
        utterances,
        face_records,
        cluster_assignments,
        cluster_matches,
        centroids=centroids,
        embeddings=embeddings,
        role_names=role_names,
        ref_count_by_name=ref_count_by_name,
        face_verify_threshold=args.face_verify_threshold,
    )
    write_json(output_dir / "pipeline_summary.json", summary)
    write_csv(output_dir / "cluster_matches.csv", cluster_rows)
    write_csv(output_dir / "utterance_predictions.csv", utterance_rows)
    write_cluster_gallery(
        output_dir=output_dir,
        face_records=face_records,
        assignments=cluster_assignments,
        cluster_rows=cluster_rows,
        args=args,
    )
    write_markdown_report(output_dir / "pipeline_report.md", args=args, pipeline_summary=summary, reference_bank=reference_bank)
    write_json(
        output_dir / "run_config.json",
        {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "txt_path": str(txt_path),
            "video_dir": str(video_dir),
            "reference_root": str(reference_root),
            "output_dir": str(output_dir),
        },
    )

    print(f"[facenet_cluster_pipeline] output_dir={output_dir}")
    print(f"[facenet_cluster_pipeline] total_faces={len(face_records)} total_clusters={len(cluster_matches)}")
    print(f"[facenet_cluster_pipeline] utterances_with_face_ratio={summary['utterances_with_face_ratio']:.2%}")
    print(f"[facenet_cluster_pipeline] speaker_hit_ratio={summary['speaker_hit_ratio']:.2%}")
    print(f"[facenet_cluster_pipeline] participant_hit_ratio={summary['participant_hit_ratio']:.2%}")
    print(f"[facenet_cluster_pipeline] matched_role_ratio={summary['matched_role_ratio']:.2%}")


if __name__ == "__main__":
    main()
