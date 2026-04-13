#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig  # noqa: E402
from src.parser import parse_dev_txt  # noqa: E402


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _check(condition: bool, ok_msg: str, fail_msg: str) -> None:
    if condition:
        print(f"[ok] {ok_msg}", flush=True)
        return
    raise SystemExit(f"[error] {fail_msg}")


@dataclass(frozen=True)
class WorkflowPaths:
    dataset_root: Path
    speaker_split: str
    target_split: str
    speaker_split_dir: Path
    target_split_dir: Path
    speaker_txt: Path
    target_txt: Path
    speaker_video_dir: Path
    target_video_dir: Path
    generated_dir: Path
    main_config_path: Path
    dataset_build_config_path: Path
    facenet_train_config_path: Path
    feature_root: Path
    merged_output_dir: Path
    face_work_dir: Path
    face_image_root: Path
    face_manifest_dir: Path
    face_meta_dir: Path
    face_review_dir: Path
    facenet_split_dir: Path
    facenet_output_dir: Path
    face_checkpoint: Path


def build_workflow_paths(
    dataset_root: Path,
    speaker_split: str,
    target_split: str,
    generated_dir: Path,
    feature_root: Optional[Path],
    merged_output_dir: Optional[Path],
    face_work_dir: Optional[Path],
) -> WorkflowPaths:
    speaker_split_dir = dataset_root / speaker_split
    target_split_dir = dataset_root / target_split
    default_face_work_dir = PROJECT_ROOT / "face_name_id" / "artifacts_zh" / f"{speaker_split}_speaker_model"
    resolved_face_work_dir = (face_work_dir or default_face_work_dir).resolve()
    resolved_generated_dir = generated_dir.resolve()
    resolved_feature_root = (
        feature_root.resolve()
        if feature_root is not None
        else (PROJECT_ROOT / "artifacts_zh" / "features" / target_split).resolve()
    )
    resolved_merged_output = (
        merged_output_dir.resolve()
        if merged_output_dir is not None
        else target_split_dir.resolve()
    )

    return WorkflowPaths(
        dataset_root=dataset_root.resolve(),
        speaker_split=speaker_split,
        target_split=target_split,
        speaker_split_dir=speaker_split_dir.resolve(),
        target_split_dir=target_split_dir.resolve(),
        speaker_txt=(speaker_split_dir / f"{speaker_split}.txt").resolve(),
        target_txt=(target_split_dir / f"{target_split}.txt").resolve(),
        speaker_video_dir=(speaker_split_dir / f"Video_{speaker_split}").resolve(),
        target_video_dir=(target_split_dir / f"Video_{target_split}").resolve(),
        generated_dir=resolved_generated_dir,
        main_config_path=(resolved_generated_dir / f"main_{target_split}_zh.yaml").resolve(),
        dataset_build_config_path=(resolved_generated_dir / f"dataset_build_{speaker_split}_zh.yaml").resolve(),
        facenet_train_config_path=(resolved_generated_dir / f"facenet_fr_{speaker_split}_zh.yaml").resolve(),
        feature_root=(resolved_feature_root / f"Video_{target_split}_face_scene_fr").resolve(),
        merged_output_dir=resolved_merged_output,
        face_work_dir=resolved_face_work_dir,
        face_image_root=(resolved_face_work_dir / "dataset" / "images").resolve(),
        face_manifest_dir=(resolved_face_work_dir / "dataset" / "manifests").resolve(),
        face_meta_dir=(resolved_face_work_dir / "dataset" / "meta").resolve(),
        face_review_dir=(resolved_face_work_dir / "dataset" / "review").resolve(),
        facenet_split_dir=(resolved_face_work_dir / "facenet_fr" / "splits").resolve(),
        facenet_output_dir=(resolved_face_work_dir / "facenet_fr" / "outputs").resolve(),
        face_checkpoint=(resolved_face_work_dir / "facenet_fr" / "outputs" / "checkpoints" / "best.pt").resolve(),
    )


def validate_inputs(paths: WorkflowPaths) -> None:
    _print_header("Validate Inputs")
    checks = [
        (paths.dataset_root.exists(), f"dataset_root exists: {paths.dataset_root}", f"dataset_root not found: {paths.dataset_root}"),
        (paths.speaker_split_dir.exists(), f"speaker split dir exists: {paths.speaker_split_dir}", f"speaker split dir not found: {paths.speaker_split_dir}"),
        (paths.target_split_dir.exists(), f"target split dir exists: {paths.target_split_dir}", f"target split dir not found: {paths.target_split_dir}"),
        (paths.speaker_txt.exists(), f"speaker txt exists: {paths.speaker_txt}", f"speaker txt not found: {paths.speaker_txt}"),
        (paths.target_txt.exists(), f"target txt exists: {paths.target_txt}", f"target txt not found: {paths.target_txt}"),
        (paths.speaker_video_dir.exists(), f"speaker video dir exists: {paths.speaker_video_dir}", f"speaker video dir not found: {paths.speaker_video_dir}"),
        (paths.target_video_dir.exists(), f"target video dir exists: {paths.target_video_dir}", f"target video dir not found: {paths.target_video_dir}"),
    ]
    for condition, ok_msg, fail_msg in checks:
        _check(condition, ok_msg, fail_msg)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    print(f"[ok] wrote config: {path}", flush=True)


def generate_configs(paths: WorkflowPaths, smoke_max_dialogues: int) -> None:
    _print_header("Generate Configs")
    dataset_build_payload = {
        "project": {"seed": 42},
        "paths": {
            "host_project_root": str(PROJECT_ROOT),
            "dev_txt": str(paths.speaker_txt),
            "video_dir": str(paths.speaker_video_dir),
            "work_dir": str(paths.face_work_dir),
            "image_root": str(paths.face_image_root),
            "manifest_dir": str(paths.face_manifest_dir),
            "meta_dir": str(paths.face_meta_dir),
            "review_dir": str(paths.face_review_dir),
        },
        "labeling": {
            "strategy": "topk_plus_other",
            "topk_speakers": 12,
            "whitelist_names": [],
            "other_label": "other",
        },
        "sampling": {
            "frames_per_utterance": 6,
            "strategy": "uniform",
            "resize_short_side": 720,
        },
        "face_detection": {
            "device": "cpu",
            "image_size": 160,
            "margin": 12,
            "min_face_size": 24,
            "thresholds": [0.6, 0.7, 0.7],
            "keep_all": True,
        },
        "build": {
            "keep_single_face_only": True,
            "min_face_confidence": 0.95,
            "jpeg_quality": 95,
            "skip_existing": True,
            "filename_pattern": "{label}__d{dialogue_id:04d}_u{utterance_idx:04d}_f{frame_idx:04d}.jpg",
            "write_manifest": True,
            "write_meta": True,
            "verbose_log": True,
        },
        "augmentation": {
            "enable_multi_face_weak_label": False,
            "min_anchor_samples_per_label": 20,
            "min_similarity": 0.72,
        },
        "cleaning": {
            "enable": False,
            "method": "kmeans2",
            "dbscan_eps": 0.65,
            "dbscan_min_samples": 5,
            "min_keep_per_label": 20,
        },
        "web_review": {
            "host": "127.0.0.1",
            "port": 8765,
        },
    }
    facenet_train_payload = {
        "model": {"pretrained": "vggface2", "topk": 3},
        "split": {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1, "seed": 42},
        "data": {"image_size": 160, "batch_size": 32, "num_workers": 4},
        "train": {
            "epochs": 20,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "dropout": 0.2,
            "freeze_backbone": False,
            "label_smoothing": 0.05,
            "early_stop_patience": 5,
        },
        "paths": {
            "image_root": str(paths.face_image_root),
            "rejected_root": str(paths.face_review_dir / "rejected_images"),
            "split_dir": str(paths.facenet_split_dir),
            "output_dir": str(paths.facenet_output_dir),
        },
    }
    main_payload = {
        "paths": {
            "split_name": paths.target_split,
            "dev_txt": str(paths.target_txt),
            "video_dir": str(paths.target_video_dir),
            "feat_out": str(paths.feature_root),
        },
        "extractor": {
            "active_type": "face_scene_fr",
        },
        "visual_clip_config": {
            "model_name": "ViT-B/32",
            "device": "auto",
            "clip_output_dim": 512,
            "target_dim": 1024,
            "frame_sampling": {
                "strategy": "uniform",
                "num_frames": 8,
                "aggregation": "mean",
            },
        },
        "face_scene_fr_config": {
            "device": "auto",
            "face_checkpoint": str(paths.face_checkpoint),
            "clip_model_name": "ViT-B/32",
            "frame_sampling": {
                "strategy": "uniform",
                "num_frames": 8,
                "aggregation": "mean",
            },
            "person_num_frames": 3,
            "mtcnn_image_size": 160,
            "mtcnn_margin": 12,
            "mtcnn_min_face_size": 24,
            "mtcnn_thresholds": [0.6, 0.7, 0.7],
            "mtcnn_keep_all": True,
            "face_batch_size": 64,
            "min_detection_confidence": 0.95,
            "classification_strategy": "top1",
            "min_classification_confidence": 0.5,
            "unknown_person_strategy": "other_mean",
            "other_label_name": "other",
        },
        "non_speaker": {
            "strategy": "context_video",
            "fallback": "zero",
        },
        "pipeline": {
            "skip_existing": False,
            "show_progress": True,
        },
        "workflow_hints": {
            "speaker_split_for_face_model": paths.speaker_split,
            "target_split_for_feature_extraction": paths.target_split,
            "smoke_max_dialogues": smoke_max_dialogues,
        },
    }
    write_yaml(paths.dataset_build_config_path, dataset_build_payload)
    write_yaml(paths.facenet_train_config_path, facenet_train_payload)
    write_yaml(paths.main_config_path, main_payload)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2), flush=True)


def build_python_command(
    script_path: Path,
    script_args: Sequence[str],
    use_conda_run: bool,
    conda_env: str,
) -> list[str]:
    if use_conda_run:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u", str(script_path), *script_args]
    return [sys.executable, str(script_path), *script_args]


def run_step(title: str, cmd: Sequence[str]) -> None:
    _print_header(title)
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=PROJECT_ROOT, check=True)


def run_role_stats(paths: WorkflowPaths, use_conda_run: bool, conda_env: str) -> None:
    out_csv = paths.generated_dir / f"role_stats_{paths.speaker_split}.csv"
    out_json = paths.generated_dir / f"role_stats_{paths.speaker_split}.json"
    cmd = build_python_command(
        PROJECT_ROOT / "scripts" / "stat_role_frequencies.py",
        ["--txt-path", str(paths.speaker_txt), "--out-csv", str(out_csv), "--out-json", str(out_json)],
        use_conda_run,
        conda_env,
    )
    run_step("Role Stats", cmd)
    _check(out_csv.exists(), f"role stats csv created: {out_csv}", f"missing role stats csv: {out_csv}")
    _check(out_json.exists(), f"role stats json created: {out_json}", f"missing role stats json: {out_json}")


def run_build_face_dataset(paths: WorkflowPaths, use_conda_run: bool, conda_env: str) -> None:
    cmd = build_python_command(
        PROJECT_ROOT / "face_name_id" / "scripts" / "build_dataset.py",
        ["--config", str(paths.dataset_build_config_path)],
        use_conda_run,
        conda_env,
    )
    run_step("Build Face Dataset", cmd)
    image_count = len(list(paths.face_image_root.glob("*.jpg")))
    _check(image_count > 0, f"face dataset built with {image_count} images", f"no face images found under: {paths.face_image_root}")


def run_train_face_model(paths: WorkflowPaths, use_conda_run: bool, conda_env: str) -> None:
    cmd = build_python_command(
        PROJECT_ROOT / "face_name_id" / "scripts" / "train_facenet_fr.py",
        ["--config", str(paths.facenet_train_config_path)],
        use_conda_run,
        conda_env,
    )
    run_step("Train Face Model", cmd)
    _check(paths.face_checkpoint.exists(), f"face checkpoint created: {paths.face_checkpoint}", f"face checkpoint missing: {paths.face_checkpoint}")


def run_extract(paths: WorkflowPaths, use_conda_run: bool, conda_env: str, smoke: bool, max_dialogues: int) -> None:
    args = ["--config", str(paths.main_config_path)]
    stage_name = "Extract Features"
    if smoke:
        args.extend(["--smoke", "--max-dialogues", str(max_dialogues)])
        stage_name = "Extract Features (Smoke)"
    cmd = build_python_command(PROJECT_ROOT / "main.py", args, use_conda_run, conda_env)
    run_step(stage_name, cmd)
    pt_count = len(list(paths.feature_root.glob("C_*/*.pt")))
    _check(pt_count > 0, f"feature tensors generated: {pt_count}", f"no feature tensors found under: {paths.feature_root}")


def run_merge(paths: WorkflowPaths, use_conda_run: bool, conda_env: str) -> None:
    cmd = build_python_command(
        PROJECT_ROOT / "scripts" / "merge_video_dev_features.py",
        ["--config", str(paths.main_config_path), "--output-dir", str(paths.merged_output_dir)],
        use_conda_run,
        conda_env,
    )
    run_step("Merge Features", cmd)
    emb_path = paths.merged_output_dir / f"video_embedding_{paths.target_split}.npy"
    map_path = paths.merged_output_dir / f"video_id_mapping_{paths.target_split}.npy"
    _check(emb_path.exists(), f"merged embedding created: {emb_path}", f"merged embedding missing: {emb_path}")
    _check(map_path.exists(), f"merged mapping created: {map_path}", f"merged mapping missing: {map_path}")


def verify_outputs(paths: WorkflowPaths) -> None:
    _print_header("Verify Outputs")
    target_dialogues = parse_dev_txt(paths.target_txt)
    expected_utterances = sum(len(dialogue.utterances) for dialogue in target_dialogues)
    actual_pt = len(list(paths.feature_root.glob("C_*/*.pt")))
    print(f"[info] expected utterances: {expected_utterances}", flush=True)
    print(f"[info] actual .pt files  : {actual_pt}", flush=True)
    if paths.face_checkpoint.exists():
        print(f"[ok] face checkpoint exists: {paths.face_checkpoint}", flush=True)
    else:
        print(f"[warn] face checkpoint not found yet: {paths.face_checkpoint}", flush=True)

    if actual_pt == expected_utterances and expected_utterances > 0:
        print("[ok] per-utterance feature count matches target split", flush=True)
    else:
        print("[warn] per-utterance feature count does not match target split yet", flush=True)

    emb_path = paths.merged_output_dir / f"video_embedding_{paths.target_split}.npy"
    map_path = paths.merged_output_dir / f"video_id_mapping_{paths.target_split}.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        print(f"[ok] merged embedding shape={emb.shape} dtype={emb.dtype}", flush=True)
    else:
        print(f"[warn] merged embedding not found: {emb_path}", flush=True)
    if map_path.exists():
        mapping = np.load(map_path, allow_pickle=True).item()
        print(f"[ok] merged mapping entries={len(mapping)}", flush=True)
    else:
        print(f"[warn] merged mapping not found: {map_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate configs and run the Chinese dataset feature workflow step by step.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="例如 /data16T_1/.../data_zh")
    parser.add_argument("--speaker-split", type=str, default="train", help="用于人脸识别模型训练的数据划分，默认 train")
    parser.add_argument("--target-split", type=str, default="dev", help="要提取视频特征的数据划分，默认 dev")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "generated_configs" / "zh_workflow")
    parser.add_argument("--feature-root", type=Path, default=None, help="中间 .pt 特征输出根目录")
    parser.add_argument("--merged-output-dir", type=Path, default=None, help="最终 .npy 输出目录，默认写回目标 split 目录")
    parser.add_argument("--face-work-dir", type=Path, default=None, help="中文人脸数据集与 checkpoint 产物目录")
    parser.add_argument("--smoke-max-dialogues", type=int, default=2)
    parser.add_argument("--use-conda-run", action="store_true", help="通过 conda run 调起各阶段脚本")
    parser.add_argument("--conda-env", type=str, default="security")
    parser.add_argument(
        "--run",
        nargs="*",
        default=[],
        choices=[
            "generate-configs",
            "role-stats",
            "build-face-dataset",
            "train-face-model",
            "extract-smoke",
            "extract-full",
            "merge",
            "verify",
        ],
        help="留空时只做输入检查并生成配置；传入后按顺序执行这些阶段。",
    )
    args = parser.parse_args()

    paths = build_workflow_paths(
        dataset_root=args.dataset_root,
        speaker_split=args.speaker_split,
        target_split=args.target_split,
        generated_dir=args.generated_dir,
        feature_root=args.feature_root,
        merged_output_dir=args.merged_output_dir,
        face_work_dir=args.face_work_dir,
    )
    validate_inputs(paths)
    generate_configs(paths, smoke_max_dialogues=args.smoke_max_dialogues)

    for stage in args.run:
        if stage == "generate-configs":
            continue
        if stage == "role-stats":
            run_role_stats(paths, args.use_conda_run, args.conda_env)
        elif stage == "build-face-dataset":
            run_build_face_dataset(paths, args.use_conda_run, args.conda_env)
        elif stage == "train-face-model":
            run_train_face_model(paths, args.use_conda_run, args.conda_env)
        elif stage == "extract-smoke":
            run_extract(paths, args.use_conda_run, args.conda_env, smoke=True, max_dialogues=args.smoke_max_dialogues)
        elif stage == "extract-full":
            run_extract(paths, args.use_conda_run, args.conda_env, smoke=False, max_dialogues=args.smoke_max_dialogues)
        elif stage == "merge":
            run_merge(paths, args.use_conda_run, args.conda_env)
        elif stage == "verify":
            verify_outputs(paths)


if __name__ == "__main__":
    main()
