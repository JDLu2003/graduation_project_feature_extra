#!/usr/bin/env python3
"""Merge per-utterance visual feature tensors into dev-level numpy files.

Steps:
1. Parse dev annotations with src.parser.parse_dev_txt.
2. Validate every utterance tensor (.pt): existence, 2D shape, and row count.
3. Merge rows in dev order.
4. Add a zero padding row at index 0 for direct 1-based indexing.
5. Save to feat_out/merge by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig  # noqa: E402
from src.parser import parse_dev_txt  # noqa: E402


def _utt_key(dialogue_id: int, utterance_idx: int) -> str:
    return f"C_{dialogue_id}_U_{utterance_idx}"


def _expected_output_dim_from_config(app_config: AppConfig) -> int | None:
    if app_config.extractor.active_type == "visual_clip":
        cfg = app_config.extractor.visual_clip_config
        return None if cfg is None else int(cfg.target_dim)
    if app_config.extractor.active_type == "face_scene_fr":
        return 1024
    return None


def validate_and_collect(
    app_config: AppConfig,
    feat_root: Path,
) -> Tuple[List[np.ndarray], Dict[str, List[int]], int]:
    dialogues = parse_dev_txt(app_config.paths.dev_txt)
    expected_dim = _expected_output_dim_from_config(app_config)

    rows: List[np.ndarray] = []
    mapping: Dict[str, List[int]] = {}
    running_idx = 1
    feature_dim: int | None = None
    total_utterances = 0
    source_dtype: np.dtype | None = None

    for dialogue in dialogues:
        for utt in dialogue.utterances:
            total_utterances += 1
            key = _utt_key(utt.dialogue_id, utt.utterance_idx)
            pt_path = feat_root / f"C_{utt.dialogue_id}" / f"{key}.pt"

            if not pt_path.exists():
                raise FileNotFoundError(f"Missing feature file: {pt_path}")

            tensor = torch.load(pt_path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Not a torch.Tensor: {pt_path} ({type(tensor)})")
            if tensor.ndim != 2:
                raise ValueError(f"Expected 2D tensor at {pt_path}, got {tensor.ndim}D")

            expected_rows = 1 + len(utt.listeners)
            if tensor.shape[0] != expected_rows:
                raise ValueError(
                    f"Row mismatch at {pt_path}: expected {expected_rows}, got {tensor.shape[0]}"
                )

            cur_dim = int(tensor.shape[1])
            if feature_dim is None:
                feature_dim = cur_dim
                if expected_dim is not None and feature_dim != expected_dim:
                    raise ValueError(
                        f"Feature dim mismatch: expected {expected_dim}, got {feature_dim} ({pt_path})"
                    )
            elif cur_dim != feature_dim:
                raise ValueError(
                    f"Inconsistent feature dim at {pt_path}: expected {feature_dim}, got {cur_dim}"
                )

            arr_raw = tensor.detach().cpu().numpy()
            cur_dtype = arr_raw.dtype
            if source_dtype is None:
                source_dtype = cur_dtype
                print(f"[merge] Detected source dtype: {source_dtype}")
                if source_dtype != np.float64:
                    raise ValueError(
                        f"Source dtype must be float64, but got {source_dtype} at {pt_path}"
                    )
            elif cur_dtype != source_dtype:
                raise ValueError(
                    f"Inconsistent source dtype at {pt_path}: expected {source_dtype}, got {cur_dtype}"
                )

            arr = arr_raw.astype(np.float64, copy=False)
            rows.append(arr)

            end_idx = running_idx + expected_rows - 1
            mapping[key] = list(range(running_idx, end_idx + 1))
            running_idx = end_idx + 1

    if not rows or feature_dim is None:
        raise RuntimeError("No feature rows collected.")

    print(
        f"[merge] Validation passed: {total_utterances} utterances, "
        f"{running_idx - 1} rows, feature_dim={feature_dim}."
    )
    return rows, mapping, feature_dim


def save_outputs(rows: List[np.ndarray], mapping: Dict[str, List[int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "video_embedding_dev.npy"
    map_path = output_dir / "video_id_mapping_dev.npy"

    merged = np.concatenate(rows, axis=0).astype(np.float64, copy=False)
    zero_row = np.zeros((1, merged.shape[1]), dtype=np.float64)
    merged_with_padding = np.concatenate([zero_row, merged], axis=0)

    np.save(emb_path, merged_with_padding)
    np.save(map_path, mapping)

    print(
        f"[merge] Saved embedding: {emb_path} "
        f"shape={merged_with_padding.shape} dtype={merged_with_padding.dtype}"
    )
    print(f"[merge] Saved mapping:   {map_path} entries={len(mapping)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge dev visual features to npy")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--feat-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: feat_out/merge",
    )
    args = parser.parse_args()

    app_config = AppConfig.from_yaml(args.config)
    feat_root = args.feat_root.resolve() if args.feat_root else app_config.paths.feat_out
    output_dir = args.output_dir.resolve() if args.output_dir else (feat_root.parent / "merge")

    print(f"[merge] dev_txt   : {app_config.paths.dev_txt}")
    print(f"[merge] feat_root : {feat_root}")
    print(f"[merge] output_dir: {output_dir}")

    rows, mapping, _ = validate_and_collect(app_config, feat_root)
    save_outputs(rows, mapping, output_dir)


if __name__ == "__main__":
    main()
