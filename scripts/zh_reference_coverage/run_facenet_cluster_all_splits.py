#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "zh_reference_coverage" / "run_facenet_cluster_pipeline.py"
DEFAULT_DATASET_ROOT = Path("/data16T_1/sunshengzhe/lujiading/data_zh")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "zh_reference_coverage" / "artifacts"
DEFAULT_REFERENCE_ROOT = Path("/data16T_2/sunshengzhe/reference_face_zh")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="在 train/dev/test 三个 split 上运行 FaceNet 聚类匹配评估并汇总指标。")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="中文数据集根目录。")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="reference 根目录。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录。")
    parser.add_argument("--run-prefix", type=str, default="pipeline_all_splits", help="各 split 输出目录名前缀。")
    parser.add_argument("--device", type=str, default="cuda", help="cpu / cuda / cuda:0。")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--min-detect-confidence", type=float, default=0.95)
    parser.add_argument("--detect-batch-size", type=int, default=8)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--reference-embed-batch-size", type=int, default=256)
    parser.add_argument("--cluster-threshold", type=float, default=0.72)
    parser.add_argument("--reference-match-threshold", type=float, default=0.80)
    parser.add_argument("--reference-match-margin", type=float, default=0.03)
    parser.add_argument("--face-verify-threshold", type=float, default=0.75)
    parser.add_argument("--max-faces-per-frame", type=int, default=8)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    for split in ["train", "dev", "test"]:
        txt_path = (args.dataset_root / split / f"{split}.txt").resolve()
        video_dir = (args.dataset_root / split / f"Video_{split}").resolve()
        run_name = f"{args.run_prefix}_{split}"
        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--txt-path", str(txt_path),
            "--video-dir", str(video_dir),
            "--reference-root", str(args.reference_root.resolve()),
            "--output-root", str(args.output_root.resolve()),
            "--run-name", run_name,
            "--device", args.device,
            "--num-frames", str(args.num_frames),
            "--min-detect-confidence", str(args.min_detect_confidence),
            "--detect-batch-size", str(args.detect_batch_size),
            "--embed-batch-size", str(args.embed_batch_size),
            "--reference-embed-batch-size", str(args.reference_embed_batch_size),
            "--cluster-threshold", str(args.cluster_threshold),
            "--reference-match-threshold", str(args.reference_match_threshold),
            "--reference-match-margin", str(args.reference_match_margin),
            "--face-verify-threshold", str(args.face_verify_threshold),
            "--max-faces-per-frame", str(args.max_faces_per_frame),
        ]
        if args.use_fp16:
            cmd.append("--use-fp16")
        if args.overwrite:
            cmd.append("--overwrite")
        print("[all_splits] run:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary_path = args.output_root.resolve() / run_name / "pipeline_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["summary_path"] = str(summary_path)
        summaries.append(payload)

    aggregate = {
        "splits": summaries,
        "macro_avg": {
            "utterances_with_face_ratio": sum(float(x["utterances_with_face_ratio"]) for x in summaries) / len(summaries),
            "speaker_hit_ratio": sum(float(x["speaker_hit_ratio"]) for x in summaries) / len(summaries),
            "matched_listener_role_ratio": sum(float(x["matched_listener_role_ratio"]) for x in summaries) / len(summaries),
            "avg_listener_recall_per_utterance_non_empty": sum(float(x["avg_listener_recall_per_utterance_non_empty"]) for x in summaries) / len(summaries),
            "participant_hit_ratio": sum(float(x["participant_hit_ratio"]) for x in summaries) / len(summaries),
            "all_participants_hit_ratio": sum(float(x["all_participants_hit_ratio"]) for x in summaries) / len(summaries),
            "covered_reference_identity_ratio": sum(float(x["covered_reference_identity_ratio"]) for x in summaries) / len(summaries),
        },
    }
    out_path = args.output_root.resolve() / f"{args.run_prefix}_summary.json"
    write_json(out_path, aggregate)
    print(f"[all_splits] aggregate summary written to {out_path}")


if __name__ == "__main__":
    main()
