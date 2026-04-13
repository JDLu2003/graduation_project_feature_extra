#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import DialogueRecord, PersonEntry, parse_dev_txt  # noqa: E402

DEFAULT_TXT_PATH = Path("/data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt")
DEFAULT_REFERENCE_ROOT = Path("/data16T_2/sunshengzhe/reference_face_zh")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "zh_reference_coverage" / "artifacts"
IGNORED_REFERENCE_FILES = {"identity_bank.pkl", "_placeholder.jpg"}


@dataclass(frozen=True)
class RoleStats:
    name: str
    dialogue_count: int
    utterance_count: int
    speaker_count: int
    listener_count: int

    @property
    def total_count(self) -> int:
        return self.speaker_count + self.listener_count


@dataclass(frozen=True)
class ReferenceIdentity:
    name: str
    image_count: int
    directory: str
    status: str


@dataclass(frozen=True)
class ThresholdCoverage:
    min_ref_images: int
    covered_roles: int
    total_roles: int
    covered_dialogues: int
    total_dialogues: int
    covered_utterances: int
    total_utterances: int
    covered_speaker_turns: int
    total_speaker_turns: int
    covered_listener_turns: int
    total_listener_turns: int

    @property
    def role_coverage_ratio(self) -> float:
        return safe_div(self.covered_roles, self.total_roles)

    @property
    def dialogue_coverage_ratio(self) -> float:
        return safe_div(self.covered_dialogues, self.total_dialogues)

    @property
    def utterance_coverage_ratio(self) -> float:
        return safe_div(self.covered_utterances, self.total_utterances)

    @property
    def speaker_coverage_ratio(self) -> float:
        return safe_div(self.covered_speaker_turns, self.total_speaker_turns)

    @property
    def listener_coverage_ratio(self) -> float:
        return safe_div(self.covered_listener_turns, self.total_listener_turns)


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


def normalize_name(name: str) -> str:
    return name.strip()


def compute_role_stats(dialogues: Iterable[DialogueRecord]) -> tuple[list[RoleStats], dict[str, set[int]]]:
    dialogue_sets: dict[str, set[int]] = {}
    utterance_counter: Counter[str] = Counter()
    speaker_counter: Counter[str] = Counter()
    listener_counter: Counter[str] = Counter()

    for dialogue in dialogues:
        for utterance in dialogue.utterances:
            collect_person(dialogue_sets, utterance_counter, speaker_counter, dialogue.dialogue_id, utterance.speaker)
            for listener in utterance.listeners:
                collect_person(dialogue_sets, utterance_counter, listener_counter, dialogue.dialogue_id, listener)

    names = sorted(set(dialogue_sets) | set(utterance_counter) | set(speaker_counter) | set(listener_counter))
    rows: list[RoleStats] = []
    for name in names:
        rows.append(
            RoleStats(
                name=name,
                dialogue_count=len(dialogue_sets.get(name, set())),
                utterance_count=utterance_counter[name],
                speaker_count=speaker_counter[name],
                listener_count=listener_counter[name],
            )
        )
    rows.sort(key=lambda x: (-x.utterance_count, -x.dialogue_count, x.name))
    return rows, dialogue_sets


def collect_person(
    dialogue_sets: dict[str, set[int]],
    utterance_counter: Counter[str],
    role_counter: Counter[str],
    dialogue_id: int,
    person: PersonEntry,
) -> None:
    name = normalize_name(person.name)
    dialogue_sets.setdefault(name, set()).add(dialogue_id)
    utterance_counter[name] += 1
    role_counter[name] += 1


def scan_reference_identities(reference_root: Path) -> list[ReferenceIdentity]:
    rows: list[ReferenceIdentity] = []
    if not reference_root.exists():
        raise FileNotFoundError(f"reference root not found: {reference_root}")

    for child in sorted(reference_root.iterdir(), key=lambda p: p.name):
        if child.name in IGNORED_REFERENCE_FILES:
            continue
        if not child.is_dir():
            continue
        image_files = sorted(
            p for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        status = "ok" if image_files else "empty"
        rows.append(
            ReferenceIdentity(
                name=normalize_name(child.name),
                image_count=len(image_files),
                directory=str(child.resolve()),
                status=status,
            )
        )
    return rows


def build_threshold_coverages(
    role_rows: list[RoleStats],
    role_dialogue_sets: dict[str, set[int]],
    ref_rows: list[ReferenceIdentity],
    thresholds: list[int],
) -> list[ThresholdCoverage]:
    total_dialogues = len(set().union(*role_dialogue_sets.values())) if role_dialogue_sets else 0
    total_utterances = sum(row.utterance_count for row in role_rows)
    total_speaker_turns = sum(row.speaker_count for row in role_rows)
    total_listener_turns = sum(row.listener_count for row in role_rows)
    total_roles = len(role_rows)
    ref_count_by_name = {row.name: row.image_count for row in ref_rows}

    results: list[ThresholdCoverage] = []
    for threshold in thresholds:
        covered = [row for row in role_rows if ref_count_by_name.get(row.name, 0) >= threshold]
        covered_dialogue_ids = set()
        for row in covered:
            covered_dialogue_ids.update(role_dialogue_sets.get(row.name, set()))
        results.append(
            ThresholdCoverage(
                min_ref_images=threshold,
                covered_roles=len(covered),
                total_roles=total_roles,
                covered_dialogues=len(covered_dialogue_ids),
                total_dialogues=total_dialogues,
                covered_utterances=sum(row.utterance_count for row in covered),
                total_utterances=total_utterances,
                covered_speaker_turns=sum(row.speaker_count for row in covered),
                total_speaker_turns=total_speaker_turns,
                covered_listener_turns=sum(row.listener_count for row in covered),
                total_listener_turns=total_listener_turns,
            )
        )
    return results


def infer_alias_groups(role_rows: list[RoleStats], ref_rows: list[ReferenceIdentity]) -> list[dict[str, object]]:
    role_names = {row.name for row in role_rows if row.name}
    ref_names = {row.name for row in ref_rows if row.name}
    all_names = sorted(role_names | ref_names)
    seen: set[tuple[str, str]] = set()
    suspicious: list[dict[str, object]] = []

    for i, left in enumerate(all_names):
        for right in all_names[i + 1:]:
            if (left, right) in seen:
                continue
            if left and right and (left in right or right in left):
                shorter, longer = sorted([left, right], key=len)
                if len(shorter) >= 2:
                    suspicious.append(
                        {
                            "left": left,
                            "right": right,
                            "reason": "substring_overlap",
                            "left_in_reference": left in ref_names,
                            "right_in_reference": right in ref_names,
                        }
                    )
                    seen.add((left, right))
    return suspicious


def build_role_rows_with_reference(role_rows: list[RoleStats], ref_rows: list[ReferenceIdentity]) -> list[dict[str, object]]:
    ref_map = {row.name: row for row in ref_rows}
    rows: list[dict[str, object]] = []
    for idx, role in enumerate(role_rows, start=1):
        ref = ref_map.get(role.name)
        ref_count = ref.image_count if ref is not None else 0
        rows.append(
            {
                "rank": idx,
                "name": role.name,
                "dialogue_count": role.dialogue_count,
                "utterance_count": role.utterance_count,
                "speaker_count": role.speaker_count,
                "listener_count": role.listener_count,
                "total_count": role.total_count,
                "reference_image_count": ref_count,
                "covered_at_1": ref_count >= 1,
                "covered_at_5": ref_count >= 5,
                "covered_at_10": ref_count >= 10,
                "covered_at_20": ref_count >= 20,
            }
        )
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_report(
    path: Path,
    txt_path: Path,
    reference_root: Path,
    role_rows: list[RoleStats],
    ref_rows: list[ReferenceIdentity],
    thresholds: list[ThresholdCoverage],
    role_rows_with_ref: list[dict[str, object]],
    suspicious_aliases: list[dict[str, object]],
) -> None:
    ensure_dir(path.parent)
    top_missing = [row for row in role_rows_with_ref if not row["covered_at_1"]][:20]
    top_thin = [row for row in role_rows_with_ref if row["covered_at_1"] and row["reference_image_count"] < 10][:20]
    best_threshold = next((row for row in thresholds if row.min_ref_images == 10), thresholds[0] if thresholds else None)

    lines = [
        "# 中文 reference 覆盖面评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集标注文件：`{txt_path}`",
        f"- reference 根目录：`{reference_root}`",
        f"- 数据集角色数：`{len(role_rows)}`",
        f"- reference 身份数：`{len(ref_rows)}`",
        "",
        "## 技术路线",
        "",
        "本评估不直接跑人脸模型，而是先评估 reference 库对中文数据集角色分布的理论覆盖面。",
        "",
        "1. 解析 `dev.txt`，统计每个角色在 dialogue / utterance / speaker / listener 四个维度的出现次数。",
        "2. 扫描 `reference_face_zh`，统计每个身份目录下的有效图片数量，忽略 `identity_bank.pkl` 和 `_placeholder.jpg`。",
        "3. 以“reference 图片数 >= 阈值”作为身份可用性的近似条件，分别统计角色覆盖率、utterance 覆盖率、speaker 覆盖率、listener 覆盖率。",
        "4. 输出缺失角色、高频但样本稀薄角色、疑似别名角色，帮助后续清洗 reference 库与设计分层识别策略。",
        "",
        "## 阈值覆盖率",
        "",
        "| min_ref | role_cov | utt_cov | speaker_cov | listener_cov |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in thresholds:
        lines.append(
            "| {thr} | {role:.2%} ({r1}/{r2}) | {utt:.2%} ({u1}/{u2}) | {spk:.2%} ({s1}/{s2}) | {lst:.2%} ({l1}/{l2}) |".format(
                thr=row.min_ref_images,
                role=row.role_coverage_ratio,
                r1=row.covered_roles,
                r2=row.total_roles,
                utt=row.utterance_coverage_ratio,
                u1=row.covered_utterances,
                u2=row.total_utterances,
                spk=row.speaker_coverage_ratio,
                s1=row.covered_speaker_turns,
                s2=row.total_speaker_turns,
                lst=row.listener_coverage_ratio,
                l1=row.covered_listener_turns,
                l2=row.total_listener_turns,
            )
        )

    lines.extend(["", "## 重点结论", ""])
    if best_threshold is not None:
        lines.extend(
            [
                f"- 以 `min_ref_images >= {best_threshold.min_ref_images}` 作为较稳妥门槛时，角色覆盖率约为 `{best_threshold.role_coverage_ratio:.2%}`。",
                f"- 同一门槛下，按 utterance 计的覆盖率约为 `{best_threshold.utterance_coverage_ratio:.2%}`。",
                f"- speaker 侧覆盖率约为 `{best_threshold.speaker_coverage_ratio:.2%}`，listener 侧覆盖率约为 `{best_threshold.listener_coverage_ratio:.2%}`。",
            ]
        )
    empty_dirs = [row for row in ref_rows if row.status == "empty"]
    lines.append(f"- 空 reference 目录数量为 `{len(empty_dirs)}`，这些角色当前不会贡献任何识别能力。")
    lines.append("")

    lines.append("## 高频但缺失 reference 的角色")
    lines.append("")
    if top_missing:
        lines.append("| rank | 角色 | utterance | speaker | listener |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in top_missing:
            lines.append(
                f"| {row['rank']} | {row['name'] or '(空名)'} | {row['utterance_count']} | {row['speaker_count']} | {row['listener_count']} |"
            )
    else:
        lines.append("- 无。")
    lines.append("")

    lines.append("## 已覆盖但样本仍偏少的高频角色")
    lines.append("")
    if top_thin:
        lines.append("| rank | 角色 | ref_images | utterance | speaker |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in top_thin:
            lines.append(
                f"| {row['rank']} | {row['name'] or '(空名)'} | {row['reference_image_count']} | {row['utterance_count']} | {row['speaker_count']} |"
            )
    else:
        lines.append("- 无。")
    lines.append("")

    lines.append("## 疑似别名 / 重复标注")
    lines.append("")
    if suspicious_aliases:
        lines.append("| left | right | reason |")
        lines.append("| --- | --- | --- |")
        for row in suspicious_aliases[:20]:
            lines.append(f"| {row['left']} | {row['right']} | {row['reason']} |")
    else:
        lines.append("- 未发现明显候选。")
    lines.append("")

    lines.append("## 建议")
    lines.append("")
    lines.append("1. 先补齐高频缺失角色，再考虑继续扩 reference 尾部角色。")
    lines.append("2. 对 `ref_images < 10` 的高频角色优先补多样性样本，而不是只补总数。")
    lines.append("3. 对疑似别名角色先做人名合并，否则会影响 gallery 检索与覆盖统计。")
    lines.append("4. 真正落地识别时，建议采用“reference 检索 + 阈值拒识 + unknown 聚类”的开放集方案。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def choose_thresholds(user_thresholds: Optional[list[int]]) -> list[int]:
    values = user_thresholds or [1, 3, 5, 10, 15, 20]
    uniq = sorted(set(v for v in values if v >= 0))
    return uniq or [1]


def build_output_dir(output_root: Path, run_name: Optional[str], overwrite: bool) -> Path:
    if run_name:
        out_dir = output_root / run_name
    else:
        out_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"output dir already exists: {out_dir} (pass --overwrite to reuse)")
    ensure_dir(out_dir)
    return out_dir


def summarize(role_rows: list[RoleStats], ref_rows: list[ReferenceIdentity], thresholds: list[ThresholdCoverage]) -> dict[str, object]:
    covered_names = {row.name for row in ref_rows if row.image_count > 0}
    total_ref_images = sum(row.image_count for row in ref_rows)
    empty_ref_dirs = sum(1 for row in ref_rows if row.image_count == 0)
    return {
        "dataset_roles": len(role_rows),
        "reference_identities": len(ref_rows),
        "reference_non_empty_identities": len(covered_names),
        "reference_total_images": total_ref_images,
        "reference_empty_directories": empty_ref_dirs,
        "thresholds": [
            {
                **asdict(row),
                "role_coverage_ratio": row.role_coverage_ratio,
                "dialogue_coverage_ratio": row.dialogue_coverage_ratio,
                "utterance_coverage_ratio": row.utterance_coverage_ratio,
                "speaker_coverage_ratio": row.speaker_coverage_ratio,
                "listener_coverage_ratio": row.listener_coverage_ratio,
            }
            for row in thresholds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估中文 reference 脸库对数据集角色分布的覆盖面。")
    parser.add_argument("--txt-path", type=Path, default=DEFAULT_TXT_PATH, help="中文数据集 txt 路径。")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="reference_face_zh 根目录。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录。")
    parser.add_argument("--run-name", type=str, default="latest", help="输出子目录名，默认 latest。")
    parser.add_argument("--threshold", type=int, action="append", default=None, help="可重复传入多个 reference 图片数门槛。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录。")
    args = parser.parse_args()

    txt_path = args.txt_path.resolve()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    thresholds = choose_thresholds(args.threshold)
    output_dir = build_output_dir(output_root, args.run_name, overwrite=args.overwrite)

    dialogues = parse_dev_txt(txt_path)
    role_rows, role_dialogue_sets = compute_role_stats(dialogues)
    ref_rows = scan_reference_identities(reference_root)
    threshold_rows = build_threshold_coverages(role_rows, role_dialogue_sets, ref_rows, thresholds)
    role_rows_with_ref = build_role_rows_with_reference(role_rows, ref_rows)
    suspicious_aliases = infer_alias_groups(role_rows, ref_rows)
    summary = summarize(role_rows, ref_rows, threshold_rows)

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "reference_inventory.json", [asdict(row) for row in ref_rows])
    write_json(output_dir / "suspicious_aliases.json", suspicious_aliases)
    write_csv(output_dir / "role_reference_coverage.csv", role_rows_with_ref)
    write_csv(
        output_dir / "threshold_coverage.csv",
        [
            {
                "min_ref_images": row.min_ref_images,
                "covered_roles": row.covered_roles,
                "total_roles": row.total_roles,
                "role_coverage_ratio": row.role_coverage_ratio,
                "covered_dialogues": row.covered_dialogues,
                "total_dialogues": row.total_dialogues,
                "dialogue_coverage_ratio": row.dialogue_coverage_ratio,
                "covered_utterances": row.covered_utterances,
                "total_utterances": row.total_utterances,
                "utterance_coverage_ratio": row.utterance_coverage_ratio,
                "covered_speaker_turns": row.covered_speaker_turns,
                "total_speaker_turns": row.total_speaker_turns,
                "speaker_coverage_ratio": row.speaker_coverage_ratio,
                "covered_listener_turns": row.covered_listener_turns,
                "total_listener_turns": row.total_listener_turns,
                "listener_coverage_ratio": row.listener_coverage_ratio,
            }
            for row in threshold_rows
        ],
    )
    write_report(
        output_dir / "report.md",
        txt_path=txt_path,
        reference_root=reference_root,
        role_rows=role_rows,
        ref_rows=ref_rows,
        thresholds=threshold_rows,
        role_rows_with_ref=role_rows_with_ref,
        suspicious_aliases=suspicious_aliases,
    )

    print(f"[zh_reference_coverage] txt_path={txt_path}")
    print(f"[zh_reference_coverage] reference_root={reference_root}")
    print(f"[zh_reference_coverage] output_dir={output_dir}")
    print(f"[zh_reference_coverage] dataset_roles={len(role_rows)} reference_identities={len(ref_rows)}")
    for row in threshold_rows:
        print(
            "[zh_reference_coverage] min_ref={thr} role_cov={role:.2%} utt_cov={utt:.2%} "
            "speaker_cov={spk:.2%} listener_cov={lst:.2%}".format(
                thr=row.min_ref_images,
                role=row.role_coverage_ratio,
                utt=row.utterance_coverage_ratio,
                spk=row.speaker_coverage_ratio,
                lst=row.listener_coverage_ratio,
            )
        )


if __name__ == "__main__":
    main()
