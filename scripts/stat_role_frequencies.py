#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AppConfig  # noqa: E402
from src.parser import DialogueRecord, parse_dev_txt  # noqa: E402


@dataclass(frozen=True)
class RoleFrequency:
    name: str
    dialogue_count: int
    utterance_count: int
    speaker_count: int
    listener_count: int

    @property
    def total_count(self) -> int:
        return self.speaker_count + self.listener_count


def load_dialogues(config_path: Path) -> list[DialogueRecord]:
    app_config = AppConfig.from_yaml(config_path)
    return parse_dev_txt(app_config.paths.dev_txt)


def compute_role_frequencies(dialogues: Iterable[DialogueRecord]) -> list[RoleFrequency]:
    dialogue_sets: dict[str, set[int]] = defaultdict(set)
    utterance_counter: Counter[str] = Counter()
    speaker_counter: Counter[str] = Counter()
    listener_counter: Counter[str] = Counter()

    for dialogue in dialogues:
        for utterance in dialogue.utterances:
            speaker_name = utterance.speaker.name
            dialogue_sets[speaker_name].add(dialogue.dialogue_id)
            utterance_counter[speaker_name] += 1
            speaker_counter[speaker_name] += 1

            for listener in utterance.listeners:
                listener_name = listener.name
                dialogue_sets[listener_name].add(dialogue.dialogue_id)
                utterance_counter[listener_name] += 1
                listener_counter[listener_name] += 1

    names = sorted(set(dialogue_sets) | set(utterance_counter) | set(speaker_counter) | set(listener_counter))
    return [
        RoleFrequency(
            name=name,
            dialogue_count=len(dialogue_sets[name]),
            utterance_count=utterance_counter[name],
            speaker_count=speaker_counter[name],
            listener_count=listener_counter[name],
        )
        for name in names
    ]


def sort_roles(rows: list[RoleFrequency], sort_key: str) -> list[RoleFrequency]:
    if sort_key == "dialogues":
        key_fn = lambda r: (-r.dialogue_count, -r.utterance_count, r.name)
    elif sort_key == "speakers":
        key_fn = lambda r: (-r.speaker_count, -r.utterance_count, r.name)
    elif sort_key == "listeners":
        key_fn = lambda r: (-r.listener_count, -r.utterance_count, r.name)
    elif sort_key == "name":
        key_fn = lambda r: (r.name,)
    else:
        key_fn = lambda r: (-r.utterance_count, -r.dialogue_count, -r.speaker_count, -r.listener_count, r.name)
    return sorted(rows, key=key_fn)


def write_csv(rows: list[RoleFrequency], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "name",
                "dialogue_count",
                "utterance_count",
                "speaker_count",
                "listener_count",
                "total_count",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": idx,
                    "name": row.name,
                    "dialogue_count": row.dialogue_count,
                    "utterance_count": row.utterance_count,
                    "speaker_count": row.speaker_count,
                    "listener_count": row.listener_count,
                    "total_count": row.total_count,
                }
            )


def write_json(rows: list[RoleFrequency], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "rank": idx,
            "name": row.name,
            "dialogue_count": row.dialogue_count,
            "utterance_count": row.utterance_count,
            "speaker_count": row.speaker_count,
            "listener_count": row.listener_count,
            "total_count": row.total_count,
        }
        for idx, row in enumerate(rows, 1)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_table(rows: list[RoleFrequency]) -> None:
    headers = [
        "rank",
        "name",
        "dialogues",
        "utterances",
        "speaker",
        "listener",
        "total",
    ]
    table_rows = []
    for idx, row in enumerate(rows, 1):
        table_rows.append(
            [
                str(idx),
                row.name,
                str(row.dialogue_count),
                str(row.utterance_count),
                str(row.speaker_count),
                str(row.listener_count),
                str(row.total_count),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(format_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="统计所有角色在对话数据中的出现频次。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML 配置文件路径，默认读取当前目录下的 config.yaml。",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="utterances",
        choices=["utterances", "dialogues", "speakers", "listeners", "name"],
        help="结果排序方式，默认按语句出现次数降序排列。",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="可选：将统计结果写入 CSV 文件。",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="可选：将统计结果写入 JSON 文件。",
    )
    args = parser.parse_args()

    try:
        dialogues = load_dialogues(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stat_role_frequencies] 配置或解析失败: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[stat_role_frequencies] 未知错误: {exc}")
        sys.exit(1)

    rows = compute_role_frequencies(dialogues)
    rows = sort_roles(rows, args.sort)

    print(f"[stat_role_frequencies] 共统计到 {len(rows)} 个角色。")
    print_table(rows)

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print(f"[stat_role_frequencies] CSV 已写入: {args.out_csv}")
    if args.out_json is not None:
        write_json(rows, args.out_json)
        print(f"[stat_role_frequencies] JSON 已写入: {args.out_json}")


if __name__ == "__main__":
    main()
