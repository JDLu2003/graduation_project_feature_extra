#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


UTTERANCE_RE = re.compile(
    r"Processing utterance C_(?P<dialogue>\d+)_U_(?P<utt>\d+) "
    r"\(speaker: (?P<speaker>.*), listeners: (?P<listeners>\d+)\)"
)
STATUS_RE = re.compile(
    r"\[face_scene_fr\]\[(?P<tag>C_\d+_U_\d+)\] "
    r"person='(?P<person>.*)' role=(?P<role>speaker|listener) "
    r"feature_status=(?P<status>[A-Z_]+)"
)
SAVED_RE = re.compile(
    r"Saved features for dialogue (?P<dialogue>\d+), utterance (?P<utt>\d+) to (?P<path>.+)$"
)

NON_ZERO_STATUSES = {"FOUND", "OTHER_MEAN"}


@dataclass
class UtteranceStat:
    dialogue_id: int
    utterance_idx: int
    speaker_name: str
    listeners_expected: int
    persons: list[tuple[str, str, str]] = field(default_factory=list)  # (person, role, status)
    saved: bool = False
    output_path: str = ""

    @property
    def tag(self) -> str:
        return f"C_{self.dialogue_id}_U_{self.utterance_idx}"

    @property
    def expected_persons(self) -> int:
        return self.listeners_expected + 1


def parse_log(log_path: Path) -> dict[str, UtteranceStat]:
    utterances: dict[str, UtteranceStat] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_u = UTTERANCE_RE.search(line)
            if m_u:
                d = int(m_u.group("dialogue"))
                u = int(m_u.group("utt"))
                tag = f"C_{d}_U_{u}"
                utterances[tag] = UtteranceStat(
                    dialogue_id=d,
                    utterance_idx=u,
                    speaker_name=m_u.group("speaker"),
                    listeners_expected=int(m_u.group("listeners")),
                )
                continue

            m_s = STATUS_RE.search(line)
            if m_s:
                tag = m_s.group("tag")
                if tag not in utterances:
                    # 日志可能被截断；尽量补一个最小记录防止丢失统计。
                    dd, uu = tag.split("_")[1], tag.split("_")[3]
                    utterances[tag] = UtteranceStat(
                        dialogue_id=int(dd),
                        utterance_idx=int(uu),
                        speaker_name="",
                        listeners_expected=0,
                    )
                utterances[tag].persons.append(
                    (m_s.group("person"), m_s.group("role"), m_s.group("status"))
                )
                continue

            m_sv = SAVED_RE.search(line)
            if m_sv:
                d = int(m_sv.group("dialogue"))
                u = int(m_sv.group("utt"))
                tag = f"C_{d}_U_{u}"
                if tag not in utterances:
                    utterances[tag] = UtteranceStat(
                        dialogue_id=d,
                        utterance_idx=u,
                        speaker_name="",
                        listeners_expected=0,
                    )
                utterances[tag].saved = True
                utterances[tag].output_path = m_sv.group("path")

    return utterances


def compute_report(utterances: dict[str, UtteranceStat]) -> dict:
    status_counter: Counter[str] = Counter()
    role_status_counter: Counter[str] = Counter()
    person_status_counter: dict[str, Counter[str]] = defaultdict(Counter)
    dialogue_status_counter: dict[int, Counter[str]] = defaultdict(Counter)
    dialogue_person_total: Counter[int] = Counter()
    dialogue_utt_total: Counter[int] = Counter()

    utt_all_non_zero = 0
    utt_has_zero = 0
    utt_incomplete_logs = 0
    saved_count = 0

    for utt in utterances.values():
        dialogue_utt_total[utt.dialogue_id] += 1
        if utt.saved:
            saved_count += 1

        observed = len(utt.persons)
        if observed < utt.expected_persons:
            utt_incomplete_logs += 1

        zero_in_utt = False
        all_non_zero = observed > 0

        for person, role, status in utt.persons:
            status_counter[status] += 1
            role_status_counter[f"{role}:{status}"] += 1
            person_status_counter[person][status] += 1
            dialogue_status_counter[utt.dialogue_id][status] += 1
            dialogue_person_total[utt.dialogue_id] += 1

            if status not in NON_ZERO_STATUSES:
                zero_in_utt = True
                all_non_zero = False

        if observed == utt.expected_persons and all_non_zero:
            utt_all_non_zero += 1
        if zero_in_utt:
            utt_has_zero += 1

    total_utterances = len(utterances)
    total_person_records = sum(status_counter.values())
    non_zero_count = sum(status_counter[s] for s in NON_ZERO_STATUSES)
    zero_count = status_counter["ZERO"]

    speaker_total = sum(v for k, v in role_status_counter.items() if k.startswith("speaker:"))
    speaker_zero = role_status_counter["speaker:ZERO"]
    listener_total = sum(v for k, v in role_status_counter.items() if k.startswith("listener:"))
    listener_zero = role_status_counter["listener:ZERO"]

    summary = {
        "total_utterances": total_utterances,
        "saved_utterances": saved_count,
        "total_person_records": total_person_records,
        "status_counts": dict(status_counter),
        "overall_non_zero_rate": (non_zero_count / total_person_records) if total_person_records else 0.0,
        "overall_zero_rate": (zero_count / total_person_records) if total_person_records else 0.0,
        "speaker_zero_rate": (speaker_zero / speaker_total) if speaker_total else 0.0,
        "listener_zero_rate": (listener_zero / listener_total) if listener_total else 0.0,
        "utterance_all_non_zero_rate": (utt_all_non_zero / total_utterances) if total_utterances else 0.0,
        "utterance_has_zero_rate": (utt_has_zero / total_utterances) if total_utterances else 0.0,
        "utterance_incomplete_log_count": utt_incomplete_logs,
    }

    per_person_rows = []
    for person, c in sorted(person_status_counter.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
        total = sum(c.values())
        nz = c["FOUND"] + c["OTHER_MEAN"]
        per_person_rows.append(
            {
                "person": person,
                "total": total,
                "found": c["FOUND"],
                "other_mean": c["OTHER_MEAN"],
                "zero": c["ZERO"],
                "non_zero_rate": (nz / total) if total else 0.0,
            }
        )

    per_dialogue_rows = []
    for did, c in sorted(dialogue_status_counter.items(), key=lambda kv: kv[0]):
        total = dialogue_person_total[did]
        nz = c["FOUND"] + c["OTHER_MEAN"]
        per_dialogue_rows.append(
            {
                "dialogue_id": did,
                "utterances": dialogue_utt_total[did],
                "person_records": total,
                "found": c["FOUND"],
                "other_mean": c["OTHER_MEAN"],
                "zero": c["ZERO"],
                "non_zero_rate": (nz / total) if total else 0.0,
            }
        )

    per_utt_rows = []
    for tag, utt in sorted(
        utterances.items(),
        key=lambda kv: (kv[1].dialogue_id, kv[1].utterance_idx),
    ):
        c = Counter(status for _, _, status in utt.persons)
        total = len(utt.persons)
        nz = c["FOUND"] + c["OTHER_MEAN"]
        per_utt_rows.append(
            {
                "tag": tag,
                "dialogue_id": utt.dialogue_id,
                "utterance_idx": utt.utterance_idx,
                "speaker_name": utt.speaker_name,
                "expected_persons": utt.expected_persons,
                "observed_person_logs": total,
                "found": c["FOUND"],
                "other_mean": c["OTHER_MEAN"],
                "zero": c["ZERO"],
                "non_zero_rate": (nz / total) if total else 0.0,
                "saved": utt.saved,
                "output_path": utt.output_path,
            }
        )

    return {
        "summary": summary,
        "per_person": per_person_rows,
        "per_dialogue": per_dialogue_rows,
        "per_utterance": per_utt_rows,
    }


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze face_scene_fr extraction log quality.")
    parser.add_argument("--log", type=Path, required=True, help="Path to extraction .log file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to save JSON/CSV report files (default: logs/)",
    )
    args = parser.parse_args()

    assert args.log.exists(), f"log not found: {args.log}"
    utterances = parse_log(args.log)
    report = compute_report(utterances)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.log.stem
    json_path = args.out_dir / f"{stem}.summary.json"
    person_csv = args.out_dir / f"{stem}.per_person.csv"
    dialogue_csv = args.out_dir / f"{stem}.per_dialogue.csv"
    utt_csv = args.out_dir / f"{stem}.per_utterance.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report["summary"], f, ensure_ascii=False, indent=2)
    write_csv(report["per_person"], person_csv)
    write_csv(report["per_dialogue"], dialogue_csv)
    write_csv(report["per_utterance"], utt_csv)

    s = report["summary"]
    print(f"[report] log={args.log}")
    print(f"[report] total_utterances={s['total_utterances']} saved_utterances={s['saved_utterances']}")
    print(f"[report] total_person_records={s['total_person_records']} status_counts={s['status_counts']}")
    print(
        "[report] overall_non_zero_rate={:.4f} overall_zero_rate={:.4f}".format(
            s["overall_non_zero_rate"], s["overall_zero_rate"]
        )
    )
    print(
        "[report] speaker_zero_rate={:.4f} listener_zero_rate={:.4f}".format(
            s["speaker_zero_rate"], s["listener_zero_rate"]
        )
    )
    print(
        "[report] utterance_all_non_zero_rate={:.4f} utterance_has_zero_rate={:.4f} incomplete_log_count={}".format(
            s["utterance_all_non_zero_rate"],
            s["utterance_has_zero_rate"],
            s["utterance_incomplete_log_count"],
        )
    )
    print(f"[report] wrote: {json_path}")
    print(f"[report] wrote: {person_csv}")
    print(f"[report] wrote: {dialogue_csv}")
    print(f"[report] wrote: {utt_csv}")


if __name__ == "__main__":
    main()
