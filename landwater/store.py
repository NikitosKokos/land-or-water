"""On-disk results: one append-only JSONL per (model, step), plus batch bookkeeping.

results/<model-key>/step2.jsonl          one line per answered point (last write wins)
results/<model-key>/step2.batches.json   batch IDs submitted and whether collected
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from .core import fmt_num

ROOT = Path("results")


def set_root(path: str | Path) -> None:
    global ROOT
    ROOT = Path(path)


def _path(key: str, step: float, suffix: str) -> Path:
    # Plain string join, not with_suffix(): "step0.5" would lose its ".5".
    return ROOT / key / f"step{fmt_num(step)}{suffix}"


def results_path(key: str, step: float) -> Path:
    return _path(key, step, ".jsonl")


def batches_path(key: str, step: float) -> Path:
    return _path(key, step, ".batches.json")


def load_records(key: str, step: float) -> dict[str, dict[str, Any]]:
    path = results_path(key, step)
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from an interrupted run
            records[rec["id"]] = rec
    return records


def write_record(f: TextIO, rec: dict[str, Any]) -> None:
    f.write(json.dumps(rec) + "\n")


def append_records(key: str, step: float, records: Iterable[dict[str, Any]]) -> int:
    path = results_path(key, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            write_record(f, rec)
            n += 1
    return n


def load_batches(key: str, step: float) -> list[dict[str, Any]]:
    path = batches_path(key, step)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_batches(key: str, step: float, batches: list[dict[str, Any]]) -> None:
    path = batches_path(key, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(batches, indent=2), encoding="utf-8")


def models_with_results(step: float) -> list[str]:
    if not ROOT.exists():
        return []
    return [p.parent.name for p in ROOT.glob(f"*/step{fmt_num(step)}.jsonl")]
