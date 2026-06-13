"""Small JSON/JSONL helpers for examples and scoring scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: str | Path, *, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def index_annotations(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index annotation rows by sample id.

    Rows may use either the compact example schema with `sample_id` or the flat
    generated-dataset schema with `vid`.
    """
    index = {}
    for record in records:
        key = record.get("sample_id") or record.get("vid")
        if key:
            index[str(key)] = record
    return index
