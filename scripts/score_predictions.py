#!/usr/bin/env python3
"""Score prediction JSON/JSONL against annotation JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from grounding_video_reasoning.io import index_annotations, read_json, read_jsonl, write_json
from grounding_video_reasoning.metrics import aggregate_scores, score_sample


def load_predictions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    return payload.get("predictions", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = index_annotations(read_json(args.annotations))
    predictions = load_predictions(args.predictions)

    scores = []
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        if sample_id not in annotations:
            raise KeyError(f"No annotation found for sample_id={sample_id!r}")
        scores.append(score_sample(annotations[sample_id], prediction))

    report = aggregate_scores(scores)
    report["sample_scores"] = [score.to_dict() for score in scores]
    write_json(report, args.output)
    print(f"Wrote metrics report to {args.output}")


if __name__ == "__main__":
    main()
