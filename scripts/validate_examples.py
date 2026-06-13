#!/usr/bin/env python3
"""Validate bundled JSON examples and run a tiny scoring smoke test."""

from __future__ import annotations

from pathlib import Path

from grounding_video_reasoning.io import index_annotations, read_json, read_jsonl
from grounding_video_reasoning.metrics import aggregate_scores, score_sample


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    annotations = read_json(ROOT / "examples/annotations/mini_annotations.json")
    predictions = read_jsonl(ROOT / "examples/predictions/mini_predictions.jsonl")
    prompts = read_json(ROOT / "examples/prompts/real_prompt_examples.json")
    results = read_json(ROOT / "examples/results/paper_results_snapshot.json")

    require(len(annotations) >= 2, "mini_annotations.json should contain multiple rows")
    require(len(predictions) >= 2, "mini_predictions.jsonl should contain multiple rows")
    require(len(prompts) >= 3, "real_prompt_examples.json should contain real prompt examples")
    require("prompt_families" in results, "paper_results_snapshot.json missing prompt_families")

    annotation_index = index_annotations(annotations)
    scores = [score_sample(annotation_index[row["sample_id"]], row) for row in predictions]
    report = aggregate_scores(scores)
    require("original" in report["conditions"], "smoke report missing original condition")
    print("Examples and scoring smoke test look OK.")


if __name__ == "__main__":
    main()
