#!/usr/bin/env python3
"""Print reproduction commands from a path-neutral JSON config."""

from __future__ import annotations

import argparse
import json
import signal
import shlex
from pathlib import Path


def q(value: object) -> str:
    return shlex.quote(str(value))


def command(parts: list[object]) -> str:
    return " ".join(q(part) for part in parts)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    python = cfg.get("python", "python")
    paths = cfg["paths"]
    conditions = cfg["conditions"]

    print("# Generate perturbation videos for a local smoke test")
    print(
        command(
            [
                python,
                "-m",
                "grounding_video_reasoning.video_perturbation",
                "--input_dir",
                paths["source_videos_dir"],
                "--output_dir",
                f"{paths['dataset_base'].rstrip('/')}/videos",
                "--conditions",
                *conditions,
            ]
        )
    )
    print()

    print("# Inference and metrics using the public JSON contract")
    for prompt_condition, prompt_cfg in cfg["prompt_families"].items():
        results_base = f"{paths['results_root'].rstrip('/')}/{prompt_cfg['results_subdir']}"
        for dataset in cfg["datasets"]:
            for model in cfg["models"]:
                annotations_path = f"{paths['dataset_base'].rstrip('/')}/{dataset}/annotations_all.json"
                prediction_path = f"{results_base}/{dataset}/{model}/predictions.jsonl"
                metrics_path = f"{results_base}/{dataset}/{model}/metrics.json"
                print(
                    command(
                        [
                            python,
                            "scripts/run_inference_template.py",
                            "--annotations",
                            annotations_path,
                            "--output",
                            prediction_path,
                            "--model_name",
                            model,
                            "--prompt_condition",
                            prompt_condition,
                        ]
                    )
                )
                print(
                    command(
                        [
                            python,
                            "scripts/score_predictions.py",
                            "--annotations",
                            annotations_path,
                            "--predictions",
                            prediction_path,
                            "--output",
                            metrics_path,
                        ]
                    )
                )


if __name__ == "__main__":
    main()
