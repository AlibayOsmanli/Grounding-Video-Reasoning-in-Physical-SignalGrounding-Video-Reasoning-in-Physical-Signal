#!/usr/bin/env python3
"""Path-neutral inference template for the paper output contract.

This script is intentionally backend-agnostic. It builds the same prompt
families used by the benchmark and writes prediction JSONL in the expected
format. Replace `predict_video` with the model loading/generation code for your
machine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from grounding_video_reasoning.io import read_json, write_jsonl
from grounding_video_reasoning.prompting import build_neutral_rstr_prompt, build_vstar_like_prompt


def build_physics_prompt(sample: dict[str, Any]) -> str:
    duration = sample.get("duration_sec")
    event = sample.get("a_what", "")
    if duration is not None:
        q_when = (
            f"The video is {float(duration):.1f} seconds long. "
            "At what start and end time (in seconds, as floats) does the following event occur? "
            f"Event: {event}\n"
            'Output ONLY a JSON object: {"start_sec": <float>, "end_sec": <float>}'
        )
    else:
        q_when = sample.get("q_when", "When does the described event happen?")

    return (
        "You are evaluating a physical AI task. Watch the video carefully and answer each question in order.\n\n"
        f"[Step 1 - What]  {sample.get('q_what', 'What physical event happens in the video?')}\n"
        f"[Step 2 - When]  {q_when}\n"
        f"[Step 3 - Where] {sample.get('q_where', 'Where is the primary object during the relevant time span?')}\n\n"
        "Provide your answers as a JSON object with keys 'a_what', "
        "'a_when' ({\"start_sec\": <float>, \"end_sec\": <float>}), "
        "and 'a_where' ({\"x\": <int>, \"y\": <int>, \"w\": <int>, \"h\": <int>}).\n"
        "Return ONLY the JSON object. Do not include markdown, code fences, or reasoning."
    )


def build_prompt(sample: dict[str, Any], prompt_condition: str) -> str:
    duration = sample.get("duration_sec")
    if prompt_condition == "physics":
        return build_physics_prompt(sample)
    if prompt_condition == "neutral_rstr":
        return build_neutral_rstr_prompt(sample, video_duration_sec=duration)
    if prompt_condition == "vstar_like":
        return build_vstar_like_prompt(sample, video_duration_sec=duration)
    raise ValueError(f"Unknown prompt_condition: {prompt_condition}")


def predict_video(video_path: str, prompt: str) -> dict[str, Any]:
    """CHANGE HERE: call your video-language model and return parsed JSON.

    Expected return shape:
        {
          "a_what": "...",
          "a_when": {"start_sec": 0.0, "end_sec": 1.0},
          "a_where": {"x": 0, "y": 0, "w": 10, "h": 10}
        }
    """
    raise NotImplementedError(
        "Connect your model backend in predict_video(video_path, prompt)."
    )


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or sample.get("vid"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--prompt_condition",
        choices=["physics", "neutral_rstr", "vstar_like"],
        default="physics",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    samples = read_json(args.annotations)
    if args.max_samples:
        samples = samples[: args.max_samples]

    rows = []
    for sample in samples:
        prompt = build_prompt(sample, args.prompt_condition)
        start = time.time()
        prediction = predict_video(sample["video_path"], prompt)
        latency_ms = (time.time() - start) * 1000.0
        rows.append(
            {
                "sample_id": sample_id(sample),
                "video_id": sample.get("video_id", ""),
                "condition": sample.get("condition", ""),
                "video_path": sample["video_path"],
                "model_name": args.model_name,
                "prompt_condition": args.prompt_condition,
                "latency_ms": latency_ms,
                "prompt": prompt,
                "prediction": prediction,
            }
        )

    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} prediction row(s) to {args.output}")


if __name__ == "__main__":
    main()
