# Grounding Video Reasoning in Physical Signals

Code and compact artifacts for reproducing the benchmark protocol from
**Grounding Video Reasoning in Physical Signals**.

Paper: https://arxiv.org/abs/2604.21873

This repository is meant to be small enough to upload to GitHub while still
containing real code and real examples. It does not include raw videos, model
checkpoints, cache directories, private cluster paths, or Slurm logs.

## Contents

```text
grounding_video_reasoning/
  schema.py                  RSTR annotation schema and prompt builder
  prompting.py               physics-neutral and V-STaR-like prompt builders
  metrics.py                 Acc, tIoU, sIoU, LGM, SBI, PRI, SPI scoring
  video_perturbation.py      original/shuffled/ablated/frame-masked videos
  bbox_normalization.py      model-specific bbox normalization helpers
  prediction_validation.py   invalid/refusal payload checks
  io.py                      JSON and JSONL helpers

scripts/
  run_inference_template.py  backend hook for model inference
  score_predictions.py       score predictions against annotations
  validate_examples.py       validate bundled examples
  make_reproduction_commands.py

configs/
  dataset_card.json
  model_registry.json
  reproduction_config.example.json

examples/
  annotations/mini_annotations.json
  predictions/mini_predictions.jsonl
  prompts/real_prompt_examples.json
  results/paper_results_snapshot.json
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For video perturbation generation, install the video extra:

```bash
pip install -e ".[video]"
```

## Validate The Repository

```bash
python scripts/validate_examples.py
```

This loads the bundled JSON examples and runs a small scoring smoke test.

## Score The Included Mini Example

```bash
python scripts/score_predictions.py \
  --annotations examples/annotations/mini_annotations.json \
  --predictions examples/predictions/mini_predictions.jsonl \
  --output examples/results/mini_metrics.generated.json
```

## Data Layout For A Full Run

After preparing source videos, generated data should follow this layout:

```text
<DATA_ROOT>/<dataset>/
  videos/
  annotations_original.json
  annotations_shuffled.json
  annotations_ablated.json
  annotations_frame_masked.json
  annotations_all.json
  checkpoint.json
```

Set local paths outside the repo:

```bash
export DATASETS_ROOT=/path/to/raw/source/datasets
export DATA_ROOT=/path/to/generated_dataset
export RESULTS_ROOT=/path/to/results
```

Do not commit those local paths.

## Generate Perturbed Videos

For a folder of source MP4 files:

```bash
python -m grounding_video_reasoning.video_perturbation \
  --input_dir /path/to/source_mp4s \
  --output_dir "$DATA_ROOT/videos" \
  --conditions original shuffled ablated frame_masked
```

The full dataset construction used source-specific adapters for SSV2,
YouCook2, HoloAssist, and Roundabout-TAU. The public contract is the JSON schema
shown in `examples/annotations/mini_annotations.json`.

## Run Model Inference

Use `scripts/run_inference_template.py` as the model hook. The only required
edit is inside `predict_video(...)`.

```python
def predict_video(video_path: str, prompt: str) -> dict:
    # CHANGE HERE: load/call your local VLM backend and parse the response.
    ...
```

The script writes JSONL rows with this contract:

```json
{
  "sample_id": "31244_original",
  "video_id": "31244",
  "condition": "original",
  "model_name": "gemma4_26b_a4b_it",
  "prompt_condition": "physics",
  "prompt": "...",
  "prediction": {
    "a_what": "...",
    "a_when": {"start_sec": 0.0, "end_sec": 3.7},
    "a_where": {"x": 36, "y": 239, "w": 251, "h": 420}
  }
}
```

## Reproduce The Paper Matrix

The paper evaluates:

- datasets: `ssv2`, `youcook2`, `holoassist`, `roundabout`
- conditions: `original`, `shuffled`, `ablated`, `frame_masked`
- prompt families: `physics`, `neutral_rstr`, `vstar_like`
- model keys listed in `configs/model_registry.json`

Generate path-neutral command templates:

```bash
python scripts/make_reproduction_commands.py configs/reproduction_config.example.json
```

Edit `configs/reproduction_config.example.json` for your paths before using the
commands.

## Real Examples

- `examples/prompts/real_prompt_examples.json` contains real prompt/prediction
  examples from completed runs across SSV2, YouCook2, HoloAssist, and
  Roundabout-TAU.
- `examples/results/paper_results_snapshot.json` contains the compact
  paper-facing result snapshot for all three prompt families and ten models.

These examples are intentionally small. Full prediction JSONL files are large
and should be released separately if needed.
