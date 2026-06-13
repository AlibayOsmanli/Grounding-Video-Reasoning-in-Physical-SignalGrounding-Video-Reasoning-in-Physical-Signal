"""Metrics used for grounded video reasoning evaluation.

The implementation mirrors the V-STaR 2.0 scoring path: text accuracy for the
``what`` answer, temporal IoU for ``when``, spatial IoU for ``where``, and a
logarithmic geometric mean (LGM) composite.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .bbox_normalization import normalize_bbox_for_model
from .prediction_validation import get_prediction_invalid_reason

EPS = 1e-7


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_text_accuracy(prediction: str, ground_truth: str) -> float:
    """Token-F1 score between prediction and ground-truth event text."""
    if not prediction or not ground_truth:
        return 0.0
    pred_tokens = set(normalize_text(prediction).split())
    gt_tokens = set(normalize_text(ground_truth).split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Temporal intersection-over-union for two second-based intervals."""
    if pred_end <= pred_start or gt_end <= gt_start:
        return 0.0
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def compute_spatial_iou(pred_box: Mapping[str, Any], gt_box: Mapping[str, Any]) -> float:
    """Spatial intersection-over-union for axis-aligned xywh boxes."""
    def to_xyxy(box: Mapping[str, Any]) -> tuple[int, int, int, int]:
        x1 = int(box.get("x", 0))
        y1 = int(box.get("y", 0))
        x2 = x1 + int(box.get("w", 0))
        y2 = y1 + int(box.get("h", 0))
        return x1, y1, x2, y2

    px1, py1, px2, py2 = to_xyxy(pred_box)
    gx1, gy1, gx2, gy2 = to_xyxy(gt_box)
    if px2 <= px1 or py2 <= py1 or gx2 <= gx1 or gy2 <= gy1:
        return 0.0

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def compute_lgm(acc: float, t_iou: float, s_iou: float, eps: float = EPS) -> float:
    """Logarithmic geometric mean of semantic, temporal, and spatial scores."""
    def clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    acc = clamp(acc)
    t_iou = clamp(t_iou)
    s_iou = clamp(s_iou)
    value = -1.0 / 3.0 * (
        math.log(max(eps, 1.0 - acc))
        + math.log(max(eps, 1.0 - t_iou))
        + math.log(max(eps, 1.0 - s_iou))
    )
    return float(max(0.0, value))


@dataclass
class SampleScore:
    sample_id: str
    video_id: str
    condition: str
    acc: float
    t_iou: float
    s_iou: float
    lgm: float
    invalid_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_sample(annotation: Mapping[str, Any], prediction_record: Mapping[str, Any]) -> SampleScore:
    """Score one prediction record against one annotation record."""
    prediction = prediction_record.get("prediction", {})
    invalid_reason = get_prediction_invalid_reason(prediction) or ""
    video_id = str(prediction_record.get("video_id") or annotation.get("video_id") or "")
    sample_id = str(prediction_record.get("sample_id") or annotation.get("sample_id") or annotation.get("vid"))
    condition = str(prediction_record.get("condition") or annotation.get("condition") or "")

    if invalid_reason:
        return SampleScore(sample_id, video_id, condition, 0.0, 0.0, 0.0, 0.0, invalid_reason)

    gt_when = annotation["a_when"]
    gt_where = annotation["a_where"]
    pred_when = prediction.get("a_when") or {}
    pred_where = normalize_bbox_for_model(
        prediction.get("a_where"),
        native_width=int(annotation.get("width", 0) or 0),
        native_height=int(annotation.get("height", 0) or 0),
        model_name=str(prediction_record.get("model_name", "")),
    )

    acc = compute_text_accuracy(str(prediction.get("a_what", "")), str(annotation.get("a_what", "")))
    t_iou = compute_temporal_iou(
        float(pred_when.get("start_sec", 0.0)),
        float(pred_when.get("end_sec", 0.0)),
        float(gt_when.get("start_sec", 0.0)),
        float(gt_when.get("end_sec", 0.0)),
    )
    s_iou = compute_spatial_iou(pred_where or {}, gt_where)
    return SampleScore(sample_id, video_id, condition, acc, t_iou, s_iou, compute_lgm(acc, t_iou, s_iou))


def aggregate_scores(scores: Iterable[SampleScore]) -> dict[str, Any]:
    """Aggregate sample scores by condition and compute diagnostic indices."""
    by_condition: dict[str, list[SampleScore]] = {}
    for score in scores:
        by_condition.setdefault(score.condition, []).append(score)

    conditions = {}
    for condition, rows in sorted(by_condition.items()):
        n = len(rows)
        if n == 0:
            continue
        conditions[condition] = {
            "n_samples": n,
            "mean_acc": sum(r.acc for r in rows) / n,
            "mean_tiou": sum(r.t_iou for r in rows) / n,
            "mean_siou": sum(r.s_iou for r in rows) / n,
            "mean_lgm": sum(r.lgm for r in rows) / n,
            "invalid_count": sum(1 for r in rows if r.invalid_reason),
        }

    original = conditions.get("original", {}).get("mean_lgm", 0.0)
    shuffled = conditions.get("shuffled", {}).get("mean_lgm", 0.0)
    ablated = conditions.get("ablated", {}).get("mean_lgm", 0.0)
    masked = conditions.get("frame_masked", {}).get("mean_lgm", 0.0)

    diagnostic_indices = {
        "sbi": 1.0 - (original - shuffled) if original else 0.0,
        "pri": ablated / original if original else 0.0,
        "spi": masked / original if original else 0.0,
    }
    return {"conditions": conditions, "diagnostic_indices": diagnostic_indices}
