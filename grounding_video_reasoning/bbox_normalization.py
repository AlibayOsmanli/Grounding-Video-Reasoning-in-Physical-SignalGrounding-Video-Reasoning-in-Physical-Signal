"""
Bounding-box normalization helpers.

These utilities keep model-specific bbox coordinate fixes in one place so the
same logic can be applied both during inference-time serialization and during
metrics-time rescoring of already-saved prediction files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

INTERNVL35_REFERENCE_GRID = 1000


def _coerce_xywh_box(box: Any) -> Optional[Dict[str, int]]:
    """Return a best-effort ``{x, y, w, h}`` dict with integer values."""
    if not isinstance(box, dict):
        return None
    try:
        return {
            "x": int(round(float(box.get("x", 0)))),
            "y": int(round(float(box.get("y", 0)))),
            "w": int(round(float(box.get("w", 0)))),
            "h": int(round(float(box.get("h", 0)))),
        }
    except (TypeError, ValueError):
        return None


def rescale_xywh_box(
    box: Dict[str, int],
    *,
    native_width: int,
    native_height: int,
    reference_grid: int,
) -> Dict[str, int]:
    """Scale an ``xywh`` box from a fixed reference grid into native pixels."""
    if reference_grid <= 0 or native_width <= 0 or native_height <= 0:
        return dict(box)
    return {
        "x": int(round(box["x"] * native_width / reference_grid)),
        "y": int(round(box["y"] * native_height / reference_grid)),
        "w": int(round(box["w"] * native_width / reference_grid)),
        "h": int(round(box["h"] * native_height / reference_grid)),
    }


def normalize_bbox_for_model(
    box: Any,
    *,
    native_width: int = 0,
    native_height: int = 0,
    model_name: str = "",
) -> Optional[Dict[str, int]]:
    """Normalize a model-emitted bbox into native video pixels.

    `internvl35` is currently treated specially: its `a_where` predictions are
    interpreted as being emitted on a `0..1000` reference grid and rescaled to
    the video's native resolution before scoring/serialization.
    """
    normalized = _coerce_xywh_box(box)
    if normalized is None:
        return None

    model_key = (model_name or "").lower()
    if "internvl35" in model_key and native_width > 0 and native_height > 0:
        return rescale_xywh_box(
            normalized,
            native_width=native_width,
            native_height=native_height,
            reference_grid=INTERNVL35_REFERENCE_GRID,
        )

    return normalized
