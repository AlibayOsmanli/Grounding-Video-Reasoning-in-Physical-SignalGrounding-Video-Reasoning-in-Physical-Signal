"""
Shared prediction-payload validation utilities.

These helpers are intentionally lightweight so they can be used by both the
inference path and the metrics path without importing model-specific modules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

REFUSAL_PATTERNS = (
    "i'm unable to provide",
    "i am unable to provide",
    "i'm unable to determine",
    "i am unable to determine",
    "i cannot provide",
    "i can't provide",
    "i cannot determine",
    "i can't determine",
    "i do not have access to the video",
    "i cannot see the video",
    "i can't see the video",
    "i cannot actually see the video",
    "i can help explain the concept",
    "let me know how you would like to proceed",
)


def is_refusal_text(text: Any) -> bool:
    """Return True when ``text`` matches known non-answer / refusal patterns."""
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def get_prediction_invalid_reason(prediction: Any) -> Optional[str]:
    """Return a stable invalid-reason string, or ``None`` when valid."""
    if not isinstance(prediction, dict):
        return "prediction_not_dict"

    error = prediction.get("error")
    if isinstance(error, str) and error.strip():
        return "error_payload"

    a_what = prediction.get("a_what")
    a_when = prediction.get("a_when")
    a_where = prediction.get("a_where")
    raw = prediction.get("raw", "")

    has_what = isinstance(a_what, str) and a_what.strip() != ""
    has_when = (
        isinstance(a_when, dict)
        and "start_sec" in a_when
        and "end_sec" in a_when
    )
    has_where = (
        isinstance(a_where, dict)
        and all(k in a_where for k in ("x", "y", "w", "h"))
    )

    if not (has_what or has_when or has_where):
        return "empty_payload"
    if is_refusal_text(raw):
        return "refusal_text"
    return None


def is_valid_prediction_payload(prediction: Dict[str, Any]) -> bool:
    """Return True when the payload is usable for scoring / resuming."""
    return get_prediction_invalid_reason(prediction) is None

