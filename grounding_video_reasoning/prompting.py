"""
Prompt utilities for V-STaR 2.0 inference.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional


_AUXILIARY_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "happen",
    "happening",
    "happens",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "must",
    "of",
    "on",
    "or",
    "should",
    "clip",
    "event",
    "moment",
    "occur",
    "occurring",
    "occurs",
    "scene",
    "segment",
    "that",
    "the",
    "their",
    "then",
    "they",
    "this",
    "to",
    "video",
    "was",
    "were",
    "what",
    "when",
    "where",
    "while",
    "will",
    "with",
    "would",
}

_BANNED_STEMS = (
    "accelerat",
    "ballistic",
    "collid",
    "collis",
    "deform",
    "fluid",
    "force",
    "friction",
    "gravit",
    "kinetic",
    "moment",
    "pressur",
    "static",
    "torque",
    "traject",
    "velocit",
    "viscos",
)

_SAFE_OBJECT_REPLACEMENTS = {
    "fluid": "liquid",
    "fluids": "liquid",
}

_DOMAIN_DEFAULT_VERBS = {
    "gravity": "move downward",
    "fluids": "spread or collect",
    "collisions": "move or make contact",
    "deformation": "change shape",
    "friction": "move across the surface",
    "state_changes": "change",
}

_IRREGULAR_VERBS = {
    "begins": "begin",
    "changing": "change",
    "changes": "change",
    "closing": "close",
    "closes": "close",
    "deviating": "deviate",
    "drops": "drop",
    "falls": "fall",
    "fills": "fill",
    "goes": "go",
    "hits": "hit",
    "lands": "land",
    "melts": "melt",
    "moving": "move",
    "moves": "move",
    "opening": "open",
    "opens": "open",
    "pours": "pour",
    "rotating": "rotate",
    "rolls": "roll",
    "sliding": "slide",
    "spinning": "spin",
    "slides": "slide",
    "spills": "spill",
    "spreads": "spread",
    "turns": "turn",
    "violates": "violate",
    "violating": "violate",
}

_GERUND_OVERRIDES = {
    "grab": "grabbing",
    "run": "running",
    "sit": "sitting",
    "slip": "slipping",
    "spin": "spinning",
    "stop": "stopping",
}

_DOMAIN_DEFAULT_EVENTS = {
    "gravity": "the {object} moves downward",
    "fluids": "the {object} moves or spreads",
    "collisions": "the {object} makes contact",
    "deformation": "the {object} changes shape",
    "friction": "the {object} moves across the surface",
    "state_changes": "the {object} changes",
}

_SOURCE_VERB_EVENT_MAP = {
    "grab": "the {object} is grabbed",
    "hold": "the {object} is held",
    "pick": "the {object} is picked up",
    "pickup": "the {object} is picked up",
    "lift": "the {object} is lifted",
    "place": "the {object} is placed down",
    "put": "the {object} is placed down",
    "move": "the {object} moves",
    "open": "the {object} is opened",
    "close": "the {object} is closed",
    "turn": "the {object} is turned",
    "rotate": "the {object} rotates",
    "push": "the {object} is pushed",
    "pull": "the {object} is pulled",
}


def _sample_value(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _metadata_value(sample: Any, key: str, default: Any = None) -> Any:
    metadata = _sample_value(sample, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        return metadata.get(key, default)
    return default


def _normalize_domain(domain: Optional[str]) -> str:
    if not domain:
        return ""
    return str(domain).strip().lower().replace(" ", "_")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_banned_stem(text: str) -> bool:
    token = text.lower()
    return any(stem in token for stem in _BANNED_STEMS)


def _normalize_verb(token: str) -> str:
    token = token.lower().strip()
    if not token:
        return ""
    if token in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _sanitize_object_name(raw_object: Any) -> str:
    text = _normalize_whitespace(str(raw_object or "object")).lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w]
    if not words:
        return "object"

    sanitized = []
    for word in words[:4]:
        replacement = _SAFE_OBJECT_REPLACEMENTS.get(word, word)
        if _contains_banned_stem(replacement):
            continue
        sanitized.append(replacement)

    return " ".join(sanitized) if sanitized else "object"


def _extract_event_verb(sample: Any, domain_key: str) -> str:
    default = _DOMAIN_DEFAULT_VERBS.get(domain_key, "move or change")
    object_name = _sanitize_object_name(
        _metadata_value(sample, "object", _sample_value(sample, "object", "object"))
    )
    object_words = re.findall(r"[a-z0-9]+", object_name)
    object_tokens = set(object_words)

    candidates = [
        _sample_value(sample, "a_what", ""),
        _sample_value(sample, "q_what", ""),
        _metadata_value(sample, "temporal_question", ""),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        text = _normalize_whitespace(str(candidate).lower())
        if (
            "directional constraints" in text
            or "direction violation" in text
            or "violat" in text
        ):
            if re.search(r"\bslip\w*|\bskid\w*|\bdrift\w*", text):
                return "slip during the turn"
            return "deviate from the intended turn"

        if object_words:
            object_pattern = (
                r"(?:the|a|an)?\s*"
                + r"\s+".join(map(re.escape, object_words))
                + r"\s+"
            )
            text = re.sub(rf"^{object_pattern}", "", text)

        for token in re.findall(r"[a-z]+", text):
            if token in _AUXILIARY_WORDS or token in object_tokens:
                continue
            normalized = _normalize_verb(token)
            if not normalized or normalized in _AUXILIARY_WORDS:
                continue
            if _contains_banned_stem(normalized):
                return default
            return normalized

    return default


def _source_extra(sample: Any) -> Mapping[str, Any]:
    extra = _metadata_value(sample, "source_extra", {})
    if isinstance(extra, Mapping):
        return extra
    top_level = _sample_value(sample, "source_extra", {})
    if isinstance(top_level, Mapping):
        return top_level
    return {}


def _text_candidates(sample: Any) -> list[str]:
    values = [
        _sample_value(sample, "q_what", ""),
        _sample_value(sample, "a_what", ""),
        _metadata_value(sample, "temporal_question", ""),
        _metadata_value(sample, "spatial_question", ""),
    ]
    return [str(v) for v in values if v]


def _joined_candidate_text(sample: Any) -> str:
    return _normalize_whitespace(" ".join(_text_candidates(sample)).lower())


def _to_gerund(verb: str) -> str:
    token = _normalize_verb(verb.strip().lower())
    if not token:
        return ""
    if token in _GERUND_OVERRIDES:
        return _GERUND_OVERRIDES[token]
    if token.endswith("ing"):
        return token
    if token == "be":
        return "being"
    if token == "make":
        return "making"
    if token == "move":
        return "moving"
    if token == "slide":
        return "sliding"
    if token == "change":
        return "changing"
    if token == "close":
        return "closing"
    if token == "open":
        return "opening"
    if token == "rotate":
        return "rotating"
    if token.endswith("ie"):
        return token[:-2] + "ying"
    if token.endswith("e") and not token.endswith("ee"):
        return token[:-1] + "ing"
    return token + "ing"


def _to_third_person(verb: str) -> str:
    token = _normalize_verb(verb.strip().lower())
    if not token:
        return ""
    if token == "be":
        return "is"
    if token == "have":
        return "has"
    if token.endswith("y") and len(token) > 2 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith(("s", "sh", "ch", "x", "z", "o")):
        return token + "es"
    return token + "s"


def _to_third_person_phrase(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    words[0] = _to_third_person(words[0])
    return " ".join(words)


def _normalize_clause(text: str) -> str:
    return _normalize_whitespace(text.strip().rstrip(".?!,;:"))


def _source_verb_to_event_answer(object_name: str, verb: str) -> Optional[str]:
    clean = _normalize_verb(verb)
    template = _SOURCE_VERB_EVENT_MAP.get(clean)
    if not template:
        return None
    return template.format(object=object_name)


def _extract_direction_phrase(text: str) -> str:
    patterns = (
        r"from left to right",
        r"from right to left",
        r"left to right",
        r"right to left",
        r"up and down",
        r"upward",
        r"downward",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            phrase = match.group(0)
            if phrase == "left to right":
                return "from left to right"
            if phrase == "right to left":
                return "from right to left"
            return phrase
    return ""


def _derive_vstar_like_event_answer(sample: Any, object_name: str, domain_key: str) -> str:
    extra = _source_extra(sample)
    verb = extra.get("verb")
    if verb:
        event_answer = _source_verb_to_event_answer(object_name, str(verb))
        if event_answer:
            return event_answer

    text = _joined_candidate_text(sample)
    direction = _extract_direction_phrase(text)
    if direction:
        return f"the {object_name} moves {direction}"
    if re.search(r"\bcollid\w*|\bcrash\w*", text):
        return f"the {object_name} makes contact"
    if "rotational motion" in text or re.search(r"\brotate\w*|\bspin\w*", text):
        return f"the {object_name} starts to rotate"
    if re.search(r"\bvibrat\w*", text):
        return f"the {object_name} vibrates"
    if re.search(r"\bslip\w*|\bskid\w*|\bdrift\w*", text):
        return f"the {object_name} slips during the turn"
    if re.search(r"\bgrab\w*|\bgrip\w*|\bhold\w*", text):
        return f"the {object_name} is grabbed"
    if re.search(r"\bopen\w*", text):
        return f"the {object_name} is opened"
    if re.search(r"\bclose\w*", text):
        return f"the {object_name} is closed"
    if re.search(r"\bfill\w*", text):
        return f"the {object_name} is filled"
    if re.search(r"\bpour\w*|\bspill\w*|\bspread\w*|\bflow\w*", text):
        return f"the {object_name} moves or spreads"
    if re.search(r"\bfall\w*|\bdrop\w*|\bdescend\w*", text):
        return f"the {object_name} moves downward"
    if re.search(r"\bhit\w*|\bcontact\b|\btouch\w*|\bimpact\w*", text):
        return f"the {object_name} makes contact"
    if re.search(r"\bbend\w*|\bdeform\w*|\bcompress\w*|\bstretch\w*|\bsquash\w*", text):
        return f"the {object_name} changes shape"
    if re.search(r"\btransition\w*|\bchange\w*", text):
        return f"the {object_name} changes"

    event_verb = _extract_event_verb(sample, domain_key)
    event_verb = event_verb.split(" or ")[0].strip()
    if not event_verb:
        default_event = _DOMAIN_DEFAULT_EVENTS.get(domain_key, "the {object} changes")
        return default_event.format(object=object_name)
    if event_verb.startswith("make contact"):
        return f"the {object_name} makes contact"
    if event_verb.startswith(("move ", "change ", "spread", "collect")) or event_verb in {"move", "change"}:
        return f"the {object_name} {_to_third_person_phrase(event_verb)}"
    return f"the {object_name} starts to {event_verb}"


def _derive_vstar_like_what_question(object_name: str, event_answer: str) -> str:
    lower = event_answer.lower()
    move_like_markers = (
        f"the {object_name} moves",
        f"the {object_name} starts to",
        f"the {object_name} rotates",
    )
    if any(lower.startswith(marker) for marker in move_like_markers):
        return f"What does the {object_name} do?"
    return f"What happens to the {object_name}?"


def _derive_action_phrase(event_answer: str, object_name: str) -> str:
    clean_answer = _normalize_clause(event_answer.lower())
    object_pattern = rf"^(?:the\s+)?{re.escape(object_name)}\s+"
    remainder = re.sub(object_pattern, "", clean_answer, flags=re.IGNORECASE).strip()

    if not remainder:
        return f"happening to the {object_name}"
    if remainder.startswith("is "):
        return "being " + remainder[3:].strip()
    if remainder.startswith("are "):
        return "being " + remainder[4:].strip()
    if remainder.startswith("starts to "):
        return "starting to " + remainder[len("starts to "):].strip()
    if remainder.startswith("begins to "):
        return "beginning to " + remainder[len("begins to "):].strip()

    words = remainder.split()
    if not words:
        return f"happening to the {object_name}"
    words[0] = _to_gerund(words[0])
    return " ".join(words)


def _derive_prompt_aligned_action_answer(sample: Any) -> str:
    """Return a short semantic action answer for non-physics prompt families.

    This is used only for prompt-aligned evaluation of ``a_what`` on
    ``neutral_rstr`` and ``vstar_like``.  It intentionally strips the object
    prefix and physics-heavy wording so that models answering with concise
    action phrases are not unfairly scored against the original physics answer.
    """

    domain_key = _normalize_domain(_sample_value(sample, "domain", ""))
    object_name = _sanitize_object_name(
        _metadata_value(sample, "object", _sample_value(sample, "object", "object"))
    )
    event_answer = _normalize_clause(
        _derive_vstar_like_event_answer(sample, object_name, domain_key)
    )

    object_pattern = rf"^(?:the\s+)?{re.escape(object_name)}\s+"
    answer = re.sub(object_pattern, "", event_answer, flags=re.IGNORECASE).strip()
    if answer.startswith("is "):
        answer = answer[3:].strip()
    elif answer.startswith("are "):
        answer = answer[4:].strip()

    return _normalize_clause(answer or event_answer)


def derive_prompt_aligned_a_what(sample: Any, prompt_condition: str) -> str:
    """Return the evaluation target for ``a_what`` under a prompt family.

    ``physics`` keeps the original annotation answer.
    ``neutral_rstr`` and ``vstar_like`` use a deterministic, prompt-aligned
    semantic target derived from existing metadata and physics annotations.
    """

    normalized_condition = _normalize_clause(prompt_condition).lower()
    if normalized_condition == "physics":
        return _normalize_clause(str(_sample_value(sample, "a_what", "")))
    if normalized_condition in {"neutral_rstr", "vstar_like"}:
        return _derive_prompt_aligned_action_answer(sample)
    raise ValueError(f"Unsupported prompt condition for a_what derivation: {prompt_condition}")


def _rounded_clip_duration(sample: Any, video_duration_sec: Optional[float]) -> int:
    if video_duration_sec is not None and video_duration_sec > 0:
        return max(1, int(round(video_duration_sec)))

    a_when = _sample_value(sample, "a_when", None)
    end_sec = _sample_value(a_when, "end_sec", 0.0)
    try:
        end_value = float(end_sec)
    except Exception:
        end_value = 0.0
    return max(1, int(round(end_value))) if end_value > 0 else 1


def build_neutral_rstr_prompt(
    sample: Any,
    video_duration_sec: Optional[float] = None,
) -> str:
    """Build a neutral what/when/where prompt without physics vocabulary."""

    NEUTRAL_TEMPLATES = (
        "What does the {object} do during the part where it {event_verb}?",
        "How does the {object} move or change during the part where it {event_verb}?",
        "What happens to the {object} during the part where it {event_verb}?",
        "What does the {object} do in the moment when it {event_verb}?",
        "What happens to the {object} after it {event_verb}?",
        "How does the {object} look or change while it {event_verb}?",
        "How does the {object} move while it {event_verb}?",
        "What change does the {object} go through while it {event_verb}?",
    )

    domain_to_template_ids = {
        "gravity": (0, 1),
        "fluids": (2,),
        "collisions": (3, 4),
        "deformation": (5,),
        "friction": (6,),
        "state_changes": (7,),
    }

    domain_key = _normalize_domain(_sample_value(sample, "domain", ""))
    template_ids = domain_to_template_ids.get(domain_key, (0,))

    sample_id = _sample_value(sample, "sample_id", "") or _sample_value(sample, "video_id", "")
    selector = sum(ord(ch) for ch in str(sample_id)) % len(template_ids)
    template = NEUTRAL_TEMPLATES[template_ids[selector]]

    object_name = _sanitize_object_name(
        _metadata_value(sample, "object", _sample_value(sample, "object", "object"))
    )
    event_verb = _extract_event_verb(sample, domain_key)
    if event_verb.startswith("starts to ") or event_verb.startswith("begins to "):
        event_phrase = event_verb
    else:
        event_phrase = f"starts to {event_verb}"

    q_what = template.format(object=object_name, event_verb=event_phrase)
    event_description = f"the moment when the {object_name} {event_phrase}"

    if video_duration_sec is not None:
        q_when = (
            f"The video is {video_duration_sec:.1f} seconds long. "
            f"At what start and end time (in seconds, as floats) does the "
            f"following event occur? "
            f"Event: {event_description}\n"
            f'Output ONLY a JSON object: {{"start_sec": <float>, "end_sec": <float>}}'
        )
    else:
        q_when = (
            f"When does {event_description} happen in the video?\n"
            f'Output ONLY a JSON object: {{"start_sec": <float>, "end_sec": <float>}}'
        )

    q_where = _sample_value(
        sample,
        "q_where",
        "Where is the primary object during the relevant time span?",
    )

    return (
        "You are evaluating a video understanding task. Watch the video carefully "
        "and answer each question in order.\n\n"
        f"[Step 1 - What]  {q_what}\n"
        f"[Step 2 - When]  {q_when}\n"
        f"[Step 3 - Where] {q_where}\n\n"
        "Provide your answers as a JSON object with keys "
        "'a_what', 'a_when' ({\"start_sec\": <float>, \"end_sec\": <float>}), "
        "and 'a_where' ({\"x\": <int>, \"y\": <int>, \"w\": <int>, \"h\": <int>}).\n"
        "Return ONLY the JSON object. Do not include markdown, code fences, or reasoning."
    )


def build_vstar_like_prompt(
    sample: Any,
    video_duration_sec: Optional[float] = None,
) -> str:
    """Build a deterministic V-STaR-like prompt from existing annotations."""

    domain_key = _normalize_domain(_sample_value(sample, "domain", ""))
    object_name = _sanitize_object_name(
        _metadata_value(sample, "object", _sample_value(sample, "object", "object"))
    )
    event_answer = _normalize_clause(
        _derive_vstar_like_event_answer(sample, object_name, domain_key)
    )
    q_what = _derive_vstar_like_what_question(object_name, event_answer)
    q_when = f"When is the moment '{event_answer}' occurring?"
    action_phrase = _derive_action_phrase(event_answer, object_name)
    clip_duration_rounded = _rounded_clip_duration(sample, video_duration_sec)
    q_where = (
        f"Where is the {object_name} during the event of {action_phrase} "
        f"between 0s and {clip_duration_rounded}s?"
    )

    return (
        "You are evaluating a video understanding task. Watch the video carefully "
        "and answer each question in order.\n\n"
        f"[Step 1 - What]  {q_what}\n"
        f"[Step 2 - When]  {q_when}\n"
        f"[Step 3 - Where] {q_where}\n\n"
        "Provide your answers as a JSON object with keys "
        "'a_what', 'a_when' ({\"start_sec\": <float>, \"end_sec\": <float>}), "
        "and 'a_where' ({\"x\": <int>, \"y\": <int>, \"w\": <int>, \"h\": <int>}).\n"
        "Return ONLY the JSON object. Do not include markdown, code fences, or reasoning."
    )
