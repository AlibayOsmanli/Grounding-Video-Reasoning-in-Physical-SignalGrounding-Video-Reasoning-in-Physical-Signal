"""
Ground-truth JSON schema and prompt builder
===========================================
Implements the **Reverse Spatio-Temporal Reasoning (RSTR)** annotation format
used by Grounding Video Reasoning in Physical Signals.

The RSTR chain uses the following evaluation order:
  What -> When -> Where

Pydantic v2 dataclasses are used so each schema object can be serialised to
JSON with full field validation.  A :class:`RSTRDatasetBuilder` helper collects
individual :class:`RSTRSample` objects and persists them to / loads them from
a single ``annotations.json`` file.

Typical usage
-------------
    from grounding_video_reasoning.schema import RSTRSample, TemporalSpan, BoundingBox, RSTRDatasetBuilder

    sample = RSTRSample(
        video_id   = "fluid_001",
        condition  = "original",
        video_path = "/outputs/fluid_001/fluid_001_original.mp4",
        q_what     = "What physical action is occurring?",
        a_what     = "A stream of water fills the container.",
        q_when     = "When does the water filling occur?",
        a_when     = TemporalSpan(start_sec=1.2, end_sec=4.8),
        q_where    = "Where is the container during [1.2 s - 4.8 s]?",
        a_where    = BoundingBox(x=120, y=80, w=200, h=310),
    )

    builder = RSTRDatasetBuilder(output_path=Path("annotations.json"))
    builder.add(sample)
    builder.save()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Accepted condition strings (mirrors perturbation module to avoid circular import)
ConditionType = Literal["original", "shuffled", "ablated", "frame_masked"]


# ---------------------------------------------------------------------------
# Sub-schema: Temporal Span
# ---------------------------------------------------------------------------
class TemporalSpan(BaseModel):
    """Closed interval [start_sec, end_sec] in video-time seconds.

    Attributes
    ----------
    start_sec : Start time in seconds (>= 0).
    end_sec   : End time in seconds (> start_sec).
    """
    start_sec: float = Field(..., ge=0.0, description="Event start time (seconds).")
    end_sec:   float = Field(...,          description="Event end time (seconds).")

    @model_validator(mode="after")
    def _end_after_start(self) -> "TemporalSpan":
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"end_sec ({self.end_sec}) must be strictly greater than "
                f"start_sec ({self.start_sec})."
            )
        return self

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ---------------------------------------------------------------------------
# Sub-schema: Bounding Box
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (XYWH format).

    Attributes
    ----------
    x : Top-left x coordinate (pixels, >= 0).
    y : Top-left y coordinate (pixels, >= 0).
    w : Box width  in pixels (> 0).
    h : Box height in pixels (> 0).
    frame_ref : Optional reference frame index or timestamp the box is staked to.
    """
    x:         int   = Field(..., ge=0,  description="Top-left x (pixels).")
    y:         int   = Field(..., ge=0,  description="Top-left y (pixels).")
    w:         int   = Field(..., gt=0,  description="Width (pixels).")
    h:         int   = Field(..., gt=0,  description="Height (pixels).")
    frame_ref: Optional[int] = Field(None, ge=0, description="Reference frame index.")

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format."""
        return self.x, self.y, self.x + self.w, self.y + self.h


# ---------------------------------------------------------------------------
# Core RSTR Sample
# ---------------------------------------------------------------------------
class RSTRSample(BaseModel):
    """A single RSTR ground-truth annotation for one video / condition pair.

    Fields follow the What -> When -> Where reasoning chain.

    Attributes
    ----------
    video_id   : Unique identifier for the source video clip.
    condition  : Perturbation condition (original | shuffled | ablated).
    video_path : Absolute or relative path to the video file.
    domain     : Optional physics domain tag such as fluids, collisions, or gravity.
    source_fps : FPS of the referenced video (used to convert timestamps).
    q_what     : "What physical action is occurring?"
    a_what     : Free-text description of the physical event.
    q_when     : "When does <a_what> happen?"
    a_when     : :class:`TemporalSpan` with start / end in seconds.
    q_where    : "Where is the object during [<a_when>]?"
    a_where    : :class:`BoundingBox` for the primary object of interest.
    metadata   : Free-form key-value store for extra annotation fields.
    """
    video_id:   str           = Field(..., description="Unique video identifier.")
    condition:  ConditionType = Field(..., description="Perturbation condition.")
    video_path: str           = Field(..., description="Path to the processed video file.")
    domain:     Optional[str] = Field(None, description="Physics domain tag.")
    source_fps: Optional[float] = Field(None, gt=0.0, description="Video frame rate.")

    # RSTR chain.
    q_what: str = Field(
        default="What physical action is occurring?",
        description="Chain step 1 - What question."
    )
    a_what: str = Field(..., description="Ground-truth answer to q_what.")

    q_when: str = Field(
        default="When does the described physical action happen?",
        description="Chain step 2 - When question."
    )
    a_when: TemporalSpan = Field(..., description="Ground-truth temporal span.")

    q_where: str = Field(
        default="Where is the primary object during the relevant time span?",
        description="Chain step 3 - Where question."
    )
    a_where: BoundingBox = Field(..., description="Ground-truth bounding box.")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra fields such as annotator or confidence.",
    )

    # Derived helpers.
    @property
    def sample_id(self) -> str:
        """Globally unique sample key: ``<video_id>_<condition>``."""
        return f"{self.video_id}_{self.condition}"

    def build_prompt(self, video_duration_sec: Optional[float] = None) -> str:
        """Return a structured natural-language prompt for a Video-LLM.

        Parameters
        ----------
        video_duration_sec :
            Actual duration of the video clip in seconds.  When provided the
            Step-2 temporal question is replaced with an explicit, duration-aware
            instruction that asks for float timestamps and embeds the event
            description, dramatically reducing the incidence of the model
            returning a hardcoded fallback like ``{"start_sec": 0, "end_sec": 5}``.
        """
        if video_duration_sec is not None:
            when_q = (
                f"The video is {video_duration_sec:.1f} seconds long. "
                f"At what start and end time (in seconds, as floats) does the "
                f"following event occur? "
                f"Event: {self.a_what}\n"
                f'Output ONLY a JSON object: {{"start_sec": <float>, "end_sec": <float>}}'
            )
        else:
            when_q = self.q_when

        return (
            f"You are evaluating a physical AI task. Watch the video carefully "
            f"and answer each question in order.\n\n"
            f"[Step 1 - What]  {self.q_what}\n"
            f"[Step 2 - When]  {when_q}\n"
            f"[Step 3 - Where] {self.q_where}\n\n"
            f"Provide your answers as a JSON object with keys "
            f"'a_what', 'a_when' ({{\"start_sec\": <float>, \"end_sec\": <float>}}), "
            f"and 'a_where' ({{\"x\": <int>, \"y\": <int>, \"w\": <int>, \"h\": <int>}}).\n"
            f"Return ONLY the JSON object. Do not include markdown, code fences, or reasoning."
        )


# ---------------------------------------------------------------------------
# Dataset builder / loader
# ---------------------------------------------------------------------------
class RSTRDataset(BaseModel):
    """Container for a collection of :class:`RSTRSample` objects."""
    version:  str             = Field(default="2.0.0")
    samples:  List[RSTRSample] = Field(default_factory=list)

    # Fast lookup by sample_id
    def get(self, sample_id: str) -> Optional[RSTRSample]:
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        return None

    def filter_by_condition(self, condition: ConditionType) -> List[RSTRSample]:
        return [s for s in self.samples if s.condition == condition]

    def filter_by_video(self, video_id: str) -> List[RSTRSample]:
        return [s for s in self.samples if s.video_id == video_id]


class RSTRDatasetBuilder:
    """Accumulates :class:`RSTRSample` objects and serialises the dataset.

    Parameters
    ----------
    output_path : Path where ``annotations.json`` will be written.
    """

    def __init__(self, output_path: Path):
        self._output_path = Path(output_path)
        self._dataset = RSTRDataset()

    def add(self, sample: RSTRSample) -> "RSTRDatasetBuilder":
        """Append a sample (chainable)."""
        self._dataset.samples.append(sample)
        return self

    def add_batch(self, samples: List[RSTRSample]) -> "RSTRDatasetBuilder":
        self._dataset.samples.extend(samples)
        return self

    def save(self, indent: int = 2) -> Path:
        """Serialise to JSON and return the output path."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._dataset.model_dump()
        with open(self._output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False)
        logger.info("Saved %d RSTR sample(s) to %s", len(self._dataset.samples), self._output_path)
        return self._output_path

    @property
    def dataset(self) -> RSTRDataset:
        return self._dataset

    # Class-method loaders.
    @classmethod
    def load(cls, path: Path) -> RSTRDataset:
        """Load an existing ``annotations.json`` into an :class:`RSTRDataset`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        dataset = RSTRDataset.model_validate(raw)
        logger.info("Loaded %d sample(s) from %s", len(dataset.samples), path)
        return dataset


# ---------------------------------------------------------------------------
# Helper: build a trio of samples (one per condition) from raw annotations
# ---------------------------------------------------------------------------
def make_condition_trio(
    video_id:   str,
    output_dir: Path,
    a_what:     str,
    a_when:     TemporalSpan,
    a_where:    BoundingBox,
    domain:     Optional[str]  = None,
    source_fps: Optional[float] = None,
    metadata:   Optional[Dict[str, Any]] = None,
) -> List[RSTRSample]:
    """Convenience factory that creates :class:`RSTRSample` objects for all four
    conditions given a single set of ground-truth labels.

    Parameters
    ----------
    video_id    : Source video identifier (without extension).
    output_dir  : Root output directory produced by the perturbation pipeline.
    a_what / a_when / a_where : Ground-truth annotation values.
    domain      : Optional physics domain string.
    source_fps  : Source video frame rate.
    metadata    : Extra annotation fields propagated to all three samples.
    """
    conditions: List[ConditionType] = ["original", "shuffled", "ablated", "frame_masked"]
    metadata   = metadata or {}
    samples    = []

    for cond in conditions:
        video_path = str(Path(output_dir) / video_id / f"{video_id}_{cond}.mp4")
        samples.append(
            RSTRSample(
                video_id   = video_id,
                condition  = cond,
                video_path = video_path,
                domain     = domain,
                source_fps = source_fps,
                a_what     = a_what,
                a_when     = a_when,
                a_where    = a_where,
                metadata   = metadata,
            )
        )
    return samples
