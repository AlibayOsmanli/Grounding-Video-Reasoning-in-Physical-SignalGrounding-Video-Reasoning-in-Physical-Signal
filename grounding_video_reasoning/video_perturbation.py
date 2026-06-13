"""
Video perturbation pipeline
===========================
Generates three testing conditions for each input .mp4 video:

  Condition A  (original)  - unchanged RGB video
  Condition B  (shuffled)  - frames randomly shuffled along the time axis
  Condition C  (ablated)   - texture stripped; structural edges preserved
                              via grayscale, bilateral filtering, and blur
  Condition D  (frame_masked) - selected frames replaced by black frames

Usage (CLI)
-----------
  python -m grounding_video_reasoning.video_perturbation \\
      --input_dir  /path/to/raw_videos \\
      --output_dir /path/to/output \\
      --num_workers 8

Architecture
------------
* PerturbationConfig  - frozen dataclass that holds all hyperparameters
* VideoWriter         - thin context-manager wrapper around cv2.VideoWriter
* FrameProcessor      - stateless transforms
* VideoPerturbator    - orchestrates per-video processing with multiprocessing
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITION_ORIGINAL    = "original"
CONDITION_SHUFFLED    = "shuffled"
CONDITION_ABLATED     = "ablated"
CONDITION_FRAME_MASKED = "frame_masked"

ALL_CONDITIONS = (
    CONDITION_ORIGINAL,
    CONDITION_SHUFFLED,
    CONDITION_ABLATED,
    CONDITION_FRAME_MASKED,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PerturbationConfig:
    """All hyperparameters for the perturbation pipeline.

    Attributes
    ----------
    bilateral_d       : Diameter of each pixel neighbourhood for bilateral filter.
    bilateral_sigma_c : Filter sigma in colour space.
    bilateral_sigma_s : Filter sigma in coordinate space.
    gaussian_ksize    : Kernel size (odd integer) for the Gaussian blur.
    gaussian_sigma    : Standard deviation for the Gaussian blur.
    seed              : Random seed for reproducible temporal shuffling.
    fourcc            : FourCC codec for output video files.
    fps_override      : If > 0, override the source video FPS.
    max_frames        : Cap on frames loaded into memory (0 = no cap).
    conditions        : Which conditions to generate.
    frame_mask_rate   : Fraction of frames to mask with zeros (default 0.5).
    """
    bilateral_d:       int            = 9
    bilateral_sigma_c: float          = 75.0
    bilateral_sigma_s: float          = 75.0
    gaussian_ksize:    int            = 5
    gaussian_sigma:    float          = 1.2
    seed:              Optional[int]  = 42
    fourcc:            str            = "mp4v"
    fps_override:      float          = 0.0
    max_frames:        int            = 0
    conditions:        Tuple[str,...] = field(default=ALL_CONDITIONS)
    frame_mask_rate:   float          = 0.5

    def __post_init__(self):
        if self.gaussian_ksize % 2 == 0:
            raise ValueError("gaussian_ksize must be an odd integer.")
        unknown = set(self.conditions) - set(ALL_CONDITIONS)
        if unknown:
            raise ValueError(f"Unknown conditions: {unknown}")
        if not 0.0 <= self.frame_mask_rate < 1.0:
            raise ValueError("frame_mask_rate must be in [0, 1).")


# ---------------------------------------------------------------------------
# Frame-level transforms
# ---------------------------------------------------------------------------
class FrameProcessor:
    """Stateless, vectorised frame transforms."""

    @staticmethod
    def apply_ablation(
        frame: np.ndarray,
        bilateral_d: int,
        bilateral_sigma_c: float,
        bilateral_sigma_s: float,
        gaussian_ksize: int,
        gaussian_sigma: float,
    ) -> np.ndarray:
        """Texture ablation: grayscale, bilateral filtering, blur, then BGR.

        The bilateral filter preserves hard structural edges while smoothing
        homogeneous colour regions (skin, fabric, etc.).  The subsequent mild
        Gaussian blurs away residual high-frequency textures.

        Returns a 3-channel BGR uint8 array (same shape as input) so the
        downstream VideoWriter requires no branch logic.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Bilateral needs an 8-bit single-channel or 3-channel image.
        bilateral = cv2.bilateralFilter(
            gray,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_c,
            sigmaSpace=bilateral_sigma_s,
        )
        blurred = cv2.GaussianBlur(
            bilateral,
            ksize=(gaussian_ksize, gaussian_ksize),
            sigmaX=gaussian_sigma,
        )
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_shuffle(
        frames: List[np.ndarray],
        seed: Optional[int],
    ) -> List[np.ndarray]:
        """Randomly permute *all* frames along the temporal axis."""
        rng = random.Random(seed)
        indices = list(range(len(frames)))
        rng.shuffle(indices)
        return [frames[i] for i in indices]

    @staticmethod
    def apply_frame_mask(
        frames: List[np.ndarray],
        mask_rate: float,
        seed: Optional[int],
    ) -> List[np.ndarray]:
        """Replace ``mask_rate`` fraction of frames with black (zero) frames.

        The output list has exactly the same length as the input; temporal
        structure (duration, frame count, fps) is preserved entirely.  Only
        the pixel content of selected frames is zeroed out.

        This is the critical difference from frame-dropping: the video clock
        keeps ticking; the selected frames simply go black.
        """
        rng = random.Random((seed or 0) + 9999)  # different seed from shuffle
        n_mask = round(len(frames) * mask_rate)
        masked_indices = set(rng.sample(range(len(frames)), n_mask))
        result = []
        for i, f in enumerate(frames):
            if i in masked_indices:
                result.append(np.zeros_like(f))
            else:
                result.append(f)
        return result


# ---------------------------------------------------------------------------
# VideoWriter context manager
# ---------------------------------------------------------------------------
class VideoWriter:
    """Thin RAII wrapper around cv2.VideoWriter."""

    def __init__(self, path: Path, fps: float, width: int, height: int, fourcc: str = "mp4v"):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise IOError(f"cv2.VideoWriter could not open: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Core per-video logic (must be module-level for pickling)
# ---------------------------------------------------------------------------
def _process_single_video(
    src: Path,
    output_dir: Path,
    cfg: PerturbationConfig,
) -> dict:
    """Process one video file and write all requested condition outputs.

    Returns a result dict with keys: video_id, paths, success, error.
    """
    video_id = src.stem
    result = {"video_id": video_id, "paths": {}, "success": False, "error": None}

    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {src}")

        fps    = cfg.fps_override if cfg.fps_override > 0 else cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read all frames into memory.
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if cfg.max_frames > 0 and len(frames) >= cfg.max_frames:
                break
        cap.release()

        if not frames:
            raise ValueError("No frames could be decoded.")

        vid_dir = output_dir / video_id
        vid_dir.mkdir(parents=True, exist_ok=True)

        frame_count = len(frames)

        # Condition A: original.
        if CONDITION_ORIGINAL in cfg.conditions:
            out_path = vid_dir / f"{video_id}_{CONDITION_ORIGINAL}.mp4"
            with VideoWriter(out_path, fps, width, height, cfg.fourcc) as vw:
                for f in frames:
                    vw.write(f)
            result["paths"][CONDITION_ORIGINAL] = str(out_path)

        # Condition B: shuffled.
        if CONDITION_SHUFFLED in cfg.conditions:
            shuffled = FrameProcessor.apply_shuffle(frames, cfg.seed)
            out_path = vid_dir / f"{video_id}_{CONDITION_SHUFFLED}.mp4"
            with VideoWriter(out_path, fps, width, height, cfg.fourcc) as vw:
                for f in shuffled:
                    vw.write(f)
            result["paths"][CONDITION_SHUFFLED] = str(out_path)
            del shuffled  # free reordered-reference list immediately

        # Condition C: texture-ablated.
        if CONDITION_ABLATED in cfg.conditions:
            out_path = vid_dir / f"{video_id}_{CONDITION_ABLATED}.mp4"
            with VideoWriter(out_path, fps, width, height, cfg.fourcc) as vw:
                for f in frames:
                    ablated = FrameProcessor.apply_ablation(
                        f,
                        cfg.bilateral_d,
                        cfg.bilateral_sigma_c,
                        cfg.bilateral_sigma_s,
                        cfg.gaussian_ksize,
                        cfg.gaussian_sigma,
                    )
                    vw.write(ablated)
            result["paths"][CONDITION_ABLATED] = str(out_path)

        # Condition D: frame-masked.
        if CONDITION_FRAME_MASKED in cfg.conditions:
            masked = FrameProcessor.apply_frame_mask(frames, cfg.frame_mask_rate, cfg.seed)
            del frames  # frames no longer needed after masked is built
            out_path = vid_dir / f"{video_id}_{CONDITION_FRAME_MASKED}.mp4"
            with VideoWriter(out_path, fps, width, height, cfg.fourcc) as vw:
                for f in masked:
                    vw.write(f)
            result["paths"][CONDITION_FRAME_MASKED] = str(out_path)
            del masked  # free zero-frame arrays immediately
        else:
            del frames  # no frame_masked condition, free now

        result["success"] = True
        logger.info("OK %s (%d frames, %.1f fps)", video_id, frame_count, fps)

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        logger.error("FAILED %s - %s", src.name, exc)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class VideoPerturbator:
    """Orchestrate batch processing of a video directory.

    Parameters
    ----------
    input_dir    : Directory containing raw .mp4 files (searched recursively).
    output_dir   : Root output directory; per-video sub-directories are created.
    cfg          : Frozen :class:`PerturbationConfig`.
    num_workers  : Number of parallel worker processes (``None`` means CPU count).
    recurse      : Search ``input_dir`` recursively for .mp4 files.
    """

    def __init__(
        self,
        input_dir:   Path,
        output_dir:  Path,
        cfg:         PerturbationConfig  = PerturbationConfig(),
        num_workers: Optional[int]       = None,
        recurse:     bool                = True,
    ):
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.cfg         = cfg
        self.num_workers = num_workers
        self.recurse     = recurse
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _gather_videos(self) -> List[Path]:
        pattern = "**/*.mp4" if self.recurse else "*.mp4"
        videos  = sorted(self.input_dir.glob(pattern))
        if not videos:
            logger.warning("No .mp4 files found in %s", self.input_dir)
        return videos

    def run(self) -> List[dict]:
        """Process all videos and return a list of result dicts."""
        videos  = self._gather_videos()
        results = []

        logger.info("Found %d video(s). Using %s worker(s).", len(videos), self.num_workers or "all CPU")

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_process_single_video, v, self.output_dir, self.cfg): v
                for v in videos
            }
            for future in as_completed(futures):
                results.append(future.result())

        ok  = sum(r["success"] for r in results)
        err = len(results) - ok
        logger.info("Perturbation complete. Success: %d | Errors: %d", ok, err)
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grounding Video Reasoning video perturbation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",      type=Path, required=True, help="Directory of raw .mp4 videos.")
    p.add_argument("--output_dir",     type=Path, required=True, help="Root output directory.")
    p.add_argument("--num_workers",    type=int,  default=None,  help="Parallel worker processes.")
    p.add_argument("--seed",           type=int,  default=42,    help="Shuffle RNG seed.")
    p.add_argument("--max_frames",     type=int,  default=0,     help="Max frames per video (0=all).")
    p.add_argument("--fps_override",   type=float,default=0.0,   help="Override source FPS (0=auto).")
    p.add_argument("--no_recurse",     action="store_true",       help="Do not search sub-directories.")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=list(ALL_CONDITIONS),
        choices=ALL_CONDITIONS,
        help="Which conditions to generate.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PerturbationConfig(
        seed=args.seed,
        max_frames=args.max_frames,
        fps_override=args.fps_override,
        conditions=tuple(args.conditions),
    )
    perturbator = VideoPerturbator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        num_workers=args.num_workers,
        recurse=not args.no_recurse,
    )
    perturbator.run()
