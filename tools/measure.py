"""Measure matte quality on a clip, reproducibly.

Every quality number quoted in this project -- the 0.902 -> 0.150 contamination
table in `matting.py`, the 0.077-at-full-res-vs-0.082-at-720 trade in
`decontaminate` -- was originally measured by hand and could not be re-checked
after a change. This script is that check.

It drives `MattingEngine` directly rather than going through `pipeline.run`, on
purpose: the shipped path encodes to video, which quantises alpha to 8 bits and
discards the float foreground entirely. Both are exactly what the contamination
metric needs, so the measurement has to tap the pipeline before the encoder.
The matting itself is the same code the real run uses.

Usage:
    python tools/measure.py Practice.mp4
    python tools/measure.py IMG_6377.MOV --start 50 --variant decontam --variant temporal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bgremover import compositing, models          # noqa: E402
from bgremover.matting import MattingEngine        # noqa: E402
from bgremover.video import FrameReader, capped_size, probe   # noqa: E402

# The band that counts as "edge" for every metric here. Deliberately wide: a
# narrow band would miss the soft halo that is the whole point of measuring.
EDGE_LOW, EDGE_HIGH = 0.05, 0.95
SOLID = 0.99      # alpha above this is trusted as pure subject
EMPTY = 0.02      # alpha below this is trusted as pure background


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def edge_contamination(foreground: np.ndarray, alpha: np.ndarray,
                       source: np.ndarray) -> tuple[float, int]:
    """How far the edge band's colour has drifted toward the background.

    0 means the edge carries the subject's own colour; 1 means it carries
    entirely the background's. This is the quantity the halo *is*: a bright wall
    behind a dark subject shows up as a bright ring precisely because the
    reconstructed foreground in the band sits near the wall's colour.

    Both reference colours are local, not global. A subject is not one colour --
    hair, skin and shirt differ -- so comparing against a frame-wide mean would
    report drift that is really just the subject's own variation. Instead the
    trusted interior colour is extrapolated outward into the band, and so is the
    trusted background colour, using the same push-pull fill `decontaminate`
    relies on. Each edge pixel is then scored against the two colours that
    actually meet at it.

    Returns (mean contamination, pixels scored).
    """
    band = (alpha > EDGE_LOW) & (alpha < EDGE_HIGH)
    if not band.any():
        return float("nan"), 0

    # Local subject colour: grown outward from solidly opaque pixels.
    w_fg = (alpha > SOLID).astype(np.float32)
    # Local background colour: grown inward from solidly transparent ones. The
    # source frame is used here, not the foreground -- outside the subject the
    # network's foreground output is meaningless.
    w_bg = (alpha < EMPTY).astype(np.float32)
    if w_fg.sum() < 64 or w_bg.sum() < 64:
        return float("nan"), 0    # nothing trustworthy to compare against

    f_local = compositing._push_pull_fill(foreground * w_fg[..., None], w_fg)
    b_local = compositing._push_pull_fill(source * w_bg[..., None], w_bg)

    axis = b_local - f_local
    denom = np.einsum("...c,...c->...", axis, axis)

    # Where subject and background are nearly the same colour there is no axis
    # to project onto and contamination is undefined, not zero. Those pixels are
    # dropped rather than scored as clean, which would flatter the result.
    usable = band & (denom > 1e-3)
    if not usable.any():
        return float("nan"), 0

    delta = foreground - f_local
    projection = np.einsum("...c,...c->...", delta, axis)[usable] / denom[usable]
    return float(np.clip(projection, 0.0, 1.0).mean()), int(usable.sum())


def edge_width(alpha: np.ndarray) -> float:
    """Mean thickness of the alpha transition, in pixels.

    Measured as the area of the soft band divided by the length of the
    silhouette, which is what "how many pixels does it take to go from opaque to
    transparent" means when the edge is an arbitrary shape.

    Read this as a two-sided number. Guided refinement should pull it down,
    because RVM's upsampled matte is softer than the image edge underneath it --
    but a *large* drop means real soft detail (flyaway hair, motion blur) has
    been hardened into a cutout, which is a worse artifact than the one being
    fixed.
    """
    solid = (alpha > 0.5).astype(np.uint8)
    eroded = cv2.erode(solid, np.ones((3, 3), np.uint8))
    perimeter = float((solid - eroded).sum())
    if perimeter < 1:
        return float("nan")
    band = float(((alpha > EDGE_LOW) & (alpha < EDGE_HIGH)).sum())
    return band / perimeter


def flicker(alpha: np.ndarray, previous: np.ndarray) -> tuple[float, float]:
    """Frame-to-frame alpha change, split by where it happens.

    Returns (edge, interior). Movement legitimately changes the edge, so the
    edge figure is only comparable between runs over the same clip. The interior
    figure has no such excuse: a pixel deep inside the subject is opaque in both
    frames and any change there is pure noise, so it should sit near zero and
    any rise is a real regression.
    """
    if previous is None or previous.shape != alpha.shape:
        return float("nan"), float("nan")
    delta = np.abs(alpha - previous)

    band = ((alpha > EDGE_LOW) & (alpha < EDGE_HIGH)) | \
           ((previous > EDGE_LOW) & (previous < EDGE_HIGH))
    interior = (alpha > SOLID) & (previous > SOLID)

    edge_val = float(delta[band].mean()) if band.any() else float("nan")
    inner_val = float(delta[interior].mean()) if interior.any() else float("nan")
    return edge_val, inner_val


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

def lag(smoothed: np.ndarray, unsmoothed: np.ndarray) -> float:
    """How far temporal smoothing moved the matte away from this frame's truth.

    Flicker alone cannot judge a smoother: the way to drive frame-to-frame
    change to zero is to stop updating, and an averaged matte that trails a
    moving arm scores beautifully on stability while looking obviously wrong.
    This is the opposing metric. The network's own output for *this* frame is
    the reference, and any deviation from it is smoothing that has overreached.

    A good smoother lowers flicker while keeping this near zero; one that only
    trades the first for the second has achieved nothing.
    """
    band = (unsmoothed > EDGE_LOW) & (unsmoothed < EDGE_HIGH)
    if not band.any():
        return float("nan")
    return float(np.abs(smoothed - unsmoothed)[band].mean())


# Each variant is a set of post-processing switches applied to the same matte,
# so a run can compare them without re-decoding or re-matting from scratch.
#
# `knee` is what separates the two smoothing variants: a knee far above any
# achievable alpha change means the per-pixel scaling never engages, which is
# precisely the flat exponential average the smoother used to be. That makes
# "temporal-flat" an honest before to "temporal"'s after.
VARIANTS = {
    "raw":            dict(decontaminate=False, temporal=0.0, knee=0.25),
    "decontam":       dict(decontaminate=True,  temporal=0.0, knee=0.25),
    "temporal-flat":  dict(decontaminate=True,  temporal=0.5, knee=1e9),
    "temporal":       dict(decontaminate=True,  temporal=0.5, knee=0.25),
}


def measure(path: Path, variant: str, opts: argparse.Namespace) -> dict:
    """Run one variant over the clip and return its metrics."""
    config = VARIANTS[variant]
    info = probe(path)
    width, height = capped_size(info.width, info.height, opts.resolution)
    scale_to = (width, height) if (width, height) != (info.width, info.height) else None

    engine = MattingEngine(models.resolve(opts.model), threads=opts.threads)
    ratio = engine.downsample_for(width, height)
    smoother = compositing.TemporalSmoother(config["temporal"], knee=config["knee"])

    reader = FrameReader(path, info.width, info.height,
                         start=opts.start or None,
                         duration=(opts.frames + opts.warmup + 1) / float(info.fps),
                         scale_to=scale_to)

    contamination: list[float] = []
    widths: list[float] = []
    edge_flick: list[float] = []
    inner_flick: list[float] = []
    lags: list[float] = []
    previous: np.ndarray | None = None
    kept = 0
    elapsed = 0.0

    try:
        for i in range(opts.warmup + opts.frames):
            raw = reader.read()
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

            start = time.perf_counter()
            foreground, alpha = engine(frame, ratio)
            unsmoothed = alpha
            alpha = smoother(alpha)
            if config["decontaminate"]:
                foreground = compositing.decontaminate(foreground, alpha)
            elapsed += time.perf_counter() - start

            # The recurrent state is still settling; measuring here would report
            # the warm-up, not the model.
            if i < opts.warmup:
                previous = alpha
                continue

            source = frame.astype(np.float32) / 255.0
            value, scored = edge_contamination(foreground, alpha, source)
            if scored:
                contamination.append(value)
            widths.append(edge_width(alpha))
            lags.append(lag(alpha, unsmoothed))
            e, n = flicker(alpha, previous)
            if e == e:      # not NaN
                edge_flick.append(e)
                inner_flick.append(n)

            previous = alpha
            kept += 1
    finally:
        reader.close()

    def mean(values: list[float]) -> float:
        clean = [v for v in values if v == v]
        return float(np.mean(clean)) if clean else float("nan")

    return {
        "variant": variant,
        "frames": kept,
        "contamination": mean(contamination),
        "edge_width": mean(widths),
        "flicker_edge": mean(edge_flick),
        "flicker_inner": mean(inner_flick),
        "lag": mean(lags),
        "fps": kept / elapsed if elapsed else float("nan"),
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure matte quality on a clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, nargs="+", help="Clip(s) to measure.")
    parser.add_argument("--variant", action="append", choices=list(VARIANTS),
                        help="Which configurations to measure (repeatable).")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Seconds into the clip to measure from. Worth setting: "
                             "a clip that opens on an empty room gives a matte with "
                             "no solid pixels, and every metric here comes back NaN.")
    parser.add_argument("--frames", type=int, default=120,
                        help="Frames to measure after warm-up.")
    parser.add_argument("--warmup", type=int, default=24,
                        help="Frames to discard while the recurrent state settles.")
    parser.add_argument("--resolution", type=int, default=1080,
                        help="Cap the short edge, matching the CLI's own default.")
    parser.add_argument("--model", default="mobilenetv3")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    variants = args.variant or list(VARIANTS)

    header = (f"{'variant':<14} {'frames':>6} {'contam':>8} {'edge_px':>8} "
              f"{'flick_e':>8} {'flick_i':>8} {'lag':>8} {'fps':>7}")

    for path in args.input:
        if not path.exists():
            print(f"skipping {path}: not found", file=sys.stderr)
            continue
        print(f"\n{path.name}  ({args.frames} frames @ short edge {args.resolution})")
        print(header)
        print("-" * len(header))
        for variant in variants:
            row = measure(path, variant, args)
            print(f"{row['variant']:<14} {row['frames']:>6} "
                  f"{row['contamination']:>8.3f} {row['edge_width']:>8.2f} "
                  f"{row['flicker_edge']:>8.4f} {row['flicker_inner']:>8.4f} "
                  f"{row['lag']:>8.4f} {row['fps']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
