"""The matting engine: Robust Video Matting running on ONNX Runtime.

RVM is a *recurrent* network. Each frame's hidden state feeds into the next one,
which is what keeps edges from shimmering between frames the way per-frame
segmentation models do. That also means frames must be fed in order, and that a
short warm-up run lets the state settle before output is kept.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Names come from the official RVM ONNX export.
_OUTPUTS = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"]
_STATE_IN = ["r1i", "r2i", "r3i", "r4i"]


# Speed presets, expressed as the scale the backbone runs at.
#
# These used to be absolute sizes -- a target of 256/320/512 pixels on the long
# edge -- which is the wrong quantity to hold fixed. What the refinement stage
# has to do is bridge the gap between the backbone and the full frame, so the
# thing that decides edge quality is the *ratio* between them, and a fixed
# target silently becomes a smaller and smaller ratio as the source grows. On
# 640x360 a 320px target is a ratio of 0.5 and perfectly fine; on 4K phone
# footage the same number is 0.083, and the refinement stage is left inventing
# twelve pixels of edge for every one it was given.
#
# Measured on that 4K footage, the foreground colour recovered in the edge band
# sat this far toward the background (0 = clean, 1 = entirely background):
#
#     ratio       0.167   0.25    0.375   0.5
#     as-is       0.902   0.674   0.609   0.489
#     decontam.   0.150   0.102   0.077   0.066
#
# So the ratio drives the halo, and decontaminating the edge afterwards takes
# most of the remaining sting out of choosing a cheap one. 0.375 is also RVM's
# own recommendation for HD.
SPEED_RATIOS = {
    "fast": 0.25,
    "balanced": 0.375,   # RVM's own recommendation for HD
    "best": 0.5,
    "max": None,         # no downsampling at all
}

# A ratio stops meaning much at the extremes: on a small frame it can shrink the
# backbone below the size the network can recognise a person at, and on a very
# large one it inflates the cost chasing detail the lens never captured. Keep
# the backbone's long edge inside this band. 320 is the low end because that is
# the smallest backbone measured to be indistinguishable from full resolution on
# 640x360 footage; below it there is no evidence either way.
BACKBONE_MIN, BACKBONE_MAX = 320, 1600


def auto_downsample_ratio(width: int, height: int, ratio: float = 0.375) -> float:
    """Clamp a preset ratio to one that keeps the backbone a sensible size.

    The backbone runs at this fraction of the frame while the refinement stage
    produces a full-resolution matte from it, guided by the original frame. The
    ratio is returned unchanged for anything between roughly VGA and 4K; it is
    only raised for very small frames and lowered for very large ones.
    """
    long_edge = max(width, height, 1)
    lowest = min(BACKBONE_MIN / long_edge, 1.0)
    highest = min(BACKBONE_MAX / long_edge, 1.0)
    # On a frame small enough that both bite, the floor wins: too little detail
    # is a worse failure than too much cost.
    return float(min(max(ratio, lowest), max(highest, lowest)))


def physical_cores() -> int:
    """Best-effort physical (not logical) core count.

    Hyper-threads share execution units, so counting them tends to hurt: ONNX
    convolutions end up fighting over the same cache and vector units.
    """
    try:
        import psutil  # optional
        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except Exception:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


class SceneCutDetector:
    """Spots a hard cut between shots, so the recurrent state can be dropped.

    The same memory that makes RVM stable within a shot works against it across
    one. At a cut the hidden state still describes the previous scene -- where
    the subject was, what the background looked like -- and the network spends
    the next second or so being confidently wrong about a frame that has nothing
    to do with it. Since `WARMUP_FRAMES` is 24, that is roughly a second of
    visibly bad matte after every cut, and it is invisible in testing on
    single-shot footage, which is what most test footage is.

    Detection runs on a small greyscale thumbnail. Comparing full frames would
    fire on camera shake and on a hand sweeping past the lens; at 64 pixels a
    side, motion within a shot averages away and only a wholesale change of
    content clears the threshold. It also makes the check free next to matting.
    """

    THUMB = 64

    def __init__(self, threshold: float = 0.25):
        self.threshold = float(threshold)
        self._previous: np.ndarray | None = None
        self.cuts = 0

    def _thumb(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2
        grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(grey, (self.THUMB, self.THUMB), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) * np.float32(1.0 / 255.0)

    def __call__(self, frame_rgb: np.ndarray) -> bool:
        """True when this frame begins a new shot."""
        if self.threshold <= 0:
            return False
        thumb = self._thumb(frame_rgb)
        previous, self._previous = self._previous, thumb
        # The first frame of a clip is a cut by definition, but the engine is
        # already in a reset state there, so reporting one would be noise.
        if previous is None:
            return False
        if float(np.abs(thumb - previous).mean()) < self.threshold:
            return False
        self.cuts += 1
        return True

    def reset(self) -> None:
        self._previous = None


class MattingEngine:
    """Wraps one ONNX Runtime session plus the recurrent state it carries."""

    def __init__(self, model_path: Path, *, threads: int | None = None,
                 providers: list[str] | None = None, downsample: float | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or physical_cores()
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Busy-waiting between frames burns cycles that the next frame needs.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

        available = set(ort.get_available_providers())
        if providers:
            chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        else:
            preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                         "OpenVINOExecutionProvider", "CPUExecutionProvider"]
            chosen = [p for p in preferred if p in available]

        self.session = ort.InferenceSession(str(model_path), opts, providers=chosen)
        self.provider = self.session.get_providers()[0]
        self.threads = opts.intra_op_num_threads
        self._fixed_downsample = downsample
        self.reset()

    def reset(self) -> None:
        """Clear temporal memory. Call between unrelated clips."""
        zero = np.zeros([1, 1, 1, 1], dtype=np.float32)
        self._state = [zero, zero, zero, zero]

    def downsample_for(self, width: int, height: int) -> float:
        if self._fixed_downsample is not None:
            return self._fixed_downsample
        return auto_downsample_ratio(width, height)

    def __call__(self, frame_rgb: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Matte one frame.

        Takes HxWx3 uint8 RGB. Returns (foreground HxWx3 float32 0-1,
        alpha HxW float32 0-1). The foreground is colour-decontaminated by the
        network, so edge pixels do not carry a tint from whatever was behind them.
        """
        src = frame_rgb.astype(np.float32, copy=False)
        src = src.transpose(2, 0, 1)[None] * np.float32(1.0 / 255.0)
        src = np.ascontiguousarray(src)

        feed = {"src": src, "downsample_ratio": np.array([ratio], dtype=np.float32)}
        feed.update(dict(zip(_STATE_IN, self._state)))

        fgr, pha, *state = self.session.run(_OUTPUTS, feed)
        self._state = state

        fgr = fgr[0].transpose(1, 2, 0)   # -> HxWx3
        pha = pha[0, 0]                   # -> HxW
        return fgr, pha
