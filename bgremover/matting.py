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


# Speed presets, expressed as the longest edge the backbone should see. Scaling
# by a target size rather than a fixed multiplier keeps the presets meaningful
# across source resolutions.
#
# 512 is RVM's own recommendation. Measured on 640x360 talking-head footage,
# dropping the backbone to a 320px long edge changed the matte by a mean of
# 0.0016 alpha (99th percentile 0.054) against a full-resolution reference, for
# roughly 6x the throughput -- so 'balanced' sits there rather than at 512.
SPEED_TARGETS = {
    "fast": 256,
    "balanced": 320,
    "best": 512,
    "max": None,     # no downsampling at all
}


def auto_downsample_ratio(width: int, height: int, target_long_edge: int = 320) -> float:
    """Shrink so the backbone's longest side lands near `target_long_edge`.

    The backbone runs at this reduced size while the refinement stage still
    produces a full-resolution matte guided by the original frame -- which is
    why the edge stays sharp even when the backbone sees very little.
    """
    return max(0.1, min(target_long_edge / max(width, height), 1.0))


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
