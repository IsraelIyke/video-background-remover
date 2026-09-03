"""Matte refinement and background compositing."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .video import FrameReader

# Common colours plus the two chroma shades broadcast tools expect.
NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
    # Studio chroma values -- these key far more cleanly than pure 0/255 green.
    "greenscreen": (0, 177, 64),
    "bluescreen": (0, 71, 187),
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_color(text: str) -> tuple[int, int, int]:
    """Accept '#rrggbb', '#rgb', 'r,g,b' or a name from NAMED_COLORS."""
    key = text.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]

    hex_match = re.fullmatch(r"#?([0-9a-f]{6})", key)
    if hex_match:
        value = hex_match.group(1)
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    short = re.fullmatch(r"#?([0-9a-f]{3})", key)
    if short:
        value = short.group(1)
        return tuple(int(c * 2, 16) for c in value)  # type: ignore[return-value]

    triple = re.fullmatch(r"(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})", key)
    if triple:
        rgb = tuple(min(255, int(g)) for g in triple.groups())
        return rgb  # type: ignore[return-value]

    raise ValueError(
        f"Could not read {text!r} as a colour. Use a hex value like '#101820', "
        f"'r,g,b', or one of: {', '.join(sorted(NAMED_COLORS))}"
    )


def fit_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale and centre-crop so the image fills WxH without distortion."""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    new_w, new_h = int(np.ceil(src_w * scale)), int(np.ceil(src_h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    x0 = (new_w - width) // 2
    y0 = (new_h - height) // 2
    return resized[y0:y0 + height, x0:x0 + width]


# --------------------------------------------------------------------------- #
# Matte refinement
# --------------------------------------------------------------------------- #

def keep_largest_regions(alpha: np.ndarray, count: int = 1,
                         threshold: float = 0.05) -> np.ndarray:
    """Zero every part of the matte except the `count` biggest connected blobs.

    RVM segments *people* -- all of them. A picture-in-picture inset, a poster on
    the wall, or a thumbnail burned into the footage will each be kept, correctly
    by the model's logic but rarely by the user's. Keeping only the largest
    region isolates the presenter, and since stray speckle is by definition small
    and disconnected, it vanishes in the same pass.

    The mask is built at a deliberately low threshold: at 0.5 a subject's
    semi-transparent halo would fall outside its own component and be cut loose,
    hardening exactly the soft edge a keyer needs.
    """
    mask = (alpha > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= count + 1:      # label 0 is the background; nothing to drop
        return alpha

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.argsort(areas)[::-1][:count] + 1     # +1 to skip the background label
    lookup = np.zeros(n_labels, dtype=np.float32)  # a LUT beats np.isin here
    lookup[keep] = 1.0
    return alpha * lookup[labels]


def refine_alpha(alpha: np.ndarray, *, choke: float = 0.0, feather: float = 0.0,
                 gamma: float = 1.0, low: float = 0.0, high: float = 1.0,
                 denoise: bool = False, main_subject: int = 0) -> np.ndarray:
    """Post-process the raw matte.

    choke   pixels to shrink (negative) or grow (positive) the silhouette
    feather gaussian softening applied to the edge, in pixels
    gamma   <1 makes semi-transparent areas more opaque, >1 more transparent
    low/high    levels remap; pulls near-0 to 0 and near-1 to 1 to kill haze
    denoise small median filter to remove speckle
    main_subject    keep only this many of the largest regions (0 = keep all)
    """
    out = alpha

    if denoise:
        out = cv2.medianBlur((out * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

    if low > 0.0 or high < 1.0:
        span = max(high - low, 1e-6)
        out = np.clip((out - low) / span, 0.0, 1.0)

    # Run before choke/feather so the blob analysis sees a clean matte and any
    # later softening applies only to the subjects that survived.
    if main_subject > 0:
        out = keep_largest_regions(out, main_subject)

    if choke:
        size = max(1, int(round(abs(choke))) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        out = cv2.erode(out, kernel) if choke < 0 else cv2.dilate(out, kernel)

    if feather > 0:
        sigma = float(feather)
        radius = int(sigma * 3) | 1
        out = cv2.GaussianBlur(out, (radius, radius), sigma)

    if gamma != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), gamma, dtype=np.float32)

    if out is alpha:
        return out  # untouched; already in range straight from the network
    return np.clip(out, 0.0, 1.0, out=out)


# Nothing here consults the source frame when deciding alpha, which looks like
# an omission: the obvious next move is a guided filter over the matte with the
# frame as guide, to snap a soft silhouette onto the hard edge beneath it. It
# was implemented and measured, and it is worse -- monotonically so, on both
# test clips, at every radius and epsilon tried:
#
#     Practice.mp4          contamination   edge width (px)
#     as-is                     0.019            2.10
#     guided r=1 eps=1e-6       0.042            2.29
#     guided r=2 eps=1e-6       0.054            2.75
#     guided r=4 eps=1e-6       0.075            3.97
#
# The premise was simply wrong. RVM's `downsample_ratio` input drives a Deep
# Guided Filter inside the network: the upsample from backbone resolution to
# full frame *is* a guided filter already, with coefficients learned end to end
# rather than fitted over a box window. Adding a second one on top does not
# sharpen an unrefined matte, it blurs a refined one. If the edge needs work,
# raise the ratio (see SPEED_RATIOS in matting.py) so the learned filter has
# more to work with -- do not post-process it here.


# --------------------------------------------------------------------------- #
# Foreground colour decontamination
# --------------------------------------------------------------------------- #

def _push_pull_fill(weighted: np.ndarray, weights: np.ndarray,
                    levels: int = 6, gain: float = 2.0) -> np.ndarray:
    """Extrapolate outward into the region where `weights` is ~0.

    Takes the values *already multiplied* by their weights, alongside the
    weights, because that is the pair that survives being resampled: scaling
    `values` and `weights` separately and multiplying afterwards computes
    mean(v)*mean(w) where the weighted average needs mean(v*w).

    A pyramid "push-pull": weighted averages are pushed down to coarse levels
    where every hole is covered, then pulled back up, each level preferring its
    own data where it has enough weight and falling back on the coarser fill
    where it does not. The result is a smooth extension of the trusted colours
    across the untrusted region, at a cost that is linear in pixels.

    `gain` sets how little weight a level needs before it is believed over the
    coarser fill. Raising it sounds right -- prefer the closest colour -- but
    measured on this project's footage it is mildly worse (contamination 0.088
    at gain 32 against 0.077 at gain 2), because the nearest pixels are the ones
    just inside the edge, which are themselves the most contaminated. Leaning on
    the coarser, deeper-interior average is the better trade.
    """
    vw = [weighted]
    ws = [weights]
    for _ in range(levels):
        if min(ws[-1].shape[:2]) < 8:
            break
        vw.append(cv2.pyrDown(vw[-1]))
        ws.append(cv2.pyrDown(ws[-1]))

    filled = vw[-1] / np.maximum(ws[-1], 1e-5)[..., None]
    for level in range(len(ws) - 2, -1, -1):
        height, width = ws[level].shape
        coarse = cv2.pyrUp(filled, dstsize=(width, height))
        direct = vw[level] / np.maximum(ws[level], 1e-5)[..., None]
        trust = np.clip(ws[level] * gain, 0.0, 1.0)[..., None]
        filled = direct * trust + coarse * (1.0 - trust)
    return filled


def decontaminate(foreground: np.ndarray, alpha: np.ndarray, *,
                  low: float = 0.90, high: float = 0.995,
                  levels: int = 6, gain: float = 2.0,
                  fill_long_edge: int = 720) -> np.ndarray:
    """Strip background colour out of the soft edge, which is what a halo *is*.

    RVM's refinement stage reconstructs the full-resolution foreground as a
    local linear function of the source frame. Inside the subject that is
    exactly right. Across the semi-transparent edge it is not: the only colours
    available locally are a mix of subject and background, so the reconstructed
    foreground drifts toward whatever was behind the person. Measured on this
    project's 4K footage the edge band's colour sat 89% of the way from the
    subject's own colour to the background's -- a dark subject against a bright
    wall therefore comes out ringed in bright wall, and the ring changes colour
    as they move across the wall, which is what makes it so obvious.

    The fix is the one compositors have always used: throw the edge colours away
    and re-grow them from the pixels we trust. Only solidly opaque pixels are
    kept, and their colour is extrapolated outward to cover everything else.
    Alpha is untouched -- the silhouette, hair and softness all stay exactly as
    the network produced them; only the colour underneath changes. On that same
    footage this takes the edge band from 89% background-coloured to 8%.

    Solving the matting equation instead -- estimating the background too and
    recovering F from I = aF + (1-a)B -- was tried and is worse here (20%): it
    needs alpha to be accurate enough to divide by, and RVM's is not, so it
    trades a colour error for an amplified-noise one.

    low/high        alpha range over which trust ramps from none to full
    levels          pyramid depth; deeper reaches further for a colour to borrow
    fill_long_edge  resolution the extrapolation itself is computed at
    """
    foreground = np.ascontiguousarray(foreground, dtype=np.float32)
    span = max(high - low, 1e-6)
    trust = np.clip((alpha - low) / span, 0.0, 1.0)
    trust = trust * trust * (3.0 - 2.0 * trust)      # smoothstep; no hard seam

    # Nothing solid to borrow from (an empty frame, or a subject that is all
    # soft edge) -- extrapolating from noise would be worse than leaving it.
    if float(trust.max()) < 0.05:
        return foreground

    # The extrapolation is a smooth field by construction, so computing it at
    # full resolution is most of the cost for very little of the benefit. On a
    # 1080x1920 frame: 1430ms for a contamination of 0.077 at full resolution,
    # against 308ms for 0.082 built at a 720px long edge -- roughly the size the
    # backbone itself runs at, which is all the detail the matte can justify.
    weight = trust[..., None]
    weighted = foreground * weight

    height, width = alpha.shape
    long_edge = max(height, width)
    if fill_long_edge and long_edge > fill_long_edge:
        scale = fill_long_edge / long_edge
        small = (max(8, int(width * scale)), max(8, int(height * scale)))
        filled = _push_pull_fill(
            cv2.resize(weighted, small, interpolation=cv2.INTER_AREA),
            cv2.resize(trust, small, interpolation=cv2.INTER_AREA),
            levels, gain)
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        filled = _push_pull_fill(weighted, trust, levels, gain)

    out = weighted
    out += filled * (1.0 - weight)
    return np.clip(out, 0.0, 1.0, out=out)


class TemporalSmoother:
    """Optional smoothing of the matte across frames, applied per pixel.

    RVM is already temporally stable, so this stays off by default; it earns its
    keep on noisy or low-light footage where the edge still crawls a little.

    The blend is motion-adaptive rather than a flat exponential average. A
    uniform EMA cannot tell edge crawl from the subject actually moving, so it
    smooths both: the flicker goes away and the silhouette drags a frame or two
    behind an arm that swings. That lag is far more objectionable than the
    shimmer it was meant to cure, which is why a flat version has to be kept at
    a strength too low to accomplish much.

    Weighting the blend by how much each pixel changed separates the two cases.
    A pixel whose alpha barely moved is either interior or a static edge, and
    smoothing it costs nothing; a pixel that swung hard is on a moving boundary,
    where the new value is signal and the old one is stale. So the strength is
    scaled down toward zero exactly where motion is, leaving still regions fully
    smoothed and moving ones untouched.
    """

    # How large an alpha change counts as motion rather than noise. Frame-to-
    # frame flicker on the test clips sits at 0.08-0.20 in the edge band and
    # essentially 0 in the interior, so the knee belongs above the noise floor
    # and below a real edge sweep.
    KNEE = 0.25

    def __init__(self, strength: float = 0.0, knee: float = KNEE):
        self.strength = float(np.clip(strength, 0.0, 0.95))
        self.knee = max(float(knee), 1e-3)
        self._previous: np.ndarray | None = None

    def __call__(self, alpha: np.ndarray) -> np.ndarray:
        if self.strength <= 0:
            return alpha
        if self._previous is None or self._previous.shape != alpha.shape:
            self._previous = alpha.copy()
            return alpha

        delta = np.abs(alpha - self._previous)
        local = np.clip(delta * (1.0 / self.knee), 0.0, 1.0)
        local *= -self.strength
        local += self.strength          # strength * (1 - clip(delta/knee))

        blended = self._previous * local + alpha * (1.0 - local)
        self._previous = blended
        return blended

    def reset(self) -> None:
        self._previous = None


# --------------------------------------------------------------------------- #
# Background sources
# --------------------------------------------------------------------------- #

class Background:
    """Supplies a background plate for each frame."""

    is_transparent = False

    def frame(self, index: int, source_rgb: np.ndarray,
              alpha: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TransparentBackground(Background):
    is_transparent = True

    def frame(self, index, source_rgb, alpha):
        return None


class ColorBackground(Background):
    def __init__(self, rgb: tuple[int, int, int], width: int, height: int):
        self.rgb = rgb
        self._plate = np.empty((height, width, 3), dtype=np.float32)
        self._plate[:] = np.asarray(rgb, dtype=np.float32) / 255.0

    def frame(self, index, source_rgb, alpha):
        return self._plate


class ImageBackground(Background):
    def __init__(self, path: Path, width: int, height: int):
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Could not open background image: {path}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self._plate = fit_cover(rgb, width, height).astype(np.float32) / 255.0

    def frame(self, index, source_rgb, alpha):
        return self._plate


class VideoBackground(Background):
    """Streams a background clip, looping it for as long as the input runs."""

    def __init__(self, path: Path, width: int, height: int):
        from .video import probe
        info = probe(path)
        self.width, self.height = width, height
        self._reader = FrameReader(
            path, info.width, info.height, loop=True, scale_to=(width, height),
        )
        self._last = np.zeros((height, width, 3), dtype=np.float32)

    def frame(self, index, source_rgb, alpha):
        raw = self._reader.read()
        if raw is None:
            return self._last  # clip ended unexpectedly; hold the final frame
        plate = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._last = plate.astype(np.float32) / 255.0
        return self._last

    def close(self):
        self._reader.close()


class BlurBackground(Background):
    """Blurs the original background without smearing the subject into it.

    A naive blur of the whole frame drags the person's colours outward and
    leaves a visible halo around them. Instead the subject is masked out and the
    remaining pixels are blurred with a normalised convolution, so surrounding
    background flows in to fill the hole before the blur is applied.
    """

    def __init__(self, strength: float, width: int, height: int):
        # Scale the kernel with frame size so the look holds across resolutions.
        base = max(width, height) / 640.0
        self.sigma = max(1.0, strength * base)
        self.radius = int(self.sigma * 3) | 1

    def frame(self, index, source_rgb, alpha):
        source = source_rgb.astype(np.float32) / 255.0
        weight = (1.0 - alpha).astype(np.float32)

        masked = source * weight[..., None]
        blurred = cv2.GaussianBlur(masked, (self.radius, self.radius), self.sigma)
        norm = cv2.GaussianBlur(weight, (self.radius, self.radius), self.sigma)
        filled = blurred / np.maximum(norm, 1e-3)[..., None]

        # Where the subject fills most of the frame there is barely any real
        # background within blur range, and the normalised result is amplified
        # noise. Fade back to a plain blur wherever coverage is that thin.
        plain = cv2.GaussianBlur(source, (self.radius, self.radius), self.sigma)
        trust = np.clip(norm / 0.15, 0.0, 1.0)[..., None]
        merged = filled * trust + plain * (1.0 - trust)

        # A second, lighter pass evens out the filled region.
        return cv2.GaussianBlur(merged, (self.radius, self.radius), self.sigma * 0.5)


def build_background(spec: str, width: int, height: int) -> Background:
    """Turn a --background value into a Background instance."""
    text = spec.strip()
    lowered = text.lower()

    if lowered in ("none", "transparent", "alpha"):
        return TransparentBackground()

    if lowered.startswith("blur"):
        strength = 18.0
        if ":" in lowered:
            try:
                strength = float(lowered.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("Blur strength must be a number, e.g. --background blur:25") from exc
        return BlurBackground(strength, width, height)

    path = Path(text).expanduser()
    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            return VideoBackground(path, width, height)
        if suffix in IMAGE_SUFFIXES:
            return ImageBackground(path, width, height)
        raise ValueError(f"Unsupported background file type: {suffix}")

    return ColorBackground(parse_color(text), width, height)


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #

def composite(foreground: np.ndarray, alpha: np.ndarray,
              background: np.ndarray) -> np.ndarray:
    """Standard matting equation, returned as uint8 RGB.

    Written as bg + (fg-bg)*a rather than fg*a + bg*(1-a): same result, one
    fewer full-frame multiply. Compositing is a few percent of a frame's cost at
    high matting resolutions but climbs to a noticeable share at low ones.
    """
    a = alpha[..., None]
    out = background + (foreground - background) * a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def to_rgba(foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight (non-premultiplied) RGBA, which is what PNG/ProRes/VP9 expect."""
    height, width = alpha.shape
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return out


def to_matte(alpha: np.ndarray) -> np.ndarray:
    """Greyscale matte as RGB, for luma-key / track-matte workflows."""
    grey = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)
