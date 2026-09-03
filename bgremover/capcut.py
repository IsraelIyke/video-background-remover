"""Reuse CapCut's own background-removal result instead of computing one.

CapCut caches the output of its auto-cutout on disk, per draft, as a sequence of
raw alpha masks. If a clip has already been cut out in CapCut, the expensive part
is done and the masks can be applied to the video here — no model, no inference,
no GPU.

What that actually saves, stated carefully: the *matting step* goes from ~1.7
s/frame (RVM mobilenetv3 at 1080x1920 on this CPU) to essentially free — a
114 KB file read and one resize. It does **not** make the whole run 80x faster,
because matting stops being the bottleneck and everything else takes over:
decoding 4K HEVC, colour decontamination, and encoding. Measured end to end on
this 2-core i5-5200U against a 4K source, the pipeline ran 4.5-6 s/frame — but
that was with CapCut and Firefox open and the CPU already at 65%, which
USAGE.md documents as costing this project roughly 7x. Treat those figures as a
floor, not a benchmark, and measure on an idle machine before quoting them.

Format, decoded from a CapCut 7.1.0 draft on Windows:

    <draft>/matting/<media-hash>/<level>/
        mask/<microsecond-timestamp>      raw, headerless, 8-bit single channel
        maskinfo/<same-timestamp>         JSON; boundingBox states width/height
        matting_result.json               result_time_range covered, in µs
        mocf                              small binary, purpose unidentified

Each mask on the measured draft is 256x448 = 114,688 bytes, matching exactly what
its maskinfo reports, and filenames are source-media timestamps spaced
41,666.67 µs apart — 24 fps, the source's own rate. Masks are therefore looked up
by source time rather than by index, which keeps this correct when the timeline
fps and the media fps disagree (the measured draft was a 24 fps source on a
30 fps timeline).

Two things about this data are worth knowing before trusting it:

* Geometry is a **direct stretch** to the full frame, not an aspect-fit with
  padding, even though 256/448 = 0.5714 does not match the 9:16 source's 0.5625.
  Verified by scoring mean frame-gradient magnitude along the alpha boundary for
  both hypotheses: 0.356 for a stretch against 0.293 for 2 px of side padding.

* The masks are a **quality ceiling**. 256x448 to 1080x1920 is a 4.22x upscale,
  so fine detail is simply absent — on the measured footage (short twists against
  a light wall) every individual strand is a smooth blob. Measured soft-edge
  area: CapCut bicubic 1.115%, CapCut guided 0.484%, RVM at full resolution
  1.514%. If edge detail matters more than time, use bgremove.py instead.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Re-exported so bgcapcut.py can call capcut.encoder_keeps_alpha; the single
# implementation lives in video.py. See the note at the end of this module.
from .video import encoder_keeps_alpha  # noqa: F401

# Where CapCut keeps its drafts on Windows. Overridable, because a portable
# install or a non-default user data directory moves it.
DEFAULT_DRAFT_ROOT = (Path(os.environ.get("LOCALAPPDATA", "")) / "CapCut"
                      / "User Data" / "Projects" / "com.lveditor.draft")

# CapCut writes the matting directory as a token plus a relative path, so a draft
# folder stays valid if it is moved. The source media path, by contrast, is
# absolute — which is why a copied draft folder is not self-contained.
_PLACEHOLDER = re.compile(r"^##_draftpath_placeholder_[0-9A-Fa-f-]+_##[/\\]?")


class DraftError(RuntimeError):
    pass


@dataclass
class MattingParams:
    """The clip's matting settings, as CapCut recorded them."""
    feather: float = 0.0
    expansion: float = 0.0
    stroke: bool = False
    reverse: bool = False
    flag: int = 0

    @property
    def is_plain(self) -> bool:
        """True when the alpha needs no edge post-processing to match CapCut."""
        return not self.feather and not self.expansion and not self.stroke


@dataclass
class Clip:
    """One video segment of a draft that carries a matting result."""
    index: int
    source: Path
    mask_dir: Path
    source_start_us: int
    source_duration_us: int
    speed: float
    width: int                     # source media dimensions, as displayed
    height: int
    matting: MattingParams
    material_name: str = ""

    @property
    def duration_s(self) -> float:
        return self.source_duration_us / 1e6

    @property
    def start_s(self) -> float:
        return self.source_start_us / 1e6


@dataclass
class Draft:
    name: str
    path: Path
    canvas_width: int
    canvas_height: int
    timeline_fps: float
    clips: list[Clip] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Discovery and parsing
# --------------------------------------------------------------------------- #

def find_drafts(root: Path | None = None) -> list[Path]:
    """Every draft folder that has a timeline, newest first."""
    root = Path(root) if root else DEFAULT_DRAFT_ROOT
    if not root.is_dir():
        return []
    drafts = [p for p in root.iterdir()
              if p.is_dir() and (p / "draft_content.json").is_file()]
    return sorted(drafts, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_draft(target: str, root: Path | None = None) -> Path:
    """Accept a draft name or a path to one."""
    candidate = Path(target).expanduser()
    if (candidate / "draft_content.json").is_file():
        return candidate
    root = Path(root) if root else DEFAULT_DRAFT_ROOT
    named = root / target
    if (named / "draft_content.json").is_file():
        return named
    available = [p.name for p in find_drafts(root)]
    raise DraftError(
        f"no CapCut draft called {target!r}.\n"
        + (f"  Available: {', '.join(available[:12])}" if available
           else f"  Looked in: {root}")
    )


@dataclass
class MaskCache:
    """A mask sequence found on disk, whether or not a timeline still cites it."""
    draft: str
    draft_dir: Path
    media_hash: str
    mask_dir: Path
    n_masks: int
    linked: bool = False           # still referenced by draft_content.json
    media: dict | None = None      # from CapCut's import cache, if available


# CapCut's own media-info cache, keyed by the same hash as the matting directory.
_MEDIAINFO = (Path(os.environ.get("LOCALAPPDATA", "")) / "CapCut" / "User Data"
              / "Cache" / "importcache3" / "mediainfo")


def media_info_for_hash(media_hash: str) -> dict | None:
    """Dimensions / fps / duration / byte size for a media hash, if cached.

    The matting directory is named after CapCut's hash of the source media, and
    the import cache is keyed the same way — so this recovers what the media
    *was* even after the timeline that referenced it has been emptied. It does
    not store the file path, so the size is the useful part: it identifies the
    right file on disk when nothing else can.
    """
    path = _MEDIAINFO / f"{media_hash}.json"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    header: dict = {}
    body: dict = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "veAVInfo" in parsed:
            body = parsed
        elif "s" in parsed and not header:
            header = parsed

    video = ((body.get("veAVInfo") or {}).get("videoInfo") or [{}])[0]
    if not video and not header:
        return None
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    # Stored orientation plus a rotation flag, same as the container itself.
    if int(video.get("rotation", 0)) % 180 == 90:
        width, height = height, width
    return {
        "width": width, "height": height,
        "fps": float(video.get("fps", 0) or 0),
        "duration_ms": int(video.get("duration", 0) or 0),
        "size_bytes": int(header.get("s", 0) or 0),
        "format": (body.get("veAVInfo") or {}).get("formatName", ""),
    }


def find_mask_caches(root: Path | None = None) -> list[MaskCache]:
    """Every mask sequence on disk, including ones no timeline references.

    Worth having as a separate path from read_draft, because draft_content.json
    tracks the *live* timeline: clearing or re-cutting a project rewrites it and
    the matting reference disappears, while the 95 MB of masks stays exactly
    where it was. Observed directly — a draft went from one cut-out clip to
    `videos: []` while its mask directory was untouched. Discovery therefore
    reads the disk, and treats the JSON only as extra information.
    """
    root = Path(root) if root else DEFAULT_DRAFT_ROOT
    if not root.is_dir():
        return []

    linked: set[Path] = set()
    found: list[MaskCache] = []
    for draft_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        matting_root = draft_dir / "matting"
        if not matting_root.is_dir():
            continue

        if (draft_dir / "draft_content.json").is_file():
            try:
                for clip in read_draft(draft_dir).clips:
                    linked.add(clip.mask_dir.resolve())
            except DraftError:
                pass

        for media_dir in sorted(matting_root.iterdir()):
            if not media_dir.is_dir():
                continue
            for level in sorted(media_dir.iterdir()):
                mask_root = level / "mask"
                if not (level.is_dir() and mask_root.is_dir()):
                    continue
                count = sum(1 for n in os.listdir(mask_root) if n.isdigit())
                if not count:
                    continue      # an aborted or cleared cutout
                found.append(MaskCache(
                    draft=draft_dir.name, draft_dir=draft_dir,
                    media_hash=media_dir.name, mask_dir=level, n_masks=count,
                    linked=level.resolve() in linked,
                    media=media_info_for_hash(media_dir.name),
                ))
    return found


def _mask_dir_from_token(draft_dir: Path, raw: str) -> Path | None:
    """Turn CapCut's placeholder-prefixed matting path into a real directory.

    The recorded path points at `<draft>/matting/<hash>`; the mask files live one
    level below that in a numbered subdirectory, so this returns the directory
    that actually contains `mask/`.
    """
    if not raw:
        return None
    relative = _PLACEHOLDER.sub("", raw).replace("\\", "/").lstrip("/")
    base = draft_dir / relative
    if not base.is_dir():
        return None
    if (base / "mask").is_dir():
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "mask").is_dir():
            return child
    return None


def read_draft(draft_dir: Path) -> Draft:
    """Parse draft_content.json into the bits needed to re-apply its cutouts."""
    draft_dir = Path(draft_dir)
    content_path = draft_dir / "draft_content.json"
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DraftError(f"cannot read {content_path}") from exc
    except json.JSONDecodeError as exc:
        raise DraftError(
            f"{content_path} is not valid JSON. CapCut may have been mid-save; "
            f"close the project and try again."
        ) from exc

    canvas = content.get("canvas_config") or {}
    materials = {m.get("id"): m for m in (content.get("materials") or {}).get("videos", [])}

    clips: list[Clip] = []
    for track in content.get("tracks") or []:
        if track.get("type") != "video":
            continue
        for segment in track.get("segments") or []:
            material = materials.get(segment.get("material_id"))
            if not material:
                continue
            mask_dir = _mask_dir_from_token(
                draft_dir, (material.get("matting") or {}).get("path", ""))
            if mask_dir is None:
                continue          # this clip has no cutout cached
            source_range = segment.get("source_timerange") or {}
            matting = material.get("matting") or {}
            clips.append(Clip(
                index=len(clips),
                source=Path(str(material.get("path", "")).replace("/", os.sep)),
                mask_dir=mask_dir,
                source_start_us=int(source_range.get("start", 0)),
                source_duration_us=int(source_range.get("duration", 0)),
                speed=float(segment.get("speed", 1.0) or 1.0),
                width=int(material.get("width", 0)),
                height=int(material.get("height", 0)),
                material_name=str(material.get("material_name", "")),
                matting=MattingParams(
                    feather=float(matting.get("feather", 0) or 0),
                    expansion=float(matting.get("expansion", 0) or 0),
                    stroke=bool(matting.get("enable_matting_stroke", False)),
                    reverse=bool(matting.get("reverse", False)),
                    flag=int(matting.get("flag", 0) or 0),
                ),
            ))

    return Draft(
        name=draft_dir.name, path=draft_dir,
        canvas_width=int(canvas.get("width", 0)),
        canvas_height=int(canvas.get("height", 0)),
        timeline_fps=float(content.get("fps", 0) or 0),
        clips=clips,
    )


# --------------------------------------------------------------------------- #
# The mask sequence
# --------------------------------------------------------------------------- #

class MaskSequence:
    """The raw masks for one clip, addressable by source timestamp."""

    def __init__(self, mask_dir: Path):
        self.dir = Path(mask_dir)
        mask_root = self.dir / "mask"
        if not mask_root.is_dir():
            raise DraftError(f"no mask/ directory under {self.dir}")

        # Numeric sort: these are timestamps, and a lexical sort puts 9xxxxxx
        # before 10xxxxxx, which would silently shuffle the sequence.
        names = [n for n in os.listdir(mask_root) if n.isdigit()]
        if not names:
            raise DraftError(f"no mask frames in {mask_root}")
        self.timestamps = np.array(sorted(int(n) for n in names), dtype=np.int64)
        self._paths = {int(n): mask_root / n for n in names}
        self.width, self.height = self._dimensions(mask_root)
        self._cache: tuple[int, np.ndarray] | None = None

    def _dimensions(self, mask_root: Path) -> tuple[int, int]:
        """Take the mask size from maskinfo, and verify it against the bytes."""
        size = (mask_root / str(self.timestamps[0])).stat().st_size
        info_path = self.dir / "maskinfo" / str(self.timestamps[0])
        try:
            box = json.loads(info_path.read_text(encoding="utf-8"))["boundingBox"]
            width, height = int(box["width"]), int(box["height"])
            if width * height == size:
                return width, height
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

        # No usable sidecar. The masks are single-channel and 9:16-ish, so look
        # for the factor pair closest to the draft's own aspect rather than
        # guessing; better to fail loudly than to reshape bytes into noise.
        for width in range(64, 1025):
            if size % width == 0:
                height = size // width
                if 1.2 <= height / width <= 2.4:
                    return width, height
        raise DraftError(
            f"cannot determine mask dimensions: {size} bytes per frame in "
            f"{mask_root}, and maskinfo did not say."
        )

    @property
    def covered_us(self) -> tuple[int, int]:
        return int(self.timestamps[0]), int(self.timestamps[-1])

    def __len__(self) -> int:
        return int(self.timestamps.size)

    def at(self, source_us: int) -> np.ndarray:
        """Nearest mask to a source timestamp, as float32 0-1 at mask resolution."""
        pos = int(np.searchsorted(self.timestamps, source_us))
        if pos <= 0:
            pos = 0
        elif pos >= self.timestamps.size:
            pos = self.timestamps.size - 1
        elif (source_us - self.timestamps[pos - 1]) <= (self.timestamps[pos] - source_us):
            pos -= 1
        stamp = int(self.timestamps[pos])

        if self._cache is not None and self._cache[0] == stamp:
            return self._cache[1]
        raw = np.fromfile(self._paths[stamp], dtype=np.uint8)
        expected = self.width * self.height
        if raw.size != expected:
            raise DraftError(
                f"mask {stamp} is {raw.size} bytes, expected {expected} "
                f"({self.width}x{self.height})"
            )
        alpha = raw.reshape(self.height, self.width).astype(np.float32) / 255.0
        self._cache = (stamp, alpha)
        return alpha

    def gap_before(self, source_us: int) -> int:
        """How far the nearest mask is from this timestamp, in µs.

        Large values mean the requested time falls outside what CapCut cached —
        worth reporting, because the alpha will be a held frame rather than a
        matte for the frame actually being composited.
        """
        pos = int(np.clip(np.searchsorted(self.timestamps, source_us), 1,
                          self.timestamps.size - 1))
        return int(min(abs(source_us - self.timestamps[pos]),
                       abs(source_us - self.timestamps[pos - 1])))


# --------------------------------------------------------------------------- #
# Upsampling
# --------------------------------------------------------------------------- #

def guided_upsample(mask: np.ndarray, guide_grey: np.ndarray,
                    radius: int = 8, eps: float = 1e-4,
                    tighten: float = 1.6) -> np.ndarray:
    """Upsample a low-resolution mask, snapping its edge onto real image edges.

    compositing.py argues at length against guided-filtering a matte, and it is
    right — about RVM's matte. RVM takes a `downsample_ratio` and runs a learned
    Deep Guided Filter internally to get from backbone resolution to full frame,
    so its output has already been guided-upsampled with coefficients trained end
    to end. A box-window guided filter on top of that blurs a refined matte, and
    their measurements show exactly that.

    CapCut's masks are the opposite case: a raw 256x448 mask with no refinement
    of any kind, needing a 4.22x upscale. Here there *is* no learned filter to
    duplicate, and the guide is doing the job nothing else has done. Measured on
    the same frame, soft-edge area went from 1.046% (plain bicubic) to 0.444% —
    a tighter edge sitting on the actual boundary rather than a wide interpolated
    ramp.

    `tighten` restores edge contrast afterwards, because the filter necessarily
    softens globally. 2.2 was tried first and visibly eroded the shoulders on the
    test clip, so the default is 1.6.
    """
    height, width = guide_grey.shape[:2]
    coarse = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)
    coarse = np.clip(coarse, 0.0, 1.0)

    k = (radius, radius)
    mean_g = cv2.boxFilter(guide_grey, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(coarse, cv2.CV_32F, k)
    cov = cv2.boxFilter(guide_grey * coarse, cv2.CV_32F, k) - mean_g * mean_p
    var = cv2.boxFilter(guide_grey * guide_grey, cv2.CV_32F, k) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_p - a * mean_g
    out = cv2.boxFilter(a, cv2.CV_32F, k) * guide_grey + cv2.boxFilter(b, cv2.CV_32F, k)

    if tighten and tighten != 1.0:
        out = (out - 0.5) * float(tighten) + 0.5
    return np.clip(out, 0.0, 1.0, out=out)


def bicubic_upsample(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Plain bicubic upscale — what CapCut's own compositor effectively does."""
    return np.clip(cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC),
                   0.0, 1.0)


def alpha_for_frame(masks: MaskSequence, source_us: int, frame_rgb: np.ndarray, *,
                    mode: str = "guided", tighten: float = 1.6,
                    reverse: bool = False) -> np.ndarray:
    """Fetch, upsample and orient the alpha for one decoded frame."""
    mask = masks.at(source_us)
    height, width = frame_rgb.shape[:2]
    if mode == "bicubic":
        alpha = bicubic_upsample(mask, width, height)
    else:
        grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha = guided_upsample(mask, grey, tighten=tighten)
    if reverse:
        alpha = 1.0 - alpha
    return alpha


# --------------------------------------------------------------------------- #
# Alpha probe
# --------------------------------------------------------------------------- #
#
# This module carried its own corrected copy of the probe while
# video.encoder_keeps_alpha still read its test file back with ffmpeg's default
# decoder -- which for WebM is the native `vp9` decoder, and that one does not
# surface VP9's alpha side-channel, so the probe saw an opaque plane and blamed
# the encoder for alpha it had in fact written. That is fixed at the source now:
# video.encoder_keeps_alpha names the container's real decoder, so both tools
# agree and WebM is allowed. The name is re-exported (see the imports above) so
# callers here keep working, and there is one implementation to maintain.
