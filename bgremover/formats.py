"""Output format definitions.

A note on MP4 and transparency, because this trips everyone up:

MP4/H.264 has no usable alpha channel. Apple ships HEVC-with-alpha in MP4, but
ffmpeg cannot encode it and almost nothing outside Apple's ecosystem plays it.
So an MP4 always needs *something* behind the subject -- a colour, an image, a
video, or a blur. If genuine transparency is the goal, use webm / mov / png, or
use the `matte` and `stacked` formats which carry the alpha as visible pixels
that an editor or shader can key off.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OutputFormat:
    name: str
    suffix: str
    supports_alpha: bool
    pix_fmt_in: str                      # raw pixel format we pipe into ffmpeg
    encoder: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    is_sequence: bool = False
    description: str = ""

    def encoder_args(self, quality: int, hwenc: bool = False,
                     threads: int | None = None) -> list[str]:
        args = list(self.encoder)
        if self.name in ("mp4", "matte", "stacked"):
            if hwenc:
                args = ["-c:v", "h264_qsv", "-global_quality", str(quality),
                        "-pix_fmt", "nv12"]
            else:
                args = [a if a != "__CRF__" else str(quality) for a in args]
            args += ["-movflags", "+faststart"]
        elif self.name == "webm":
            args = [a if a != "__CRF__" else str(quality + 6) for a in args]

        # When several workers share a machine, an encoder that helps itself to
        # every core starves the inference processes that are the real bottleneck.
        if threads and not hwenc:
            args += ["-threads", str(threads)]
        return args


FORMATS: dict[str, OutputFormat] = {
    "mp4": OutputFormat(
        name="mp4", suffix=".mp4", supports_alpha=False, pix_fmt_in="rgb24",
        encoder=["-c:v", "libx264", "-preset", "medium", "-crf", "__CRF__",
                 "-pix_fmt", "yuv420p"],
        audio=["-c:a", "aac", "-b:a", "192k"],
        description="H.264 MP4 with a replaced background. Plays everywhere.",
    ),
    "webm": OutputFormat(
        name="webm", suffix=".webm", supports_alpha=True, pix_fmt_in="rgba",
        encoder=["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                 "-b:v", "0", "-crf", "__CRF__", "-row-mt", "1"],
        audio=["-c:a", "libopus", "-b:a", "128k"],
        description="VP9 WebM with real transparency. For browsers and most editors.",
    ),
    "mov": OutputFormat(
        name="mov", suffix=".mov", supports_alpha=True, pix_fmt_in="rgba",
        encoder=["-c:v", "prores_ks", "-profile:v", "4444", "-vendor", "apl0",
                 "-pix_fmt", "yuva444p10le", "-alpha_bits", "16"],
        audio=["-c:a", "pcm_s16le"],
        description="ProRes 4444 with alpha. For Premiere / Resolve / Final Cut.",
    ),
    "mkv": OutputFormat(
        name="mkv", suffix=".mkv", supports_alpha=True, pix_fmt_in="rgba",
        encoder=["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgba"],
        audio=["-c:a", "copy"],
        description="Lossless FFV1 with alpha. Archival quality, large files.",
    ),
    "png": OutputFormat(
        name="png", suffix=".png", supports_alpha=True, pix_fmt_in="rgba",
        encoder=["-c:v", "png"], audio=[], is_sequence=True,
        description="RGBA PNG sequence. Lossless alpha, imports into anything.",
    ),
    "matte": OutputFormat(
        name="matte", suffix=".mp4", supports_alpha=False, pix_fmt_in="rgb24",
        encoder=["-c:v", "libx264", "-preset", "medium", "-crf", "__CRF__",
                 "-pix_fmt", "yuv420p"],
        audio=[],
        description="Black-and-white alpha matte as MP4, for luma-key / track mattes.",
    ),
    "stacked": OutputFormat(
        name="stacked", suffix=".mp4", supports_alpha=False, pix_fmt_in="rgb24",
        encoder=["-c:v", "libx264", "-preset", "medium", "-crf", "__CRF__",
                 "-pix_fmt", "yuv420p"],
        audio=["-c:a", "aac", "-b:a", "192k"],
        description="MP4 with colour on top and the matte below. For shaders / Unity / canvas.",
    ),
}

ALPHA_FORMATS = [name for name, fmt in FORMATS.items() if fmt.supports_alpha]


def audio_args_for(fmt: OutputFormat, source_codec: str | None) -> list[str]:
    """Prefer stream copy when the container accepts the source codec as-is."""
    if not fmt.audio:
        return []
    if fmt.name == "mp4" and source_codec == "aac":
        return ["-c:a", "copy"]
    if fmt.name == "mov" and source_codec in ("aac", "pcm_s16le"):
        return ["-c:a", "copy"]
    if fmt.name == "webm" and source_codec in ("opus", "vorbis"):
        return ["-c:a", "copy"]
    return list(fmt.audio)
