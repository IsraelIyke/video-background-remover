"""Orchestration: decode -> matte -> composite -> encode.

Two execution paths share one segment worker:

* one process streams the whole clip straight into the encoder;
* several processes each take a contiguous span, encode it to a temporary
  segment, and the segments are joined without re-encoding.

Splitting the work pays off because ONNX convolutions scale poorly across
threads on small frames -- two single-threaded processes beat one process using
every thread. Each span decodes a short run-up before its first kept frame so
RVM's recurrent state has settled by then and the seams stay invisible.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import compositing, formats, models
from .video import FrameReader, FrameWriter, VideoInfo, concat, probe

WARMUP_FRAMES = 24  # ~0.8s at 30fps; RVM's state converges well before this


class PartialResult(KeyboardInterrupt):
    """Raised when Ctrl+C stopped a run that had already encoded usable frames.

    Subclasses KeyboardInterrupt so any caller that only knows about the
    interrupt still behaves correctly; callers that care can report the file.
    """

    def __init__(self, path: Path, frames: int):
        super().__init__(f"interrupted after {frames} frames -> {path}")
        self.path = path
        self.frames = frames


@dataclass
class Settings:
    """Everything a worker needs to process a span of frames."""
    input: Path
    output: Path
    model: str = "mobilenetv3"
    fmt: str = "mp4"
    background: str = "#000000"
    downsample: float | None = None
    threads: int | None = None
    quality: int = 18
    hwenc: bool = False
    io_threads: int | None = None   # cap ffmpeg's own threads when workers share cores

    # Resolution the whole pipeline works at. 0 means "the source's own".
    # Matting a 4K frame is four times the work of an HD one for an edge that is
    # no better -- what sharpens the edge is the *ratio* the backbone runs at,
    # not the pixel count around it -- so capping this is usually a pure win.
    work_width: int = 0
    work_height: int = 0

    # matte refinement
    decontaminate: bool = True
    choke: float = 0.0
    feather: float = 0.0
    gamma: float = 1.0
    levels_low: float = 0.0
    levels_high: float = 1.0
    denoise: bool = False
    temporal: float = 0.0
    main_subject: int = 0

    # span
    start_frame: int = 0
    end_frame: int = 0
    include_audio: bool = True
    audio_codec: str | None = None
    png_start_number: int = 0


def _frame_span_to_time(index: int, fps: Fraction) -> float:
    return float(index / fps)


def process_span(settings: Settings, info: VideoInfo,
                 progress: "mp.Queue | None" = None) -> int:
    """Matte and encode frames [start_frame, end_frame). Returns frames written."""
    from .matting import MattingEngine

    fmt = formats.FORMATS[settings.fmt]
    width = settings.work_width or info.width
    height = settings.work_height or info.height
    scale_to = (width, height) if (width, height) != (info.width, info.height) else None

    model_path = models.resolve(settings.model)
    engine = MattingEngine(model_path, threads=settings.threads,
                           downsample=settings.downsample)
    ratio = engine.downsample_for(width, height)

    background = compositing.build_background(settings.background, width, height)
    smoother = compositing.TemporalSmoother(settings.temporal)

    # A background given for an alpha-capable format is honoured, and the frames
    # simply come out fully opaque. The reverse is not possible.
    if not fmt.supports_alpha and background.is_transparent:
        raise ValueError(
            f"{fmt.name} cannot store transparency. Give --background a colour, "
            f"image, video or 'blur', or pick one of: {', '.join(formats.ALPHA_FORMATS)}"
        )

    warmup = min(settings.start_frame, WARMUP_FRAMES)
    decode_start = settings.start_frame - warmup
    n_keep = settings.end_frame - settings.start_frame
    n_decode = warmup + n_keep

    out_height = height * 2 if settings.fmt == "stacked" else height

    reader = FrameReader(
        settings.input, info.width, info.height,
        start=_frame_span_to_time(decode_start, info.fps) if decode_start else None,
        duration=(n_decode + 1) / float(info.fps),
        scale_to=scale_to,
        threads=settings.io_threads,
    )

    audio_source = None
    audio_args: list[str] = []
    if settings.include_audio and info.has_audio and fmt.audio:
        audio_source = settings.input
        audio_args = formats.audio_args_for(fmt, settings.audio_codec)

    encoder_args = fmt.encoder_args(settings.quality, settings.hwenc,
                                    threads=settings.io_threads)
    if fmt.is_sequence:
        encoder_args = encoder_args + ["-start_number", str(settings.png_start_number)]

    writer = FrameWriter(
        settings.output, width, out_height, info.fps,
        pix_fmt_in=fmt.pix_fmt_in, encoder_args=encoder_args,
        audio_from=audio_source, audio_args=audio_args,
        audio_start=_frame_span_to_time(settings.start_frame, info.fps) or None,
        # A PNG sequence is many files matched by a pattern, so there is no
        # single path to swap into place at the end.
        atomic=not fmt.is_sequence,
    )

    written = 0
    try:
        for i in range(n_decode):
            raw = reader.read()
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

            foreground, alpha = engine(frame, ratio)

            if i < warmup:
                continue  # state still settling; nothing kept

            alpha = compositing.refine_alpha(
                alpha, choke=settings.choke, feather=settings.feather,
                gamma=settings.gamma, low=settings.levels_low,
                high=settings.levels_high, denoise=settings.denoise,
                main_subject=settings.main_subject,
            )
            alpha = smoother(alpha)

            # Must run on the final alpha: it decides which pixels are solid
            # enough to borrow colour from. Pointless for 'matte', which throws
            # the colour away.
            if settings.decontaminate and settings.fmt != "matte":
                foreground = compositing.decontaminate(foreground, alpha)

            if settings.fmt == "matte":
                out = compositing.to_matte(alpha)
            elif settings.fmt == "stacked":
                colour = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)
                out = np.vstack([colour, compositing.to_matte(alpha)])
            elif background.is_transparent:
                out = compositing.to_rgba(foreground, alpha)
            else:
                plate = background.frame(written, frame, alpha)
                rgb = compositing.composite(foreground, alpha, plate)
                out = (np.dstack([rgb, np.full(alpha.shape, 255, np.uint8)])
                       if fmt.pix_fmt_in == "rgba" else rgb)

            writer.write(out.tobytes())
            written += 1
            if progress is not None and written % 4 == 0:
                progress.put(4)

        if progress is not None and written % 4:
            progress.put(written % 4)
    except KeyboardInterrupt:
        # Ctrl+C after twenty minutes of matting should not cost the user all
        # twenty minutes. Let the encoder finish its index and keep what is done.
        if written:
            raise PartialResult(writer.close_partial(), written) from None
        writer.abort()
        raise
    except BaseException:
        # Don't let ffmpeg linger holding a half-written file, and don't mask
        # the original failure with a secondary encoder error.
        writer.abort()
        raise
    finally:
        reader.close()
        background.close()

    writer.close()
    return written


class Progress:
    """Single-line progress with a live rate and ETA."""

    def __init__(self, total: int, label: str = "matting"):
        self.total = max(total, 1)
        self.label = label
        self.done = 0
        self.start = time.perf_counter()
        self._last_draw = 0.0

    def advance(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        # Redrawing costs real CPU in the terminal emulator, which on a machine
        # with few cores competes with the matting itself.
        if now - self._last_draw < 0.5 and self.done < self.total:
            return
        self._last_draw = now
        self.draw()

    def draw(self) -> None:
        elapsed = time.perf_counter() - self.start
        frac = min(self.done / self.total, 1.0)
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        filled = int(frac * 28)
        bar = "█" * filled + "░" * (28 - filled)
        print(f"\r  {self.label} [{bar}] {frac*100:5.1f}%  "
              f"{self.done}/{self.total}  {rate:4.1f} fps  ETA {_clock(eta)}   ",
              end="", file=sys.stderr, flush=True)

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.start
        rate = self.done / elapsed if elapsed else 0
        filled = 28
        print(f"\r  {self.label} [{'█'*filled}] 100.0%  {self.done}/{self.total}  "
              f"{rate:4.1f} fps  in {_clock(elapsed)}      ", file=sys.stderr, flush=True)


def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _DirectProgress:
    """Stands in for the worker queue when everything runs in this process."""

    def __init__(self, bar: Progress):
        self._bar = bar

    def put(self, item) -> None:
        if isinstance(item, tuple):
            return  # error sentinels propagate as exceptions here anyway
        self._bar.advance(int(item))


def run(settings: Settings, info: VideoInfo, *, workers: int = 1,
        show_progress: bool = True) -> Path:
    """Process the requested span, in parallel when it helps."""
    total = settings.end_frame - settings.start_frame
    fmt = formats.FORMATS[settings.fmt]

    if fmt.is_sequence:
        workers = 1  # frames land in a directory; no joining needed

    if workers > 1:
        return _run_parallel(settings, info, workers, total, show_progress)

    bar = Progress(total) if show_progress else None
    process_span(settings, info, _DirectProgress(bar) if bar else None)
    if bar:
        bar.finish()
    return settings.output


def _run_parallel(settings: Settings, info: VideoInfo, workers: int,
                  total: int, show_progress: bool) -> Path:
    fmt = formats.FORMATS[settings.fmt]
    temp_dir = Path(tempfile.mkdtemp(prefix="bgremove_"))
    bar = Progress(total) if show_progress else None

    try:
        # Contiguous spans of equal length keep every worker busy for the same time.
        bounds = np.linspace(settings.start_frame, settings.end_frame,
                             workers + 1).round().astype(int)
        jobs = []
        for i in range(workers):
            span_start, span_end = int(bounds[i]), int(bounds[i + 1])
            if span_end <= span_start:
                continue
            seg = temp_dir / f"segment_{i:03d}{fmt.suffix}"
            sub = dataclasses.replace(
                settings, output=seg, start_frame=span_start, end_frame=span_end,
                include_audio=False,   # audio is muxed once, at the join
                threads=1,             # one core each; that is the whole point
                io_threads=1,          # and ffmpeg must not grab the rest
            )
            jobs.append(sub)

        ctx = mp.get_context("spawn")
        progress_q = ctx.Queue()
        info_dict = dataclasses.asdict(info)
        info_dict["path"] = str(info_dict["path"])
        info_dict["fps"] = str(info.fps)

        procs = []
        for job in jobs:
            job_dict = dataclasses.asdict(job)
            job_dict["input"] = str(job_dict["input"])
            job_dict["output"] = str(job_dict["output"])
            p = ctx.Process(target=_worker_entry,
                            args=(job_dict, info_dict, progress_q), daemon=True)
            p.start()
            procs.append(p)

        errors: list[str] = []
        alive = len(procs)
        while alive > 0:
            try:
                item = progress_q.get(timeout=0.5)
            except Exception:
                alive = sum(1 for p in procs if p.is_alive())
                if alive == 0:
                    break
                continue
            if isinstance(item, tuple) and item and item[0] == "error":
                errors.append(item[1])
            elif bar:
                bar.advance(int(item))

        for p in procs:
            p.join()
        if bar:
            bar.finish()

        failed = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if errors or failed:
            detail = "\n  ".join(errors) or f"worker exit codes: {failed}"
            raise RuntimeError(f"A worker failed while matting:\n  {detail}")

        segments = [job.output for job in jobs if job.output.exists()]
        if not segments:
            raise RuntimeError("No segments were produced.")

        audio_source = None
        audio_args: list[str] = []
        if settings.include_audio and info.has_audio and fmt.audio:
            audio_source = settings.input
            audio_args = formats.audio_args_for(fmt, settings.audio_codec)

        # The per-segment encode set +faststart, but the join re-muxes, so the
        # joined file needs it again or the index lands at the end of the file.
        extra = ["-movflags", "+faststart"] if fmt.suffix == ".mp4" else []
        concat(segments, settings.output, audio_from=audio_source,
               audio_args=audio_args, list_file=temp_dir / "segments.txt",
               extra_out_args=extra)
        return settings.output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _worker_entry(settings_dict: dict, info_dict: dict, progress: "mp.Queue") -> None:
    """Child-process entry point.

    Paths and Fractions are passed as strings because `spawn` has to pickle
    everything, and rebuilding them here keeps the parent's send cheap.
    """
    settings_dict = dict(settings_dict)
    settings_dict["input"] = Path(settings_dict["input"])
    settings_dict["output"] = Path(settings_dict["output"])
    info_dict = dict(info_dict)
    info_dict["path"] = Path(info_dict["path"])
    info_dict["fps"] = Fraction(info_dict["fps"])

    try:
        process_span(Settings(**settings_dict), VideoInfo(**info_dict), progress)
    except Exception as exc:  # the parent reports this instead of a bare exit code
        progress.put(("error", f"{type(exc).__name__}: {exc}"))
        raise
