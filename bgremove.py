#!/usr/bin/env python3
"""Remove the background from a video, locally.

Uses Robust Video Matting (RVM) through ONNX Runtime. Nothing is uploaded and no
account is needed. Run --help for the full option list, or see README.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

from bgremover import __version__, formats, models
from bgremover.matting import SPEED_RATIOS, auto_downsample_ratio, physical_cores
from bgremover.pipeline import PartialResult, Settings, run
from bgremover.video import (FFmpegMissing, capped_size, encoder_keeps_alpha,
                             probe, require_ffmpeg, verify_playable)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgremove",
        description="Remove a video's background locally, with proper edge quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
  bgremove Practice.mp4
        black background, MP4 out, audio kept

  bgremove Practice.mp4 --background blur
        keep the real background but throw it out of focus

  bgremove Practice.mp4 --background office.jpg -o talk.mp4
        drop the subject onto a photo

  bgremove Practice.mp4 --background greenscreen
        chroma green, for keying in an editor later

  bgremove Practice.mp4 --transparent
        no background at all -- ProRes 4444 with a real alpha channel

  bgremove Practice.mp4 --main-subject
        keep only the biggest person, dropping picture-in-picture insets

  bgremove Practice.mp4 --preview 10
        try the settings on the first 10 seconds before committing

a note on transparency
  MP4 cannot store an alpha channel -- that is a limit of the format, not this
  tool. Use --transparent (or --format mov / mkv / png) for real alpha. To carry
  the matte through an MP4, use --format matte (black & white matte) or
  --format stacked (colour above, matte below).

  WebM is listed but ffmpeg 8 dropped VP9's alpha side-channel: it accepts the
  request and silently returns opaque video. The tool probes your ffmpeg before
  a transparent run and refuses rather than let you find out afterwards.
""",
    )

    parser.add_argument("input", type=Path, nargs="?", help="Video file to process.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path. Defaults to <input>_nobg.<ext>.")
    parser.add_argument("-f", "--format", dest="fmt", default=None,
                        choices=sorted(formats.FORMATS),
                        help="Output format (default: mp4, or mov with --transparent).")
    parser.add_argument("-b", "--background", "--bg", dest="background", default=None,
                        help="Colour ('black', '#101820', '12,34,56'), an image or "
                             "video file, 'blur[:strength]', or 'none' for "
                             "transparency. Default: black for mp4, transparent otherwise.")
    parser.add_argument("-t", "--transparent", "--no-bg", dest="transparent",
                        action="store_true",
                        help="No background at all -- keep a real alpha channel. "
                             "Selects mov (ProRes 4444) unless --format says otherwise.")

    quality = parser.add_argument_group("quality / speed")
    quality.add_argument("--speed", choices=list(SPEED_RATIOS), default="balanced",
                         help="The scale the network runs at: fast/balanced/best/max "
                              "= 0.25/0.375/0.5/1.0 of the frame (default: balanced). "
                              "This, not --resolution, is what decides how clean the "
                              "edge is.")
    quality.add_argument("--model", default="mobilenetv3",
                         choices=["mobilenetv3", "resnet50"],
                         help="mobilenetv3 is 3-6x faster; resnet50 is slightly cleaner "
                              "on fine hair but painfully slow without a GPU.")
    quality.add_argument("--resolution", default="1080", metavar="N",
                         help="Process at most N pixels on the short edge (default: "
                              "1080, i.e. HD). 'source' keeps the input's own size. "
                              "4K costs 4x the time and yields no better an edge -- "
                              "edge quality comes from --speed, not from frame size.")
    quality.add_argument("--downsample", type=float, default=None,
                         help="Override the internal matting scale (0-1). Overrides --speed.")
    quality.add_argument("--crf", type=int, default=18,
                         help="Encoder quality, lower is better (default: 18).")
    quality.add_argument("--hwenc", action="store_true",
                         help="Encode H.264 on the Intel/AMD GPU instead of the CPU.")

    matte = parser.add_argument_group("matte refinement")
    matte.add_argument("--no-decontaminate", dest="decontaminate", action="store_false",
                       help="Keep the network's own edge colours. Those carry the "
                            "background the subject was shot against, which is what a "
                            "halo is made of -- only useful for comparison.")
    matte.add_argument("--choke", type=float, default=0.0,
                       help="Shrink (negative) or grow (positive) the cutout, in pixels.")
    matte.add_argument("--feather", type=float, default=0.0,
                       help="Soften the edge, in pixels.")
    matte.add_argument("--alpha-gamma", type=float, default=1.0, dest="gamma",
                       help="<1 firms up semi-transparent areas, >1 softens them.")
    matte.add_argument("--levels", default=None, metavar="LOW,HIGH",
                       help="Remap the matte, e.g. '0.05,0.95' to clear edge haze.")
    matte.add_argument("--denoise", action="store_true",
                       help="Median-filter the matte to remove speckle.")
    matte.add_argument("--temporal", type=float, default=0.0,
                       help="Blend the matte with the previous frame (0-0.9) to "
                            "settle a crawling edge.")
    matte.add_argument("--main-subject", type=int, nargs="?", const=1, default=0,
                       metavar="N", dest="main_subject",
                       help="Keep only the N largest subjects (default 1). Drops "
                            "people inside picture-in-picture insets, posters and "
                            "burned-in thumbnails, and removes stray speckle too.")

    span = parser.add_argument_group("what to process")
    span.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    span.add_argument("--duration", type=float, default=None,
                      help="How many seconds to process.")
    span.add_argument("--preview", type=float, nargs="?", const=10.0, default=None,
                      metavar="SECONDS",
                      help="Process a short sample (default 10s) to check settings.")
    span.add_argument("--no-audio", action="store_true", help="Drop the audio track.")

    perf = parser.add_argument_group("performance")
    perf.add_argument("-j", "--workers", type=int, default=None,
                      help="Parallel worker processes (default: one per physical core, max 4).")
    perf.add_argument("--threads", type=int, default=None,
                      help="Threads per worker (default: 1 when parallel, else all cores).")

    misc = parser.add_argument_group("other")
    misc.add_argument("--benchmark", action="store_true",
                      help="Time a few settings on this machine and exit.")
    misc.add_argument("--list-formats", action="store_true", help="Show output formats and exit.")
    misc.add_argument("--quiet", action="store_true", help="Suppress the progress bar.")
    misc.add_argument("--version", action="version", version=f"bgremove {__version__}")
    return parser


def show_formats() -> None:
    eprint("Output formats:\n")
    for name, fmt in formats.FORMATS.items():
        alpha = "yes" if fmt.supports_alpha else "no "
        eprint(f"  {name:<9} alpha:{alpha}  {fmt.description}")
    eprint("\n  MP4 has no alpha channel, so mp4/matte/stacked always need a background.")


def benchmark(args) -> int:
    """Measure this machine so the time estimates are real, not guesses."""
    import numpy as np
    from bgremover.matting import MattingEngine

    width, height = 640, 360
    if args.input and args.input.exists():
        info = probe(args.input)
        # Measure the size a real run would work at, or the estimate for a 4K
        # phone clip comes out four times too pessimistic.
        try:
            cap = None if str(args.resolution).strip().lower() in (
                "source", "native", "full", "none", "0") else int(args.resolution)
        except ValueError:
            cap = 1080
        width, height = capped_size(info.width, info.height, cap)
        total = info.n_frames
    else:
        total = 6571

    cores = physical_cores()
    eprint(f"Benchmarking at {width}x{height} on {cores} physical core(s)\n")

    header = (f"{'model':<13}{'--speed':<10}{'scale':>7}{'backbone':>11}"
              f"{'ms/frame':>10}{'fps':>7}   est. for {total} frames")
    eprint(header)
    eprint("-" * len(header))

    for model_name in ("mobilenetv3", "resnet50"):
        try:
            path = models.resolve(model_name)
        except Exception as exc:
            eprint(f"  {model_name}: unavailable ({exc})")
            continue
        presets = list(SPEED_RATIOS) if model_name == "mobilenetv3" else ["balanced"]
        for preset in presets:
            preset_ratio = SPEED_RATIOS[preset]
            ratio = (1.0 if preset_ratio is None
                     else auto_downsample_ratio(width, height, preset_ratio))
            engine = MattingEngine(path, threads=cores)
            frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
            for _ in range(3):
                engine(frame, ratio)
            samples = []
            for _ in range(9):
                t0 = time.perf_counter()
                engine(frame, ratio)
                samples.append(time.perf_counter() - t0)
            # The fastest sample is the one least disturbed by whatever else the
            # machine was doing; a median just measures the background load.
            per_frame = min(samples)
            backbone = f"{int(width*ratio)}x{int(height*ratio)}"
            eprint(f"{model_name:<13}{preset:<10}{ratio:>7.3f}{backbone:>11}"
                   f"{per_frame*1000:>10.0f}{1/per_frame:>7.1f}   {_clock(per_frame*total)}")
    eprint("\n  Matting is ~95% of the run time, so these estimates track reality closely.")
    return 0


# Bytes per pixel per frame, measured on this project's 640x360 source. Only
# used for a heads-up, because "real alpha" costs 20-50x an MP4 and finding that
# out after a half-hour render is a poor way to learn it.
_ALPHA_WEIGHT = {"mov": 1.03, "mkv": 0.48, "png": 0.87}


def _size_hint(fmt_name: str, width: int, height: int, frames: int) -> str | None:
    weight = _ALPHA_WEIGHT.get(fmt_name)
    if weight is None:
        return None
    total = weight * width * height * frames
    return f"~{total / 1e9:.1f} GB" if total >= 1e9 else f"~{total / 1e6:.0f} MB"


def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def resolve_output(args, fmt: formats.OutputFormat, stem: str, input_path: Path) -> Path:
    if args.output:
        out = Path(args.output)
    else:
        # matte and stacked also end in .mp4, so they need naming apart from a
        # plain run or the second one would quietly overwrite the first.
        parts = [stem]
        if fmt.name in ("matte", "stacked"):
            parts.append(fmt.name)
        elif not args.preview:
            parts.append("nobg")
        if args.preview:
            parts.append("preview")
        out = input_path.with_name("_".join(parts) + fmt.suffix)
    if fmt.is_sequence:
        directory = out if out.suffix == "" else out.with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "%05d.png"
    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_formats:
        show_formats()
        return 0

    try:
        require_ffmpeg()
    except FFmpegMissing as exc:
        eprint(f"error: {exc}")
        return 1

    if args.benchmark:
        return benchmark(args)

    if args.input is None:
        parser.print_help()
        return 1
    if not args.input.exists():
        eprint(f"error: input not found: {args.input}")
        return 1

    if args.transparent and args.background is not None:
        eprint("error: --transparent and --background ask for opposite things.")
        eprint("       Drop one: --transparent for a real alpha channel, or")
        eprint("       --background for a colour / image / video / blur behind the subject.")
        return 1

    # mov (ProRes 4444) is the transparent default because it is the one alpha
    # container that has stayed reliable across ffmpeg releases -- FFmpeg 8
    # dropped WebM's alpha side-channel -- and it is what editors want anyway.
    if args.fmt is None:
        args.fmt = "mov" if args.transparent else "mp4"
    fmt = formats.FORMATS[args.fmt]

    if args.transparent and not fmt.supports_alpha:
        eprint(f"error: --transparent needs a format that can store alpha, and "
               f"{args.fmt} cannot.")
        eprint(f"       Use --format {' / '.join(formats.ALPHA_FORMATS)},")
        eprint("       or --format matte / stacked to carry the alpha as visible pixels.")
        return 1

    background = "none" if args.transparent else args.background
    if background is None:
        background = "none" if fmt.supports_alpha else "black"
    transparent_out = background.lower() in ("none", "transparent", "alpha")
    if not fmt.supports_alpha and transparent_out:
        eprint(f"error: {args.fmt} cannot store transparency.")
        eprint(f"       Use --format {' / '.join(formats.ALPHA_FORMATS)} for real alpha,")
        eprint("       or give --background a colour, image, video, or 'blur'.")
        return 1

    # Ask this ffmpeg whether it can really do it, rather than trusting the
    # format table -- see encoder_keeps_alpha. Better a second now than a silently
    # opaque file after half an hour of matting.
    if transparent_out and not encoder_keeps_alpha(
            fmt.encoder_args(args.crf, args.hwenc), fmt.suffix, fmt.is_sequence):
        eprint(f"error: this ffmpeg build encodes {args.fmt} without an alpha channel.")
        eprint("       It accepts the request and silently returns opaque video, so the")
        eprint("       run would look fine and the transparency would simply be missing.")
        working = [name for name in formats.ALPHA_FORMATS
                   if name != args.fmt and encoder_keeps_alpha(
                       formats.FORMATS[name].encoder_args(args.crf, args.hwenc),
                       formats.FORMATS[name].suffix,
                       formats.FORMATS[name].is_sequence)]
        if working:
            eprint(f"       Formats that do keep alpha here: {' / '.join(working)}")
        eprint("       (ffmpeg 8 removed WebM alpha; mov/mkv/png are unaffected.)")
        return 1

    try:
        info = probe(args.input)
    except Exception as exc:
        eprint(f"error: {exc}")
        return 1

    fps = float(info.fps)
    total_frames = info.n_frames or int(info.duration * fps)
    if total_frames <= 0:
        eprint("error: could not determine the video length.")
        return 1

    start_frame = int(round(args.start * fps))
    if args.preview is not None:
        end_frame = min(total_frames, start_frame + int(round(args.preview * fps)))
    elif args.duration is not None:
        end_frame = min(total_frames, start_frame + int(round(args.duration * fps)))
    else:
        end_frame = total_frames
    if end_frame <= start_frame:
        eprint("error: the selected range contains no frames.")
        return 1

    resolution = str(args.resolution).strip().lower()
    if resolution in ("source", "native", "full", "none", "0"):
        cap = None
    else:
        try:
            cap = int(resolution)
        except ValueError:
            eprint("error: --resolution wants a pixel count (e.g. 1080) or 'source'.")
            return 1
        if cap < 64:
            eprint("error: --resolution below 64 pixels is not useful.")
            return 1
    work_w, work_h = capped_size(info.width, info.height, cap)

    # The ratio has to be derived from the size the network will actually see,
    # not the source's, or capping the resolution would silently shrink the
    # backbone along with it and give back the soft edge we just paid to fix.
    if args.downsample is not None:
        downsample = max(0.05, min(1.0, args.downsample))
    else:
        preset_ratio = SPEED_RATIOS[args.speed]
        downsample = (1.0 if preset_ratio is None
                      else auto_downsample_ratio(work_w, work_h, preset_ratio))

    levels_low, levels_high = 0.0, 1.0
    if args.levels:
        try:
            low_text, high_text = args.levels.split(",")
            levels_low, levels_high = float(low_text), float(high_text)
        except ValueError:
            eprint("error: --levels wants two numbers, e.g. --levels 0.05,0.95")
            return 1

    cores = physical_cores()
    if args.workers is not None:
        workers = max(1, args.workers)
    else:
        # Measured: on a 2-core machine the decoder, encoder and inference
        # already saturate both cores, and splitting the work gained nothing.
        # Extra processes only pay off once there are cores to spare.
        workers = min(cores, 4) if cores >= 4 else 1
    if fmt.is_sequence:
        workers = 1
    n_frames = end_frame - start_frame
    if n_frames < 60:
        workers = 1  # not worth the process spin-up

    stem = args.input.stem
    output = resolve_output(args, fmt, stem, args.input)
    if output.resolve() == args.input.resolve():
        eprint("error: the output would overwrite the input. Pass -o with a different name.")
        return 1

    settings = Settings(
        input=args.input, output=output, model=args.model, fmt=args.fmt,
        background=background, downsample=downsample, threads=args.threads,
        quality=args.crf, hwenc=args.hwenc,
        work_width=work_w, work_height=work_h,
        decontaminate=args.decontaminate,
        choke=args.choke, feather=args.feather, gamma=args.gamma,
        levels_low=levels_low, levels_high=levels_high, denoise=args.denoise,
        temporal=args.temporal, main_subject=args.main_subject,
        start_frame=start_frame, end_frame=end_frame,
        include_audio=not args.no_audio, audio_codec=info.audio_codec,
    )

    seconds = n_frames / fps
    eprint(f"bgremove {__version__}")
    eprint(f"  input       {args.input.name}  "
           f"{info.width}x{info.height} @ {fps:.3f}fps  {_clock(info.duration)}")
    eprint(f"  processing  {n_frames} frames ({_clock(seconds)})"
           + (f"  from {args.start:g}s" if start_frame else ""))
    if (work_w, work_h) != (info.width, info.height):
        eprint(f"  working at  {work_w}x{work_h}  "
               f"(--resolution source to keep {info.width}x{info.height})")
    eprint(f"  model       RVM {args.model}  scale {downsample:.3f}"
           f"  ({int(work_w*downsample)}x{int(work_h*downsample)} internally)")
    eprint(f"  output      {args.fmt} -> {output}")
    eprint(f"  background  {background}"
           + ("  (real alpha channel)" if fmt.supports_alpha and background == "none" else ""))
    if args.main_subject:
        eprint(f"  subjects    keeping the {args.main_subject} largest; "
               f"smaller people and speckle dropped")
    if transparent_out:
        hint = _size_hint(args.fmt, work_w, work_h, n_frames)
        if hint:
            eprint(f"  size        {hint} -- lossless alpha is bulky; an MP4 with a "
                   f"colour behind is ~50x smaller")
    eprint(f"  workers     {workers} x {args.threads or (1 if workers > 1 else cores)} thread(s)")
    if info.has_audio and not args.no_audio and fmt.audio:
        eprint("  audio       kept from the source")
    elif not fmt.audio:
        eprint("  audio       not supported by this format")
    eprint()

    # Fetch the weights here rather than inside the workers, so parallel runs
    # can't race each other downloading the same file.
    try:
        models.resolve(args.model)
    except Exception as exc:
        eprint(f"error: {exc}")
        return 1

    start_time = time.perf_counter()
    try:
        run(settings, info, workers=workers, show_progress=not args.quiet)
    except PartialResult as partial:
        eprint(f"\ninterrupted after {partial.frames} of {n_frames} frames.")
        eprint(f"  The finished part was kept and is playable:  {partial.path}")
        return 130
    except KeyboardInterrupt:
        eprint("\ninterrupted before any frame was encoded; nothing written.")
        return 130
    except Exception as exc:
        eprint(f"\nerror: {exc}")
        return 1

    elapsed = time.perf_counter() - start_time
    final = output.parent if fmt.is_sequence else output
    if not fmt.is_sequence:
        # Confirm the thing we are about to call "Done" actually opens.
        try:
            verify_playable(output)
        except RuntimeError as exc:
            eprint(f"\nerror: {exc}")
            return 1
    eprint(f"\nDone in {_clock(elapsed)}  ->  {final}")
    if fmt.is_sequence:
        eprint(f"  {n_frames} RGBA frames written.")
    else:
        try:
            size_mb = output.stat().st_size / 1e6
            eprint(f"  {size_mb:.1f} MB")
        except OSError:
            pass
    if args.preview is not None:
        eprint("  This was a preview. Drop --preview to process the whole clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
