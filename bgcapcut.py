#!/usr/bin/env python3
"""Apply a cutout CapCut already computed, without recomputing it.

Sibling to bgremove.py. Where bgremove runs Robust Video Matting to *produce* a
matte, this one reads the matte CapCut cached when you used its auto-cutout on a
clip, and composites with it. No model, no inference — roughly 80x faster on this
machine, at the cost of being limited to CapCut's 256x448 mask resolution.

Run --help for the options, --list-drafts to see what is available, or read
COMMANDS.txt for the full command reference for both tools.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from bgremover import __version__, capcut, compositing, formats
from bgremover.pipeline import Progress
from bgremover.video import (FFmpegMissing, FrameReader, FrameWriter,
                             capped_size, probe, require_ffmpeg, verify_playable)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgcapcut",
        description="Composite with the cutout CapCut already computed for a draft.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
  bgcapcut --list-drafts
        show every CapCut draft that has a cached cutout

  bgcapcut 0809
        black background, MP4 out -- the draft's own trim and audio

  bgcapcut 0809 --transparent
        real alpha channel, ProRes 4444

  bgcapcut 0809 --background greenscreen --crf 14
        chroma green, ready to key

  bgcapcut 0809 --upsample bicubic
        skip the guided upsample, matching CapCut's own softer edge

  bgcapcut 0809 --preview 5
        try the settings on 5 seconds first

when to use this instead of bgremove.py
  Use this when the clip has already been cut out in CapCut and you want that
  same result out as a file -- it is ~80x faster because the matte already
  exists. Use bgremove.py when you need the best edge: CapCut caches its masks
  at 256x448, which on a 1080x1920 frame is a 4.22x upscale, and fine hair
  detail is simply not present in the data. See COMMANDS.txt.
""",
    )

    parser.add_argument("draft", nargs="?",
                        help="CapCut draft name (or path to a draft folder).")
    parser.add_argument("--masks", type=Path, default=None,
                        help="Use a matting directory directly, bypassing draft "
                             "lookup. Needs --video too.")
    parser.add_argument("--video", type=Path, default=None,
                        help="Source video, when using --masks.")
    parser.add_argument("--segment", type=int, default=0, metavar="N",
                        help="Which of the draft's cut-out clips to render (default 0).")
    parser.add_argument("--draft-root", type=Path, default=None,
                        help=f"Where CapCut keeps drafts (default: {capcut.DEFAULT_DRAFT_ROOT}).")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--output", type=Path, default=None,
                     help="Output path. Defaults to <source>_capcut_nobg.<ext>.")
    out.add_argument("-f", "--format", dest="fmt", default=None,
                     choices=sorted(formats.FORMATS),
                     help="Output format (default: mp4, or mov with --transparent).")
    out.add_argument("-b", "--background", "--bg", dest="background", default=None,
                     help="Colour, image, video, 'blur[:n]', or 'none'. "
                          "Same values as bgremove.")
    out.add_argument("-t", "--transparent", "--no-bg", dest="transparent",
                     action="store_true",
                     help="Real alpha channel; selects mov (ProRes 4444).")
    out.add_argument("--crf", type=int, default=18,
                     help="Encoder quality, lower is better (default: 18).")
    out.add_argument("--hwenc", action="store_true",
                     help="Encode H.264 on the Intel/AMD GPU.")

    mask = parser.add_argument_group("mask handling")
    mask.add_argument("--upsample", choices=["guided", "bicubic"], default="guided",
                      help="How to get from CapCut's 256x448 mask to full frame. "
                           "'guided' snaps the edge onto real image edges "
                           "(measured soft-edge 0.444%% against bicubic's 1.046%%); "
                           "'bicubic' reproduces CapCut's own softer edge.")
    mask.add_argument("--tighten", type=float, default=1.6, metavar="F",
                      help="Edge contrast restored after the guided filter "
                           "(default 1.6). Higher is crisper but erodes the "
                           "silhouette; 2.2 visibly thinned shoulders in testing.")
    mask.add_argument("--resolution", default="1080", metavar="N",
                      help="Cap the working size to N px on the short edge "
                           "(default 1080). 'source' keeps the input's own.")

    matte = parser.add_argument_group("matte refinement")
    matte.add_argument("--no-decontaminate", dest="decontaminate", action="store_false",
                       help="Keep the original edge colours. The masks are applied "
                            "to untouched source pixels, so edge pixels really do "
                            "contain the old background -- this is on by default.")
    matte.add_argument("--choke", type=float, default=None,
                       help="Shrink (negative) or grow (positive) the cutout, in "
                            "pixels. Defaults to the draft's own 'expansion'.")
    matte.add_argument("--feather", type=float, default=None,
                       help="Soften the edge, in pixels. Defaults to the draft's own.")
    matte.add_argument("--alpha-gamma", type=float, default=1.0, dest="gamma",
                       help="<1 firms up semi-transparent areas, >1 softens.")
    matte.add_argument("--levels", default=None, metavar="LOW,HIGH",
                       help="Remap the matte, e.g. '0.05,0.95'.")
    matte.add_argument("--denoise", action="store_true",
                       help="Median-filter the matte to remove speckle.")
    matte.add_argument("--temporal", type=float, default=0.0,
                       help="Blend with the previous frame (0-0.9). Rarely needed: "
                            "CapCut's masks come from a video model and are "
                            "already temporally stable.")
    matte.add_argument("--main-subject", type=int, nargs="?", const=1, default=0,
                       metavar="N", dest="main_subject",
                       help="Keep only the N largest subjects.")

    span = parser.add_argument_group("what to process")
    span.add_argument("--start", type=float, default=None,
                      help="Start time within the source, in seconds. Defaults to "
                           "the draft's own trim point.")
    span.add_argument("--duration", type=float, default=None,
                      help="How many seconds. Defaults to the draft's trim length.")
    span.add_argument("--full-source", action="store_true",
                      help="Ignore the draft's trim and process everything the "
                           "cached masks cover.")
    span.add_argument("--preview", type=float, nargs="?", const=10.0, default=None,
                      metavar="SECONDS", help="Process a short sample (default 10s).")
    span.add_argument("--no-audio", action="store_true", help="Drop the audio track.")

    misc = parser.add_argument_group("other")
    misc.add_argument("--list-drafts", action="store_true",
                      help="List CapCut drafts with cached cutouts, and exit.")
    misc.add_argument("--info", action="store_true",
                      help="Describe the draft's clips and masks, and exit.")
    misc.add_argument("--quiet", action="store_true", help="Suppress the progress bar.")
    misc.add_argument("--version", action="version", version=f"bgcapcut {__version__}")
    return parser


def _draft_label(mask_dir: Path) -> str:
    """Name the draft a mask directory belongs to, for reporting.

    Layout is <draft>/matting/<media-hash>/<level>, so the level's own name is
    a bare number and useless on its own.
    """
    parts = Path(mask_dir).resolve().parts
    if "matting" in parts:
        index = parts.index("matting")
        if index > 0:
            return parts[index - 1]
    return Path(mask_dir).name


def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def list_drafts(root: Path | None) -> int:
    """Every cached cutout on disk, linked to a timeline or not."""
    caches = capcut.find_mask_caches(root)
    if not caches:
        eprint(f"No cached CapCut cutouts found in {root or capcut.DEFAULT_DRAFT_ROOT}")
        eprint("  Apply 'Remove background' to a clip in CapCut and let it finish.")
        return 1

    eprint(f"Cached CapCut cutouts in {root or capcut.DEFAULT_DRAFT_ROOT}\n")
    eprint(f"  {'draft':<20}{'masks':>7}{'link':>7}   media (from CapCut's cache)")
    eprint("  " + "-" * 76)
    orphans = 0
    for cache in caches:
        media = cache.media or {}
        detail = "-"
        if media.get("width"):
            detail = (f"{media['width']}x{media['height']} @ {media['fps']:.2f}fps  "
                      f"{media['duration_ms'] / 1000:.1f}s  "
                      f"{media['size_bytes'] / 1e6:.0f} MB")
        state = "yes" if cache.linked else "ORPHAN"
        orphans += 0 if cache.linked else 1
        eprint(f"  {cache.draft:<20}{cache.n_masks:>7}{state:>7}   {detail}")

    eprint("\n  Render a linked one with:\n    python bgcapcut.py <draft>")
    if orphans:
        eprint("\n  ORPHAN means the masks are still on disk but no timeline references")
        eprint("  them any more -- clearing or re-cutting a project in CapCut rewrites")
        eprint("  draft_content.json and drops the reference, leaving the masks intact.")
        eprint("  The source path is dropped with it, so pass the video yourself:")
        eprint("    python bgcapcut.py --masks <mask-dir> --video <source> --transparent")
        eprint("  Use the size and dimensions above to identify the right source file.")
        eprint("  Mask directories:")
        for cache in caches:
            if not cache.linked:
                eprint(f"    {cache.mask_dir}")
    return 0


def show_info(draft: capcut.Draft) -> int:
    eprint(f"draft       {draft.name}")
    eprint(f"  path      {draft.path}")
    eprint(f"  canvas    {draft.canvas_width}x{draft.canvas_height} "
           f"@ {draft.timeline_fps:g} fps (timeline)")
    if not draft.clips:
        eprint("\n  No clip in this draft has a cached cutout.")
        eprint("  Apply 'Remove background' to a clip in CapCut, let it finish, "
               "then save.")
        return 1
    for clip in draft.clips:
        masks = capcut.MaskSequence(clip.mask_dir)
        first, last = masks.covered_us
        eprint(f"\n  clip {clip.index}")
        eprint(f"    source     {clip.source}")
        eprint(f"               {'exists' if clip.source.exists() else 'MISSING'}"
               f"  {clip.width}x{clip.height}")
        eprint(f"    trim       {clip.start_s:.3f}s for {clip.duration_s:.3f}s"
               f"  (speed {clip.speed:g})")
        eprint(f"    masks      {len(masks)} x {masks.width}x{masks.height}"
               f"  covering {first / 1e6:.3f}-{last / 1e6:.3f}s of the source")
        m = clip.matting
        eprint(f"    matting    feather {m.feather:g}  expansion {m.expansion:g}"
               f"  stroke {m.stroke}  reverse {m.reverse}"
               + ("  (plain alpha)" if m.is_plain else ""))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        require_ffmpeg()
    except FFmpegMissing as exc:
        eprint(f"error: {exc}")
        return 1

    if args.list_drafts:
        return list_drafts(args.draft_root)

    # ---- locate the clip and its masks ------------------------------------
    if args.masks:
        if not args.video:
            eprint("error: --masks also needs --video (the source it was cut from).")
            return 1
        try:
            masks = capcut.MaskSequence(args.masks)
        except capcut.DraftError as exc:
            eprint(f"error: {exc}")
            return 1
        first, last = masks.covered_us
        clip = capcut.Clip(
            index=0, source=args.video, mask_dir=args.masks,
            source_start_us=first, source_duration_us=max(last - first, 1),
            speed=1.0, width=0, height=0, matting=capcut.MattingParams(),
        )
        draft = capcut.Draft(name=_draft_label(args.masks), path=args.masks,
                             canvas_width=0, canvas_height=0, timeline_fps=0.0,
                             clips=[clip])
    else:
        if not args.draft:
            parser.print_help()
            eprint("\nStart with:  python bgcapcut.py --list-drafts")
            return 1
        try:
            draft = capcut.read_draft(capcut.resolve_draft(args.draft, args.draft_root))
        except capcut.DraftError as exc:
            eprint(f"error: {exc}")
            return 1
        if args.info:
            try:
                return show_info(draft)
            except capcut.DraftError as exc:
                eprint(f"error: {exc}")
                return 1
        if not draft.clips:
            # The timeline no longer cites a cutout, but CapCut leaves the masks
            # on disk when a project is cleared or re-cut. If they are still
            # there the job is perfectly doable -- only the source path is lost,
            # because that goes with the timeline.
            orphans = [c for c in capcut.find_mask_caches(args.draft_root)
                       if c.draft_dir.resolve() == draft.path.resolve()]
            if not orphans:
                eprint(f"error: no clip in draft {draft.name!r} has a cached cutout.")
                eprint("       Apply 'Remove background' to the clip in CapCut, let it")
                eprint("       finish, and save the project. Then try again.")
                return 1
            cache = max(orphans, key=lambda c: c.n_masks)
            if not args.video:
                media = cache.media or {}
                eprint(f"error: draft {draft.name!r} has {cache.n_masks} cached masks, but its")
                eprint("       timeline no longer references them -- so the source video path")
                eprint("       is gone too. Point at the source yourself:")
                eprint(f"\n         python bgcapcut.py {args.draft} --video <source>"
                       f" --transparent\n")
                if media.get("width"):
                    eprint(f"       The media was {media['width']}x{media['height']} @ "
                           f"{media['fps']:.2f}fps, {media['duration_ms'] / 1000:.1f}s, "
                           f"{media['size_bytes'] / 1e6:.0f} MB")
                    eprint("       (from CapCut's import cache -- use it to identify the file).")
                eprint(f"       Masks: {cache.mask_dir}")
                return 1
            try:
                masks_probe = capcut.MaskSequence(cache.mask_dir)
            except capcut.DraftError as exc:
                eprint(f"error: {exc}")
                return 1
            first, last = masks_probe.covered_us
            eprint(f"note: draft {draft.name!r} has an emptied timeline; using its "
                   f"orphaned mask cache")
            eprint(f"      ({cache.n_masks} masks) with the source you supplied.\n")
            draft.clips = [capcut.Clip(
                index=0, source=args.video, mask_dir=cache.mask_dir,
                source_start_us=first, source_duration_us=max(last - first, 1),
                speed=1.0, width=0, height=0, matting=capcut.MattingParams(),
            )]
        if not 0 <= args.segment < len(draft.clips):
            eprint(f"error: --segment {args.segment} is out of range; this draft has "
                   f"{len(draft.clips)} cut-out clip(s).")
            return 1
        clip = draft.clips[args.segment]
        try:
            masks = capcut.MaskSequence(clip.mask_dir)
        except capcut.DraftError as exc:
            eprint(f"error: {exc}")
            return 1

    if not clip.source.exists():
        eprint(f"error: the draft's source video is missing:\n         {clip.source}")
        eprint("       CapCut stores this as an absolute path, so moving or renaming")
        eprint("       the media breaks the link. Pass --masks and --video to point")
        eprint("       at it directly.")
        return 1

    # ---- format and background, same rules as bgremove --------------------
    if args.transparent and args.background is not None:
        eprint("error: --transparent and --background ask for opposite things.")
        return 1
    if args.fmt is None:
        args.fmt = "mov" if args.transparent else "mp4"
    fmt = formats.FORMATS[args.fmt]

    if args.transparent and not fmt.supports_alpha:
        eprint(f"error: --transparent needs a format that can store alpha, and "
               f"{args.fmt} cannot.")
        eprint(f"       Use --format {' / '.join(formats.ALPHA_FORMATS)}.")
        return 1

    background_spec = "none" if args.transparent else args.background
    if background_spec is None:
        background_spec = "none" if fmt.supports_alpha else "black"
    transparent_out = background_spec.lower() in ("none", "transparent", "alpha")
    if transparent_out and not fmt.supports_alpha:
        eprint(f"error: {args.fmt} cannot store transparency.")
        return 1

    if transparent_out and not capcut.encoder_keeps_alpha(
            fmt.encoder_args(args.crf, args.hwenc), fmt.suffix, fmt.is_sequence):
        eprint(f"error: this ffmpeg build encodes {args.fmt} without a usable alpha "
               f"channel.")
        working = [n for n in formats.ALPHA_FORMATS
                   if n != args.fmt and capcut.encoder_keeps_alpha(
                       formats.FORMATS[n].encoder_args(args.crf, args.hwenc),
                       formats.FORMATS[n].suffix, formats.FORMATS[n].is_sequence)]
        if working:
            eprint(f"       Formats that keep alpha here: {' / '.join(working)}")
        return 1

    # ---- geometry and span -------------------------------------------------
    try:
        info = probe(clip.source)
    except Exception as exc:
        eprint(f"error: {exc}")
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
    width, height = capped_size(info.width, info.height, cap)

    src_fps = float(info.fps)
    covered_first, covered_last = masks.covered_us

    if args.full_source:
        start_us, duration_us = covered_first, max(covered_last - covered_first, 1)
    else:
        start_us = clip.source_start_us
        duration_us = clip.source_duration_us
    if args.start is not None:
        start_us = int(round(args.start * 1e6))
    if args.duration is not None:
        duration_us = int(round(args.duration * 1e6))
    if args.preview is not None:
        duration_us = min(duration_us, int(round(args.preview * 1e6)))
    if duration_us <= 0:
        eprint("error: the selected range contains no frames.")
        return 1

    speed = clip.speed if clip.speed > 0 else 1.0
    out_fps = info.fps                        # source-native: masks are 1:1 with it
    n_frames = max(1, math.ceil(duration_us / 1e6 / speed * src_fps))

    # ---- resolve draft-derived refinement defaults ------------------------
    choke = clip.matting.expansion if args.choke is None else args.choke
    feather = clip.matting.feather if args.feather is None else args.feather
    levels_low, levels_high = 0.0, 1.0
    if args.levels:
        try:
            low_text, high_text = args.levels.split(",")
            levels_low, levels_high = float(low_text), float(high_text)
        except ValueError:
            eprint("error: --levels wants two numbers, e.g. --levels 0.05,0.95")
            return 1

    # ---- output path -------------------------------------------------------
    if args.output:
        output = Path(args.output)
    else:
        parts = [clip.source.stem, "capcut"]
        if fmt.name in ("matte", "stacked"):
            parts.append(fmt.name)
        elif args.preview is None:
            parts.append("nobg")
        if args.preview is not None:
            parts.append("preview")
        output = clip.source.with_name("_".join(parts) + fmt.suffix)
    if fmt.is_sequence:
        directory = output if output.suffix == "" else output.with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "%05d.png"
    elif output.resolve() == clip.source.resolve():
        eprint("error: the output would overwrite the source. Pass -o.")
        return 1

    # ---- report ------------------------------------------------------------
    eprint(f"bgcapcut {__version__}   (CapCut's cached cutout; no inference)")
    eprint(f"  draft       {draft.name}  clip {clip.index}")
    eprint(f"  source      {clip.source.name}  {info.width}x{info.height} "
           f"@ {src_fps:.3f}fps")
    eprint(f"  masks       {len(masks)} x {masks.width}x{masks.height}  "
           f"({width / masks.width:.2f}x upscale to {width}x{height})")
    eprint(f"  processing  {n_frames} frames ({_clock(duration_us / 1e6)}) "
           f"from {start_us / 1e6:.3f}s")
    eprint(f"  upsample    {args.upsample}"
           + (f"  tighten {args.tighten:g}" if args.upsample == "guided" else ""))
    eprint(f"  output      {args.fmt} -> {output}")
    eprint(f"  background  {background_spec}"
           + ("  (real alpha channel)" if transparent_out else ""))
    if choke or feather:
        eprint(f"  matte       choke {choke:g}  feather {feather:g}"
               + ("  (from the draft)" if args.choke is None and args.feather is None
                  else ""))
    if clip.matting.reverse:
        eprint("  matting     draft says reverse -- alpha inverted")

    # A request outside what CapCut cached would silently composite a held mask.
    end_us = start_us + duration_us
    if start_us < covered_first - 50_000 or end_us > covered_last + 50_000:
        eprint(f"  WARNING     masks only cover {covered_first / 1e6:.3f}-"
               f"{covered_last / 1e6:.3f}s of the source; outside that the "
               f"nearest mask is held.")
    eprint()

    # ---- run ---------------------------------------------------------------
    background = compositing.build_background(background_spec, width, height)
    smoother = compositing.TemporalSmoother(args.temporal)
    scale_to = (width, height) if (width, height) != (info.width, info.height) else None

    reader = FrameReader(
        clip.source, info.width, info.height,
        start=start_us / 1e6, duration=(n_frames + 2) / src_fps * speed + 1.0,
        scale_to=scale_to,
    )
    audio_source = clip.source if (not args.no_audio and info.has_audio and fmt.audio) else None
    encoder_args = fmt.encoder_args(args.crf, args.hwenc)
    if fmt.is_sequence:
        encoder_args = encoder_args + ["-start_number", "0"]

    writer = FrameWriter(
        output, width, height * 2 if args.fmt == "stacked" else height, out_fps,
        pix_fmt_in=fmt.pix_fmt_in, encoder_args=encoder_args,
        audio_from=audio_source,
        audio_args=formats.audio_args_for(fmt, info.audio_codec) if audio_source else None,
        audio_start=start_us / 1e6 or None,
        audio_duration=duration_us / 1e6,
        atomic=not fmt.is_sequence,
    )

    bar = None if args.quiet else Progress(n_frames, label="compositing")
    started = time.perf_counter()
    written = 0
    frame: np.ndarray | None = None
    src_index = -1

    try:
        for k in range(n_frames):
            # Which source frame this output frame draws from. Handles a speed
            # change and any fps mismatch by advancing (or holding) the decoder.
            want = int(round(k * speed * src_fps / float(out_fps)))
            while src_index < want:
                raw = reader.read()
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                src_index += 1
            if frame is None:
                break

            source_us = int(round(start_us + k * speed * 1e6 / float(out_fps)))
            alpha = capcut.alpha_for_frame(
                masks, source_us, frame, mode=args.upsample,
                tighten=args.tighten, reverse=clip.matting.reverse)

            alpha = compositing.refine_alpha(
                alpha, choke=choke, feather=feather, gamma=args.gamma,
                low=levels_low, high=levels_high, denoise=args.denoise,
                main_subject=args.main_subject)
            alpha = smoother(alpha)

            foreground = frame.astype(np.float32) / 255.0
            if args.decontaminate and args.fmt != "matte":
                foreground = compositing.decontaminate(foreground, alpha)

            if args.fmt == "matte":
                out_frame = compositing.to_matte(alpha)
            elif args.fmt == "stacked":
                colour = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)
                out_frame = np.vstack([colour, compositing.to_matte(alpha)])
            elif background.is_transparent:
                out_frame = compositing.to_rgba(foreground, alpha)
            else:
                plate = background.frame(written, frame, alpha)
                rgb = compositing.composite(foreground, alpha, plate)
                out_frame = (np.dstack([rgb, np.full(alpha.shape, 255, np.uint8)])
                             if fmt.pix_fmt_in == "rgba" else rgb)

            writer.write(out_frame.tobytes())
            written += 1
            if bar:
                bar.advance()
    except KeyboardInterrupt:
        reader.close()
        background.close()
        if written:
            kept = writer.close_partial()
            eprint(f"\ninterrupted after {written} of {n_frames} frames.")
            eprint(f"  The finished part was kept and is playable:  {kept}")
            return 130
        writer.abort()
        eprint("\ninterrupted before any frame was encoded; nothing written.")
        return 130
    except Exception as exc:
        reader.close()
        background.close()
        writer.abort()
        eprint(f"\nerror: {exc}")
        return 1

    reader.close()
    background.close()
    try:
        final = writer.close()
    except RuntimeError as exc:
        eprint(f"\nerror: {exc}")
        return 1
    if bar:
        bar.finish()

    if not fmt.is_sequence:
        try:
            verify_playable(final)
        except RuntimeError as exc:
            eprint(f"\nerror: {exc}")
            return 1

    elapsed = time.perf_counter() - started
    eprint(f"\nDone in {_clock(elapsed)}  ->  "
           f"{output.parent if fmt.is_sequence else final}")
    if not fmt.is_sequence:
        try:
            eprint(f"  {final.stat().st_size / 1e6:.1f} MB  "
                   f"{written} frames  {written / max(elapsed, 1e-6):.1f} fps")
        except OSError:
            pass
    if args.preview is not None:
        eprint("  This was a preview. Drop --preview for the whole clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
