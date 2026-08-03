#!/usr/bin/env python3
"""
Local video background remover for talking-head / webcam footage.

Produces a TRANSPARENT (alpha) result using Robust Video Matting (RVM):
  https://github.com/PeterL1n/RobustVideoMatting

Everything runs on your machine. No uploads, no cloud, no account.

Output formats:
  png   RGBA PNG sequence            (lossless, true alpha; most compatible)
  mov   Apple ProRes 4444 .mov       (alpha, great for Premiere/Resolve/FCP)
  webm  VP9 .webm with alpha         (for the web / browsers)

The PNG sequence is always the source of truth for the alpha channel; the
mov/webm formats are transcoded from it losslessly with ffmpeg.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def detect_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def probe_fps(input_path: Path) -> str:
    """Return the source frame rate as an ffmpeg-friendly string (e.g. '30000/1001')."""
    if shutil.which("ffprobe") is None:
        return "30"
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "0", "-of", "csv=p=0",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                str(input_path),
            ],
            text=True,
        ).strip()
        return out or "30"
    except Exception:
        return "30"


def run_matting(model, convert_video, input_path: Path, png_dir: Path,
                device: str, seq_chunk: int, downsample):
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    png_dir.mkdir(parents=True, exist_ok=True)
    convert_video(
        model,
        input_source=str(input_path),
        output_type="png_sequence",
        output_composition=str(png_dir),   # writes RGBA frames with true alpha
        downsample_ratio=downsample,       # None => RVM auto-picks per resolution
        seq_chunk=seq_chunk,
        progress=True,
        device=device,
        dtype=dtype,
    )


def encode_mov(png_dir: Path, out_path: Path, fps: str):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", fps,
        "-start_number", "0",
        "-i", str(png_dir / "%04d.png"),
        "-c:v", "prores_ks", "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le", "-alpha_bits", "16",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def encode_webm(png_dir: Path, out_path: Path, fps: str):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", fps,
        "-start_number", "0",
        "-i", str(png_dir / "%04d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-b:v", "0", "-crf", "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Remove a person's background from a video, locally, with a transparent result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the input video.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (png => a folder; mov/webm => a file). "
                             "Defaults to '<input>_nobg.<ext>'.")
    parser.add_argument("--format", choices=["png", "mov", "webm"], default="mov",
                        help="Output container. png=RGBA frames, mov=ProRes4444, webm=VP9+alpha.")
    parser.add_argument("--model", choices=["resnet50", "mobilenetv3"], default="resnet50",
                        help="resnet50 = best quality, mobilenetv3 = faster / lighter.")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--seq-chunk", type=int, default=None,
                        help="Frames processed at once. Higher=faster, more memory. "
                             "Default: 12 on GPU, 4 on CPU/MPS.")
    parser.add_argument("--downsample", type=float, default=None,
                        help="0.0-1.0 internal matting resolution. Leave unset for auto. "
                             "Tips: 1080p~0.25, 4K~0.125, SD~0.4.")
    parser.add_argument("--keep-frames", action="store_true",
                        help="For mov/webm, keep the intermediate PNG frames instead of deleting them.")
    args = parser.parse_args()

    if not args.input.exists():
        eprint(f"error: input not found: {args.input}")
        sys.exit(1)

    if args.format in ("mov", "webm") and shutil.which("ffmpeg") is None:
        eprint("error: ffmpeg is required for mov/webm output but was not found on PATH.")
        eprint("       Install ffmpeg, or use '--format png' which needs no ffmpeg.")
        sys.exit(1)

    try:
        import torch  # noqa: F401
    except ImportError:
        eprint("error: PyTorch is not installed. Run:  pip install -r requirements.txt")
        sys.exit(1)

    device = detect_device(args.device)
    seq_chunk = args.seq_chunk if args.seq_chunk is not None else (12 if device == "cuda" else 4)
    eprint(f"Device: {device}   Model: {args.model}   seq_chunk: {seq_chunk}")

    # Resolve output path
    stem = args.input.stem
    if args.format == "png":
        out = Path(args.output or args.input.with_name(f"{stem}_nobg_frames"))
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = Path(args.output or args.input.with_name(f"{stem}_nobg.{args.format}"))
        out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    eprint("Loading RVM (first run downloads model weights from GitHub)...")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", args.model)
    model = model.eval().to(device)
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    if args.format == "png":
        eprint(f"Writing RGBA frames to: {out}")
        run_matting(model, convert_video, args.input, out, device, seq_chunk, args.downsample)
        eprint("Done. Transparent RGBA PNG sequence is ready.")
        return

    # mov / webm: matte to a temp PNG dir, then transcode with ffmpeg
    fps = probe_fps(args.input)
    tmp_dir = Path(tempfile.mkdtemp(prefix="rvm_frames_"))
    try:
        eprint("Matting frames...")
        run_matting(model, convert_video, args.input, tmp_dir, device, seq_chunk, args.downsample)
        eprint(f"Encoding {args.format} at {fps} fps -> {out}")
        if args.format == "mov":
            encode_mov(tmp_dir, out, fps)
        else:
            encode_webm(tmp_dir, out, fps)
        eprint(f"Done. Transparent {args.format.upper()} written to: {out}")
    finally:
        if args.keep_frames:
            kept = args.input.with_name(f"{stem}_nobg_frames")
            shutil.move(str(tmp_dir), str(kept))
            eprint(f"Kept intermediate frames at: {kept}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()