"""ffmpeg-backed video I/O.

Frames move through pipes as raw bytes, so nothing is ever written to disk as an
intermediate image sequence. That alone is a large speedup over decode-to-PNG
workflows, and it keeps memory flat regardless of clip length.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# Keep console windows from flashing open on Windows for every child process.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class FFmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegMissing(
            f"{' and '.join(missing)} not found on PATH.\n"
            "  Install with:  winget install Gyan.FFmpeg\n"
            "  (then open a new terminal so PATH is refreshed)"
        )


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    n_frames: int
    has_audio: bool
    pix_fmt: str
    audio_codec: str | None = None

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def probe(path: Path) -> VideoInfo:
    """Read stream metadata. Falls back gracefully when fields are absent."""
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, creationflags=_NO_WINDOW)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe could not read {path}") from exc

    data = json.loads(raw)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_audio = audio is not None

    width, height = int(video["width"]), int(video["height"])

    # Phone footage is stored in one orientation with a Display Matrix telling
    # players to rotate it. ffmpeg applies that rotation when decoding, so the
    # frames arriving on the pipe are already in display orientation -- and for
    # a quarter turn the byte count is identical, so a mismatch here reshapes
    # the pixels into noise instead of raising. Report display dimensions.
    rotation = 0
    for side_data in video.get("side_data_list", []):
        try:
            rotation = int(float(side_data.get("rotation", 0)))
        except (TypeError, ValueError):
            pass
    if rotation % 180 == 90:
        width, height = height, width

    fps_raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"
    if fps_raw in ("0/0", "0"):
        fps_raw = video.get("r_frame_rate") or "30/1"
    fps = Fraction(fps_raw)
    if fps <= 0:
        fps = Fraction(30, 1)

    duration = 0.0
    for source in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue

    n_frames = 0
    try:
        n_frames = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        pass
    if n_frames <= 0 and duration:
        n_frames = int(round(duration * float(fps)))

    return VideoInfo(
        path=path, width=width, height=height, fps=fps, duration=duration,
        n_frames=n_frames, has_audio=has_audio, pix_fmt=video.get("pix_fmt", "yuv420p"),
        audio_codec=(audio or {}).get("codec_name"),
    )


def capped_size(width: int, height: int, short_edge: int | None) -> tuple[int, int]:
    """Scale WxH down until its shorter side is at most `short_edge`.

    Phone video is portrait and desktop video is landscape, but "1080p" means
    the short side either way, so the cap is expressed on the short side rather
    than on width or height. Dimensions are forced even, because several
    encoders reject odd ones.
    """
    if not short_edge:
        return width, height
    short = min(width, height)
    if short <= short_edge:
        return width, height
    scale = short_edge / short
    return (max(2, int(round(width * scale)) // 2 * 2),
            max(2, int(round(height * scale)) // 2 * 2))


class FrameReader:
    """Decode a clip to raw RGB24 frames on stdout."""

    def __init__(self, path: Path, width: int, height: int, *,
                 start: float | None = None, duration: float | None = None,
                 loop: bool = False, scale_to: tuple[int, int] | None = None,
                 threads: int | None = None):
        self.width, self.height = (scale_to or (width, height))
        self.frame_bytes = self.width * self.height * 3

        cmd = ["ffmpeg", "-v", "error", "-nostdin"]
        if threads:
            cmd += ["-threads", str(threads)]
        if loop:
            cmd += ["-stream_loop", "-1"]
        # -ss before -i is the fast (keyframe) seek; accurate enough here because
        # we always decode a warm-up run-up before the frames we actually keep.
        if start:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", str(path)]
        if duration:
            cmd += ["-t", f"{duration:.6f}"]
        if scale_to:
            cmd += ["-vf", f"scale={self.width}:{self.height}:flags=bicubic"]
        cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 4, creationflags=_NO_WINDOW,
        )

    def read(self) -> bytes | None:
        """Return one frame's raw bytes, or None at end of stream."""
        buf = self.proc.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return None
        return buf

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # Drain stderr so the pipe buffer never blocks the child.
        try:
            self.proc.stderr.read()
        except (OSError, ValueError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FrameWriter:
    """Encode raw frames arriving on stdin, optionally muxing audio from a source clip."""

    def __init__(self, out_path: Path, width: int, height: int, fps: Fraction, *,
                 pix_fmt_in: str = "rgb24", encoder_args: list[str],
                 audio_from: Path | None = None, audio_args: list[str] | None = None,
                 audio_start: float | None = None, audio_duration: float | None = None,
                 atomic: bool = True):
        self.frame_bytes = width * height * (4 if pix_fmt_in == "rgba" else 3)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # An MP4's index -- the moov atom -- is only written once ffmpeg exits
        # cleanly, so a run that is interrupted or killed leaves a pile of
        # frames that no player will open. Encoding to a sibling temp file and
        # moving it into place only on success means the destination holds
        # either a finished video or nothing, never a corpse that looks done.
        # The real suffix has to stay last: ffmpeg picks the muxer from it.
        self.final_path = out_path
        self.temp_path = (out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
                          if atomic else out_path)
        if self.temp_path != self.final_path:
            self.temp_path.unlink(missing_ok=True)

        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", pix_fmt_in,
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        ]
        if audio_from is not None:
            if audio_start:
                cmd += ["-ss", f"{audio_start:.6f}"]
            cmd += ["-i", str(audio_from)]
            if audio_duration:
                cmd += ["-t", f"{audio_duration:.6f}"]
            cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
            cmd += (audio_args or ["-c:a", "copy"])
        else:
            cmd += ["-map", "0:v:0", "-an"]

        cmd += encoder_args + [str(self.temp_path)]
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 4, creationflags=_NO_WINDOW,
        )

    def write(self, data: bytes) -> None:
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            err = b""
            try:
                err = self.proc.stderr.read()
            except (OSError, ValueError):
                pass
            raise RuntimeError(
                "ffmpeg stopped accepting frames:\n" + err.decode("utf-8", "replace")
            ) from exc

    def _finalize(self) -> None:
        """Close stdin and wait, so ffmpeg gets to write its trailer/index."""
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        err = b""
        try:
            err = self.proc.stderr.read()
        except (OSError, ValueError):
            pass
        code = self.proc.wait()
        if code != 0:
            self._discard()
            raise RuntimeError(
                f"ffmpeg encoding failed (exit {code}):\n"
                + err.decode("utf-8", "replace")
                + "\ncommand: " + " ".join(self.cmd)
            )

    def _commit(self, destination: Path) -> Path:
        if self.temp_path == self.final_path:
            return self.final_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temp_path, destination)   # atomic within one filesystem
        return destination

    def _discard(self) -> None:
        if self.temp_path != self.final_path:
            try:
                self.temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> Path:
        """Flush, wait for the encoder, and move the result into place."""
        self._finalize()
        return self._commit(self.final_path)

    def close_partial(self) -> Path:
        """Finalize after an interrupt, keeping the frames encoded so far.

        Killing ffmpeg here would throw away a perfectly good clip, so let it
        write its index and land the result beside the intended output under a
        name that cannot be mistaken for a complete run.
        """
        self._finalize()
        stem = self.final_path.stem
        return self._commit(self.final_path.with_name(f"{stem}_partial{self.final_path.suffix}"))

    def abort(self) -> None:
        """Tear the encoder down without reporting its exit status."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._discard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is not None:
            # Something upstream blew up; don't mask it with an encoder error.
            self.abort()
            return
        self.close()


def concat(segments: list[Path], out_path: Path, *, audio_from: Path | None = None,
           audio_args: list[str] | None = None, list_file: Path | None = None,
           extra_out_args: list[str] | None = None) -> None:
    """Join pre-encoded segments without re-encoding the video."""
    list_file = list_file or out_path.with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as fh:
        for seg in segments:
            escaped = str(seg.resolve()).replace("\\", "/").replace("'", r"'\''")
            fh.write(f"file '{escaped}'\n")

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
           "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if audio_from is not None:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
        cmd += (audio_args or ["-c:a", "copy"])
    else:
        cmd += ["-map", "0:v:0", "-an"]
    # Same reasoning as FrameWriter: never let a failed join leave something
    # unplayable sitting at the name the user is going to double-click.
    temp_out = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
    temp_out.unlink(missing_ok=True)
    cmd += ["-c:v", "copy"] + (extra_out_args or []) + [str(temp_out)]

    result = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        temp_out.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg could not join the segments:\n"
            + result.stderr.decode("utf-8", "replace")
        )
    os.replace(temp_out, out_path)


_ALPHA_SUPPORT: dict[tuple, bool] = {}


def encoder_keeps_alpha(encoder_args: list[str], suffix: str,
                        is_sequence: bool = False) -> bool:
    """Encode two tiny frames carrying a known alpha ramp and see if it survives.

    A full run here is measured in tens of minutes, so spending a third of a
    second to find out first is a bargain -- and which codecs carry alpha varies
    by build, which is why this asks the local ffmpeg rather than assuming.

    The read-back has to name the decoder. VP9 keeps alpha in a side-channel that
    ffmpeg's *native* `vp9` decoder does not surface, so decoding without saying
    which decoder to use hands back a solid 255 plane for a file whose alpha is
    perfectly intact -- and this function then blamed the encoder for it. That
    false negative is why WebM was believed to be broken and documented as such;
    asking libvpx-vp9 explicitly shows alpha spanning 0..255. Verified against
    real output:

        ffmpeg -c:v libvpx-vp9 -i out.webm -vf alphaextract -frames:v 1 a.png

    So each candidate decoder is tried and the format passes if any of them
    recovers the ramp. A genuinely alpha-less encode fails all of them.
    """
    key = (tuple(encoder_args), suffix, is_sequence)
    if key in _ALPHA_SUPPORT:
        return _ALPHA_SUPPORT[key]

    import numpy as np

    width, height = 64, 32
    ramp = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, 0)
    rgba = np.dstack([np.full((height, width), 200, np.uint8),
                      np.full((height, width), 100, np.uint8),
                      np.full((height, width), 50, np.uint8), ramp])
    payload = rgba.tobytes() * 2
    # An image sequence would need a %0Nd pattern to take more than one frame,
    # so cap it there; one frame proves the point for a still format anyway.
    limit = ["-frames:v", "1"] if is_sequence else []

    tmp = Path(tempfile.gettempdir()) / f"bgremove_alphaprobe_{os.getpid()}{suffix}"
    try:
        encode = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
             "-r", "12", "-i", "-", "-an"] + encoder_args + limit + [str(tmp)],
            input=payload, capture_output=True, creationflags=_NO_WINDOW)
        if encode.returncode != 0 or not tmp.exists():
            return _ALPHA_SUPPORT.setdefault(key, False)

        # The container's real decoder first, then whatever ffmpeg would pick.
        decoders: list[list[str]] = [[]]
        if suffix == ".webm":
            decoders.insert(0, ["-c:v", "libvpx-vp9"])

        for decoder in decoders:
            decode = subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin"] + decoder
                + ["-i", str(tmp), "-frames:v", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                capture_output=True, creationflags=_NO_WINDOW)
            raw = decode.stdout
            if len(raw) < width * height * 4:
                continue
            alpha = np.frombuffer(raw[:width * height * 4], np.uint8) \
                      .reshape(height, width, 4)[..., 3]
            # A codec that dropped alpha hands back a solid 255 plane.
            if alpha.min() < 16 and alpha.max() > 239 and len(np.unique(alpha)) > 8:
                return _ALPHA_SUPPORT.setdefault(key, True)
        return _ALPHA_SUPPORT.setdefault(key, False)
    except (OSError, subprocess.SubprocessError):
        return _ALPHA_SUPPORT.setdefault(key, False)
    finally:
        tmp.unlink(missing_ok=True)


def verify_playable(path: Path) -> int:
    """Confirm a finished file actually decodes. Returns its frame count.

    Cheap insurance against reporting success over a file that no player will
    open -- an MP4 missing its moov atom looks fine on disk and fails silently
    at the only moment anyone cares.
    """
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-count_packets", "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_packets", str(path)]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE,
                                      creationflags=_NO_WINDOW)
        streams = json.loads(raw).get("streams", [])
        frames = int(streams[0]["nb_read_packets"]) if streams else 0
    except (subprocess.CalledProcessError, ValueError, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"the encoder produced a file that will not decode: {path}"
        ) from exc
    if frames <= 0:
        raise RuntimeError(f"the encoder produced a file with no video frames: {path}")
    return frames
