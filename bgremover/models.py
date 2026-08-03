"""Model weights: locating, downloading and caching."""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_BASE = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"

# name -> (filename, url, approximate size in bytes for progress display)
MODELS = {
    "mobilenetv3": ("rvm_mobilenetv3_fp32.onnx", f"{_BASE}/rvm_mobilenetv3_fp32.onnx", 14_326_000),
    "resnet50": ("rvm_resnet50_fp32.onnx", f"{_BASE}/rvm_resnet50_fp32.onnx", 102_500_000),
}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _download(url: str, dest: Path, expected: int) -> None:
    """Download to a temp file then move into place, so a partial file is never used."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"  downloading {dest.name} ({_human(expected)}) ...", file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bgremover/2.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or expected)
            done = 0
            last_pct = -5
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                pct = int(done * 100 / total) if total else 0
                if pct >= last_pct + 5:
                    last_pct = pct
                    bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
                    print(f"\r  [{bar}] {pct:3d}%  {_human(done)}", end="", file=sys.stderr)
        print(file=sys.stderr)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the model from {url}\n"
            f"  reason: {exc}\n"
            f"  You can also download it manually and save it to: {dest}"
        ) from exc

    shutil.move(str(tmp), str(dest))


def resolve(name: str, model_dir: Path | None = None) -> Path:
    """Return a local path to the requested model, downloading it on first use.

    `name` may also be a path to a user-supplied .onnx file.
    """
    candidate = Path(name)
    if candidate.suffix == ".onnx":
        if not candidate.exists():
            raise FileNotFoundError(f"Model file not found: {candidate}")
        return candidate

    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}")

    filename, url, size = MODELS[name]
    directory = model_dir or MODEL_DIR
    path = directory / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        return path

    _download(url, path, size)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
