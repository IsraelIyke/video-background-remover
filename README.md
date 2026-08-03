# bgremove

Removes the background from a video, entirely on your own machine. No uploads,
no account, no watermark, no per-minute pricing.

Built on [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
(RVM) running through ONNX Runtime.

```
python bgremove.py Practice.mp4
```

That writes `Practice_nobg.mp4` — H.264, audio preserved, subject on a black
background.

**[USAGE.md](USAGE.md) is the complete reference** — every flag, measured speeds
and file sizes for this machine, recipes, and which formats can actually carry
an alpha channel on your ffmpeg build.

---

## Why the previous script didn't work

Three separate problems, all fixed here:

| Problem | Cause | Fix |
| --- | --- | --- |
| The `.mov` wouldn't open | It was **ProRes 4444**, which nothing on Windows plays by default — not Media Player, not Films & TV. The file was fine; Windows just can't decode it. | MP4/H.264 is now the default output. |
| Painfully slow | It defaulted to the **ResNet-50** model in PyTorch, then wrote **6,571 PNG files** to disk and re-read them to encode. On this laptop that is several hours of work. | MobileNetV3 on ONNX Runtime, frames streamed through pipes, nothing touching disk. |
| Audio disappeared | A PNG sequence carries no audio, and nothing re-attached it. | Audio is copied from the source automatically. |

Measured on this machine (Intel i5-5200U, 2 cores, no usable GPU), for the
3m39s / 6,571-frame `Practice.mp4`:

| | Time |
| --- | --- |
| Old script (ResNet-50 + PNG round-trip) | several hours |
| `bgremove` default | **~20–25 minutes** |

The old `background_remover.py` is left in place but is fully superseded — you
can delete it.

---

## Installing

```bash
pip install -r requirements.txt
winget install Gyan.FFmpeg      # ffmpeg is required and is not a pip package
```

The matting model (14 MB) downloads automatically the first time you run it and
is cached in `models/`.

---

## MP4 and transparency — read this once

**MP4 cannot store transparency.** That is a limitation of the format, not of
this tool. H.264 has no usable alpha channel; Apple's HEVC-with-alpha only plays
on Apple hardware and ffmpeg cannot even encode it.

So an MP4 always needs *something* behind the subject. You have four ways to
work with that:

**1. Replace the background** (most common — plays everywhere)

```bash
python bgremove.py Practice.mp4 --background "#101820"      # any colour
python bgremove.py Practice.mp4 --background office.jpg     # a photo
python bgremove.py Practice.mp4 --background beach.mp4      # a video, looped
python bgremove.py Practice.mp4 --background blur           # the real background, defocused
python bgremove.py Practice.mp4 --background greenscreen    # chroma green, to key later
```

**2. Real transparency in a format that supports it**

```bash
python bgremove.py Practice.mp4 --format webm   # VP9 + alpha — browsers, most editors
python bgremove.py Practice.mp4 --format mov    # ProRes 4444 — Premiere / Resolve / FCP
python bgremove.py Practice.mp4 --format png    # RGBA frames — imports into anything
python bgremove.py Practice.mp4 --format mkv    # lossless FFV1 + alpha
```

**3. Carry the matte through an MP4**

```bash
python bgremove.py Practice.mp4 --format matte     # black & white matte on its own
python bgremove.py Practice.mp4 --format stacked   # colour on top, matte below
```

`matte` is what you feed a luma key or track matte in any editor. `stacked` is
the usual trick for shaders, Unity, and `<canvas>`.

> Note on WebM: `ffprobe` will report the pixel format as `yuv420p` and appear to
> show no alpha. That is an ffmpeg display quirk — WebM keeps alpha in a side
> stream that the native decoder doesn't surface. Browsers read it correctly. To
> verify it yourself, decode with the libvpx decoder explicitly:
> `ffmpeg -c:v libvpx-vp9 -i out.webm -vframes 1 -pix_fmt rgba check.png`

---

## Speed

Matting is ~95% of the run time, and its cost is set almost entirely by how much
detail the network sees — the `--speed` setting.

```bash
python bgremove.py Practice.mp4 --speed fast       # ~256px long edge
python bgremove.py Practice.mp4 --speed balanced   # ~320px  (default)
python bgremove.py Practice.mp4 --speed best       # ~512px  (RVM's own recommendation)
python bgremove.py Practice.mp4 --speed max        # full resolution
```

The default is `balanced` rather than `best` because it was measured, not
guessed. Against a full-resolution reference on this footage, a 320px backbone
changed the matte by a mean of **0.0016 alpha** (99th percentile 0.054) — visually
indistinguishable — while running about **6× faster**. `best` and `max` exist for
long flyaway hair or high-resolution sources where the extra detail earns its cost.

Check your own machine before committing to a long run:

```bash
python bgremove.py Practice.mp4 --benchmark
```

And always try settings on a short sample first:

```bash
python bgremove.py Practice.mp4 --preview 10
```

### Getting more speed

- **A GPU is worth far more than any setting here.** If you have a DirectX 12 GPU
  (including Intel/AMD integrated), try `pip uninstall onnxruntime` then
  `pip install onnxruntime-directml`. NVIDIA: `onnxruntime-gpu`. The tool picks
  the best available provider automatically.
- `--workers N` splits the clip across processes. This helps on 4+ cores. On a
  2-core machine it does nothing — decode, encode and inference already saturate
  both — so it defaults to 1 there.
- Close background CPU hogs. Docker, browsers and emulators make a real dent on
  a 2-core laptop.

---

## Edge quality

The raw matte is usually good as-is. When it isn't:

```bash
--choke -1        # shrink the cutout by a pixel (kills a bright background fringe)
--feather 1.5     # soften the edge
--levels 0.05,0.95  # clear haze: force near-transparent to 0, near-opaque to 1
--alpha-gamma 0.8   # firm up semi-transparent areas
--denoise           # remove speckle
--temporal 0.5      # blend with the previous frame if the edge crawls
```

A dark halo on a light background usually means `--choke -1`. A hard, cut-out
look usually wants `--feather 1`.

RVM outputs a colour-decontaminated foreground, so edge pixels don't carry a
tint from whatever was behind them — you can composite onto a light background
without a dark rim.

---

## Options

```
positional
  input                     video file to process

output
  -o, --output PATH         default: <input>_nobg.<ext>
  -f, --format FORMAT       mp4 (default), webm, mov, mkv, png, matte, stacked
  -b, --background SPEC     colour / image / video / blur[:n] / none

quality and speed
  --speed {fast,balanced,best,max}
  --model {mobilenetv3,resnet50}    resnet50 is 3-6x slower; rarely worth it on CPU
  --downsample FLOAT        override the internal matting scale directly
  --crf INT                 encoder quality, lower is better (default 18)
  --hwenc                   encode H.264 on the GPU instead of the CPU

matte refinement
  --choke, --feather, --alpha-gamma, --levels, --denoise, --temporal

what to process
  --start SECONDS           start time
  --duration SECONDS        how much to process
  --preview [SECONDS]       short sample, default 10s
  --no-audio                drop the audio track

performance
  -j, --workers N           parallel processes
  --threads N               threads per worker

other
  --benchmark               time this machine and exit
  --list-formats            show output formats
  --quiet                   no progress bar
```

---

## How it works

```
ffmpeg decode ──► RVM matting ──► composite ──► ffmpeg encode ──► MP4 (+ audio)
   (pipe)         (ONNX Runtime)                    (pipe)
```

Frames stream through pipes as raw bytes. Nothing is written to disk in between,
so memory stays flat no matter how long the clip is, and there is no PNG
round-trip.

RVM is a *recurrent* network: each frame's hidden state feeds the next, which is
what stops the edge shimmering between frames the way per-frame segmentation
models do. It also means frames must be processed in order — so when `--workers`
splits the clip, each worker decodes about a second of run-up before its first
kept frame to let the state settle. Measured at a chunk boundary, the seam is
indistinguishable from the single-process result.

---

## Troubleshooting

**"ffmpeg not found"** — install it, then open a *new* terminal so `PATH` refreshes.

**The output won't play** — you probably asked for `mov` (ProRes) or `mkv` (FFV1).
Neither plays in Windows' built-in players. Use `--format mp4`, or open them in
VLC or your editor.

If it's an MP4 that a *previous* version of this tool wrote, it is very likely
truncated: an MP4's index lives in a `moov` atom that is only written when the
encoder shuts down cleanly, so a run that was killed left a file full of frames
with nothing to describe them. You can confirm it with
`ffprobe yourfile.mp4` — "moov atom not found" means exactly that. Those bytes
cannot be repaired by this tool; just run the job again. Current versions cannot
produce that file: frames are encoded to `<name>.part.mp4` and only moved onto
the real name once the encoder has exited cleanly and the result has been probed,
so the destination is either a finished video or nothing at all.

**I had to stop a long run** — press Ctrl+C once and wait a moment. Everything
matted so far is finalised into `<name>_partial.mp4`, which plays normally. The
run does not resume, but you keep the work rather than losing all of it.

**Model download fails** — grab
`rvm_mobilenetv3_fp32.onnx` from the
[RVM releases page](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0)
and drop it in `models/`.

**The matte is poor** — RVM is trained on people. It won't segment objects,
pets, or several overlapping subjects well. For a person, try `--speed best`, and
make sure they're reasonably lit and separated from the background.

**It's still slow** — it's the CPU. Check `--benchmark`, then see "Getting more
speed" above. A GPU changes this by an order of magnitude.

---

## Credit

[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) —
Shanchuan Lin, Linjie Yang, Imran Saleemi, Soumyadip Sengupta. Model weights are
released under GPL-3.0; check that before commercial use.
