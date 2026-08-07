# bgremove — complete reference

Removes a video's background locally. No upload, no account, no API key.
Everything below was measured on this machine (Intel i5-5200U, 2 cores, no
usable GPU) against `Practice.mp4` — 640×360, 30 fps, 6571 frames, 3m39s.

---

## Quick start

```bash
# green screen, ready to key in an editor          <- what you have now
python bgremove.py Practice.mp4 --background greenscreen --crf 14

# no background at all, real alpha channel
python bgremove.py Practice.mp4 --transparent

# only the main presenter, dropping picture-in-picture people
python bgremove.py Practice.mp4 --background greenscreen --main-subject

# try settings on 10 seconds before committing 25 minutes
python bgremove.py Practice.mp4 --preview 10 --background greenscreen
```

Output defaults to `<input>_nobg.<ext>` next to the input. `--preview` writes
`<input>_preview.<ext>` so it never clobbers a real render.

---

## The two flags added most recently

### `--main-subject [N]`

Keeps only the **N largest connected regions** of the matte (default 1).

RVM segments *people* — all of them. If your footage has a picture-in-picture
inset, a poster, or a burned-in thumbnail containing a person, that person is
kept as foreground. Correct by the model's logic, rarely by yours.

Measured on `Practice.mp4` at t=77.5s, where a "BRAVO!" card inset appears:

| region | area | with `--main-subject` |
|---|---|---|
| presenter | 72,795 px | kept |
| inset person | 32,152 px | dropped |
| speckle | 33 px | dropped |
| speckle | 1 px | dropped |

Because stray speckle is by definition small and disconnected, this also
removes it — so `--main-subject` largely subsumes `--denoise` for that purpose.

Use `--main-subject 2` for a two-person interview, and so on.

**Caveats.** Regions are computed per frame, so if two subjects trade places as
"largest" between frames you'll get flicker — use `-N` large enough to keep
both. A subject fully split by an occluder (someone walking in front of them)
counts as two regions, and the smaller half would be dropped. The mask is built
at alpha > 0.05, deliberately low, so a subject's soft edge stays inside its own
region instead of being cut loose and hardened.

### `--transparent` / `--no-bg` / `-t`

No background at all — a real alpha channel. Shorthand for `--background none`
plus a format that can hold it. Selects **mov (ProRes 4444)** unless `--format`
says otherwise.

Contradictory combinations are refused up front, not silently resolved:

```
$ python bgremove.py Practice.mp4 --transparent --background greenscreen
error: --transparent and --background ask for opposite things.
```

---

## Transparency: what actually works here

**MP4 cannot store alpha.** That is the container, not this tool. Apple ships
HEVC-with-alpha in MP4; ffmpeg cannot encode it and little outside Apple plays it.

More surprising: **WebM alpha is broken in ffmpeg 8.** `libvpx-vp9` still lists
`yuva420p` among its pixel formats, accepts the request, exits 0 — and hands
back opaque video. VP8 fails the same way. Nothing in the command line reveals
it. Probed on this build:

| `--format` | codec | alpha | full-clip size |
|---|---|---|---|
| `mov` | ProRes 4444 | **works** | ~1.6 GB |
| `mkv` | FFV1 | **works** | ~0.7 GB |
| `png` | PNG sequence | **works** | ~1.3 GB |
| `webm` | VP9 | **silently dropped** | — |
| `mp4` | H.264 | not possible | 33 MB |

Because a full run costs ~25 minutes, the tool now **probes your ffmpeg** before
starting a transparent render — it encodes two tiny frames carrying a known
alpha ramp and checks the ramp survives. If it doesn't, you get an error in one
second instead of a wrong file in half an hour:

```
$ python bgremove.py Practice.mp4 --transparent --format webm
error: this ffmpeg build encodes webm without an alpha channel.
       It accepts the request and silently returns opaque video, so the
       run would look fine and the transparency would simply be missing.
       Formats that do keep alpha here: mov / mkv / png
```

The probe asks your local ffmpeg rather than assuming, so a build with working
VP9 alpha will be allowed through.

### Alpha costs ~50× the file size

Real alpha means effectively lossless, and 640×360 ProRes 4444 runs ~1.6 GB for
this clip against 33 MB for green-screen MP4. If size matters more than a
perfect edge, green + key is the pragmatic choice at this resolution. The run
summary prints an estimate before starting.

### Carrying a matte through an MP4

If you need MP4's size but also the matte, two formats encode alpha as visible
pixels:

- `--format matte` — black-and-white matte only, for luma-key / track mattes
- `--format stacked` — colour on top, matte below, for shaders / Unity / canvas

---

## All flags

### Input / output

| flag | default | notes |
|---|---|---|
| `input` | — | the video file |
| `-o`, `--output` | `<input>_nobg.<ext>` | output path |
| `-f`, `--format` | `mp4` (`mov` with `--transparent`) | `mp4` `webm` `mov` `mkv` `png` `matte` `stacked` |
| `-b`, `--background`, `--bg` | `black` for mp4, else transparent | see below |
| `-t`, `--transparent`, `--no-bg` | off | real alpha channel |

**`--background` accepts:**

| value | result |
|---|---|
| `greenscreen` | studio chroma `(0,177,64)` — keys cleaner than pure green |
| `bluescreen` | studio chroma `(0,71,187)` |
| `black` `white` `grey` `red` `green` `blue` `cyan` `magenta` `yellow` | named |
| `#101820`, `#abc` | hex |
| `12,34,56` | r,g,b |
| `office.jpg` | image, scaled and centre-cropped to fill |
| `loop.mp4` | video, looped for as long as the input runs |
| `blur` / `blur:25` | the real background, defocused |
| `none` | transparency (alpha formats only) |

`blur` masks the subject out and uses a normalised convolution, so surrounding
background flows in to fill the hole before blurring — no halo dragged off the
person's outline.

### Quality / speed

| flag | default | notes |
|---|---|---|
| `--speed` | `balanced` | scale the backbone runs at: `fast` `balanced` `best` `max` = 0.25/0.375/0.5/1.0 |
| `--resolution` | `1080` | cap the working size to N px on the short edge; `source` keeps the input's |
| `--model` | `mobilenetv3` | `resnet50` is cleaner on fine hair, 3–6× slower |
| `--downsample` | — | override the matting scale (0–1); overrides `--speed` |
| `--crf` | `18` | lower is better; **use 14 if you'll key it** |
| `--hwenc` | off | H.264 on the Intel/AMD GPU |

`--speed` sets what the backbone sees, not the output resolution — the
refinement stage rebuilds a full-resolution matte from it, guided by the original
frame. But the gap between those two sizes is exactly what softens the edge and
lets background colour bleed into it, so `--speed` is the setting that decides
edge quality. The presets used to be absolute pixel targets, which becomes a
smaller and smaller ratio as the source grows: on 4K the old `balanced` target
of 320px meant a scale factor of 0.083, and the edge band came out 89%
background-coloured. The ratio is clamped so the backbone's long edge stays
between 320 and 1600 px, which leaves it untouched from VGA up to 4K.

`--resolution` is the other half of that. Matting a 4K frame costs 4× an HD one
and does *not* buy a better edge — so the default caps work at 1080 on the short
edge. A 4K phone clip is processed at 1080×1920, several times faster than
before and with a cleaner edge. Pass `--resolution source` if you specifically
need the pixels, and budget for it: 4K ProRes 4444 is ~22 GB per two minutes.

### Matte refinement

| flag | default | notes |
|---|---|---|
| `--no-decontaminate` | on | keep the network's own edge colours, which carry the old background |
| `--main-subject [N]` | off | keep the N largest subjects |
| `--choke` | `0` | shrink (−) or grow (+) the cutout, in pixels |
| `--feather` | `0` | soften the edge, in pixels |
| `--alpha-gamma` | `1.0` | <1 firms up semi-transparent areas, >1 softens |
| `--levels LOW,HIGH` | — | remap the matte, e.g. `0.05,0.95` to clear haze |
| `--denoise` | off | median-filter the matte to remove speckle |
| `--temporal` | `0` | blend with the previous frame (0–0.9) to settle a crawling edge |

### What to process

| flag | default | notes |
|---|---|---|
| `--start` | `0` | start time, seconds |
| `--duration` | — | how many seconds |
| `--preview [SECONDS]` | `10` when bare | short sample; writes `_preview` |
| `--no-audio` | off | drop the audio track |

### Performance / other

| flag | default | notes |
|---|---|---|
| `-j`, `--workers` | 1 per physical core, max 4 | 1 on this machine — 2 cores gain nothing from splitting |
| `--threads` | 1 when parallel, else all cores | threads per worker |
| `--benchmark` | — | time several settings on this machine and exit |
| `--list-formats` | — | show formats and exit |
| `--quiet` | off | suppress the progress bar |
| `--version` | — | |

---

## Recipes

```bash
# green screen for keying in Premiere / Resolve / CapCut
python bgremove.py Practice.mp4 --background greenscreen --crf 14 \
    --denoise --levels 0.06,0.97

# same, but only the presenter -- no picture-in-picture people
python bgremove.py Practice.mp4 --background greenscreen --crf 14 --main-subject

# skip keying entirely: alpha straight into the editor
python bgremove.py Practice.mp4 --transparent --main-subject

# lossless alpha, half the size of ProRes
python bgremove.py Practice.mp4 --transparent --format mkv

# drop the subject onto a photo
python bgremove.py Practice.mp4 --background office.jpg -o talk.mp4

# blurred real background, video-call look
python bgremove.py Practice.mp4 --background blur:25

# check one tricky moment before rendering the whole clip
python bgremove.py Practice.mp4 --start 76 --duration 3 \
    --background greenscreen --main-subject -o check.mp4

# matte only, to use as a track matte over the original
python bgremove.py Practice.mp4 --format matte
```

### Tuning for a keyer

Keying punishes different things than viewing does. What helped here:

- `--crf 14` — blocking artifacts at the edge are what a keyer chokes on
- `--levels 0.06,0.97` — clamps faint alpha to 0 and near-opaque to 1, so the
  keyer isn't left guessing at haze
- `--background greenscreen` — studio chroma, not pure `(0,255,0)`
- `--main-subject` — kills speckle and insets in one pass

Or sidestep all of it with `--transparent`: no chroma subsampling, no spill, no
keyer.

---

## Speed on this machine

Matting is ~95% of the run time, so estimates track reality closely. From
`--benchmark` on this machine (matting only; the end-to-end rate is slightly
lower because decode, composite and encode share the same two cores):

| preset | backbone @640×360 | ms/frame | fps | full clip |
|---|---|---|---|---|
| `--speed fast` | 256×144 | 172 | 5.8 | 19 min |
| `--speed balanced` | 320×180 | 200 | 5.0 | 22 min |
| `--speed best` | 512×288 | 646 | 1.5 | 1h 10m |
| `--speed max` | 640×360 | 1046 | 1.0 | 1h 54m |
| `--model resnet50` | 320×180 | 846 | 1.2 | 1h 32m |

The real `balanced` run measured **4.4 fps end to end, 24m36s**. Note the jump
from `balanced` to `best` is 3×, not the gentle step the names suggest.

**Close other applications.** This is a 2-core machine. A run with Firefox and
VS Code open measured **0.6 fps against 4.4 fps idle** — 25 minutes became an
ETA of over two hours. That is the single largest speed factor here, larger than
any flag. Check with `--benchmark`.

There is no GPU path on this hardware: ONNX Runtime reports only
`AzureExecutionProvider` and `CPUExecutionProvider`. DirectML would need a newer
driver than the 2016-era one this Intel HD 5500 has.

---

## If a run is interrupted

Nothing is written to the output name until the encode has finished cleanly and
been verified, so an interrupted run **cannot** leave a broken file where a good
one should be.

- **Ctrl+C** — the encoder is allowed to finish its index, and the frames done
  so far land in `<name>_partial.mp4`, which plays normally. The run does not
  resume, but you keep the work.
- **Hard kill, crash, power loss** — Python never runs, so nothing is salvaged,
  but the half-written data sits in `<name>.part.mp4` and the real output is
  untouched. Delete the `.part` file; it has no index and cannot be repaired.

---

## Troubleshooting

**"moov atom not found"** — the file is truncated. An MP4's index is written
only when the encoder shuts down cleanly, so a killed run leaves frames with
nothing describing them. Those bytes cannot be repaired; run the job again.
Current versions cannot produce this at the output path (see above).

**The output has no transparency** — you probably asked for `webm`. See
"Transparency: what actually works here". The tool now refuses this up front.

**A second person keeps appearing** — RVM segments all people, including ones
inside insets and posters. Use `--main-subject`.

**The matte is poor** — RVM is trained on people. It won't segment objects,
pets, or several overlapping subjects well. Try `--speed best`.

**Speckle in empty areas** — `--denoise --levels 0.06,0.97`, or `--main-subject`,
which removes disconnected specks as a side effect.

**Edges crawl between frames** — `--temporal 0.3`.

**It's slow** — close other applications first; see "Speed" above.

**"ffmpeg not found"** — `winget install Gyan.FFmpeg`, then open a *new*
terminal so `PATH` refreshes.

**Model download fails** — grab `rvm_mobilenetv3_fp32.onnx` from the
[RVM releases page](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0)
and drop it in `models/`.

---

## Credit

[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) —
Shanchuan Lin, Linjie Yang, Imran Saleemi, Soumyadip Sengupta. Weights are
GPL-3.0; check that before commercial use.
