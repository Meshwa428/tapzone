---
title: Usage
aliases:
  - Commands
  - Files
tags:
  - tapzones
  - reference
updated: 2026-11-05
---

# Usage

Commands, keys, files and data formats. For step-by-step recording and retraining, see
[[training]].

## Environment

> [!tip] Installing for real
> `install.sh` sets up the service, its env in `~/.local/share/tapzones/venv`, the
> `tapzones` command and the systemd units, on Omarchy or any other desktop. See
> [[Omarchy plugin#Install]]. What follows is the development setup.

The development env is one virtual environment, made with uv on the **system** Python
(3.14), so the system's GTK4 bindings and gtk4-layer-shell are visible:

```sh
cd ~/Documents/Projects/tapzones
uv venv --python /usr/bin/python3 --system-site-packages .venv
UV_HTTP_TIMEOUT=300 uv pip install --python .venv/bin/python numpy scikit-learn matplotlib torch
```

> [!note] `taps.py` restarts itself
> It re-runs with `LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so`, because the layer-shell
> library must load before libwayland. A script that imports `taps` needs the same
> `LD_PRELOAD` already set, or its own code is lost when the process restarts.

## The overlay: `taps.py`

A full-screen overlay showing the four quadrants, each quadrant's tap curves, and a
live scope of the mic.

| Command | What it does |
|---|---|
| `.venv/bin/python taps.py` | Calibrate: 18 taps per quadrant at guided grid spots, then review |
| `.venv/bin/python taps.py test` | **Test round with the final ensemble**: 20 random points (5 per quadrant). Answers 45 ms after the hit; a tap below 0.64 confidence shows "Not sure: tap again" and the same point stays |
| `.venv/bin/python taps.py free` | **Tap anywhere with the final ensemble**: each answer flashes its quadrant, with confidence and ms after the hit |
| `.venv/bin/python taps.py test --knn` | Test round with the calibration kNN (the original model) instead |
| `.venv/bin/python taps.py test --signed` | Test round guessed by the single signed CNN |
| `.venv/bin/python taps.py test --train NAME,…` | Test round with an SVM on the older spectral features, trained on named sessions |
| `.venv/bin/python taps.py collect [PER_SPOT] [PER_ZONE]` | Collect training data: the 3×3 grid per quadrant (default 5 taps per spot), then random points (default 30 per quadrant) |
| `.venv/bin/python taps.py noise` | Record everything the mic hears, no threshold, while you make sounds that are not screen taps |
| `.venv/bin/python taps.py locate` | Tap anywhere you're shown; the x,y model draws where it heard you, with the running x and y error in pixels |
| `.venv/bin/python taps.py layout` | **The layout editor** ([[Sections#Making a layout]]): sections for each gesture, their names and commands, and feedback. Save writes `gestures.toml` and trains the section model in the background |
| `.venv/bin/python taps.py run` | **The service** (`engine.py`, no UI): listens and runs the command of the section you tapped. Stays quiet while the overlay or an editor is open. Serves the socket the front-ends use ([[Omarchy plugin#How it fits together]]) |
| `.venv/bin/python taps.py status` / `pause` / `resume` / `toggle` | Talk to the running service. Pause is kept across restarts |
| `.venv/bin/python taps.py flash` | The flash feedback as a GTK client, for desktops without the Omarchy plugin |

Installed with `install.sh`, these are `tapzones <command>`, and the service runs as the
systemd user unit `tapzones.service`. To follow its log (every tap, its section and
confidence): `journalctl --user -u tapzones -f`.

It reloads `gestures.toml` and the section model by itself when either one changes. On
Omarchy, the same editor is in the shell: click the bar's tap icon
([[Omarchy plugin#Using it on Omarchy]]).

**Layout editor keys** (when no text box has focus):

| Key | Action |
|---|---|
| 1–5 | Gesture: single, double, triple, 2-finger, 3-finger |
| P / Shift+P | Next / previous preset (commands are kept by position) |
| E | Turn the gesture on or off |
| F | Feedback: flash, notify, none |
| Delete | Remove the selected section (right-click also removes) |
| U | Undo |
| S | Save |
| Esc | Quit (press twice if there are unsaved changes) |

**Keys:**

| Key | Action |
|---|---|
| Esc | Quit (saves) |
| Backspace | Undo the last tap |
| T | New test round |
| F | Free tapping |
| V | Calibration results |
| R | Recalibrate |
| N | Free tapping: the last sound wasn't a tap. Saved as noise for the gate (`noise-free-*.npz`) |

In noise mode only Esc works; typing is part of the noise.

## Training and research: `ml.py`, `research.py`, `viz.py`, `speed.py`

To record more data and retrain step by step, follow [[training]].

| Command | What it does |
|---|---|
| `.venv/bin/python ml.py --export-final` | **Train the shipped ensemble** on every session (plus shifted and noise/gain copies) → `final.joblib`. About 6 minutes. Re-run after collecting new sessions |
| `.venv/bin/python ml.py --export` | Train the signed CNN (3 copies) on every session → `signed-cnn.pt` |
| `.venv/bin/python ml.py --export-xy` | Train the x,y position SVR → `xy-svr.joblib` |
| `.venv/bin/python ml.py --signed` | The original signed-waveform evaluation |
| `.venv/bin/python ml.py --export-gate` | **Train the shipped tap-or-not gate** on every tap vs every noise → `gate.joblib`. About 1 minute |
| `.venv/bin/python research.py gate` | Score gate variants: loud noises from a recording it never heard → `gate-results.tsv` |
| `.venv/bin/python ml.py --export-sections single` | **Train the section model** for the layout in `gestures.toml` → `sections-single.joblib`. About 8 minutes. The editor's Save runs it for you |
| `.venv/bin/python research.py layouts dead_zones double_sim` | Score section layouts → `layout-results.tsv` |
| `.venv/bin/python ml.py --gate` | The old spectral-feature gate (superseded) |
| `.venv/bin/python research.py NAME…` | Run experiments (see [[experiments]]) |
| `.venv/bin/python research.py --best 15` | Leaderboard |
| `.venv/bin/python viz.py [collect-….npz]` | Waveform, mel, similarity, spot and polarity plots → `viz-*.png` |
| `.venv/bin/python speed.py` | Per-model inference time for one tap |

Shared code in `ml.py`:
- `ml.signed(W, **ml.BEST1)` turns windows (taps × 8400 × 2) into features.
- `ml.WAVE` is the CNN's input settings; `ml.CUTOFF` is the shipped confidence cutoff.
- `ml.noise_gain()` makes the noise + gain training copies.
- `ml.cnn()` defines the CNN.
- `ml.train_cnn()` trains it.
- `ml.session_paths()` lists every labelled session.

## Files

**Project** (`~/Documents/Projects/tapzones`):

| File | What |
|---|---|
| `taps.py` | The overlay, and the entry point for every command |
| `mic.py` | The tap detector, with no UI (shared by the overlay and the service) |
| `engine.py` | The service (`taps.py run`) and its socket |
| `gestures.py` | Layouts, the config, and the models the service runs |
| `editor.py` | The GTK layout editor (`taps.py layout`) |
| `flash.py` | The GTK flash feedback (`taps.py flash`) |
| `manifest.json`, `omarchy/` | The Omarchy plugin: `Service.qml` (socket client, flash), `BarWidget.qml`, `Editor.qml`, `Layout.js` ([[Omarchy plugin]]) |
| `install.sh` | Sets up the env, the `tapzones` command and the units ([[Omarchy plugin#Install]]) |
| `tapzones.service`, `tapzones-flash.service` | systemd unit templates; `install.sh` fills in the paths |
| `requirements.txt`, `requirements-research.txt` | The service's packages; plus torch and matplotlib for the rest |
| `ml.py` | Features, models, export |
| `research.py` | The experiment harness |
| `viz.py`, `analyze.py` | Plots of the raw data |
| `speed.py` | Latency measurement |
| `results.tsv` | Every scored configuration: time, experiment, variant, model, pooled, session mean, grid → random, taps |
| `gate-results.tsv` | Every scored gate: time, features, model, taps kept, loud noise rejected overall and per recording, taps, loud noises |
| `pool.npz` | Out-of-session probabilities per model (`<model>\|P`, `<model>\|Pg`) |
| `layout-results.tsv` | Every scored section layout (`research.py layouts`, `dead_zones`, `double_sim`) |

**Config**: `~/.config/tapzones/gestures.toml`. See [[Configuration]].
| `docs/` | These documents, an Obsidian vault. Start at [[README]] |

**Data** (`~/.local/share/tapzones`):

| File | What | Keys |
|---|---|---|
| `run2-calibration.npz` | Guided calibration, 72 taps | `windows`, `labels`, `where` |
| `calibration.npz` | The current calibration (used by `taps.py test` without `--signed`) | same |
| `collect-20261104-193438.npz` | 200 taps at random points | `windows`, `where`, `truth`, `source` |
| `collect-20261104-200937.npz` | 300 taps, balanced: 180 grid (`grid spot N`) + 120 `random` | same |
| `test-*.npz` | Test rounds, 20 taps each, with the model's guesses | `model`, `windows`, `where`, `truth`, `guess`, `conf`; final-model rounds add `ms` (answer time after the hit) and `refused` (count) |
| `final.joblib` | **The shipped ensemble**, loaded by `taps.py test` / `free` | `sklearn` (svm, mlp, logistic pipelines), `cnn` (3 state dicts), `n`, `cutoff`, `taps`, `sessions` |
| `final-692taps.joblib` | The previous ensemble (692 taps), kept for comparison | same |
| `locate-*.npz` | Locate sessions; also training data now | `model`, `windows`, `where`, `guess_xy`, `truth` |
| `sections-<gesture>.joblib` | A gesture's section model, valid only for the layout it was trained on | `models`, `layout`, `hash`, `cutoff`, `accuracy`, `answered`, `answered_accuracy`, `per_section`, `taps` |
| `estimate-cells.npz` | The editor's cached 6×6 cell probabilities for the live estimate | `P`, `where`, `key` |
| `train-single.log` | The log of the last training started by the editor's Save | |
| `overlay.pid` | Set while the overlay or editor is open, so the service stays quiet | |
| `noise-*.raw` / `.json` | Continuous noise, 16-bit mono 48 kHz, with prompt times | |
| `noise-20261104-195118.npz` | Older threshold-caught noises | `windows` |
| `noise-free-*.npz` | False alarms marked with **N** in free mode, one file per run | `windows` |
| `gate.joblib` | The tap-or-not gate (`ml.py --export-gate`) | `model`, `threshold`, `keep`, `taps`, `noise`, `loud`, `recordings` |
| `signed-cnn.pt` | The live CNN (3 copies), loaded by `taps.py test --signed` | `states`, `n`, `taps`, `sessions` |
| `xy-svr.joblib` | The x,y model, loaded by `taps.py locate` | `model`, `taps`, `sessions` |

**Formats:**
- `windows` is float32 (taps × 8400 × 2): 175 ms at 48 kHz, stereo, starting 20 ms
  before the detected tap.
- `where` is the target point as fractions of the screen, (x, y) with y down.
- `truth` and `labels` are quadrants: 0 top-left, 1 top-right, 2 bottom-left,
  3 bottom-right.
