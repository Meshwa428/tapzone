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
(3.14), so the system's GTK4 bindings and gtk4-layer-shell are visible. The package is
installed editable, so `.venv/bin/tapzones` runs the code in the repo:

```sh
cd ~/Documents/Projects/tapzones
uv venv --python /usr/bin/python3 --system-site-packages .venv
UV_HTTP_TIMEOUT=300 uv pip install --python .venv/bin/python -e '.[full]'
.venv/bin/python -m unittest discover -s tests     # Layout.js ↔ Python, config, socket dir
```

> [!note] The GTK commands restart themselves
> Importing `tapzones.gtk` re-runs the process with
> `LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so`, because the layer-shell library must load
> before libwayland. The headless commands (`run`, `status`, `train`…) never import it.

## Commands: `tapzones`

One command, `tapzones` (`python -m tapzones` in the dev env). `tapzones --help` lists
everything; `tapzones COMMAND --help` explains one.

**The service and its front-ends:**

| Command | What it does |
|---|---|
| `tapzones run` | **The service** (no UI): listens and runs the command of the section you tapped. Stays quiet while the overlay or an editor is open. Serves the socket the front-ends use ([[Omarchy plugin#How it fits together]]) |
| `tapzones status` / `pause` / `resume` / `toggle` | Talk to the running service. Pause is kept across restarts |
| `tapzones layout` | **The layout editor** ([[Sections#Making a layout]]): sections for each gesture, their names and commands, and feedback. Save writes `gestures.toml` and trains the section model in the background |
| `tapzones flash` | The flash feedback as a GTK client, for desktops without the Omarchy plugin |

Installed with `install.sh`, the service runs as the systemd user unit
`tapzones.service`. To follow its log (every tap, its section and confidence):
`journalctl --user -u tapzones -f`.

It reloads `gestures.toml` and the section model by itself when either one changes. On
Omarchy, the same editor is in the shell: click the bar's tap icon
([[Omarchy plugin#Using it on Omarchy]]).

**The overlay**: a full-screen view of the four quadrants, each quadrant's tap curves,
and a live scope of the mic.

| Command | What it does |
|---|---|
| `tapzones calibrate` | Calibrate: 18 taps per quadrant at guided grid spots, then review |
| `tapzones test` | **Test round with the final ensemble**: 20 random points (5 per quadrant). Answers 45 ms after the hit; a tap below 0.64 confidence shows "Not sure: tap again" and the same point stays |
| `tapzones free` | **Tap anywhere with the final ensemble**: each answer flashes its quadrant, with confidence and ms after the hit |
| `tapzones test --knn` | Test round with the calibration kNN (the original model) instead |
| `tapzones test --signed` | Test round guessed by the single signed CNN |
| `tapzones test --train NAME,…` | Test round with an SVM on the older spectral features, trained on named sessions |
| `tapzones collect [PER_SPOT] [PER_ZONE]` | Collect training data: the 3×3 grid per quadrant (default 5 taps per spot), then random points (default 30 per quadrant) |
| `tapzones noise` | Record everything the mic hears, no threshold, while you make sounds that are not screen taps |
| `tapzones locate` | Tap anywhere you're shown; the x,y model draws where it heard you, with the running x and y error in pixels |

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

**Overlay keys:**

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

## Training and research

To record more data and retrain step by step, follow [[training]].

**Training** (`tapzones train`, in `tapzones/train.py`):

| Command | What it does |
|---|---|
| `tapzones train sections single` | **Train the section model** for the layout in `gestures.toml` → `sections-single.joblib`. About 4 minutes. The editor's Save runs it for you |
| `tapzones train gate` | **Train the shipped tap-or-not gate** on every tap vs every noise → `gate.joblib`. About 1 minute |
| `tapzones train final` | Train the overlay's ensemble on every session (plus shifted and noise/gain copies) → `final.joblib`. About 6 minutes. Needs `.[full]` (torch) |
| `tapzones train signed` | Train the signed CNN (3 copies) on every session → `signed-cnn.pt` |
| `tapzones train xy` | Train the x,y position SVR → `xy-svr.joblib` |

**Research** (`research/`, run from the repo with the dev env's Python):

| Command | What it does |
|---|---|
| `python research/experiments.py NAME…` | Run experiments (see [[experiments]]), appending to `research/results/results.tsv` |
| `python research/experiments.py --best 15` | Leaderboard |
| `python research/experiments.py gate` | Score gate variants: loud noises from a recording it never heard → `gate-results.tsv` |
| `python research/experiments.py layouts dead_zones double_sim` | Score section layouts → `layout-results.tsv` |
| `python research/evaluate.py` | The first round: log-mel models across sessions (`--deep`, `--augment`, `--signed`, `--gate` for the old spectral-feature gate) |
| `python research/viz.py [collect-….npz]` | Waveform, mel, similarity, spot and polarity plots → `viz-*.png` |
| `python research/analyze.py` | Plots of the calibration taps → `analysis.png` |
| `python research/speed.py` | Per-model inference time for one tap |

Shared code, in `tapzones.features`:
- `signed(W, **BEST1)` turns windows (taps × 8400 × 2) into features.
- `WAVE` is the CNN's input settings; `CUTOFF` is the overlay ensemble's confidence cutoff.
- `cnn()` defines the CNN.

And in `tapzones.train`: `train_cnn()` trains it, `noise_gain()` makes the noise + gain
training copies, and `session_paths()` lists every labelled session.

## Files

**Project** (`~/Documents/Projects/tapzones`):

| Path | What |
|---|---|
| `tapzones/__main__.py` | The `tapzones` command: every subcommand above |
| `tapzones/paths.py` | Where things live: config, data, state, the socket (only in a private `$XDG_RUNTIME_DIR`) |
| `tapzones/audio.py` | The tap detector, with no UI (shared by the overlay and the service) |
| `tapzones/features.py` | What the models hear: the signed waveform, the gate's features, the CNN |
| `tapzones/layout.py` | Sections and presets, layout checks, the editor's live estimate |
| `tapzones/config.py` | `gestures.toml`: gestures, defaults, load and save |
| `tapzones/models.py` | The models the service runs on each tap: the gate and the section model |
| `tapzones/train.py` | `tapzones train`: every shipped model |
| `tapzones/service/` | `engine.py`, the service and its socket protocol; `client.py`, `tapzones status` and friends |
| `tapzones/gtk/` | The GTK front-ends: `overlay.py`, `editor.py`, `flash.py`; `__init__.py` has the layer-shell restart, theme and drawing helpers |
| `manifest.json`, `omarchy/` | The Omarchy plugin: `Service.qml` (socket client, flash), `BarWidget.qml`, `Editor.qml`, `Layout.js` ([[Omarchy plugin]]) |
| `install.sh` | Sets up the env, the `tapzones` command and the units ([[Omarchy plugin#Install]]) |
| `pyproject.toml` | The package: its dependencies (`.[full]` adds torch and matplotlib) and the `tapzones` command |
| `systemd/` | `tapzones.service`, `tapzones-flash.service`: unit templates; `install.sh` fills in the paths |
| `tests/` | `python -m unittest discover -s tests`: `Layout.js` agrees with `layout.py`, the config round trip, the socket's directory check |
| `research/` | `experiments.py` (the experiment harness), `evaluate.py` (the first round), `viz.py`, `analyze.py`, `speed.py` |
| `research/results/results.tsv` | Every scored configuration: time, experiment, variant, model, pooled, session mean, grid → random, taps |
| `research/results/gate-results.tsv` | Every scored gate: time, features, model, taps kept, loud noise rejected overall and per recording, taps, loud noises |
| `research/results/pool.npz` | Out-of-session probabilities per model (`<model>\|P`, `<model>\|Pg`) |
| `research/results/layout-results.tsv` | Every scored section layout (`experiments.py layouts`, `dead_zones`, `double_sim`) |
| `docs/` | These documents, an Obsidian vault. Start at [[README]] |

**Config**: `~/.config/tapzones/gestures.toml`. See [[Configuration]].

**Data** (`~/.local/share/tapzones`):

| File | What | Keys |
|---|---|---|
| `run2-calibration.npz` | Guided calibration, 72 taps | `windows`, `labels`, `where` |
| `calibration.npz` | The current calibration (used by `tapzones test --knn`) | same |
| `collect-20261104-193438.npz` | 200 taps at random points | `windows`, `where`, `truth`, `source` |
| `collect-20261104-200937.npz` | 300 taps, balanced: 180 grid (`grid spot N`) + 120 `random` | same |
| `test-*.npz` | Test rounds, 20 taps each, with the model's guesses | `model`, `windows`, `where`, `truth`, `guess`, `conf`; final-model rounds add `ms` (answer time after the hit) and `refused` (count) |
| `final.joblib` | **The overlay's ensemble**, loaded by `tapzones test` / `free` | `sklearn` (svm, mlp, logistic pipelines), `cnn` (3 state dicts), `n`, `cutoff`, `taps`, `sessions` |
| `final-692taps.joblib` | The previous ensemble (692 taps), kept for comparison | same |
| `locate-*.npz` | Locate sessions; also training data now | `model`, `windows`, `where`, `guess_xy`, `truth` |
| `sections-<gesture>.joblib` | A gesture's section model, valid only for the layout it was trained on | `models`, `layout`, `hash`, `cutoff`, `accuracy`, `answered`, `answered_accuracy`, `per_section`, `taps` |
| `estimate-cells.npz` | The editor's cached 6×6 cell probabilities for the live estimate | `P`, `where`, `key` |
| `train-single.log` | The log of the last training started by the editor's Save | |
| `overlay.pid` | Set while the overlay or editor is open, so the service stays quiet | |
| `noise-*.raw` / `.json` | Continuous noise, 16-bit mono 48 kHz, with prompt times | |
| `noise-20261104-195118.npz` | Older threshold-caught noises | `windows` |
| `noise-free-*.npz` | False alarms marked with **N** in free mode, one file per run | `windows` |
| `gate.joblib` | The tap-or-not gate (`tapzones train gate`) | `model`, `threshold`, `keep`, `taps`, `noise`, `loud`, `recordings` |
| `signed-cnn.pt` | The live CNN (3 copies), loaded by `tapzones test --signed` | `states`, `n`, `taps`, `sessions` |
| `xy-svr.joblib` | The x,y model, loaded by `tapzones locate` | `model`, `taps`, `sessions` |

**Formats:**
- `windows` is float32 (taps × 8400 × 2): 175 ms at 48 kHz, stereo, starting 20 ms
  before the detected tap.
- `where` is the target point as fractions of the screen, (x, y) with y down.
- `truth` and `labels` are quadrants: 0 top-left, 1 top-right, 2 bottom-left,
  3 bottom-right.
