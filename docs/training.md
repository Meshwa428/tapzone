---
title: Recording data and training
aliases:
  - Training
  - Retraining
  - Recording data
tags:
  - tapzones
  - howto
updated: 2026-11-05
---

# Recording data and training

How to record more taps and more not-a-tap noise, how the shipped model is built from
them, and how to retrain, check and ship a new one. Every command runs from
`~/Documents/Projects/tapzones` with `.venv/bin/python`. Commands and file formats are
listed in [[usage]]. How this extends to the planned gestures is in
[[Calibration and training]].

## The short version

```sh
# 1. Record (as many sittings as you can; see below)
.venv/bin/python taps.py collect                # 300 labelled taps per sitting
.venv/bin/python taps.py noise                  # not-a-tap sounds, as long as you like

# 2. Look at what you recorded
.venv/bin/python viz.py ~/.local/share/tapzones/collect-<time>.npz

# 3. Score the recipe on all data, new sessions included
.venv/bin/python research.py baseline augment_kinds pool pool_cnn cutoff

# 4. Set ml.CUTOFF from the cutoff table, then train and ship
.venv/bin/python ml.py --export-final

# 5. After new noise (or N-marked false alarms): score and ship the gate
.venv/bin/python research.py gate
.venv/bin/python ml.py --export-gate

# 6. Try it live
.venv/bin/python taps.py test
```

New `collect-*.npz` and `test-*.npz` files are picked up automatically by every step.
There is no list of sessions to edit.

## Why more data, and what kind

The model is scored on **sessions it never saw**, because that's how it will be used:
a new day, a different tap, a different background. So what matters is the **number
of sittings** and their variety, more than the raw number of taps.

From the [[research#7. How much would more data help?|learning curve]] (153 → 307 →
461 → 615 training taps gave 74.3 → 77.3 → 78.1 → 78.8%), accuracy is still rising
slowly. Most of the remaining errors are **left vs right** (85% right, against 94%
for top vs bottom) and near the **far left and right edges**, so data there helps most.

### Taps: what to vary between sittings

| Vary | Why |
|---|---|
| **Tap strength**: soft, normal, firm | Strength differs a lot between sessions (the noise + gain augmentation was the one that helped) |
| **The finger**: index, middle, knuckle, fingernail | Different contact changes the push |
| **Lid angle**: upright, leaned back, nearly flat | Changes how the lid rings |
| **Where the laptop sits**: desk, lap, sofa, a hard vs soft surface | Changes the damping |
| **Background**: quiet room, fan, music, people talking | The model must hear taps through noise |
| **Time and day** | Each sitting is one "session" in the exam |
| **Power**: charger plugged in vs not | The mic's hum and floor can change |

Keep everything else normal: tap the way you would when actually using it.

### Not-a-tap noise: what to record

Anything the mic hears that **must not fire a command**, especially things close to a
screen tap:

- knocks and taps on the **desk**, near and far from the laptop;
- taps on the **palm rest, keyboard deck, lid back and edges**, which are the hardest
  to tell from screen taps;
- typing, the touchpad and its clicks, the mouse;
- setting things down (phone, mug, bottle), drawers, doors;
- chair moves, leaning on the desk;
- claps, snaps, pen clicks, talking, coughing, music;
- **long stretches of ordinary background** with nothing happening.

## Recording taps: `taps.py collect`

```sh
.venv/bin/python taps.py collect [PER_SPOT] [PER_ZONE]
```

Defaults are 5 taps per grid spot and 30 random points per quadrant: **300 taps**.

1. **Grid phase (180 taps).** A 3×3 grid of spots in each quadrant, with
   **PER_SPOT taps at each spot** (`tap here ×5`). Spots alternate between
   quadrants so no quadrant is done in one block.
2. **Random phase (120 taps).** Random points, **PER_ZONE per quadrant**, at least
   55 px apart and away from the grid, shuffled in rounds of 4 (one per quadrant).
3. Every quadrant gets exactly the same number of taps.

While recording:
- tap the **centre of the ring**, at the moment you'd naturally tap;
- **Backspace** undoes the last tap (use it right after a mistap);
- a yellow warning means **the tap clipped the mic**: tap a little softer;
- the top line shows the phase, the count, and the estimated time left;
- **Esc** saves and quits. Progress is saved every 10 taps, so a crash loses at
  most 9.

Output: `~/.local/share/tapzones/collect-<date>-<time>.npz`.
- `windows`: taps × 8400 × 2, float32. That's 175 ms at 48 kHz, starting 20 ms before
  the tap was detected.
- `where`: the target point, as screen fractions (x, y down).
- `truth`: quadrant 0–3 (top-left, top-right, bottom-left, bottom-right).
- `source`: `grid spot N` or `random`.

**Tips:**
- Shorter sittings are fine and add variety: `taps.py collect 2 10` is 72 + 40 =
  112 taps.
- To cover the centre lines and the edges, which `collect` keeps clear of, also do
  `taps.py locate` rounds. They aren't used for training yet; see
  [[#Gaps in the current pipeline|Gaps]].
- Live test rounds (`taps.py test`) are saved as `test-*.npz` and **are** used for
  training next time. Refused taps aren't saved.

## Recording noise: `taps.py noise`

```sh
.venv/bin/python taps.py noise
```

- Records **everything** the mic hears, with no tap detection and no threshold, until
  you press Esc.
- The screen shows a timer and a list of ideas. **Don't tap the screen.**
- Every key except Esc is ignored, so you can type as part of the noise.
- Output: `noise-<date>-<time>.raw` (16-bit mono, 48 kHz) and `.json` (start time,
  format).

You don't need to label noise; all of it is "not a tap". Several shorter recordings in
different places beat one long one, because the gate is scored by leaving one whole
recording out. Only sounds loud enough to fire the overlay matter: see
[[#The tap-or-not gate (noise)|the gate]], and the **N** key in free mode, which records
exactly those.

## Checking what you recorded

```sh
.venv/bin/python viz.py ~/.local/share/tapzones/collect-<time>.npz
```

This writes `viz-1` to `viz-5` PNGs in `~/.local/share/tapzones/`:
- waveforms per quadrant;
- mel spectrograms;
- the tap-to-tap similarity matrix;
- per-spot spectra and a 2D map;
- first-swing direction on the screen.

The new session should look like the old ones: in particular, the up/down
first-swing pattern in `viz-5-polarity.png`. If a whole session looks different (say
almost every tap clipped), find out why before training on it.

## The training recipe

This is what `ml.py --export-final` does.

### 1. Data

Every file from `ml.session_paths()`: `run2-calibration.npz` plus every `test-*.npz`
and `collect-*.npz` in `~/.local/share/tapzones`.

### 2. Features: `ml.signed(W, **ml.BEST1)`

For each tap's 175 ms window:

1. Mix to mono (the two channels are the same mic).
2. Find the **onset**: the first sample, within the first 60 ms, past **20%** of the
   tap's peak.
3. Cut from **2 ms before** the onset to **38 ms after** it (40 ms).
4. Subtract the mean of the 5 ms before the cut (removes any offset).
5. Average every 12 samples, giving **4 kHz** and 160 values. Nothing useful is above
   about 500 Hz.
6. Divide by the RMS, **keeping the sign**. The sign is the main signal: see
   [[research#2. Looking at the raw sound: the sign]].
7. Append log10 of the peak (loudness): **161 numbers per tap**.

The CNN gets the same without the loudness value (`ml.WAVE`, 160 numbers).

### 3. Training copies (`export_final`)

Each tap is used **9 times**:

| Copy | How |
|---|---|
| 1 | as recorded |
| 4 | the cut moved by −24, −12, +12, +24 samples (±0.25 and ±0.5 ms): tolerates onset jitter (`ml.BEST_SHIFTS`) |
| 4 | `ml.noise_gain()`: gain ±10 dB, plus a random stretch of **real background noise** at −6 to +12 dB |

The background noise bank is the first 15 ms (before the tap) of every recorded window.

With 692 taps that's 6,228 training rows.

### 4. Models

| Model | Settings |
|---|---|
| SVM | `StandardScaler` → `SVC(C=3, probability=True)` (RBF kernel) |
| MLP | `StandardScaler` → `MLPClassifier((128,), alpha=0.01, max_iter=2000)` |
| Logistic regression | `StandardScaler` → `LogisticRegression(C=0.3)` |
| CNN, 3 copies (seeds 0, 1, 2) | `ml.cnn()`: three blocks of conv1d (16 ch k7, 32 ch k5, 32 ch k5) + batch norm + ReLU + max-pool 2, then flatten, dropout 0.3, linear to 4. **Flatten, not global pooling**, so the timing of each swing counts. Trained by `ml.train_cnn()`: AdamW (lr 2e-3, weight decay 1e-2), batches of 64, 40 epochs, label smoothing 0.1, each batch shifted ±2 samples at random. **Never flip the sign in augmentation** |

**Prediction:** average the four models' probabilities (the 3 CNNs are averaged first,
so they count as one). The quadrant is the most likely one. If its probability is
below `ml.CUTOFF` (0.64), **refuse** the tap.

### 5. Output

`~/.local/share/tapzones/final.joblib` contains:
- `sklearn` (the three pipelines);
- `cnn` (3 state dicts);
- `n` (input length);
- `cutoff`, `taps`, `sessions`.

`taps.py test` and `taps.py free` load it.

Training takes **about 6 minutes** on this laptop's CPU at 692 taps. Time grows
roughly with the number of taps; the SVM grows faster.

## Scoring before you ship

> [!warning]
> Never ship a model only because it trained. Run the same exam as the research,
> which now includes the new sessions:

```sh
.venv/bin/python research.py baseline augment_kinds   # quick: ~2 min
.venv/bin/python research.py pool pool_cnn            # the ensemble: ~15 min
.venv/bin/python research.py cutoff                   # pick ml.CUTOFF
.venv/bin/python research.py --best 10
```

- **pooled** (leave one session out) is the number to watch. Compare it with the same
  experiment's earlier rows in `results.tsv`. The last column is the tap count, so you
  can see which data each row was scored on.
- Gaps under about ±1.6 points are noise at about 700 taps; this shrinks as data grows.
- **`cutoff`** prints, for the shipped 4-model ensemble, the accuracy when answering
  100%, 95%, …, 60% of taps. Pick the row you want, usually about 80% answered, and put
  its cutoff in `ml.CUTOFF` in `ml.py` before `--export-final`.

> [!caution] Stale `pool.npz`
> `pool` rewrites `pool.npz`, so run `pool` before `pool_cnn` and `cutoff`. `cutoff`
> refuses to run on a `pool.npz` from a different number of taps.

Optional checks:
- `research.py curve`: is accuracy still rising with data?
- `research.py errors`: where the mistakes are (writes `errors.png`).
- `research.py timeline`: where in the sound each axis lives.

## Shipping and trying it

```sh
.venv/bin/python ml.py --export-final
.venv/bin/python taps.py test      # 20-point round: score, refusals, ms after the hit
.venv/bin/python taps.py free      # tap anywhere
```

The round summary shows the score, how many taps were refused, and the median time
from hit to answer. Each round is saved as `test-*.npz` and becomes training data for
the next model.

## The tap-or-not gate (noise)

Before answering a quadrant, the overlay asks the gate whether the sound was a screen
tap at all. If it wasn't, the overlay shows "not a screen tap: ignored" and does
nothing. Results and the reasoning are in [[research#9. Tap or not a tap (the gate)]].

```sh
.venv/bin/python research.py gate          # score it: ~5 min, → gate-results.tsv
.venv/bin/python ml.py --export-gate       # train and ship it: ~1 min, → gate.joblib
```

**Data.**
- **Taps:** every screen tap from every session (`ml.session_paths()`).
- **Noise:** every `noise-*.raw` (cut into windows at every bump, at a much lower bar
  than the live detector, plus one random slice per second), and every `noise-*.npz`,
  including the false alarms you mark with **N** in free mode (`noise-free-*.npz`).

**Features: `ml.tap_features`** on the same early window as the quadrant model (the
first 45 ms after the hit):
- shape: `ml.signed(W, **ml.BEST1)`, 161 numbers;
- spectrum: log peak, energy share above 1 kHz and 4 kHz, centroid, zero crossings,
  decay (6 numbers);
- jaggedness: residual above 300/600/1200 Hz, extrema, zero crossings, path length
  divided by the peak (6 numbers).

**Model:** `StandardScaler` → `SVC(C=3, class_weight="balanced")` (`ml.gate_model`).

**Threshold:** `ml.GATE_KEEP` (0.97). Every session is scored by a gate trained
without it, and the threshold is set so that 97% of those taps get through. A higher
value turns away fewer real taps and lets more noise through.

**How it is scored (`research.py gate`):** each noise recording by a gate that never
heard it, counting **only noises loud enough to fire the detector** (`ml.would_fire`).
Quiet noises never reach the overlay, and counting them makes any gate look good.

**Saved in `gate.joblib`:** `model`, `threshold`, `keep`, and the counts of `taps`,
`noise`, `loud` and `recordings`. `taps.py` loads it automatically when it exists,
and the overlay's label then ends with "+ gate". It costs about 5 ms per sound. Delete
`gate.joblib` to turn the gate off.

**Recording noise that helps:**
- **Only sounds loud enough to fire the overlay count.** Of 42 knocks in one
  recording, only 7 were. Knock about as hard as you'd tap the screen.
- **The best way:** run `taps.py free` and knock, bump and type around the laptop.
  Every time it answers a quadrant for something that wasn't a tap, press **N**. Those
  are exactly the false alarms the gate needs, from your real setup. Then run
  `ml.py --export-gate`.
- Table knocks right next to the laptop are the hardest: they rock the lid like a tap.
  Record them in several sittings.

## Gaps in the current pipeline

These are known and not done yet:

1. **Gate scores are measured on 259 loud noises from 5 recordings,** 145 of them
   from one sitting. They will move as more noise comes in.
2. **Locate sessions aren't training data.** `locate-*.npz` has the same `windows`,
   `where` and `truth` as `collect`, and covers the centre lines and edges, but
   `ml.session_paths()` doesn't include it. Adding `locate-*.npz` there would use it.
3. **Refused taps aren't saved** in test rounds. They're the hardest taps, and would be
   worth keeping.
4. **No hold-out.** `results.tsv` scores and the cutoff are measured on all the data
   that exists. Before a big decision, keep a few new sittings out of everything until
   the end.
