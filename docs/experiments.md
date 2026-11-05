---
title: Experiments
aliases:
  - Experiment log
tags:
  - tapzones
  - research
  - reference
updated: 2026-11-05
---

# Experiments

The story behind these is in [[research]]; experiments still to come are in
[[Research plan]].

Every experiment in `research.py`, with its best result. Run any of them with
`.venv/bin/python research.py NAME`. Every scored configuration (355 rows) is in
`results.tsv`.

**Columns:**
- **pooled**: leave one session out, share of all 692 taps right. This is the headline.
- **session mean**: the same, averaged per session.
- **grid → random**: inside the balanced session, trained on the 180 grid taps and
  tested on the 120 random ones.

> [!info]
> Chance is 25%. Differences under about ±1.6 points are noise.

**Feature sets:**
- **best1** = `ml.BEST1`: cut at 20% of the peak with 2 ms of lead-in, 40 ms at 4 kHz,
  pre-onset level subtracted, RMS-normalised, plus log loudness.
- **best2** = best1 + training copies shifted by ±12 and ±24 samples (±0.25 and ±0.5 ms).
- **best3** = best2 + 4 training copies with real background noise and ±10 dB of gain.

## Features

| Experiment | What it tested | Best configuration | Pooled | Session mean | Grid → random | Verdict |
|---|---|---|---|---|---|---|
| `baseline` | Signed 4 kHz waveform, 40 ms, cut at 40% of peak | SVM | 76.9% | 79.1% | 76.7% | Starting point |
| `length` | 10–120 ms after the cut | 30–40 ms, SVM | 76.9% | 76.6% | 74.2% | 30–40 ms best across sessions |
| `rate` | 1–16 kHz | any, SVM | 76.9% | 79.1% | 76.7% | No difference: all below 500 Hz |
| `lead` | 0–8 ms of lead-in | 4 ms, SVM | 77.0% | 80.4% | 76.7% | Small, noise-level |
| `onset` | Cut at 10–60% of peak, or at the peak | 20%, SVM | 76.9% | 80.8% | 79.2% | **Peak cut costs 10–17 points**; 20% best |
| `norm` | Peak / RMS / none, ± loudness | RMS + loudness, SVM | 78.5% | 80.0% | 77.5% | +1.6 |
| `filters` | Block mean vs Butterworth 150–2000 Hz | 300 Hz, SVM | 77.3% | 79.4% | 76.7% | No difference |
| `baseline_dc` | Subtract the pre-onset level | SVM | 78.2% | 78.6% | 79.2% | +1.3 |
| `stereo` | Left, right, L−R, mono + L−R | mono | 76.9% | 79.1% | 76.7% | L−R is just a scaled copy; no gain |
| `combo` | The winners together | 30 ms, SVM | 77.9% | 80.6% | 78.3% | Becomes best1 |
| `shifts` | Training copies cut early/late | ±24 samples, SVM | 78.8% | 82.1% | 80.0% | +1–2, becomes best2 |
| `session_norm` | Standardise each session by its own taps | (without it) | 78.8% | 83.0% | 80.0% | **Hurts**, −1 to −4.5 |
| `bands` | Signed waveform per frequency band | best2 + bands, SVM | 76.6% | 78.0% | 76.7% | Hurts; above 200 Hz alone ≈ 35% |
| `modes` | Resonance magnitudes + phase steps (shift-invariant) | best1 + modal, SVM | 77.3% | 80.7% | 83.3% | Alone ≈ 50%: removing the phase removes the sign |
| `physics` | Lid displacement (cumsum), velocity (diff) | all three, SVM | 77.3% | 79.1% | 78.3% | Raw pressure is already best |
| `swings` | Size and time of the first 8–12 swings | best1 + swings, MLP | 77.6% | 79.8% | 77.5% | Alone 47–63% |
| `align` | Cross-correlation alignment to the average tap | SVM | 79.3% | 80.2% | 80.0% | Same as 20% onset |
| `lift` | The finger-release event | best2 + release, SVM | 76.6% | 80.8% | 78.3% | Alone ≈ 53%; no gain |
| `warp` | Time-stretch factor (right taps ring later) | best2 + stretch, SVM | 78.8% | 82.5% | 80.0% | Real but too noisy per tap |
| `augment_kinds` | Speed / EQ / noise + gain / all, one at a time | **noise + gain, SVM** | **80.8%** | 82.1% | 79.2% | **+2.0**, becomes best3 |
| `axes` | Separate features for up/down and left/right | split SVM | 80.2% | 81.1% | 80.0% | Same as one 4-way model |

## Models

| Experiment | What it tested | Best configuration | Pooled | Session mean | Grid → random | Verdict |
|---|---|---|---|---|---|---|
| `zoo` | 9 model families on best1 | MLP | 77.5% | 80.2% | 82.5% | SVM, MLP, extra trees all about 77% |
| `svm_grid` | C 0.3–30 × gamma | C=10, gamma=0.003 | 77.8% | 80.8% | 75.8% | ±1, noise-level |
| `logistic_grid` | C 0.003–1 | C=0.1 | 76.3% | 77.5% | 81.7% | Best within-session, weaker across |
| `knn_grid` | k 3–31, euclidean/cosine | k=7 euclidean | 73.8% | 77.5% | 75.8% | Weakest |
| `targets` | Hierarchical, or predict x,y then threshold | x,y SVR | 79.2% | 81.7% | 77.5% | Same as 4-way; kept only for `locate` |
| `cnn` | 1D CNN, 40/80 epochs, ± shifted copies | 40 epochs | 79.0% | 77.5% | 72.5% | Same as SVM |
| `ensemble` | Soft vote of 2–4 models (best2) | SVM+MLP+ET+LR | 80.1% | 84.6% | 83.3% | +1.3 |
| `pool` | 5 models ± test-time shifts, every ensemble, confidence cutoff | SVM + extra trees | 81.9% | 83.5% | 79.2% | Test-time shifts: no gain |
| `pool_cnn` | Adds the augmented CNN to the pool | **SVM+MLP+ET+LR+CNN** | **82.4%** | 83.2% | 81.7% | **Best overall** |

## Data and diagnostics

| Experiment | What it tested | Result |
|---|---|---|
| `curve` | Train on 25/50/75/100% of the other sessions | 74.3 → 77.3 → 78.1 → 78.8%: flattening |
| `warmup` | 1–3 taps per quadrant from the new session added to training | 80.7% → 81.9%: about +1 |
| `timeline` | Score each axis from one 5 ms slice | Up/down at 0–10 ms, left/right at 10–30 ms |
| `torsion` | Average left vs right tap | Right-side taps ring later (`images/torsion.png`) |
| `errors` | Where the mistakes are; x,y error | Left/right fails at the centre **and** at the far edges |
| `cutoff` | The shipped ensemble's accuracy when answering only its surest taps (reads `pool.npz`) | 80% answered → 89.0% right at cutoff 0.638 |
| `gate` | Tap or not: loud noises from a recording the gate never heard, at 97% of taps kept (→ `gate-results.tsv`) | shape + spectrum SVM 80.3%; shipped shape + spectrum + jaggedness 78.4% ([[research#9. Tap or not a tap (the gate)]]) |

## Section layouts

Scored on 868 taps from 14 sessions (`locate` sessions included), with an SVM on best2,
leave one session out (→ `layout-results.tsv`). The details are in
[[Sections#Measured accuracy]].

| Experiment | What it tested | Result |
|---|---|---|
| `layouts` | k rows, k columns (k = 2–6), and 2×2, 3×2, 2×3 grids | Rows: 94.6 / 83.4 / 75.8 / 70.7 / 66.9%. Columns: 87.6 / 77.8 / 72.4 / 67.7 / 61.9%. Quadrants 84.6%. The middle bands are the weak ones |
| `dead_zones` | Gaps of 0–20% of the screen along every inner border | 3 rows with 15% gaps: 91.8% (70% of the screen covered). Top/bottom with 15%: 96.2%. Quadrants with 10%: 86.9% |
| `double_sim` | Multiply the probabilities of 2 or 3 taps from the same section | Quadrants 84.6 → 92.5 → 93.2%; 5 rows 71.0 → 78.9 → 83.5% ([[Multi-tap#Combining the taps' evidence]]) |

## Reproducing

```sh
cd ~/Documents/Projects/tapzones
.venv/bin/python research.py --list           # every experiment
.venv/bin/python research.py baseline shifts  # run some; appends to results.tsv
.venv/bin/python research.py --best 15        # leaderboard
```

`pool.npz` holds each pool model's out-of-session probabilities for every tap: keys
`<model>|P` (692×4) and `<model>|Pg` (the 120 random taps of grid → random). With it,
new ensembles can be scored without retraining.
