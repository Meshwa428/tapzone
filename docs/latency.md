---
title: Latency
aliases:
  - Speed
  - Real-time
tags:
  - tapzones
  - research
  - reference
updated: 2026-11-05
---

# Latency

The goal is real-time: the command should fire within a few tens of milliseconds of the
tap. There are three delays: waiting for enough sound, the audio buffer, and computing
the answer.

## Computing the answer

One tap at a time, as live, median of 200 runs on this laptop's CPU (one thread),
models trained on all 692 taps plus copies (6,228 rows). Measured with `speed.py`.

| Step | Median | 95th percentile |
|---|---|---|
| Features (`ml.signed`, best1) | 0.48 ms | 0.81 ms |
| SVM (with probabilities, 2,115 support vectors) | 0.80 ms | 1.34 ms |
| MLP (128 hidden) | 0.39 ms | 1.11 ms |
| Logistic regression | 0.37 ms | 1.31 ms |
| CNN (one copy) | 0.18 ms | 0.30 ms |
| Extra trees, 500 trees | **38 ms** | 48 ms |
| Extra trees, 100 trees | 7.4 ms | 12.3 ms |

Extra trees is the only slow model: scikit-learn walks every tree in turn. Without it,
the ensemble scores **82.1%** against 82.4%, which is within noise.

> [!success] Decision
> **Ship SVM + MLP + logistic + CNN: about 2 ms for features and all four models.**

## Waiting for sound

The overlay currently waits **150 ms** after detecting a tap, because that's the length
of the recorded window. The features use only 40 ms after the onset. Cutting the
window shorter and comparing the features with those from the full window:

| Wait after detection | Window | Features identical |
|---|---|---|
| 30 ms | 55 ms | 10.7% of taps |
| **40 ms** | 65 ms | **100%** |
| 45 ms | 70 ms | 100% |
| 60 ms | 85 ms | 100% |

So `POST_HOPS` in `taps.py` can drop from 30 (150 ms) to **8 (40 ms)** with no change to
the model's input. 9 hops (45 ms) leaves one hop of margin.

## What shipped

- The mic thread calls `on_early` **9 hops (45 ms) after detection** with the 70 ms
  window so far (`EARLY_HOPS` in `taps.py`).
- The final ensemble answers from that window; its answer is identical to the
  full-window answer on 22/22 unseen taps.
- The full 175 ms window still arrives later and is saved, so recorded data keeps the
  same format as all the training data.
- `Final` warms itself up at start, because torch's first call costs about 30 ms.
  Measured afterwards on unseen taps: **first tap 6.3 ms, median 3.5 ms, max 6.3 ms**.

## Total

| | Before | Now |
|---|---|---|
| Wait for sound after detection | 150 ms | **45 ms** |
| Audio buffer (`parec --latency-msec=10`) | about 10 ms | about 10 ms |
| Detection granularity (5 ms steps) | up to 5 ms | up to 5 ms |
| Features + models | about 2 ms (CNN) | about 3.5 ms (4-model ensemble) |
| Tap-or-not gate ([[training#The tap-or-not gate (noise)]]) | — | about 5 ms |
| **Tap to answer** | **about 165 ms** | **about 65 ms** |

The overlay shows, for every answer, how many milliseconds after the hit it came. It
measures from the moment the tap was detected, so it excludes the audio buffer.

> [!note] The floor is set by physics
> Left/right lives in the lid's ringing 10–30 ms after the hit (see
> [[research#When in the sound is each answer?]]), so the answer can't come much sooner
> than about 35–40 ms without giving up left/right accuracy. Up/down alone is decided in
> the first 10 ms.

> [!warning] Double and triple taps add delay
> With multi-tap gestures turned on, a single tap is only known to be single once the
> grouping window has passed (about 350 ms). See [[Multi-tap#The single-tap delay]].
