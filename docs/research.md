---
title: Research
aliases:
  - Findings
tags:
  - tapzones
  - research
updated: 2026-11-05
---

# Research

How we got from "can a mic hear where I tapped?" to 82%, and what the sound actually
contains. All scores are **on sessions the model never trained on** unless noted.
Every experiment is listed in [[experiments]]. The plans built on these findings are in
[[Gesture system]].

## Hardware

- Lenovo V14-IIL, Realtek ALC236, built-in mic at the **top centre** of the bezel.
- The left and right channels are the same mic (they differ by about 0.2%), so this is
  effectively **one microphone**.
- No touchscreen and no accelerometer.
- Screen 1366×768. Captured at 48 kHz with `parec`, in 5 ms steps.

## How a tap is caught

1. Every 5 ms the mic level is compared with the background of the last second.
2. A tap is a sudden jump: more than 6× the background, more than 4× the level just
   before, and at least 300 ms after the previous tap.
3. The overlay keeps 20 ms before the jump and 150 ms after it. That window is what
   every model sees.

## How every idea is scored

Every model is scored the same way (`research/experiments.py`):

- **Leave one session out.** Train on all sessions but one, test on the one left out,
  and repeat for each session. The headline number is the share of all taps answered
  right (**pooled**). A session is one sitting, so this tests whether the model works
  on a new day, not only on taps it has seen.
- **Grid → random.** Inside the balanced 300-tap session, train on the 180 taps at
  the grid spots and test on the 120 taps at random points.
> [!info] Noise level
> With 692 taps, a difference under about **±1.6 points** is noise.

## 1. The early approach: spectra (44–47%)

The first versions turned each tap into a spectrum (how loud each pitch is) and used
kNN, SVM, random forests, a CNN on mel spectrograms, and augmentation.

- Calibration at a single spot looked great (90%), but 48 of 60 taps had clipped the mic.
- A guided 3×3 grid per quadrant gave 74% on the calibration taps themselves and only
  8 of 20 on a test round at new points.
- With more data (5 sessions, 612 taps) it plateaued at **44–47%** across sessions.

## 2. Looking at the raw sound: the sign

Plotting every tap lined up at the hit showed the real pattern:

![[viz-5-polarity.png]]

| Quadrant | First swing goes up |
|---|---|
| top-left | 3% |
| top-right | 12% |
| bottom-left | 49% |
| bottom-right | 35% |

- The lid flexes, so the first swing of the sound is either a **push or a pull**, and
  which one depends on where you tapped.
> [!important] The key insight
> A spectrum can't tell a waveform from its upside-down copy, so every earlier feature
> threw this away.

- In the similarity matrix, the black stripes are taps that are mirror images of
  each other:

![[viz-3-similarity.png]]

- Almost all the energy is **below about 300 Hz**. It's the panel moving, not a click.
  The mel settings used before gave that range only 2–3 pixels:

![[viz-2-mel.png]]

Feeding the models the **signed** waveform (low-passed to 4 kHz, sign kept) instead of
spectra: grid → random went from 43–48% to **72–78%**, and leave-one-session-out from
44–47% to **69–79%**.

## 3. Tuning the signed waveform (77% → 79%)

| Change | Effect |
|---|---|
| Cut at the first sample past **20%** of the peak, not at the peak | Peak cut: −10 to −17 points. 20% beat 40% |
| **40 ms** after the cut (30 ms is about equal) | Longer helps within a session but hurts across sessions |
| Sample rate | 1 kHz = 16 kHz. Nothing useful above about 500 Hz |
| Subtract the level just before the onset, scale by RMS, add loudness | About +1–2 points together |
| Training copies cut 0.25–0.5 ms early and late | About +1–2 points (forgives onset jitter) |

These settings are `ml.BEST1` plus `ml.BEST_SHIFTS`.

## 4. Where the remaining errors are: left vs right

Splitting the score by axis shows the bottleneck:

- **top vs bottom: 93–94% right**
- **left vs right: 83–85% right**

![[errors.png]]

Left/right mistakes happen near the vertical centre line, as expected. They also happen
**at the far left and right edges**, where taps often sound like they came from the
middle. The x position isn't a smooth function of the sound the way y is.

### When in the sound is each answer?

An SVM scored on one 5 ms slice of the tap at a time (chance is 50% per axis):

| Time after the hit | top/bottom | left/right |
|---|---|---|
| 0–10 ms | **84–90%** | 61–62% |
| 10–30 ms | 75–83% | **70–73%** |
| after 35 ms | fading | about 50% |

- **Up/down is decided by the first push.**
- **Left/right lives in the ringing 10–30 ms later.**

The average left tap and average right tap have nearly the same shape, but the
right-side taps' swings come **later**. Some asymmetry in the lid (hinge, mic offset,
stiffness) makes the right side ring later or slower:

![[torsion.png]]

Turning this into features (separate windows per axis, a time-stretch factor) didn't
help. The models already find it in the plain waveform, and per tap the cue is noisy.

## 5. What raised the score further (79% → 82.4%)

| Change | Pooled |
|---|---|
| SVM on the tuned signed waveform + shifted copies | 78.8% |
| + 4 training copies with **real background noise** and **different tap strength** | **80.8%** |
| Soft-vote ensemble: SVM + extra trees | 81.9% |
| Soft-vote ensemble: SVM + MLP + extra trees + logistic + CNN | **82.4%** |
| Same without extra trees, which is too slow live (see [[latency]]) | 82.1% |

Speed (±5%) and EQ (±6 dB) augmentation did nothing. Sessions differ mainly in **how
hard you tap** and **the background noise**, not in the lid's tone.

> [!warning] Selection bias
> The best ensemble was picked from 26 combinations on the same data, so its real gain
> is probably a bit less than the headline. The top six all lie between 81.2% and 82.4%.

## 6. Refusing doubtful taps

Each model gives a probability per quadrant. If the system only acts when the
ensemble is confident enough:

| Taps answered | Accuracy on them |
|---|---|
| 100% | 81.9% |
| 95% | 83.6% |
| 90% | 85.0% |
| 80% | **88.1%** |
| 70% | **91.1%** |
| 50% | 95.1% |

(SVM + extra trees ensemble.) For triggering commands, ignoring the least certain
20–30% of taps (you'd simply tap again) is the single best improvement left.

## 7. How much would more data help?

Learning curve (SVM, leave one session out, trained on part of the other sessions):

| Training taps | Pooled |
|---|---|
| about 153 | 74.3% |
| about 307 | 77.3% |
| about 461 | 78.1% |
| about 615 | 78.8% |

Still rising, but flattening. Doubling the data is likely worth about 1–2 points, not 10.

A **warm-up** of 1–3 taps per quadrant at the start of each session, added to training,
gave only about +1 point (80.7% → 81.9%). The remaining errors are tap-to-tap noise, not
a session-level shift.

## 8. Live results

| Test | Result |
|---|---|
| CNN on 612 taps, live test round (it hadn't seen these taps) | 18/20 |
| CNN on 632 taps, 3 live rounds | 44/60 (73%). Top-right was heard as top-left 7 of 15 times |
| x,y position model, locate mode, 22 taps | x error 159 px (median 126), y error 38 px (median 37), 17/22 quadrants right |
| Final ensemble on 692 taps, a new 112-tap collect session it never saw | 95.5% (97% up/down, 97% left/right). One sitting, so not a new baseline |
| Final ensemble on 826 taps, live test round | 18/20, 2 refused as not sure |

The x,y model predicts the tap's screen position; only the quadrant it lands in is
used. It scored the same as the plain 4-way classifier (79.2% vs 78.8%), so it is
kept only for the visual `locate` mode.

## 9. Tap or not a tap (the gate)

The quadrant model must answer one of four quadrants: its confidence is spread over
the quadrants only, so it can't say "none of these". A table knock was answered as
"top-left, 92%". A separate model, the **gate**, now asks first whether the sound was
a screen tap at all (`research/experiments.py gate`, `tapzones train gate`).

**How it is scored.** Only noises loud enough to fire the overlay's detector count:
259 of 2,019 noise windows. The rest are quiet, and an early version of the gate scored
well on them while learning only "loud means tap". Each noise recording is scored by a
gate that never heard it. Each gate's threshold is set so that **97% of taps from
sittings it never saw** get through, so the noise numbers compare directly.

Results (846 taps, 5 noise recordings, 2026-11-05, `gate-results.tsv`):

| Features | Model | Loud noise rejected | Table knocks, 1st sitting (145) | Table knocks, newest (54) | Typing, palm rest, older noises |
|---|---|---|---|---|---|
| jaggedness | SVM | 69.9% | 57% | 87% | 74–100% |
| spectrum + jaggedness | SVM | 70.7% | 61% | 81% | 71–100% |
| shape + spectrum | SVM | **80.3%** | 68% | 93% | 97–100% |
| **shape + spectrum + jaggedness** (shipped) | SVM | 78.4% | 64% | 94% | 97–100% |
| any | logistic | 60–66% | 50–59% | 56–70% | 65–100% |

- **shape**: the signed waveform, the same features the quadrant model uses (best1).
- **spectrum**: loudness, energy above 1 kHz and 4 kHz, centroid, zero crossings, decay.
- **jaggedness**: how much fast wiggle rides on the swing (residual above 300/600/1200
  Hz, extrema, path length divided by the peak).

![[gate-knocks.png]]

**Why table knocks are hard.** Knocks come in two kinds. About half are jagged, full of
fast swings a screen tap never has (bottom row), and those are easy. The other half
look like taps: one smooth, slow swing. Knocking the table next to the laptop rocks
the whole machine, lid included, and the mic hears the lid move just as it does for a
tap.

**More recordings help.** With one sitting of knocks, the first sitting was 52–56%
rejected when unheard. After two more sittings it went up to 64–68%. Jaggedness adds
nothing measurable on top of shape and spectrum (78.4% against 80.3% is within noise
at 259 loud sounds). It is kept because it describes the jagged knocks directly.

The earlier gate on the **old spectral features**, scored on all noise windows (mostly
quiet ones), reported 98% rejected. That number was inflated by the quiet windows.

## Conclusions

1. One centred mic **can** hear the quadrant: about 82% across sessions, 94% for
   top/bottom.
2. The information is the **signed, low-frequency movement of the lid** in the first
   40 ms. Spectra can't see it.
3. Left/right is limited by a small, noisy timing asymmetry of the lid. More
   cleverness on the same signal has stopped paying off.
4. The practical path is the fast ensemble plus a **confidence cutoff** (about 88% at
   80% of taps answered), plus the tap/not-a-tap gate (about 80% of loud noises
   rejected, 97% of taps kept; table knocks right next to the laptop are the hard
   part).
