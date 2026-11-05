---
title: tapzones
aliases:
  - Home
  - tapzones home
tags:
  - tapzones
  - moc
updated: 2026-11-05
---

# tapzones

Tap one of the four quadrants of a laptop screen that has **no touchscreen**, and the
built-in microphone works out which quadrant you tapped. Each quadrant can then trigger
a command.

> [!success] Where it stands
> The best model so far gets **82.4%** of quadrants right on recording sessions it never
> trained on (chance is 25%). If it ignores the taps it is least sure about, it gets
> **about 88%** right while still answering 80% of taps.

## Documents

| Note | What's in it |
|---|---|
| [[research]] | What we learned, in the order we learned it, and why the model works the way it does |
| [[experiments]] | Every experiment with its best score, including those that failed |
| [[latency]] | How fast it is, and how to make it real-time |
| [[training]] | **Recording more taps and noise, the training recipe, and retraining, checking and shipping a new model** |
| [[usage]] | Commands, files, data formats and the environment |
| [[Omarchy plugin]] | **Installing it** (Omarchy plugin or any Linux), and how the service, socket and front-ends fit together |

## Planned: the gesture system

> [!todo] Next
> Replace the fixed quadrants with **user-defined sections**, plus **double and triple
> taps** and **2- and 3-finger taps**, each with its own layout. Design, research plan
> and roadmap: [[Gesture system]].

| Note | What's in it |
|---|---|
| [[Gesture system]] | The hub: the problem, the five gestures, the pipeline |
| [[Sections]] | Layouts, dead zones, why rows beat columns, the limit of 5 |
| [[Multi-tap]] | Double and triple taps |
| [[Multi-finger]] | 2- and 3-finger taps |
| [[Configuration]] | `gestures.toml` |
| [[Calibration and training]] | Data and models for each gesture |
| [[Research plan]] | Experiments, with pass bars |
| [[Roadmap]] | Phases and tasks |

## The idea in one picture

When you tap the lid, it flexes. The mic in the top bezel hears a low "thump" that
moves **first one way or the other** depending on where you tapped. Taps near the top
almost always push one way; many taps near the bottom push the other way.

![[viz-1-waveforms.png]]

Earlier versions of this project measured *how loud each pitch was* (spectra, mel
spectrograms). Those throw the direction away, which held accuracy at 44–47%. Keeping
the direction (the **signed** waveform) raised it to about 77%, and later tuning took
it to 82%.

## Status (2026-11-05)

- 13 recorded sessions, 846 labelled taps, plus 5 noise recordings and 1 locate session.
- **Tap-or-not gate** (`gate.joblib`): turns away about 80% of noises loud enough to
  fire the detector, keeping 97% of taps. Table knocks right next to the laptop are the
  hard part. See [[research#9. Tap or not a tap (the gate)]].
- About 355 model configurations scored the same way, logged in `results.tsv`.
- **Up/down: 94% right. Left/right: 85% right.** Left/right is the weak axis.
- **Shipped model** (`final.joblib`, used by `taps.py test` and `taps.py free`):
  - soft vote of SVM + MLP + logistic regression + CNN: **82.1%** across sessions;
  - answers **45 ms after the hit**, computing in about **3.5 ms** (plus about 5 ms for
    the gate);
  - ignores taps below 0.64 confidence: answers about 80% of taps, **about 89% right**.
  - See [[latency]].

> [!warning] Known problem
> Taps near the centre of the screen trigger the wrong quadrant more often. Why, and
> the fix: [[Sections#The centre problem]].
