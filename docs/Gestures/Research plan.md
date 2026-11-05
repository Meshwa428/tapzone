---
title: Research plan
aliases:
  - Gesture research
tags:
  - tapzones
  - gestures/research
  - status/planned
status: planned
updated: 2026-11-05
---

# Research plan for gestures

Experiments to run before and while building the [[Gesture system]]. Each one becomes a
`research.py` experiment and is logged, like everything in [[experiments]]. Scoring
follows the existing rules: **leave one session out**, and only taps or noises that
would really reach the overlay ([[research#How every idea is scored]]).

> [!tip] Order
> Experiments 1–3 use **data that already exists** and can run first. 4–6 need new
> recordings ([[Calibration and training#New recording modes]]).

## 1. Score layouts from existing taps

> [!example] Method
> Relabel the 846 taps by the section their `where` falls in
> ([[Calibration and training#Relabelling existing taps]]). Score the shipped recipe
> leave-one-session-out for:
> - rows k = 1…6;
> - columns k = 1…6;
> - grids 2×2, 2×3, 3×2;
> - every preset in [[Sections#Making a layout]].

- **Output:** accuracy per layout and per section, saved as a table in this note and
  in [[experiments]].
- **Pass bar:** replaces the estimates in [[Sections#Rows beat columns]]. Confirms or
  changes the recommended maximum of 5.
- **Watch out:** taps near the new borders are rare, because collect avoids the centre
  lines. Report how many taps each section has.

## 2. Dead zones

> [!example] Method
> For the presets, remove a band of width w around each border (w = 0, 5, 10, 15% of
> the screen). Measure:
> - accuracy on the taps that remain;
> - **coverage** (the share of the screen still active);
> - for the x,y veto ([[Sections#Dead zones]]): how often a tap inside a section gets
>   wrongly vetoed.

- **Pass bar:** the layout reaches at least 90% at the chosen w, with at least 70% of the
  screen still active.

## 3. Combining the taps of a double tap

> [!example] Method
> Simulate double taps with pairs of real taps from the same section and session,
> multiplying their probabilities ([[Multi-tap#Combining the taps' evidence]]). Compare
> with single-tap accuracy at the same layout.

- **Expectation:** a clear gain, especially for columns.
- **Caveat:** two real taps of a double tap may be more alike than two random taps
  from the same section (same finger, same spot), which would make the real gain
  smaller. Experiment 4's recordings check this.

## 4. Multi-tap timing

> [!example] Method
> Record `collect --double` and `--triple` sessions ([[Calibration and training#New recording modes]]).
> Measure:
> - the gaps between taps (their distribution decides GAP);
> - how often tap 2 starts inside tap 1's 45 ms early window;
> - whether the second tap is placed as accurately as a single tap.

- **Pass bar:**
  - at least 98% of real double taps grouped correctly;
  - fewer than 2% of separate single taps wrongly merged;
  - GAP set from the data ([[Multi-tap#Grouping taps]]).

## 5. Multi-finger taps

> [!example] Method
> Record 2- and 3-finger sessions on several sittings. Then:
> - **look at them first**, as with the sign insight
>   ([[research#2. Looking at the raw sound: the sign]]), with `viz.py`;
> - train a 1/2/3-finger classifier and score it leave-one-session-out;
> - check whether multi-finger taps can be placed: top/bottom first.

- **Pass bar:** finger count at least 90% right across sittings. Otherwise the gesture
  doesn't ship ([[Multi-finger]]).
- **Look for:** each hypothesis in [[Multi-finger#What the mic might hear]].

## 6. The gate and the new gestures

> [!example] Method
> Score the current gate on the new recordings: what share of double, triple and
> multi-finger taps does it turn away? Retrain with them as taps, then re-run
> `research.py gate`.

- **Pass bar:** the gate keeps at least 97% of every gesture's taps, and its loud-noise
  rejection doesn't drop by more than 2 points ([[research#9. Tap or not a tap (the gate)]]).

## Related

- [[Roadmap]]: when each experiment runs
- [[experiments]]: where results are recorded
