---
title: Calibration and training
aliases:
  - Gesture training
tags:
  - tapzones
  - gestures/design
  - howto
status: planned
updated: 2026-11-05
---

# Calibration and training for gestures

How each part of the [[Gesture system]] gets its data and its model. Today's recipe
for the quadrant model is in [[training]]. This note covers what changes.

## Relabelling existing taps

Every tap recorded so far (846 taps, 13 sessions) stores **where** it was tapped, as
screen fractions (`where`), next to its sound (`windows`). So for any layout:

1. Look up which section of the layout each tap's `where` falls in.
2. Taps that fall in a dead zone are left out of training for that layout.
3. Train the section model on those labels with the same recipe as today: signed
   features, shifted and noise/gain copies ([[training#The training recipe]]).

> [!success] No new recording needed for single-tap layouts
> Any single-tap layout can be trained and scored from the data that exists.

> [!warning] Coverage
> Collect sessions keep their random points away from the centre lines and the edges,
> and the grid sits at ¼, ½ and ¾ of each quadrant. Layouts with borders elsewhere,
> such as 5 rows, will have few training taps near their borders. Use `locate`
> sessions, which cover the whole screen, and record more of them
> ([[training#Gaps in the current pipeline]]).

## Fast retraining

The overlay retrains when the layout changes. From the timing measurements of
2026-11-05 (714 taps, about 6,400 training rows):

| Model | Train time |
|---|---|
| SVM | 10 s |
| MLP | 21 s |
| Logistic | 11 s |
| CNN, each | 102 s |

- **On a layout change:** train SVM + logistic straight away (about 20 s, in the
  background) and use them at once.
- **The full ensemble** (MLP + 3 CNNs, about 5–6 min) trains afterwards and replaces
  them when done.
- **The layout editor's estimate** ([[Sections#The accuracy estimate]]) uses only the
  logistic model and the real taps, without the extra copies: a second or two.

## New recording modes

`tapzones collect` gains a gesture option. Each saved tap gets two new fields:

| Field | Meaning |
|---|---|
| `count` | Taps in the gesture: 1, 2 or 3 |
| `fingers` | Fingers in each tap: 1, 2 or 3 |
| `group` | For multi-taps: which gesture the tap belongs to, so the taps of one double tap stay together |

Modes:
- **`collect --double` / `--triple`:** random points; the user double- or triple-taps
  each point. Records the gaps between taps ([[Research plan#4. Multi-tap timing]]).
- **`collect --fingers 2` / `--fingers 3`:** random points, tapped with 2 or 3
  fingers. Several sittings, with varied strengths, as for single taps
  ([[training#Taps: what to vary between sittings]]).
- **`locate`** sessions become training data (`ml.session_paths()` includes
  `locate-*.npz`), for border coverage.

## The finger-count model

- Trained on 1-finger taps (all existing data) against the new 2- and 3-finger
  recordings.
- Scored leave-one-session-out, as everything else.
- It's a separate, small model saved next to the others (proposed: `fingers.joblib`).
  It only exists once [[Research plan#5. Multi-finger taps]] passes.

## The gate

- Multi-finger and multi-tap recordings are **taps** for the gate, so they are added
  to its tap data.
- Then run `research/experiments.py gate` and `tapzones train gate` as today
  ([[training#The tap-or-not gate (noise)]]).

## Files

| File | Holds |
|---|---|
| `~/.config/tapzones/gestures.toml` | Gestures, layouts, commands ([[Configuration]]) |
| `~/.local/share/tapzones/sections-<gesture>.joblib` | The section model for one gesture's layout, with the layout's hash |
| `~/.local/share/tapzones/fingers.joblib` | The finger-count model, once it exists |
| `~/.local/share/tapzones/gate.joblib` | The tap-or-not gate, as today |

## Related

- [[training]]: the current recipe and data
- [[Research plan]]: what has to be measured first
