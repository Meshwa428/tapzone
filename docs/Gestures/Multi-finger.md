---
title: Multi-finger
aliases:
  - Two-finger tap
  - Three-finger tap
  - 2-finger tap
  - 3-finger tap
tags:
  - tapzones
  - gestures/design
  - status/research
status: research
updated: 2026-11-05
---

# 2- and 3-finger taps

A **2-finger tap** is one tap with two fingers touching the screen at once, and a
**3-finger tap** uses three. They are single taps in time. What's new is **how many
fingers** made the sound.

> [!danger] Unknown whether this works
> No multi-finger tap has been recorded yet. Everything below is a hypothesis to test
> ([[Research plan#5. Multi-finger taps]]). The feature ships only if the mic can count
> fingers reliably.

## Defaults

| Gesture | Default layout | Range |
|---|---|---|
| 2-finger tap | 1 section: anywhere | 1 – 5, once placement is shown to work |
| 3-finger tap | 1 section: anywhere | 1 – 5, once placement is shown to work |

They use the same section system as every other gesture ([[Sections]]), but start with
one section. Two fingers land in two places, and nothing is known yet about where the
mic places such a tap.

## What the mic might hear

| Hypothesis | Why it might be there | How to see it |
|---|---|---|
| **More energy, a broader push** | Two or three contact points push more of the lid | Loudness, and the size of the first swing at the same tap strength |
| **Two or three onsets a few ms apart** | Fingers never land exactly together | Small steps in the first 5–10 ms, at the full 48 kHz rate |
| **A different first swing** | Spread-out force bends the lid differently from a point | The signed waveform ([[research#2. Looking at the raw sound: the sign]]) |
| **Less ringing** | More fingers damp the lid while they rest on it | Decay over 20–40 ms |

> [!question] Open questions
> - Can 1, 2 and 3 fingers be told apart at all, across sittings?
> - Is a 2-finger tap placed where the midpoint of the fingers is?
> - Does tap strength (soft 2-finger against hard 1-finger) confuse the count?

## The gate

The tap-or-not gate ([[training#The tap-or-not gate (noise)|gate]]) was trained only on
single-finger taps. Multi-finger taps may look like noise to it and be turned away.
Before shipping:
- add multi-finger taps to the gate's tap data;
- check that the gate keeps them ([[Research plan#6. The gate and the new gestures]]).

## The model

- **The finger count is its own small classifier** (1 / 2 / 3), run on the same 45 ms
  early window before the section model ([[Gesture system#How a sound becomes a command]]).
- It's trained on recordings labelled with the number of fingers
  ([[Calibration and training#New recording modes]]).
- If its confidence is low, the tap is refused, as the quadrant model does today.

## Related

- [[Gesture system]]: the hub
- [[Multi-tap]]: the timing gestures
- [[Research plan]]: experiment 5 decides whether this ships
