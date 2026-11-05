---
title: Sections
aliases:
  - Screen sections
  - Layouts
tags:
  - tapzones
  - gestures/design
  - status/built
status: built (single taps)
updated: 2026-11-05
---

# Sections

A **section** is a rectangle of the screen that means something for one gesture. Each
gesture ([[Gesture system#The five gestures]]) has its own **layout**: a list of 1 or
more sections. A tap that lands in a section runs that section's command.

## Making a layout

1. **Pick how many sections, then a preset of that many.** The presets are every
   layout straight cuts make for 1–5 sections (2, 6, 21 and 90 layouts for 2–5),
   plus **corners + centre** (a centre section over the quadrants, so the four
   corners wrap round it) and four layouts of 6.
   A name like `1 / 2` stacks rows top to bottom (one section over two side by
   side); `1 | 2` puts columns left to right (one beside two stacked). The editor
   lists each count's **usual layout first** (top / bottom, 3 rows, quadrants,
   corners + centre, 3 × 2 grid), then the rest **simplest first** (fewest, least
   nested cuts): in the Omarchy editor as a carousel (‹ ›, click or drag a
   neighbour), in both with **N** / **Shift+N** for the count (landing on its
   usual layout) and **P** / **Shift+P** for the layout. Each shows its estimate.

   > [!warning] The centre is the mic's weakest spot
   > In corners + centre the Centre section estimates ≈ 43% on its own (≈ 74%
   > overall, against ≈ 79% for quadrants): taps near the middle are the ones
   > the mic places worst ([[#Rows beat columns]]).

   The measured ones (leave-one-session-out on the recorded taps):

| Preset | Sections | Single tap, measured | Notes |
|---|---|---|---|
| Whole screen | 1 | 100% | "Anywhere". The default for triple and multi-finger taps |
| Top / bottom halves | 2 | **94.6%** | **The most reliable split** |
| Left / right halves | 2 | 87.6% | Works, but left/right is the weaker axis |
| 3 rows | 3 | 83.4% | The middle band is the weak one (72%) |
| Quadrants | 4 | 84.6% | The default for single taps |
| 5 rows | 5 | 70.7% | Use with double taps, or with gaps ([[#Measured accuracy]]) |
| 3 columns | 3 | 77.8% | Weaker than 3 rows: see [[#Rows beat columns]] |

2. **Adjust it** in the layout editor (`tapzones layout`, or the Omarchy editor):
   - new sections can't be drawn: every layout starts from a preset;
   - drag inside a section to move it, or near its edge to resize it;
   - right-click a section to delete it;
   - type its **Name** and **Command** in the panel at the bottom.
   Sections can't overlap, and must be at least 5% of the screen on each side.
3. **Leave gaps (dead zones)** where you want nothing to happen. A tap in a gap is
   ignored, not guessed ([[#Dead zones]]).
4. **Read the estimate.** While you edit, each section shows the accuracy expected
   from the existing recorded taps ([[#The accuracy estimate]]).

> [!tip] Rule of thumb
> Prefer rows to columns, fewer sections to more, and leave a gap where two sections
> meet near the vertical centre line.

## Rows beat columns

The mic sits at the top centre of the bezel. A tap's **height** decides which way the
lid first moves, which is clear and early. Its **side** only shows in a small timing
difference in the ringing 10–30 ms later ([[research#When in the sound is each answer?]]).

| Measure | Up / down (rows) | Left / right (columns) |
|---|---|---|
| Quadrant model, right | **94%** | 85% |
| Rough x,y model, mean error | **38 px = 5% of 768** | 159 px = 12% of 1366 |
| Rough x,y model, median error | **37 px** | 126 px |

The x,y errors come from one 22-tap `locate` session ([[research#8. Live results]]),
so they are rough. The direction is consistent with every other measurement.

## Measured accuracy

Measured by `research/experiments.py layouts`: SVM on the signed waveform, leave one session out,
868 taps from 14 sessions. Results are in `layout-results.tsv`.

| k | k rows | k columns |
|---|---|---|
| 2 | **94.6%** | 87.6% |
| 3 | 83.4% | 77.8% |
| 4 | 75.8% | 72.4% |
| 5 | 70.7% | 67.7% |
| 6 | 66.9% | 61.9% |

Grids: 2×2 (quadrants) 84.6%, 3 columns × 2 rows 75.6%, 2 columns × 3 rows 75.7%.

> [!warning] Correction: 5 rows is not "good"
> An earlier version of this note estimated 5 rows as "good", from the x,y model's
> 37 px median error. Measured, it is **70.7%** for single taps. The x,y error came from
> one 22-tap session, and it hid how badly the **middle bands** do. In 3 rows the middle
> band is right 72% of the time; in 5 rows the middle band is right only 39%.

Rows still beat columns at every k, but every extra section costs about 5–10 points.
What wins accuracy back:

| Layout | Plain | With dead zones (15% gaps) | Double tap | Triple tap |
|---|---|---|---|---|
| Top / bottom | 94.0% | 96.2% | 97.7% | 98.6% |
| Left / right | 87.7% | | 94.2% | 93.6% |
| Quadrants | 84.6% | 86.9% (10% gaps) | 92.5% | 93.2% |
| 3 rows | 83.4% | 91.8% (covers 70% of the screen) | | |
| 3 columns | 78.6% | | 89.4% | 91.7% |
| 5 rows | 71.0% | | 78.9% | 83.5% |

The double- and triple-tap columns are simulated from single taps (`research/experiments.py
double_sim`), see [[Multi-tap#Combining the taps' evidence]]. Wide gaps in 5 rows leave
the middle bands with too few recorded taps to measure.

## The centre problem

> [!bug] What the user sees
> Tapping near the centre of the screen often triggers the wrong section.

From the error analysis ([[research#4. Where the remaining errors are: left vs right]]):

![[errors.png]]

- **Left/right mistakes cluster at the vertical centre line,** where both sides sound
  nearly the same.
- **They also cluster at the far left and right edges,** where taps often sound like
  they came from the middle. The x position isn't a smooth function of the sound the
  way the y position is.

What the section system does about it:
1. **Dead zones** along the centre line ([[#Dead zones]]).
2. **Rows instead of columns,** when the user only needs a few actions.
3. **Confidence refusal**, as today ([[research#6. Refusing doubtful taps]]): below the
   cutoff the overlay says "not sure" instead of guessing. With sections, the cutoff
   is set per layout.
4. **Double taps** in a section are more certain than single ones, because the
   evidence from both taps is combined ([[Multi-tap#Combining the taps' evidence]]).

## Dead zones

A dead zone is any part of the screen not covered by a section. A tap there runs
nothing.

What was built is a **section classifier** (`tapzones train sections GESTURE`). It's
trained on taps labelled with the user's sections, and it drops recorded taps that fall
in gaps. A tap that really lands in a gap still gets the nearest section's name. But
gaps keep the hardest taps (the ones on borders) out of training and out of the
answer, which is where the measured gain comes from ([[#Measured accuracy]]).

The service refuses taps below the layout's cutoff ("not sure: tap again"). The cutoff
is the lowest one that reaches 90% right, while still answering at least 60% of taps.

> [!question] Not done
> An x,y model that vetoes taps landing near a border is still open. Its x is too
> coarse today ([[#Rows beat columns]]).

## Why 5 is the recommended maximum

- **Measured,** single taps drop below 75% after 4 sections, even as rows
  ([[#Measured accuracy]]). 5 is the most where double taps still get close to 80%.
- **People** remember about 5 places without looking. More sections means more
  mistakes by the user, not only by the mic.
- For more than 2–4 single-tap actions, it's better to use **another gesture** (double
  or triple tap) with its own layout than more sections.

Above `max_sections`, the editor lets the user continue, with a warning. Each section's
estimate is coloured: green at 85% or more, amber from 70%, red below that.

## The accuracy estimate

Every recorded tap stores **where** it was tapped (`where`, as screen fractions). So any
layout can be scored from the taps that already exist, with no new recording:

Training a real model for every drag takes too long (a full `--export-sections` is
about 8 minutes). So the editor uses a shortcut:

1. **Once**, it trains a model that places each tap in a 6×6 grid of cells,
   leave-one-session-out (`gestures.cell_probs`). This is cached in
   `~/.local/share/tapzones/estimate-cells.npz`, and is redone only when the recorded
   sessions change.
2. **For any layout**, a section's probability is the sum of the cells whose centre it
   covers (`gestures.quick_estimate`). This takes under 10 ms, so the estimate follows
   the drag live.
3. Sections with fewer than 10 recorded taps show "only n recorded taps here" instead
   of a number.

> [!info] The estimate runs low
> The cell model is plain logistic regression, so it reads a few points under the real
> models. For example: 2 rows 92.5% vs 94.6%; quadrants 78.8% vs 84.6%; 5 rows 56.6%
> vs 70.7%. It's good for comparing layouts. **Save** trains the real model, and the
> header then shows its measured accuracy.

## Related

- [[Gesture system]]: the hub
- [[Configuration]]: how layouts are saved
- [[Calibration and training]]: how a layout gets its model
- [[Research plan]]: the experiments that turn the estimates above into measurements
