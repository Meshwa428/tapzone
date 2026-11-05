---
title: Configuration
aliases:
  - gestures.toml
  - Config
tags:
  - tapzones
  - gestures/design
  - reference
status: planned
updated: 2026-11-05
---

# Configuration

All gestures, layouts and commands live in one file:
**`~/.config/tapzones/gestures.toml`**. The layout editor in the overlay writes it, and
it can also be edited by hand.

## Format

- One table per gesture: `single`, `double`, `triple`, `two_finger`, `three_finger`.
- Each gesture has `enabled`, and a list of `[[<gesture>.section]]` entries.
- A section has:
  - `name`: shown in the overlay;
  - `rect`: `[x, y, width, height]` as **fractions of the screen** (0–1, from the
    top-left corner), so a layout survives a resolution change;
  - `command`: a shell command, run when the gesture lands in the section;
  - `over = true` (optional): the section sits over the others and takes the area
    it covers. It must come first: a tap belongs to the first section it falls in.
    The corners + centre preset uses it for its Centre.
- Areas no section covers are dead zones ([[Sections#Dead zones]]).

## Example

```toml
max_sections = 5          # the recommended limit; more is allowed, with a warning

[single]
enabled = true
[[single.section]]
name = "Top"
rect = [0.0, 0.0, 1.0, 0.45]
command = "omarchy launch browser"
[[single.section]]
name = "Bottom"
rect = [0.0, 0.55, 1.0, 0.45]      # the band between 0.45 and 0.55 is a dead zone
command = "omarchy launch terminal"

[double]
enabled = true
[[double.section]]
name = "Anywhere"
rect = [0.0, 0.0, 1.0, 1.0]
command = "playerctl play-pause"

[triple]
enabled = false

[two_finger]
enabled = false

[three_finger]
enabled = false
```

## Rules the editor and the loader check

| Rule | What happens otherwise |
|---|---|
| An enabled gesture has **at least 1 section** | Not saved: "add a section, or turn the gesture off" |
| **Up to `max_sections`** per gesture is recommended (default 5) | More is allowed and saves; the editor warns, and the estimates show the cost ([[Sections#Why 5 is the recommended maximum]]) |
| Sections **don't overlap**, except one with `over = true`, listed first | Not saved; the overlapping area is highlighted |
| `rect` is inside the screen and at least 5% on each side | Not saved: smaller than the mic's typical error |
| Every section has a `command` (it may be empty, to only show the gesture) | — |

> [!note] Why fractions
> Training data stores tap positions as screen fractions (`where`), so a layout in
> fractions maps straight onto it ([[Calibration and training#Relabelling existing taps]]).

## Trained models per layout

Changing a layout changes what the section model has to tell apart. The model files
record which layout they were trained for (a hash of the sections). The overlay
retrains, quickly, when the config no longer matches
([[Calibration and training#Fast retraining]]).

## Related

- [[Gesture system]]: the hub
- [[Sections]]: how layouts are designed
