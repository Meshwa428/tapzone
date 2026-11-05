---
title: Roadmap
aliases:
  - Gesture roadmap
tags:
  - tapzones
  - gestures/plan
  - status/active
status: phase 2 built, being tried
updated: 2026-11-05
---

# Roadmap

The [[Gesture system]], built in the order that removes the most risk first. Each phase
ends with something usable.

## Phase 1: Know which layouts work

> [!success] Done: uses existing data only
> - [x] `research/experiments.py layouts`: [[Research plan#1. Score layouts from existing taps]]
> - [x] `research/experiments.py dead_zones`: [[Research plan#2. Dead zones]]
> - [x] `research/experiments.py double_sim`: [[Research plan#3. Combining the taps of a double tap]]
> - [x] Add `locate-*.npz` to `train.session_paths()` for border coverage
> - [x] Put the measured tables into [[Sections#Measured accuracy]] and [[experiments]]
>
> The finding: 5 rows is 70.7% for single taps, not "good", so 5 is a ceiling, not a
> target. Dead zones and double taps are what recover accuracy.

**Done when:** the preset list and the recommended maximum rest on measured numbers.

## Phase 2: Sections for single taps

> [!todo] The fix for the centre problem
> - [x] `~/.config/tapzones/gestures.toml` loader and validation ([[Configuration]]):
>       `tapzones/config.py`
> - [x] Section model: relabel, train, save with the layout hash
>       (`tapzones train sections single`, [[Calibration and training]])
> - [x] Dead zones ([[Sections#Dead zones]])
> - [ ] The border veto (needs a better x model)
> - [x] Layout editor in the overlay (`tapzones layout`): presets (no drawing new
>       sections), move and resize, the live accuracy estimate ([[Sections#Making a layout]])
> - [x] Background service (`tapzones run`, `tapzones.service`): runs each section's
>       command, with flash or notification feedback
> - [ ] Test rounds use the user's layout instead of quadrants

**Done when:** a user can choose "top / bottom halves" and stop getting false activations
near the centre.

## Phase 3: Double and triple taps

> [!todo]
> - [ ] Detector: shorter lockout, onset-only detection ([[Multi-tap#Detector changes]])
> - [ ] `collect --double` / `--triple` ([[Calibration and training#New recording modes]])
> - [ ] Record several sittings; [[Research plan#4. Multi-tap timing]]
> - [ ] Tap grouper with GAP from the data ([[Multi-tap#Grouping taps]])
> - [ ] Combine evidence across taps ([[Multi-tap#Combining the taps' evidence]])
> - [ ] No single-tap delay when multi-tap is off ([[Multi-tap#The single-tap delay]])

## Phase 4: Multi-finger taps

> [!todo] May stop here if the mic can't count fingers
> - [ ] `collect --fingers 2` / `--fingers 3`
> - [ ] Record several sittings; [[Research plan#5. Multi-finger taps]]
> - [ ] If it passes: the finger-count model (`fingers.joblib`) in the pipeline
> - [ ] If it fails: write up why in [[research]], and hide the gestures

## Phase 5: The gate, for every gesture

> [!todo]
> - [ ] Add every new gesture's taps to the gate's tap data
> - [ ] [[Research plan#6. The gate and the new gestures]]
> - [ ] Re-export `gate.joblib`

## Related

- [[Gesture system]]: the hub
- [[Research plan]]: the experiments behind each phase
