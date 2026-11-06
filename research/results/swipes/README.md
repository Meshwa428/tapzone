# Swipe experiment (paused 2026-11-06)

Can the one built-in mic hear a finger swiping across the (non-touch) screen,
and which way it went?

## Findings

- **Swipe or not: works.** A live scored round caught 11 of 12 swipes, with no
  false alarms on 9 taps or 9 "do nothing" cues. Offline, at 1% of non-swipe
  seconds firing: 75% of swipes caught, 0% of taps
  (`swipe_detector.py`, gradient-boosted trees on 1 s windows).
- **Direction: not usable yet.**
  - Scored on a sitting it never trained on: 65%. Live round: 6 of 11, leaning "→".
  - The ranking was right, though. The line between left and right moves from
    sitting to sitting: at 0.70 instead of 0.50, the live round would have
    scored 10 of 11.
  - The physical signal: over a left swipe, every band trends louder than over
    a right one, most in the 80–700 Hz bass. This was so in both sittings, but
    weaker in the second, where swipes were faster (525 ms against 730 ms).
- **What didn't work:**
  - stereo: the two channels are the same mic;
  - the tap model on the landing and lifting sounds;
  - the stick-slip clicks as micro-taps;
  - click texture.

## Next, if picked up again

1. **Re-centre each sitting** on a running average of its recent swipes: the
   live round suggests most mistakes go away.
2. **Treat the swipe as a moving tap.** Label each short slice with where the
   finger was (the arrow ran from 15% to 85% of the width), learn position
   from the signed waveform with shifted copies (about 3,000 examples from
   140 swipes), and read direction as the position's trend.
3. Record more sittings. Score every change as trained on one sitting and
   tested on another.

## Files

| File | What it is |
|---|---|
| `research/swipes.py` | Cued recorder: swipes only by default, 40 each way |
| `research/swipes_analyze.py` | First look: waveforms, loudness, stereo |
| `research/swipe_detector.py` | Swipe-or-not model → `swipe-detector.joblib` |
| `research/swipe_direction.py` | Direction from the landing, lifting and travel |
| `research/swipe_features.py` | Feature families, scored within and across sittings → `swipe-direction.joblib` |
| `research/swipes_live.py` | Live window with a scored round (G) → `live-*.json` |

The recordings (`swipes-*.raw`, 16-bit stereo at 48 kHz) are git-ignored for
size and live only on the machine that recorded them; their `.json` trial
logs are committed.
