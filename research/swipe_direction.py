#!/usr/bin/env python3
"""Which way was the swipe: left or right? One mic, like the taps: the
sound's shape, not stereo. Each swipe is taken whole, from the finger
landing to it lifting:

- where it began and ended: the tap model (DATA/sections-single.joblib, read
  only) on the swipe's first and last sharp sound. A left swipe begins on
  the right of the screen and ends on the left;
- the tap features (signed waveform) of those two sounds, for a model
  trained on the swipes themselves;
- how the sound changes as the finger travels: each band's loudness over
  the swipe, start to end.

Scored on swipes the model never trained on (5-fold × 20), and against the
same model trained on shuffled labels: what luck alone gets. The travel is
what works (70% in one sitting, 2026-11-06): the low bands get louder over
a left swipe and quieter over a right one. research/swipe_features.py
takes it further, across sittings, and saves the live model.

    python research/swipe_direction.py
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tapzones import config
from tapzones.audio import PRE_HOPS, HOP as TAP_HOP
from tapzones.features import EARLY_N, BEST1, signed
from tapzones.models import SectionModel
from swipe_detector import HERE, SR, HOP, WIN, SPAN, relative

STEPS = 10                     # the travel, resampled to this many steps
ABOVE = 8                      # dB over the room: where the swipe is
DIRECTION = os.path.join(HERE, "swipe-direction.joblib")    # written by swipe_features.py


def bounds(rel):
    """The swipe inside a second of relative dB frames: its first and last
    frame over the room (the whole second if nothing is)."""
    on = np.flatnonzero(rel[:, -1] > ABOVE)
    return (on[0], on[-1]) if len(on) > 1 else (0, len(rel) - 1)


def travel(rel, f, l):
    """How each band's loudness changes from the swipe's first frame to its
    last, in STEPS steps: the change, not how loud it was."""
    seg = rel[f: l + 1, :-1]
    if len(seg) < 2:
        seg = np.repeat(rel[f: f + 1, :-1], 2, 0)
    t = np.linspace(0, 1, len(seg))
    steps = np.stack([np.interp(np.linspace(0, 1, STEPS), t, seg[:, b]) for b in range(seg.shape[1])], 1)
    return (steps - steps.mean(0)).ravel()


def swipes(with_session=False):
    """[(kind, stereo audio, relative dB frames, first frame, last frame)],
    and the session's file name last, with_session."""
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "swipes-*.json"))):
        meta = json.load(open(path))
        raw = np.fromfile(path.removesuffix(".json") + ".raw", "<i2").astype(np.float32).reshape(-1, 2) / 32768
        for t in meta["trials"]:
            if t["bad"] or t["kind"] not in ("left", "right"):
                continue
            a = t["go"][0] // HOP - 20
            x = raw[a * HOP: (a + SPAN) * HOP + 4096]
            rel = relative(np.concatenate([raw[max(0, a * HOP - 10 * SR): a * HOP].mean(1), x.mean(1)]))[-SPAN:]
            # The swipe: the loudest second, then from its first to its last
            # frame over the room.
            power = 10 ** (np.clip(rel[:, -1], 0, None) / 10)
            s = int(np.argmax(np.convolve(power, np.ones(WIN), "valid")))
            first, last = bounds(rel[s: s + WIN])
            out.append((t["kind"], x, rel, s + first, s + last) + ((os.path.basename(path),) if with_session else ()))
    return out


def tap_window(x, frame, search=6):
    """A tap-shaped window at the sharpest sound within `search` frames of
    `frame`: what the tap model was trained on (the hit PRE_HOPS hops in)."""
    a = max(0, (frame - search) * HOP)
    seg = np.abs(x[a: (frame + search) * HOP].mean(1))
    peak = a + int(np.argmax(seg)) if len(seg) else a
    start = max(0, peak - PRE_HOPS * TAP_HOP)
    w = x[start: start + EARLY_N]
    return np.pad(w, ((0, EARLY_N - len(w)), (0, 0)))


def main():
    cfg = config.load()
    layout = [s["rect"] for s in cfg["single"]["sections"]]
    tap = SectionModel("single", layout)
    if not tap.ok:
        sys.exit(f"tap model: {tap.why}")
    # Which sections are on the left half of the screen.
    left_side = np.array([r[0] + r[2] / 2 < 0.5 for r in layout])
    data = swipes()
    kinds = np.array([k for k, *_ in data])
    y = (kinds == "left").astype(int)
    print(f"{len(data)} swipes: {(y == 1).sum()} left, {(y == 0).sum()} right; "
          f"median length {np.median([(l - f) * HOP / SR * 1000 for *_, f, l in data]):.0f} ms")

    # A: the tap model, untouched, on where each swipe began and ended.
    starts = np.array([tap.proba(tap_window(x, f))[left_side].sum() for _, x, _, f, _ in data])
    ends = np.array([tap.proba(tap_window(x, l))[left_side].sum() for _, x, _, _, l in data])
    began_right = starts < 0.5
    ended_left = ends > 0.5
    print("\nA · the tap model, as trained on taps (no swipe training):")
    print(f"  where it began says the direction for {(began_right == (y == 1)).mean():.0%} of swipes")
    print(f"  where it ended says the direction for  {(ended_left == (y == 1)).mean():.0%} of swipes")
    both = (ends - starts) > 0          # moved toward the left
    print(f"  began vs ended (moved left?)          {(both == (y == 1)).mean():.0%} of swipes   (chance 50%)")

    # B: trained on the swipes: the two sounds' tap features, the tap model's
    # answers, and the travel.
    S =np.stack([signed(tap_window(x, f)[None], **BEST1)[0] for _, x, _, f, _ in data])
    E = np.stack([signed(tap_window(x, l)[None], **BEST1)[0] for _, x, _, _, l in data])
    T = np.stack([travel(rel, f, l) for _, _, rel, f, l in data])
    A = np.stack([starts, ends, ends - starts], 1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def score(X, labels):
        steps = [StandardScaler()] + ([PCA(n_components=20)] if X.shape[1] > 40 else []) + \
                [LogisticRegression(C=0.3, max_iter=5000)]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
        return cross_val_score(make_pipeline(*steps), X, labels, cv=cv).mean()

    rng = np.random.default_rng(0)
    print("\nB · trained on the swipes (unseen swipes, 5-fold × 20; luck = the same on shuffled labels):")
    for label, X in (("tap model's answers at start and end", A), ("start sound", S), ("end sound", E),
                     ("start + end sounds", np.hstack([S, E])), ("the travel", T),
                     ("everything", np.hstack([A, S, E, T]))):
        real = score(X, y)
        luck = [score(X, rng.permutation(y)) for _ in range(20)]
        p = (1 + sum(l >= real for l in luck)) / (1 + len(luck))
        print(f"  {label:38} {real:6.1%}   luck {np.mean(luck):5.1%} (best {max(luck):5.1%})   p≈{p:.2f}")



if __name__ == "__main__":
    main()
