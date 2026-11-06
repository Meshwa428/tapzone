#!/usr/bin/env python3
"""Which way was the swipe, with more of the swipe than its loudness: each
feature family scored on swipes the model never trained on, against the
same model on shuffled labels (luck), and per direction, since a model can
score well overall by mostly answering one way.

A swipe isn't one sound like a tap: the finger sticks and slips along the
glass, thousands of tiny clicks, each made where the finger is right then.
The families:

  bands      each of 7 bands' loudness over the swipe (swipe_direction.py's travel)
  fine       the same in 40 narrow bands: the panel's resonances, not just bass vs treble
  slopes     each of the 40 bands' trend, start to end, in one number
  clicks     the stick-slip clicks as micro-taps: the tap model on each one, in order
             (does it move from right to left?), and their tap features, early vs late
  texture    how the clicks come: how many, how sharp, how bright, over the swipe

Each is also scored trained on one sitting and tested on another, which is
what a live test is. slopes holds up best (2026-11-06, 2 sittings: 65% in a
new sitting), and is saved, trained on every swipe, for
research/swipes_live.py. → research/results/swipes/swipe-direction.joblib

    python research/swipe_features.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.signal import find_peaks
from tapzones import config
from tapzones.audio import PRE_HOPS, HOP as TAP_HOP
from tapzones.features import EARLY_N, BEST1, signed
from tapzones.models import SectionModel
from swipe_detector import SR, HOP
from swipe_direction import DIRECTION, swipes, travel

FINE = np.geomspace(80, 20000, 41)
STEPS = 8
NFFT = 2048
MAX_CLICKS = 40
_f = np.fft.rfftfreq(NFFT, 1 / SR)
_bands = [(_f >= lo) & (_f < hi) for lo, hi in zip(FINE[:-1], FINE[1:])]


def fine_db(mono):
    """(frames, 40) dB, frames 10 ms apart."""
    n = max(1, 1 + (len(mono) - NFFT) // HOP)
    mono = np.pad(mono, (0, max(0, NFFT + (n - 1) * HOP - len(mono))))
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    p = np.abs(np.fft.rfft(mono[idx] * np.hanning(NFFT), axis=1)) ** 2
    return 10 * np.log10(np.stack([p[:, b].sum(1) for b in _bands], 1) + 1e-10)


def steps(seq, k=STEPS):
    """A (time, n) sequence resampled to k steps, each column's mean removed:
    how it changes over the swipe, not its level."""
    if len(seq) < 2:
        seq = np.repeat(seq[:1], 2, 0)
    t = np.linspace(0, 1, len(seq))
    out = np.stack([np.interp(np.linspace(0, 1, k), t, seq[:, j]) for j in range(seq.shape[1])], 1)
    return out - out.mean(0)


def slope(t, v):
    return np.polyfit(t, v, 1)[0] if len(t) > 2 else 0.0


def slopes(mono):
    """Each of the 40 bands' loudness trend over the swipe, start to end:
    the family that holds up best in a new sitting."""
    db = fine_db(mono)
    t = np.linspace(0, 1, len(db))
    return np.array([slope(t, db[:, b]) for b in range(db.shape[1])])


def clicks(mono):
    """The stick-slip clicks: sample positions of the sharpest peaks of the
    sound's first difference (clicks, not hum), at least 8 ms apart."""
    env = np.convolve(np.abs(np.diff(mono)), np.ones(24) / 24, "same")
    peaks, props = find_peaks(env, distance=int(0.008 * SR), height=4 * np.median(env))
    keep = np.argsort(props["peak_heights"])[-MAX_CLICKS:]
    return np.sort(peaks[keep]), env


def families(tap, left_side):
    F = {k: [] for k in ("bands", "fine", "slopes", "clicks", "texture")}
    y, sess = [], []
    for kind, x, rel, first, last, session in swipes(with_session=True):
        sess.append(session)
        seg = x[first * HOP: last * HOP + NFFT]
        mono = seg.mean(1)
        y.append(int(kind == "left"))
        F["bands"].append(travel(rel, first, last))
        db = fine_db(mono)
        F["fine"].append(steps(db).ravel())
        t = np.linspace(0, 1, len(db))
        F["slopes"].append(slopes(mono))

        pos, env = clicks(mono)
        pos = pos[(pos > PRE_HOPS * TAP_HOP) & (pos < len(seg) - EARLY_N)]
        if len(pos) < 4:
            F["clicks"].append(None)
            F["texture"].append(None)
            continue
        W = np.stack([seg[p - PRE_HOPS * TAP_HOP: p - PRE_HOPS * TAP_HOP + EARLY_N] for p in pos])
        S = signed(W, **BEST1)
        pl = np.array([np.mean([m.predict_proba(s[None])[0] for m in tap.d["models"]], axis=0)[left_side].sum() for s in S])
        tc = pos / len(seg)
        third = max(1, len(pos) // 3)
        F["clicks"].append(np.concatenate([
            [slope(tc, pl), pl[-third:].mean() - pl[:third].mean(), pl.mean()],
            S[-third:].mean(0) - S[:third].mean(0)]))
        # Texture over the swipe, in 5 parts: clicks per part, their height,
        # and how bright the sound is.
        parts = np.minimum((tc * 5).astype(int), 4)
        heights = env[pos]
        cent = (db * np.arange(db.shape[1])).sum(1) / (db - db.min()).sum(1).clip(1e-6)
        F["texture"].append(np.concatenate([
            np.bincount(parts, minlength=5) / len(pos),
            [heights[parts == k].mean() / heights.mean() if (parts == k).any() else 0 for k in range(5)],
            steps(cent[:, None], 5).ravel(),
            [slope(tc, np.log(heights)), slope(t, cent), len(pos) / (len(seg) / SR)]]))
    return F, np.array(y), np.array(sess)


def main():
    cfg = config.load()
    layout = [s["rect"] for s in cfg["single"]["sections"]]
    tap = SectionModel("single", layout)
    left_side = np.array([r[0] + r[2] / 2 < 0.5 for r in layout])
    F, y, sess = families(tap, left_side)
    ok = np.array([v is not None for v in F["clicks"]])
    print(f"{len(y)} swipes ({(y == 1).sum()} left, {(y == 0).sum()} right) from {len(set(sess))} sittings; "
          f"{ok.sum()} with enough stick-slip clicks to place")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model(dims):
        return make_pipeline(StandardScaler(), *([PCA(n_components=min(15, dims))] if dims > 15 else []),
                             LogisticRegression(C=0.3, max_iter=5000))

    def evaluate(X, labels, repeats=10):
        """Accuracy and per-direction recall on unseen swipes, averaged over reshuffled folds."""
        acc, rl, rr = [], [], []
        for r in range(repeats):
            p = cross_val_predict(model(X.shape[1]), X, labels, cv=StratifiedKFold(5, shuffle=True, random_state=r))
            acc.append((p == labels).mean()); rl.append((p[labels == 1] == 1).mean()); rr.append((p[labels == 0] == 0).mean())
        return np.mean(acc), np.mean(rl), np.mean(rr)

    def across(X, labels, groups, center=False):
        """Trained on one sitting, scored on another, every way round:
        (accuracy, left right, right right), pooled over the held-out swipes.
        center: each sitting's mean swipe taken away first (as a few
        calibration swipes in a new sitting could)."""
        if center:
            X = X.copy()
            for g in set(groups):
                X[groups == g] -= X[groups == g].mean(0)
        pred = np.zeros_like(labels)
        for g in set(groups):
            m = model(X.shape[1]).fit(X[groups != g], labels[groups != g])
            pred[groups == g] = m.predict(X[groups == g])
        return (pred == labels).mean(), (pred[labels == 1] == 1).mean(), (pred[labels == 0] == 0).mean()

    print(f"\n{'':10} {'mixed sittings':>24}   {'a new sitting':>24}   {'new sitting, re-centred':>24}")
    print(f"{'family':10} " + "   ".join([f"{'right':>7} {'← ok':>7} {'→ ok':>7}"] * 3))
    for name in ("bands", "fine", "slopes", "clicks", "texture", "all"):
        names = ("bands", "fine", "slopes", "clicks", "texture") if name == "all" else (name,)
        keep = np.all([[v is not None for v in F[n]] for n in names], axis=0)
        X = np.hstack([np.array([v for v, k in zip(F[n], keep) if k], float) for n in names])
        cells = [evaluate(X, y[keep]), across(X, y[keep], sess[keep]), across(X, y[keep], sess[keep], center=True)]
        print(f"{name:10} " + "   ".join(f"{a:7.0%} {l:7.0%} {r:7.0%}" for a, l, r in cells))

    import joblib
    X = np.array(F["slopes"], float)
    joblib.dump({"model": model(X.shape[1]).fit(X, y), "features": "slopes", "left": 1,
                 "swipes": len(y), "sittings": sorted(set(sess))}, DIRECTION)
    print(f"\nsaved {DIRECTION}: slopes, on all {len(y)} swipes from {len(set(sess))} sittings")


if __name__ == "__main__":
    main()
