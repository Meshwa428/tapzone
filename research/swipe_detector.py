#!/usr/bin/env python3
"""Swipe or not: the swipe experiment's detector. Was there a swipe on the
screen in the last second? Not where, not which way: that failed
(research/swipes_analyze.py, the two mic channels are one mic).

    python research/swipe_detector.py      train, score, save

Yes: the loudest second of each swipe trial's cued window (research/swipes.py).
No: the "get ready" second before every trial, the loudest second of each
tap, the silence, and every second of the everyday-noise recordings
(knocks, palm rest taps, typing: DATA/noise-*.raw, read only).

Scored 5-fold, each fold a stretch of the session and of every noise file
the model never trained on. → research/results/swipes/swipe-detector.joblib
(the model and its threshold), which research/swipes_live.py loads.
"""
import glob, json, os
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "swipes")
NOISE = os.path.expanduser("~/.local/share/tapzones")      # the noise recordings, read only
MODEL = os.path.join(HERE, "swipe-detector.joblib")
SR = 48000
FRAME, HOP = 1024, 480             # ~21 ms frames every 10 ms
WIN = 100                          # frames in a window: 1 s
# Every example is the loudest second of a stretch this long (a swipe's
# cued window and a little after), swipe or not: picking the loudest of
# anything makes it louder, so all of them are picked the same way.
SPAN = 260
EDGES = [100, 300, 700, 1500, 3000, 6000, 12000, 20000]
FLOOR_FRAMES, FLOOR_EVERY = 1000, 50   # the floor: median of the last 10 s, updated every 0.5 s
FA_TARGET = 0.01                   # the threshold: 1% of not-swipe seconds fire

_f = np.fft.rfftfreq(FRAME, 1 / SR)
BANDS = [(_f >= lo) & (_f < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
_win = np.hanning(FRAME).astype(np.float32)


def frame_db(mono):
    """(frames, bands + 1) log power: each band, then all of them."""
    n = 1 + (len(mono) - FRAME) // HOP
    if n < 1:
        return np.zeros((0, len(BANDS) + 1), np.float32)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    p = np.abs(np.fft.rfft(mono[idx] * _win, axis=1)) ** 2
    bands = np.stack([p[:, b].sum(1) for b in BANDS], 1)
    return 10 * np.log10(np.hstack([bands, bands.sum(1, keepdims=True)]) + 1e-10).astype(np.float32)


class Floor:
    """Each frame relative to the room's sound just before it: the median of
    the last 10 s. Fed in order, offline and live alike."""
    def __init__(self):
        self.hist = []
        self.floor = None
        self.since = 0

    def feed(self, db):
        out = np.empty_like(db)
        for i, row in enumerate(db):
            if self.floor is None or self.since >= FLOOR_EVERY:
                self.floor = np.median(np.array(self.hist[-FLOOR_FRAMES:] or [row]), axis=0)
                self.since = 0
            out[i] = row - self.floor
            self.hist.append(row)
            self.since += 1
        del self.hist[:-FLOOR_FRAMES]
        return out


def relative(mono):
    return Floor().feed(frame_db(mono))


def window_features(rel):
    """One second (WIN frames of relative dB) → what tells a swipe: how loud
    each band gets, for how long it stays up, and how sharp the peak is (a
    tap is a spike, a swipe a stretch)."""
    b, all_ = rel[:, :-1], rel[:, -1]
    above6 = all_ > 6
    run, best = 0, 0
    for v in above6:
        run = run + 1 if v else 0
        best = max(best, run)
    hi, lo = b[:, 4:].mean(1), b[:, :3].mean(1)
    active = above6 if above6.any() else np.ones_like(above6)
    return np.concatenate([
        b.max(0), np.percentile(b, 90, axis=0), np.median(b, axis=0),
        np.percentile(all_, [100, 90, 75, 50, 10]),
        [above6.mean(), (all_ > 10).mean(), best / WIN,
         all_.max() - np.percentile(all_, 75), (hi - lo)[active].mean()],
    ]).astype(np.float32)


def windows(rel, step=50):
    """Every 1 s window, step frames apart: (features, start frame)."""
    starts = range(0, len(rel) - WIN + 1, step)
    return [(window_features(rel[s: s + WIN]), s) for s in starts]


def loudest(rel, a, b):
    """The 1 s window in frames [a, b) with the most sound above the floor."""
    a, b = max(0, a), min(len(rel), b)
    if b - a < WIN:
        a = max(0, b - WIN)
    power = 10 ** (np.clip(rel[:, -1], 0, None) / 10)
    sums = np.convolve(power[a:b], np.ones(WIN), "valid")
    s = a + int(np.argmax(sums)) if len(sums) else a
    return window_features(rel[s: s + WIN])


def dataset():
    """X, y (1 = swipe), kind, fold (0-4: a stretch of its recording)."""
    X, y, kind, fold = [], [], [], []
    for path in sorted(glob.glob(os.path.join(HERE, "swipes-*.json"))):
        meta = json.load(open(path))
        raw = np.fromfile(path.removesuffix(".json") + ".raw", "<i2").astype(np.float32) / 32768
        rel = relative(raw.reshape(-1, 2).mean(1))
        f = lambda sample: sample // HOP
        trials = [t for t in meta["trials"] if not t["bad"]]
        span = max(t["go"][1] for t in trials)
        for t in trials:
            k = min(4, int(5 * t["go"][0] / span))
            g0, g1 = f(t["go"][0]), f(t["go"][1])
            if t["kind"] == "quiet":
                for a in range(g0, g1 - SPAN + 1, SPAN // 2):
                    X.append(loudest(rel, a, a + SPAN)); y.append(0); kind.append("silence"); fold.append(k)
                continue
            X.append(window_features(rel[max(0, g0 - WIN): g0]) if g0 >= WIN else loudest(rel, 0, WIN))
            y.append(0); kind.append("ready"); fold.append(k)
            X.append(loudest(rel, g0 - 20, g0 - 20 + SPAN))
            y.append(int(t["kind"] != "tap")); kind.append("swipe" if t["kind"] != "tap" else "tap"); fold.append(k)
    for path in sorted(glob.glob(os.path.join(NOISE, "noise-*.raw"))):
        rel = relative(np.fromfile(path, "<i2").astype(np.float32) / 32768)
        starts = range(0, len(rel) - SPAN + 1, SPAN)
        for i, a in enumerate(starts):
            X.append(loudest(rel, a, a + SPAN)); y.append(0); kind.append("noise"); fold.append(min(4, 5 * i // len(starts)))
    return np.array(X), np.array(y), np.array(kind), np.array(fold)


def make():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def main():
    import joblib
    X, y, kind, fold = dataset()
    print("windows: " + ", ".join(f"{k} {(kind == k).sum()}" for k in ("swipe", "tap", "ready", "silence", "noise")))
    loud = X[:, len(BANDS) * 3]       # the window's loudest frame, dB above the floor
    print(f"swipes with nothing 6 dB above the room: {(loud[kind == 'swipe'] < 6).sum()} of {(kind == 'swipe').sum()}")

    P = np.zeros(len(y))
    for k in range(5):
        m = make().fit(X[fold != k], y[fold != k])
        P[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    print("\nunseen stretches (5-fold), by threshold:")
    print(f"  {'fires on':>9}  {'swipes caught':>14}  {'taps fired':>10}  {'noise fired':>11}  {'ready fired':>11}")
    neg = P[y == 0]
    for fa in (0.005, 0.01, 0.02, 0.05, 0.10):
        thr = np.quantile(neg, 1 - fa)
        hit = lambda k: (P[kind == k] > thr).mean()
        print(f"  {fa:>8.1%}  {hit('swipe'):>14.0%}  {hit('tap'):>10.0%}  {hit('noise'):>11.1%}  {hit('ready'):>11.0%}")
    from sklearn.metrics import roc_auc_score
    print(f"  AUC swipe vs everything else: {roc_auc_score(y, P):.3f}")
    print(f"  AUC swipe vs tap:             {roc_auc_score(y[np.isin(kind, ['swipe', 'tap'])], P[np.isin(kind, ['swipe', 'tap'])]):.3f}")
    caught = P[kind == "swipe"] > np.quantile(neg, 1 - FA_TARGET)
    print(f"  at {FA_TARGET:.0%}: caught swipes' loudest frame {np.median(loud[kind == 'swipe'][caught]):.1f} dB, "
          f"missed {np.median(loud[kind == 'swipe'][~caught]):.1f} dB above the room (median)")

    model = make().fit(X, y)
    thr = float(np.quantile(neg, 1 - FA_TARGET))
    joblib.dump({"model": model, "threshold": thr, "fa_target": FA_TARGET, "win": WIN, "hop": HOP}, MODEL)
    print(f"\nsaved {MODEL}: threshold {thr:.3f} (from unseen stretches: {FA_TARGET:.0%} of not-swipe seconds fire)")


if __name__ == "__main__":
    main()
