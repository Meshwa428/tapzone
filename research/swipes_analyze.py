#!/usr/bin/env python3
"""The swipe experiment's analysis (recorded with research/swipes.py): does
the mic hear a swipe at all, does it differ from a tap, and can the two mics
tell a left swipe from a right one? Prints the numbers and writes
research/results/swipes/<session>-analysis.png.

    python research/swipes_analyze.py [research/results/swipes/swipes-<time>.json]

Each trial's sound is found inside its cued window (plus a little either
side: people react late) as the frames loud enough above the silence the
session began with, in the 1-12 kHz band where friction is.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "swipes")
BG, FG = "#0c0b0c", "#fafcfb"
COLORS = {"left": "#e06c75", "right": "#61afef", "tap": "#e5c07b"}
FRAME, HOP = 1024, 240          # ~21 ms frames every 5 ms
BAND = (1000, 12000)
LATE_S, EARLY_S = 0.6, 0.2      # look this far past the window's end, and before its start
ABOVE_DB = 8                    # a frame is "sound" this far above the silence
MAX_LAG = 40                    # samples either way for the time difference (~0.8 ms)


def load(path):
    meta = json.load(open(path))
    raw = np.fromfile(path.removesuffix(".json") + ".raw", "<i2").astype(np.float32) / 32768
    return meta, raw.reshape(-1, 2)


def stft(x):
    """Per frame and channel: power in BAND, and the complex spectra for the
    cross-channel delay."""
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.fft.rfft(x[idx] * np.hanning(FRAME)[None, :, None], axis=1)    # (n, bins, 2)
    f = np.fft.rfftfreq(FRAME, 1 / 48000)
    band = (f >= BAND[0]) & (f < BAND[1])
    return spec, band, f


def delay(spec, band):
    """GCC-PHAT per frame: how many samples the right mic lags the left."""
    cross = spec[:, :, 0] * np.conj(spec[:, :, 1])
    cross = np.where(band[None, :], cross / (np.abs(cross) + 1e-12), 0)
    cc = np.fft.irfft(cross, n=FRAME, axis=1)
    cc = np.concatenate([cc[:, -MAX_LAG:], cc[:, : MAX_LAG + 1]], axis=1)
    return np.argmax(cc, axis=1) - MAX_LAG, cc.max(axis=1)


def trial_sound(raw, t, sr, floor_db):
    """The trial's frames from its cue to a little after, and which of them
    are sound: the longest stretch above the floor, gaps under 60 ms closed."""
    a, b = t["go"][0] - int(EARLY_S * sr), t["go"][1] + int(LATE_S * sr)
    x = raw[max(0, a): b]
    spec, band, f = stft(x)
    p = (np.abs(spec[:, band]) ** 2).sum(1)              # (n, 2)
    db = 10 * np.log10(p.sum(1) + 1e-12)
    on = db > floor_db + ABOVE_DB
    # Close short gaps, then keep the stretch with the most energy.
    gap = int(0.06 * sr / HOP)
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return None
    runs, start = [], idx[0]
    for i, j in zip(idx[:-1], idx[1:]):
        if j - i > gap:
            runs.append((start, i))
            start = j
    runs.append((start, idx[-1]))
    s, e = max(runs, key=lambda r: (10 ** (db[r[0]: r[1] + 1] / 10)).sum())
    return {"x": x, "spec": spec, "band": band, "f": f, "p": p, "db": db, "s": s, "e": e + 1}


def features(snd, sr, floor_db):
    s, e, p, db = snd["s"], snd["e"], snd["p"], snd["db"]
    seg = snd["x"][s * HOP: e * HOP + FRAME].mean(1)
    energy = p[s:e].sum(1)
    w = energy / energy.sum()
    t = np.arange(e - s)
    ild = 10 * np.log10((p[s:e, 0] + 1e-12) / (p[s:e, 1] + 1e-12))     # + louder on the left mic
    lag, _ = delay(snd["spec"][s:e], snd["band"])
    k = max(1, (e - s) // 4)
    slope = lambda v: np.polyfit(t, v, 1, w=np.sqrt(w))[0] * sr / HOP if e - s > 2 else 0.0
    mag = np.abs(snd["spec"][s:e, :, :]).mean(2)
    centroid = (mag * snd["f"]).sum(1) / (mag.sum(1) + 1e-12)
    return {
        "dur_ms": (e - s) * HOP / sr * 1000,
        "peak_db": db[s:e].max() - floor_db,
        "crest": np.abs(seg).max() / (np.sqrt((seg ** 2).mean()) + 1e-12),
        "first50": energy[: int(0.05 * sr / HOP)].sum() / energy.sum(),
        "centroid_khz": (centroid * w).sum() / 1000,
        "ild_db": (ild * w).sum(),
        "ild_start": ild[:k].mean(),
        "ild_end": ild[-k:].mean(),
        "ild_change": ild[-k:].mean() - ild[:k].mean(),
        "ild_slope": slope(ild),
        "lag_start": lag[:k].mean(),
        "lag_end": lag[-k:].mean(),
        "lag_change": lag[-k:].mean() - lag[:k].mean(),
        "onset_ms": (s * HOP / sr - EARLY_S) * 1000,
    }, ild


def cv_score(X, y, seed_runs=20):
    """Mean and spread of 5-fold accuracy, repeated with reshuffles."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    s = cross_val_score(model, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=seed_runs, random_state=0))
    return s.mean(), s.std()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(DATA, "swipes-*.json")))[-1]
    meta, raw = load(path)
    sr = meta["sr"]
    trials = [t for t in meta["trials"] if not t["bad"]]
    quiet = next(t for t in trials if t["kind"] == "quiet")
    qspec, qband, _ = stft(raw[quiet["go"][0]: quiet["go"][1]])
    qdb = 10 * np.log10((np.abs(qspec[:, qband]) ** 2).sum((1, 2)) + 1e-12)
    floor_db = float(np.median(qdb))
    print(f"{os.path.basename(path)}: {len(trials) - 1} trials; silence {floor_db:.1f} dB "
          f"(95th percentile +{np.percentile(qdb, 95) - floor_db:.1f} dB) in {BAND[0] // 1000}-{BAND[1] // 1000} kHz")

    rows, kinds, curves, envs, sounds = [], [], [], [], {}
    missed = {"left": 0, "right": 0, "tap": 0}
    for t in trials:
        if t["kind"] == "quiet":
            continue
        snd = trial_sound(raw, t, sr, floor_db)
        if snd is None:
            missed[t["kind"]] += 1
            continue
        f, ild = features(snd, sr, floor_db)
        rows.append(f); kinds.append(t["kind"])
        curves.append(np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, len(ild)), ild) if len(ild) > 1 else np.full(20, ild[0]))
        env = snd["db"][snd["s"]: snd["s"] + int(1.0 * sr / HOP)] - floor_db
        envs.append(np.pad(env, (0, int(1.0 * sr / HOP) - len(env)), constant_values=np.nan))
        sounds.setdefault(t["kind"], snd)
    kinds = np.array(kinds)
    names = list(rows[0])
    F = np.array([[r[n] for n in names] for r in rows])
    print("not heard above the silence: " + ", ".join(f"{k} {v}" for k, v in missed.items()))

    print(f"\n{'':14}" + "".join(f"{k:>18}" for k in ("left", "right", "tap")))
    for j, n in enumerate(names):
        cells = []
        for k in ("left", "right", "tap"):
            v = F[kinds == k, j]
            cells.append(f"{np.median(v):8.2f} ±{np.std(v):6.2f}  ")
        print(f"{n:14}" + "".join(f"{c:>18}" for c in cells))

    swipe = kinds != "tap"
    lr = swipe
    shape = [names.index(n) for n in ("dur_ms", "peak_db", "crest", "first50", "centroid_khz")]
    stereo = [names.index(n) for n in ("ild_db", "ild_start", "ild_end", "ild_change", "ild_slope", "lag_start", "lag_end", "lag_change")]
    C = np.array(curves)
    print("\ncross-validated accuracy (5-fold × 20, logistic regression; chance shown):")
    for label, X, y in (
            ("swipe vs tap  · shape", F[:, shape], swipe),
            ("swipe vs tap  · everything", F, swipe),
            ("left vs right · stereo summary", F[lr][:, stereo], kinds[lr] == "left"),
            ("left vs right · stereo curve", C[lr], kinds[lr] == "left"),
            ("left vs right · shape only", F[lr][:, shape], kinds[lr] == "left"),
            ("left vs right · everything", np.hstack([F[lr], C[lr]]), kinds[lr] == "left")):
        m, s = cv_score(X, y)
        chance = max(y.mean(), 1 - y.mean())
        print(f"  {label:32} {m:6.1%} ± {s:4.1%}   (chance {chance:.0%}, n={len(y)})")

    plot(path, F, names, kinds, C, np.array(envs), sounds, sr, floor_db)


def plot(path, F, names, kinds, C, envs, sounds, sr, floor_db):
    plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "axes.edgecolor": "#555", "axes.facecolor": BG, "figure.facecolor": BG})
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)
    # Spectrograms, one of each.
    for i, k in enumerate(("left", "right", "tap")):
        ax = fig.add_subplot(gs[0, i])
        snd = sounds[k]
        s0 = max(0, snd["s"] - 20)
        S = 10 * np.log10(np.abs(snd["spec"][s0: s0 + 200]).mean(2) ** 2 + 1e-12).T
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=S.max() - 70, vmax=S.max(),
                  extent=[0, S.shape[1] * HOP / sr * 1000, 0, 24])
        ax.set_title(f"one {k}" + (" swipe" if k != "tap" else ""), color=COLORS[k])
        ax.set_xlabel("ms"); ax.set_ylabel("kHz")
    # Loudness over time, from the onset.
    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(envs.shape[1]) * HOP / sr * 1000
    for k in ("left", "right", "tap"):
        m = np.nanmedian(envs[kinds == k], axis=0)
        ax.plot(t, m, color=COLORS[k], label=k)
    ax.axhline(ABOVE_DB, color="#777", ls=":", lw=1)
    ax.set_title("loudness above silence (median)"); ax.set_xlabel("ms from onset"); ax.set_ylabel("dB")
    ax.legend(frameon=False)
    # Left/right balance across the sound.
    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 100, C.shape[1])
    for k in ("left", "right", "tap"):
        v = C[kinds == k]
        ax.plot(x, np.median(v, 0), color=COLORS[k], label=k)
        ax.fill_between(x, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=COLORS[k], alpha=0.15)
    ax.set_title("left mic minus right mic (median, IQR)"); ax.set_xlabel("% through the sound"); ax.set_ylabel("dB")
    ax.legend(frameon=False)
    # Feature scatters.
    for col, (a, b) in zip((2,), (("dur_ms", "crest"),)):
        ax = fig.add_subplot(gs[1, col])
        for k in ("left", "right", "tap"):
            m = kinds == k
            ax.scatter(F[m, names.index(a)], F[m, names.index(b)], s=14, color=COLORS[k], label=k)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title("swipe or tap"); ax.set_xscale("log")
    for col, (a, b) in enumerate((("ild_start", "ild_end"), ("ild_change", "lag_change"), ("ild_db", "ild_slope"))):
        ax = fig.add_subplot(gs[2, col])
        for k in ("left", "right"):
            m = kinds == k
            ax.scatter(F[m, names.index(a)], F[m, names.index(b)], s=16, color=COLORS[k], label=k)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title("left or right"); ax.legend(frameon=False)
    out = path.removesuffix(".json") + "-analysis.png"
    fig.savefig(out, dpi=90)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
