#!/usr/bin/env python3
"""Draws what the calibration taps look like, to see why quadrants are told
apart (or not). Writes research/results/analysis.png.

    python research/analyze.py [calibration.npz]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.expanduser("~/.local/share/tapzones")      # the recordings, read only
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SR, HOP, PRE_HOPS, FFT = 48000, 240, 4, 4096
ZONES = ["top-left", "top-right", "bottom-left", "bottom-right"]
COLORS = ["#e06c75", "#e5c07b", "#98c379", "#61afef"]
BG, FG = "#0c0b0c", "#fafcfb"

FREQS = np.fft.rfftfreq(FFT, 1 / SR)
EDGES = np.geomspace(80, 20000, 41)
BANDS = [(FREQS >= lo) & (FREQS < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])


def peak_of(mono):
    return int(np.argmax(np.abs(mono[: (PRE_HOPS + 8) * HOP])))


def spectrum_db(mono, p):
    seg = mono[max(0, p - 96): max(0, p - 96) + FFT]
    seg = np.pad(seg, (0, FFT - len(seg))) * np.hanning(FFT)
    spec = np.abs(np.fft.rfft(seg))
    db = 20 * np.log10(np.array([spec[b].mean() for b in BANDS]) + 1e-9)
    return db - db.max()


def envelope_db(mono, p):
    tail = mono[p: p + 12 * 192]
    tail = np.pad(tail, (0, 12 * 192 - len(tail)))
    env = 10 * np.log10((tail.reshape(12, 192) ** 2).mean(axis=1) + 1e-12)
    return env - env[0]


def features(spec, env):
    return np.concatenate([spec - spec.mean(), env * 0.5])


def lda_2d(X, y):
    """Fisher's linear discriminant: the 2D view that best pulls the
    quadrants apart relative to how much each one wobbles."""
    mu = X.mean(0)
    Sw = sum(np.cov(X[y == k].T) * (np.sum(y == k) - 1) for k in range(4))
    Sb = sum(np.sum(y == k) * np.outer(X[y == k].mean(0) - mu, X[y == k].mean(0) - mu) for k in range(4))
    vals, vecs = np.linalg.eig(np.linalg.pinv(Sw + 1e-3 * np.eye(len(Sw))) @ Sb)
    order = np.argsort(-vals.real)
    return (X - mu) @ vecs[:, order[:2]].real


def knn_mistakes(X, y, k=5):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    wrong = np.zeros(len(y), bool)
    for i in range(len(y)):
        d = np.linalg.norm(Xs - Xs[i], axis=1)
        d[i] = np.inf
        wrong[i] = np.bincount(y[np.argsort(d)[:k]], minlength=4).argmax() != y[i]
    return wrong


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "calibration.npz")
    d = np.load(path)
    windows, y = d["windows"], d["labels"].astype(int)
    mono = windows.mean(axis=2)
    peaks = [peak_of(m) for m in mono]
    specs = np.array([spectrum_db(m, p) for m, p in zip(mono, peaks)])
    envs = np.array([envelope_db(m, p) for m, p in zip(mono, peaks)])
    loud = np.array([20 * np.log10(np.abs(m).max() + 1e-9) for m in mono])
    X = np.array([features(s, e) for s, e in zip(specs, envs)])
    wrong = knn_mistakes(X, y)

    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "axes.edgecolor": "#444", "font.family": "monospace", "font.size": 9})
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"{len(y)} taps · {np.mean(~wrong):.0%} recognised (each judged by the others)", fontsize=14)

    # 1. Spectrum per quadrant, mean ± spread.
    a = ax[0, 0]
    for k in range(4):
        m, s = specs[y == k].mean(0), specs[y == k].std(0)
        a.semilogx(CENTERS, m, color=COLORS[k], lw=2, label=ZONES[k])
        a.fill_between(CENTERS, m - s, m + s, color=COLORS[k], alpha=0.12)
    a.set(title="1 · Sound fingerprint (which pitches ring)", xlabel="frequency (Hz)", ylabel="dB below loudest band")
    a.legend(frameon=False)

    # 2. Average waveform, aligned at the hit, loudness divided out.
    a = ax[0, 1]
    t = (np.arange(-96, 1440) / SR) * 1000
    for k in range(4):
        segs = [m[p - 96: p + 1440] / (np.abs(m).max() + 1e-9) for m, p, z in zip(mono, peaks, y) if z == k and p >= 96]
        a.plot(t, np.mean(segs, axis=0), color=COLORS[k], lw=1.2, label=ZONES[k])
    a.set(title="2 · What a tap looks like (first 30 ms)", xlabel="ms after the hit", ylabel="amplitude (normalised)")

    # 3. Decay.
    a = ax[0, 2]
    ms = np.arange(12) * 4
    for k in range(4):
        a.plot(ms, envs[y == k].mean(0), color=COLORS[k], lw=2, marker="o", ms=3, label=ZONES[k])
    a.set(title="3 · How fast the sound dies away", xlabel="ms after the hit", ylabel="dB vs the hit")

    # 4. Which frequencies carry the difference: between-quadrant spread
    #    divided by within-quadrant spread, per band (F-ratio).
    a = ax[1, 0]
    within = np.mean([specs[y == k].var(0) for k in range(4)], axis=0)
    between = np.var([specs[y == k].mean(0) for k in range(4)], axis=0)
    f = between / (within + 1e-9)
    a.bar(CENTERS, f, width=np.diff(EDGES) * 0.9, color="#b59790")
    a.set_xscale("log")
    a.set(title="4 · Where the useful difference is", xlabel="frequency (Hz)", ylabel="difference ÷ wobble (F-ratio)")

    # 5. All taps on the best 2D view; ✕ = recognised wrong.
    a = ax[1, 1]
    P = lda_2d(X, y)
    for k in range(4):
        ok = (y == k) & ~wrong
        a.scatter(P[ok, 0], P[ok, 1], color=COLORS[k], s=28, label=ZONES[k])
        bad = (y == k) & wrong
        a.scatter(P[bad, 0], P[bad, 1], color=COLORS[k], s=70, marker="x", linewidths=2)
    a.set(title="5 · Tap map: close dots sound alike (✕ = mistaken)", xticks=[], yticks=[])
    a.legend(frameon=False, loc="best")

    # 6. Loudness per quadrant.
    a = ax[1, 2]
    for k in range(4):
        v = loud[y == k]
        a.scatter(np.full(len(v), k) + np.random.uniform(-0.15, 0.15, len(v)), v, color=COLORS[k], s=20)
        a.hlines(v.mean(), k - 0.3, k + 0.3, color=COLORS[k], lw=2)
    a.set(title="6 · How hard each tap was", xticks=range(4), xticklabels=ZONES, ylabel="peak level (dBFS)")

    # 7. Where on the screen each tap was asked for; ✕ = recognised wrong.
    a = ax[0, 3]
    if "where" in d and len(d["where"]) == len(y):
        pos = d["where"]
        jitter = np.random.uniform(-0.012, 0.012, pos.shape)
        for k in range(4):
            ok, bad = (y == k) & ~wrong, (y == k) & wrong
            a.scatter(pos[ok, 0] + jitter[ok, 0], pos[ok, 1] + jitter[ok, 1], color=COLORS[k], s=22)
            a.scatter(pos[bad, 0], pos[bad, 1], color=COLORS[k], s=90, marker="x", linewidths=2.5)
        a.axhline(0.5, color="#444")
        a.axvline(0.5, color="#444")
        a.set(xlim=(0, 1), ylim=(1, 0), aspect=768 / 1366)
    else:
        a.text(0.5, 0.5, "no tap positions\n(calibrated before spots)", ha="center", va="center", alpha=0.5)
    a.set(title="7 · Where you tapped (✕ = mistaken)", xticks=[], yticks=[])

    # 8. Taps that hit the mic's ceiling, per quadrant.
    a = ax[1, 3]
    clipped = (np.abs(windows) >= 0.99).any(axis=(1, 2))
    a.bar(range(4), [clipped[y == k].mean() * 100 for k in range(4)], color=COLORS)
    a.set(title="8 · Taps too loud for the mic (clipped)", xticks=range(4), xticklabels=ZONES, ylabel="% of taps", ylim=(0, 100))

    for a in ax.flat:
        a.grid(alpha=0.12)
    fig.tight_layout()
    out = os.path.join(HERE, "analysis.png")
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    main()
