#!/usr/bin/env python3
"""Look at the taps themselves: waveforms, mel spectrograms, and how alike
every pair of taps is, by quadrant, grid spot and time.

    python research/viz.py [collect-….npz]   →  ~/.local/share/tapzones/viz-*.png
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tapzones import features
from tapzones.paths import DATA

ZONES = features.ZONES
COLORS = ["#e06c75", "#e5c07b", "#98c379", "#61afef"]
BG, FG = "#0c0b0c", "#fafcfb"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#444", "font.family": "monospace", "font.size": 9})


def load(path):
    d = np.load(path)
    X = np.array([features.aligned(w) for w in d["windows"]])
    return X, d["truth"].astype(int), d["source"], d["where"]


def similarity(X, ms=20, lag=24):
    """How alike two taps are: the best correlation of their first `ms`
    milliseconds, letting them slide ±0.5 ms against each other."""
    n = int(ms * features.SR / 1000)
    A = X[:, 48 - lag: 48 + n + lag]
    core = X[:, 48: 48 + n]
    core = (core - core.mean(1, keepdims=True)) / (core.std(1, keepdims=True) + 1e-9)
    best = np.full((len(X), len(X)), -1.0)
    for s in range(0, 2 * lag + 1, 2):
        B = A[:, s: s + n]
        B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
        best = np.maximum(best, core @ B.T / n)
    return np.maximum(best, best.T)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(DATA, "collect-*.npz")))[-1]
    X, y, src, where = load(path)
    name = os.path.basename(path).removesuffix(".npz")
    t_ms = (np.arange(-48, 1200) / features.SR) * 1000
    mels = np.array([features.logmel(x) for x in X])
    spot = np.array([int(s.split()[-1]) if s != "random" else 0 for s in src])

    # ---------------------------------------------------- 1. waveforms
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    fig.suptitle(f"{name}: every tap, lined up at the hit (amplitude normalised)", fontsize=13)
    for k in range(4):
        a = ax[0, k]
        seg = X[y == k][:, :1248]
        seg = seg / (np.abs(seg).max(1, keepdims=True) + 1e-9)
        for s in seg:
            a.plot(t_ms, s, color=COLORS[k], alpha=0.08, lw=0.7)
        a.plot(t_ms, seg.mean(0), color=FG, lw=1.6)
        a.set(title=f"{ZONES[k]} · {len(seg)} taps", ylim=(-1.05, 1.05), xlabel="ms")
        # zoom: first 5 ms
        a = ax[1, k]
        for s in seg:
            a.plot(t_ms[:300], s[:300], color=COLORS[k], alpha=0.12, lw=0.8)
        for j in range(4):
            m = X[y == j][:, :300]
            m = (m / (np.abs(m).max(1, keepdims=True) + 1e-9)).mean(0)
            a.plot(t_ms[:300], m, color=COLORS[j], lw=2 if j == k else 1, alpha=1 if j == k else 0.5)
        a.set(title="first 5 ms (thick: this quadrant's average,\nthin: the others' averages)", ylim=(-1.05, 1.05), xlabel="ms")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "viz-1-waveforms.png"), dpi=80)

    # -------------------------------------------- 2. mel spectrograms
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    fig.suptitle("Average mel spectrogram per quadrant (top), and its difference from the overall average (bottom)", fontsize=13)
    overall = mels.mean(0)
    ext = [0, 32 * 112 / features.SR * 1000, 0, 64]
    for k in range(4):
        m = mels[y == k].mean(0)
        a = ax[0, k]
        a.imshow(m, origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0, extent=ext)
        a.set(title=ZONES[k], xlabel="ms", ylabel="mel band (low → high pitch)")
        a = ax[1, k]
        im = a.imshow(m - overall, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=ext)
        a.set(title=f"{ZONES[k]} − average  (dB)", xlabel="ms")
        # How consistent: the difference relative to the tap-to-tap spread.
        spread = mels[y == k].std(0).mean()
        a.text(2, 58, f"typical tap-to-tap spread ±{spread:.0f} dB", color=FG, fontsize=8)
    fig.colorbar(im, ax=ax[1, :], shrink=0.6, label="dB")
    fig.savefig(os.path.join(DATA, "viz-2-mel.png"), dpi=80)

    # ------------------------------------------ 3/4. similarity matrices
    S = similarity(X)
    order_q = np.lexsort((np.arange(len(y)), spot, y))      # by quadrant, then grid spot
    order_t = np.arange(len(y))                              # as tapped
    fig, ax = plt.subplots(1, 2, figsize=(18, 8.6))
    for a, order, title in ((ax[0], order_q, "sorted by quadrant (then grid spot)"),
                            (ax[1], order_t, "in the order you tapped")):
        a.imshow(S[np.ix_(order, order)], cmap="magma", vmin=0.2, vmax=0.95)
        a.set(title=f"How alike every pair of taps is (waveform, first 20 ms)\n{title}", xticks=[], yticks=[])
        # Colour strip: which quadrant each row is.
        for i, j in enumerate(order):
            a.add_patch(plt.Rectangle((-8, i - 0.5), 6, 1, color=COLORS[y[j]], clip_on=False))
    same = S[y[:, None] == y[None, :]].mean()
    diff = S[y[:, None] != y[None, :]].mean()
    near = np.abs(np.arange(len(y))[:, None] - np.arange(len(y))[None, :]) <= 4
    fig.suptitle(f"average likeness: same quadrant {same:.3f} · different quadrants {diff:.3f} · "
                 f"tapped within 4 taps of each other {S[near & ~np.eye(len(y), dtype=bool)].mean():.3f}", fontsize=12)
    fig.savefig(os.path.join(DATA, "viz-3-similarity.png"), dpi=80)

    # ------------------------------------ 5/6. spots and a map of taps
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    a = ax[0]
    centers = np.arange(64)
    for k in range(4):
        for s_ in range(1, 10):
            sel = (y == k) & (spot == s_)
            if sel.any():
                a.plot(centers, mels[sel].mean(axis=(0, 2)), color=COLORS[k], alpha=0.7, lw=1.2)
    a.set(title="Average spectrum of each of the 36 grid spots\n(colour = quadrant)", xlabel="mel band (low → high)", ylabel="dB")

    # PCA of the mel pictures.
    F = mels.reshape(len(mels), -1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    U, s_, Vt = np.linalg.svd(F, full_matrices=False)
    P = U[:, :2] * s_[:2]
    for k in range(4):
        ax[1].scatter(P[y == k, 0], P[y == k, 1], color=COLORS[k], s=14, label=ZONES[k])
    ax[1].legend(frameon=False)
    ax[1].set(title="Every tap, squashed to 2D (PCA of the mel picture)\ncoloured by quadrant", xticks=[], yticks=[])
    sc = ax[2].scatter(P[:, 0], P[:, 1], c=np.arange(len(y)), cmap="viridis", s=14)
    fig.colorbar(sc, ax=ax[2], label="tap number (time)")
    ax[2].set(title="The same map, coloured by when you tapped", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "viz-4-spots-map.png"), dpi=80)

    # ------------------------------- 7. which way the first swing goes
    # The first sample past 40% of the peak: its sign says whether the
    # lid first pushed air towards the mic or pulled it away.
    head = X[:, :48 + 480]
    first = np.argmax(np.abs(head) > 0.4 * np.abs(head).max(1, keepdims=True), axis=1)
    sign = np.sign(head[np.arange(len(X)), first])
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    a = ax[0]
    for s_, c, lab in ((-1, "#61afef", "first swing down"), (1, "#e06c75", "first swing up")):
        sel = sign == s_
        a.scatter(where[sel, 0], where[sel, 1], color=c, s=22, label=f"{lab} ({sel.sum()})")
    a.axvline(where[:, 0].min() / 2 + where[:, 0].max() / 2, color="#444")
    a.axhline(where[:, 1].min() / 2 + where[:, 1].max() / 2, color="#444")
    a.invert_yaxis()
    a.set(title="Where you tapped, coloured by which way the first swing went", xlabel="x px", ylabel="y px")
    a.legend(frameon=False, loc="upper right")
    a = ax[1]
    up = [(sign[y == k] > 0).mean() * 100 for k in range(4)]
    a.bar(ZONES, up, color=COLORS)
    for k, v in enumerate(up):
        a.text(k, v + 1.5, f"{v:.0f}%", ha="center")
    a.set(title="% of taps whose first swing goes up", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "viz-5-polarity.png"), dpi=80)

    # Does keeping the sign help? Train on the grid taps, test on the random ones.
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    wave = X[:, :48 + 1200].reshape(len(X), -1, 12).mean(2)        # 25 ms at 4 kHz
    wave = wave / np.abs(wave).max(1, keepdims=True)
    train, test = src != "random", src == "random"
    print(f"\ngrid → random ({train.sum()} → {test.sum()} taps), chance 25%")
    for label, F_ in (("low-pass waveform, sign kept", wave),
                      ("same, sign thrown away", wave * -sign[:, None])):
        for mname, m in (("logistic", LogisticRegression(C=0.3, max_iter=3000)),
                         ("svm", SVC(C=3)), ("knn", KNeighborsClassifier(7))):
            acc = make_pipeline(StandardScaler(), m).fit(F_[train], y[train]).score(F_[test], y[test])
            print(f"  {label:30} {mname:9} {acc * 100:5.1f}%")
    print("  first-swing sign alone        " +
          f"top={((sign < 0) == (y < 2))[test].mean() * 100:.0f}% (top vs bottom only)")

    print(f"same-quadrant likeness {same:.3f}, different-quadrant {diff:.3f}")
    for k in range(4):
        row = [S[np.ix_(y == k, y == j)].mean() for j in range(4)]
        print(f"  {ZONES[k]:13}" + "  ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
