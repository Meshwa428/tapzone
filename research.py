#!/usr/bin/env python3
"""Every idea for hearing the quadrant, scored the same way, logged.

    .venv/bin/python research.py EXPERIMENT…    run them, append to results.tsv
    .venv/bin/python research.py --list         what experiments exist
    .venv/bin/python research.py --best [N]     the N best rows so far

The exam: leave one session out over every session (pooled accuracy over all
taps is the headline, chance 25%), plus grid → random inside the balanced
collect session. A model never sees the session it's scored on.
"""
import os, sys, time
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import ml

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.tsv")
SR, HOP = ml.SR, ml.HOP


def load():
    """[(name, stereo windows (n, 8400, 2), labels, source, where)] for every
    session; `where` is each tap's spot as screen fractions."""
    out = []
    for path in ml.session_paths():
        d = np.load(path)
        y = d["labels"] if "labels" in d else d["truth"]
        src = d["source"] if "source" in d else np.full(len(y), "")
        out.append((os.path.basename(path).removesuffix(".npz"), d["windows"].astype(np.float32), y.astype(int), src,
                    np.asarray(d["where"], float)))
    return out


FULL = load()
S = [s[:4] for s in FULL]
WHERE = np.concatenate([s[4] for s in FULL])
W_ALL = np.concatenate([w for _, w, _, _ in S])
Y_ALL = np.concatenate([y for _, _, y, _ in S])
SESSION = np.concatenate([np.full(len(y), i) for i, (_, _, y, _) in enumerate(S)])
BAL = max(range(len(S)), key=lambda i: ((S[i][3] == "random").any() and (S[i][3] != "random").any(), len(S[i][2])))
GRID = (SESSION == BAL) & (np.concatenate([s for *_, s in S]) != "random")
RAND = (SESSION == BAL) & (np.concatenate([s for *_, s in S]) == "random")

# ------------------------------------------------------------- features

from ml import channel, starts, lowpass, signed, BEST1

# --------------------------------------------------------------- models

def models(names=("logistic", "svm", "knn")):
    zoo = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
        "svm": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
        "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeClassifier(10)),
        "rf": lambda: RandomForestClassifier(500, random_state=0),
        "extratrees": lambda: ExtraTreesClassifier(500, random_state=0),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0)),
    }
    return {n: zoo[n] for n in names}

# ----------------------------------------------------------------- exam

def _fold(make, F, train, test, extra):
    # `extra`: more versions of every tap (e.g. shifted), training rows only.
    model = make()
    target = WHERE if getattr(model, "xy", False) else Y_ALL
    X = np.concatenate([F[train]] + [E[train] for E in extra])
    y = np.concatenate([target[train]] * (1 + len(extra)))
    return test, model.fit(X, y).predict(F[test])


def exam(F, make, extra=()):
    """(pooled leave-one-session-out, per-session mean, grid → random)."""
    jobs = [(SESSION != i, SESSION == i) for i in range(len(S))] + [(GRID, RAND)]
    res = Parallel(n_jobs=4)(delayed(_fold)(make, F, tr, te, extra) for tr, te in jobs)
    pred = np.empty(len(Y_ALL), int)
    per = []
    for (te, p), i in zip(res[:-1], range(len(S))):
        pred[te] = p
        per.append((p == Y_ALL[te]).mean())
    te, p = res[-1]
    return (pred == Y_ALL).mean(), np.mean(per), (p == Y_ALL[te]).mean(), pred


def log(experiment, variant, model, scores):
    pooled, mean, grid, pred = scores
    tb = ((pred >= 2) == (Y_ALL >= 2)).mean()        # top vs bottom right?
    lr = ((pred % 2) == (Y_ALL % 2)).mean()          # left vs right right?
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a") as f:
        if new:
            f.write("time\texperiment\tvariant\tmodel\tloso_pooled\tloso_mean\tgrid_random\ttaps\n")
        f.write(f"{time.strftime('%H:%M:%S')}\t{experiment}\t{variant}\t{model}\t{pooled:.4f}\t{mean:.4f}\t{grid:.4f}\t{len(Y_ALL)}\n")
    print(f"  {variant:44} {model:11} pooled {pooled:6.1%}  mean {mean:6.1%}  grid→rand {grid:6.1%}"
          f"  | top/bot {tb:5.1%} left/right {lr:5.1%}", flush=True)


def run(experiment, variants, names=("logistic", "svm", "knn")):
    """variants: {label: features, or (features, [extra training versions])}.
    Each scored with each model (a name, or (label, maker)), logged."""
    print(f"\n== {experiment}")
    makers = models(names) if all(isinstance(n, str) for n in names) else dict(names)
    for label, F in variants.items():
        F, extra = F if isinstance(F, tuple) else (F, ())
        for name, make in makers.items():
            log(experiment, label, name, exam(F, make, extra))

# ---------------------------------------------------------- experiments

EXPERIMENTS = {}

def experiment(fn):
    EXPERIMENTS[fn.__name__] = fn
    return fn


@experiment
def baseline():
    run("baseline", {"wave4k 40ms (ml.py --signed)": signed(W_ALL)})

@experiment
def length():
    run("length", {f"{ms} ms": signed(W_ALL, ms=ms) for ms in (10, 20, 30, 40, 60, 80, 120)})

@experiment
def rate():
    run("rate", {f"{SR // d // 1000 if d < 48 else 1} kHz (÷{d})": signed(W_ALL, dec=d) for d in (3, 6, 12, 24, 48)})

@experiment
def lead():
    run("lead", {f"{b} ms before onset": signed(W_ALL, before=b) for b in (0, 1, 2, 4, 8)})

@experiment
def onset():
    run("onset", {f"onset at {f}": signed(W_ALL, frac=f) for f in (0.1, 0.2, 0.4, 0.6, "peak")})

@experiment
def norm():
    run("norm", {"peak": signed(W_ALL), "rms": signed(W_ALL, norm="rms"),
                 "none": signed(W_ALL, norm="none"), "peak + loudness": signed(W_ALL, loud=True),
                 "rms + loudness": signed(W_ALL, norm="rms", loud=True)})

@experiment
def filters():
    run("filters", {"block mean ÷12": signed(W_ALL),
                    **{f"butterworth {c} Hz": signed(W_ALL, lp=c) for c in (2000, 1000, 500, 300, 150)}})

@experiment
def baseline_dc():
    run("baseline_dc", {"no dc removal": signed(W_ALL), "pre-onset mean removed": signed(W_ALL, dc=True)})

@experiment
def stereo():
    m = signed(W_ALL)
    d = signed(W_ALL, chan="diff")
    run("stereo", {"mono": m, "left": signed(W_ALL, chan="left"), "right": signed(W_ALL, chan="right"),
                   "L−R only": d, "mono + L−R": np.hstack([m, d])})


@experiment
def combo():
    run("combo", {
        "frac .2": signed(W_ALL, frac=0.2),
        "frac .2 + dc": signed(W_ALL, frac=0.2, dc=True),
        "frac .2 + dc + loud": signed(W_ALL, frac=0.2, dc=True, loud=True),
        "frac .2 + dc + loud + rms": signed(W_ALL, **BEST1),
        "frac .2 + dc + loud + rms, 30 ms": signed(W_ALL, **{**BEST1, "ms": 30}),
        "frac .1 + dc + loud + rms": signed(W_ALL, **{**BEST1, "frac": 0.1}),
        "frac .2 + dc + loud + rms, 1 kHz": signed(W_ALL, **{**BEST1, "dec": 48}),
    })

@experiment
def zoo():
    run("zoo", {"best1": signed(W_ALL, **BEST1)},
        ("logistic", "svm", "knn", "lda", "ridge", "rf", "extratrees", "hgb", "mlp"))

@experiment
def svm_grid():
    F = signed(W_ALL, **BEST1)
    run("svm_grid", {"best1": F}, [(f"C={c} g={g}", (lambda c=c, g=g: make_pipeline(StandardScaler(), SVC(C=c, gamma=g))))
                                   for c in (0.3, 1, 3, 10, 30) for g in ("scale", 0.001, 0.003, 0.01)])

@experiment
def logistic_grid():
    F = signed(W_ALL, **BEST1)
    run("logistic_grid", {"best1": F}, [(f"C={c}", (lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))))
                                        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1)])

@experiment
def knn_grid():
    F = signed(W_ALL, **BEST1)
    run("knn_grid", {"best1": F}, [(f"k={k} {m}", (lambda k=k, m=m: make_pipeline(StandardScaler(), KNeighborsClassifier(k, metric=m, weights="distance"))))
                                   for k in (3, 7, 15, 31) for m in ("euclidean", "cosine")])

@experiment
def shifts():
    """Train on copies of each tap cut a little early and late, so a model
    forgives the onset landing a sample or two off."""
    F = signed(W_ALL, **BEST1)
    for spread in ((-12, 12), (-24, 24), (-48, -24, 24, 48), (-24, -12, 12, 24)):
        run("shifts", {f"train shifts {spread} samples": (F, [signed(W_ALL, **BEST1, shift=s) for s in spread])})


BEST_SHIFTS = (-24, -12, 12, 24)

def best2(W=None):
    """best1 features, plus the shifted training copies."""
    W = W_ALL if W is None else W
    return signed(W, **BEST1), [signed(W, **BEST1, shift=s) for s in BEST_SHIFTS]


def per_session_z(F):
    """Each session standardised by its own taps' mean and spread: no labels
    used, so live it would be a running average over your recent taps."""
    out = F.copy()
    for i in range(len(S)):
        m = SESSION == i
        out[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
    return out


@experiment
def session_norm():
    F, extra = best2()
    run("session_norm", {"best2": (F, extra), "best2, per-session z": (per_session_z(F), [per_session_z(E) for E in extra])},
        ("logistic", "svm", "knn", "mlp"))


def multiband(W, bands=((20, 200), (200, 800), (800, 3000), (3000, 10000)), shift=0):
    """One signed waveform per frequency band, each normalised on its own so
    a quiet high band isn't drowned by the loud low wobble. Low bands keep
    40 ms at a coarse rate, high bands the first 10 ms at a fine one."""
    mono = W.mean(axis=2)
    i0 = np.maximum(starts(mono, 0.2) - 2 * SR // 1000 + shift, 0)
    parts = []
    for lo, hi in bands:
        x = sosfiltfilt(butter(4, (lo, hi), "bandpass", fs=SR, output="sos"), mono, axis=1)
        ms, dec = (40, 48) if hi <= 200 else (20, 12) if hi <= 800 else (10, 4) if hi <= 3000 else (6, 2)
        n = ms * SR // 1000
        x = np.pad(x, ((0, 0), (0, n)))
        seg = np.stack([x[k, i: i + n] for k, i in enumerate(i0)])
        seg = seg.reshape(len(seg), -1, dec).mean(2)
        e = np.sqrt((seg ** 2).mean(1, keepdims=True))
        parts += [seg / (e + 1e-9), np.log10(e + 1e-9)]
    return np.hstack(parts).astype(np.float32)


@experiment
def bands():
    F, extra = best2()
    M = multiband(W_ALL)
    Mx = [multiband(W_ALL, shift=s) for s in BEST_SHIFTS]
    hi = multiband(W_ALL, bands=((200, 800), (800, 3000), (3000, 10000)))
    run("bands", {"multiband (4 bands)": (M, Mx),
                  "high bands only (200 Hz–10 kHz)": hi,
                  "best2 + multiband": (np.hstack([F, M]), [np.hstack([a, b]) for a, b in zip(extra, Mx)])},
        ("logistic", "svm", "mlp"))


class ViaXY:
    """Learn where on the screen (x, y) the tap was, then call the quadrant
    from that. The exact spot is a richer label than the quadrant alone.
    `xy = True` tells the exam to train it on screen positions."""
    xy = True

    def __init__(self, reg):
        self.reg = reg

    def fit(self, X, where):
        self.reg.fit(X, where)
        return self

    def predict(self, X):
        p = self.reg.predict(X)
        return (p[:, 0] >= 0.5).astype(int) + 2 * (p[:, 1] >= 0.5).astype(int)


class Hierarchical:
    """Two yes/no questions instead of one four-way one: top or bottom?
    left or right? Each with its own model."""
    def __init__(self, make):
        self.tb, self.lr = make(), make()

    def fit(self, X, y):
        self.tb.fit(X, y >= 2)
        self.lr.fit(X, y % 2)
        return self

    def predict(self, X):
        return self.lr.predict(X).astype(int) + 2 * self.tb.predict(X).astype(int)


@experiment
def targets():
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.neighbors import KNeighborsRegressor
    F, extra = best2()
    svm = lambda: make_pipeline(StandardScaler(), SVC(C=3))
    run("targets", {"best2": (F, extra)}, [
        ("svm (4-way)", svm),
        ("svm hierarchical", lambda: Hierarchical(svm)),
        ("logistic hierarchical", lambda: Hierarchical(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)))),
        ("xy ridge", lambda: ViaXY(make_pipeline(StandardScaler(), Ridge(10)))),
        ("xy svr", lambda: ViaXY(make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, epsilon=0.05))))),
        ("xy knn", lambda: ViaXY(make_pipeline(StandardScaler(), KNeighborsRegressor(7, weights="distance")))),
        ("xy mlp", lambda: ViaXY(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=2000, random_state=0)))),
    ])


def modal(W, ms=80, fmax=600, n_fft=4096, frac=0.2):
    """The lid's resonances: log magnitude of each low frequency, plus the
    phase step between neighbouring frequencies with its mean removed. A
    tap cut a little early or late only adds that mean, so this part
    doesn't care where the onset landed."""
    mono = W.mean(axis=2)
    i0 = np.maximum(starts(mono, frac) - 2 * SR // 1000, 0)
    n = ms * SR // 1000
    x = np.pad(mono, ((0, 0), (0, n)))
    seg = np.stack([x[k, i: i + n] for k, i in enumerate(i0)]) * np.hanning(n)
    Z = np.fft.rfft(seg, n_fft, axis=1)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    Z = Z[:, (f >= 20) & (f <= fmax)]
    mag = np.log10(np.abs(Z) + 1e-9)
    mag -= mag.max(1, keepdims=True)
    step = Z[:, 1:] * np.conj(Z[:, :-1])
    step /= np.abs(step) + 1e-12
    mean = step.sum(1, keepdims=True)
    rel = step * np.conj(mean / (np.abs(mean) + 1e-12))       # phase steps, mean phase step removed
    return np.hstack([mag, rel.real, rel.imag]).astype(np.float32)


@experiment
def modes():
    F = signed(W_ALL, **BEST1)
    run("modes", {
        **{f"modal {ms} ms ≤{fm} Hz": modal(W_ALL, ms=ms, fmax=fm) for ms, fm in ((40, 600), (80, 600), (80, 300), (120, 400))},
        "best1 + modal 80 ms": np.hstack([F, modal(W_ALL)]),
    }, ("logistic", "svm", "mlp"))


class TorchCNN:
    """ml.cnn as a scikit-learn style model: `seeds` copies, probabilities
    averaged. One thread each, the exam runs four folds at once."""
    def __init__(self, epochs=40, seeds=3):
        self.epochs, self.seeds = epochs, seeds

    def fit(self, X, y):
        self.nets = [ml.train_cnn(X.astype(np.float32), np.asarray(y), s, self.epochs, threads=1) for s in range(self.seeds)]
        return self

    def proba(self, X):
        import torch
        with torch.no_grad():
            x = torch.tensor(X.astype(np.float32)[:, None])
            return np.mean([torch.softmax(n(x), 1).numpy() for n in self.nets], axis=0)

    def predict(self, X):
        return self.proba(X).argmax(1)


WAVE = ml.WAVE

@experiment
def cnn():
    F = signed(W_ALL, **WAVE)
    Fs = [signed(W_ALL, **WAVE, shift=s) for s in BEST_SHIFTS]
    run("cnn", {"wave 40 ms 4 kHz": F, "wave 40 ms 4 kHz + shifted copies": (F, Fs),
                "wave 40 ms (ml.py wave4k)": signed(W_ALL)},
        [("cnn 40 ep", lambda: TorchCNN(40)), ("cnn 80 ep", lambda: TorchCNN(80))])


@experiment
def curve():
    """Is data still the limit? Leave one session out, but train on only a
    fraction of the other sessions' taps."""
    F, extra = best2()
    print("\n== curve (svm, best2)")
    for frac in (0.25, 0.5, 0.75, 1.0):
        accs = []
        for seed in range(3 if frac < 1 else 1):
            keep = np.random.default_rng(seed).random(len(Y_ALL)) < frac
            correct = 0
            for i in range(len(S)):
                tr, te = (SESSION != i) & keep, SESSION == i
                m = make_pipeline(StandardScaler(), SVC(C=3)).fit(
                    np.concatenate([F[tr]] + [E[tr] for E in extra]), np.concatenate([Y_ALL[tr]] * (1 + len(extra))))
                correct += (m.predict(F[te]) == Y_ALL[te]).sum()
            accs.append(correct / len(Y_ALL))
        n = int(frac * len(Y_ALL) * (len(S) - 1) / len(S))
        print(f"  ~{n:4} training taps   pooled {np.mean(accs):6.1%}  " + "█" * int(np.mean(accs) * 50), flush=True)
        log("curve", f"{frac:.0%} of training taps", "svm", (np.mean(accs), np.nan, np.nan, np.full(len(Y_ALL), -1)))


class Ensemble:
    """Soft vote: average the members' probabilities. Each member is
    (maker, columns) so the CNN can skip the loudness column."""
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.fitted = [(make().fit(X[:, cols], y), cols) for make, cols in self.members]
        return self

    def predict(self, X):
        p = [m.proba(X[:, c]) if hasattr(m, "proba") else m.predict_proba(X[:, c]) for m, c in self.fitted]
        return np.mean(p, axis=0).argmax(1)


@experiment
def ensemble():
    F, extra = best2()
    allc, wave = slice(None), slice(0, -1)
    svm = lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0))
    mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0))
    et = lambda: ExtraTreesClassifier(500, random_state=0)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    run("ensemble", {"best2": (F, extra)}, [
        ("svm+mlp", lambda: Ensemble([(svm, allc), (mlp, allc)])),
        ("svm+mlp+et", lambda: Ensemble([(svm, allc), (mlp, allc), (et, allc)])),
        ("svm+mlp+et+lr", lambda: Ensemble([(svm, allc), (mlp, allc), (et, allc), (lr, allc)])),
        ("svm+mlp+cnn", lambda: Ensemble([(svm, allc), (mlp, allc), (lambda: TorchCNN(40), wave)])),
    ])


BANK = np.concatenate([w.mean(axis=1)[:720] for w in W_ALL])     # real background noise


def augmented_W(seed, speed=False, eq=False, noise=False, gain=False):
    """One made-up copy of every tap (stereo layout kept), with only the
    chosen effects. Per tap, so a copy only ever trains when its tap does."""
    rng = np.random.default_rng(seed)
    mono = W_ALL.mean(axis=2)
    n = mono.shape[1]
    out = mono.copy()
    for k in range(len(out)):
        y = out[k]
        if eq:
            spec = np.fft.rfft(y)
            f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 20)
            knots = np.log(np.geomspace(20, SR / 2, 6))
            spec *= 10 ** (np.interp(np.log(f), knots, rng.uniform(-6, 6, 6)) / 20)
            y = np.fft.irfft(spec, n)
        if speed:
            r = rng.uniform(0.95, 1.05)
            y = np.interp(np.arange(n) * r, np.arange(n), y, right=0)
        if gain:
            y = y * 10 ** (rng.uniform(-10, 10) / 20)
        if noise:
            s = rng.integers(0, len(BANK) - n)
            y = y + BANK[s: s + n] * 10 ** (rng.uniform(-6, 12) / 20)
        out[k] = y
    return np.repeat(out[:, :, None], 2, axis=2).astype(np.float32)


@experiment
def augment_kinds():
    F, extra = best2()
    for label, kw in (("speed ±5%", dict(speed=True)), ("eq ±6 dB", dict(eq=True)),
                      ("noise + gain", dict(noise=True, gain=True)),
                      ("all four", dict(speed=True, eq=True, noise=True, gain=True))):
        aug = [signed(augmented_W(s, **kw), **BEST1) for s in range(4)]
        run("augment_kinds", {f"best2 + 4 copies, {label}": (F, extra + aug)}, ("svm", "mlp"))


@experiment
def physics():
    """The mic hears pressure. Its running sum is closer to how far the lid
    moved; its difference, to how fast."""
    mono = W_ALL.mean(axis=2)
    disp = np.cumsum(mono - mono[:, :720].mean(1, keepdims=True), axis=1)
    vel = np.diff(mono, axis=1, prepend=mono[:, :1])
    as_W = lambda x: np.repeat(x[:, :, None], 2, axis=2)
    cut = lambda x: signed(as_W(x), **{**BEST1, "loud": False, "dc": False})
    F = signed(W_ALL, **BEST1)
    run("physics", {"displacement (cumsum)": cut(disp), "velocity (diff)": cut(vel),
                    "pressure + displacement": np.hstack([F, cut(disp)]),
                    "pressure + displacement + velocity": np.hstack([F, cut(disp), cut(vel)])},
        ("logistic", "svm", "mlp"))


def xcorr_aligned(W, iters=3, lag=48, **kw):
    """Cut each tap where it best matches the average tap (±1 ms), judged by
    |correlation| so a flipped tap aligns as well as an upright one; the
    sign itself is kept in the features."""
    shift = np.zeros(len(W), int)
    for _ in range(iters):
        F = signed(W, **{**kw, "loud": False}, shift=0) if False else np.stack(
            [signed(W[k: k + 1], **{**kw, "loud": False}, shift=int(shift[k]))[0] for k in range(len(W))])
        sign = np.sign((F * F.mean(0)).sum(1))
        tmpl = (F * sign[:, None]).mean(0)
        best = np.zeros(len(W), int)
        score = np.full(len(W), -np.inf)
        for s in range(-lag, lag + 1, 6):
            G = np.stack([signed(W[k: k + 1], **{**kw, "loud": False}, shift=int(shift[k] + s))[0] for k in range(len(W))])
            c = np.abs(G @ tmpl) / (np.linalg.norm(G, axis=1) * np.linalg.norm(tmpl) + 1e-9)
            better = c > score
            score[better], best[better] = c[better], s
        shift += best
    return shift


@experiment
def align():
    shift = xcorr_aligned(W_ALL, **WAVE)
    print(f"  cross-correlation moved taps by {np.abs(shift).mean() / 48:.2f} ms on average (max {np.abs(shift).max() / 48:.2f})")
    F = np.stack([signed(W_ALL[k: k + 1], **BEST1, shift=int(shift[k]))[0] for k in range(len(W_ALL))])
    Fx = [np.stack([signed(W_ALL[k: k + 1], **BEST1, shift=int(shift[k] + s))[0] for k in range(len(W_ALL))]) for s in BEST_SHIFTS]
    run("align", {"onset 20% (best2)": best2(), "xcorr to average tap": (F, Fx)}, ("logistic", "svm", "mlp"))


def extrema(W, n_ext=8, frac=0.2):
    """The first `n_ext` swings after the onset, as signed size (relative to
    the biggest) and time (ms): a short physical description of the tap."""
    x = lowpass(W.mean(axis=2), 1000)
    i0 = starts(W.mean(axis=2), frac)
    out = []
    for k in range(len(x)):
        seg = x[k, i0[k]: i0[k] + 60 * SR // 1000]
        turn = np.where(np.diff(np.sign(np.diff(seg))) != 0)[0] + 1
        turn = turn[np.abs(seg[turn]) > 0.02 * np.abs(seg).max()][:n_ext]
        amp = np.zeros(n_ext); t = np.full(n_ext, 60.0)
        amp[: len(turn)] = seg[turn] / (np.abs(seg).max() + 1e-9)
        t[: len(turn)] = turn / SR * 1000
        out.append(np.concatenate([amp, t]))
    return np.array(out, np.float32)


@experiment
def swings():
    E = extrema(W_ALL)
    F = signed(W_ALL, **BEST1)
    run("swings", {"first 8 swings": E, "first 12 swings": extrema(W_ALL, 12), "best1 + 8 swings": np.hstack([F, E])},
        ("logistic", "svm", "extratrees", "mlp"))


@experiment
def timeline():
    """When in the sound is each answer? Up/down and left/right scored (svm,
    leave one session out) from 5 ms slices of the tap, one at a time."""
    full = signed(W_ALL, **{**BEST1, "ms": 120, "loud": False, "dec": 12})       # 4 kHz, 20 samples per 5 ms
    print("\n== timeline (svm on one 5 ms slice; chance 50% for each axis)")
    print(f"  {'slice':>12}  {'top/bottom':>10}  {'left/right':>10}")
    for a in range(0, 120, 5):
        F = full[:, a * 4: (a + 5) * 4]
        pred = exam(F, lambda: make_pipeline(StandardScaler(), SVC(C=3)))[3]
        tb = ((pred >= 2) == (Y_ALL >= 2)).mean()
        lr = ((pred % 2) == (Y_ALL % 2)).mean()
        print(f"  {a:>4}–{a + 5:<3} ms  {tb:>10.1%}  {lr:>10.1%}  " + "▮" * int((lr - 0.5) * 100), flush=True)


def release(W, after_ms=25, ms=20):
    """The finger lifting off: the strongest movement after the first
    `after_ms`, cut signed at its own onset, plus when it came (ms)."""
    mono = lowpass(W.mean(axis=2), 1000)
    i0 = starts(W.mean(axis=2), 0.2)
    n = ms * SR // 1000
    out = []
    for k in range(len(mono)):
        a = i0[k] + after_ms * SR // 1000
        tail = mono[k, a: len(mono[k]) - n]
        j = a + int(np.argmax(np.abs(tail) > 0.5 * np.abs(tail).max())) - SR // 1000
        seg = mono[k, j: j + n].reshape(-1, 12).mean(1)
        peak = np.abs(seg).max() + 1e-9
        out.append(np.concatenate([seg / peak, [(j - i0[k]) / SR * 1000, np.log10(peak / (np.abs(mono[k]).max() + 1e-9))]]))
    return np.array(out, np.float32)


@experiment
def lift():
    F, extra = best2()
    R = release(W_ALL)
    run("lift", {"release alone": R, "best2 + release": (np.hstack([F, R]), [np.hstack([E, R]) for E in extra])},
        ("logistic", "svm", "mlp"))


@experiment
def torsion():
    """What separates left from right? The average left tap minus the
    average right tap (each half of the screen on its own, since up/down
    flips the sign), and the spectrum of that difference over 8–36 ms.
    → torsion.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = signed(W_ALL, **{**BEST1, "ms": 60, "loud": False, "dec": 1, "norm": "peak"})   # 48 kHz, 60 ms
    t = (np.arange(X.shape[1]) / SR - 0.002) * 1000
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    print("\n== torsion: strongest frequencies in (left − right), 8–36 ms")
    for half, rows, c in (("top", Y_ALL < 2, "#e06c75"), ("bottom", Y_ALL >= 2, "#61afef")):
        L = X[rows & (Y_ALL % 2 == 0)].mean(0)
        R = X[rows & (Y_ALL % 2 == 1)].mean(0)
        d = L - R
        ax[0].plot(t, L, color=c, lw=1, alpha=0.5, label=f"{half}: left")
        ax[0].plot(t, R, color=c, lw=1, alpha=0.5, ls="--", label=f"{half}: right")
        ax[0].plot(t, d, color=c, lw=2.2, label=f"{half}: left − right")
        win = (t >= 8) & (t <= 36)
        spec = np.abs(np.fft.rfft(d[win] * np.hanning(win.sum()), 8192))
        f = np.fft.rfftfreq(8192, 1 / SR)
        keep = f <= 1500
        ax[1].plot(f[keep], spec[keep] / spec[keep].max(), color=c, lw=2, label=half)
        top = np.argsort(spec[keep])[::-1]
        peaks = []
        for i in top:
            if all(abs(f[i] - p) > 25 for p in peaks):
                peaks.append(f[i])
            if len(peaks) == 3:
                break
        print(f"  {half:6} half: " + ", ".join(f"{p:.0f} Hz" for p in peaks))
    ax[0].axvspan(8, 36, color="#888", alpha=0.12)
    ax[0].set(title="Average tap, left vs right (shaded: where left/right lives)", xlabel="ms after the hit", ylabel="normalised")
    ax[0].legend(fontsize=8)
    ax[1].set(title="Spectrum of left − right, 8–36 ms", xlabel="Hz")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ml.DATA, "torsion.png"), dpi=80)


def late(W, a=8, b=36, shift=0):
    """The ringing where left/right lives (a–b ms after the hit), signed,
    4 kHz, normalised on its own so the big first push doesn't shrink it."""
    X = signed(W, **{**BEST1, "ms": b + 2, "loud": False, "norm": "none"}, shift=shift)
    seg = X[:, (a + 2) * 4: (b + 2) * 4]
    return (seg / (np.sqrt((seg ** 2).mean(1, keepdims=True)) + 1e-9)).astype(np.float32)


class AxisSplit:
    """Up/down and left/right as two questions, each looking at its own
    columns: `tb` the early-push features, `lr` whatever helps left/right."""
    def __init__(self, make, tb, lr):
        self.tb_m, self.lr_m, self.tb, self.lr = make(), make(), tb, lr

    def fit(self, X, y):
        self.tb_m.fit(X[:, self.tb], y >= 2)
        self.lr_m.fit(X[:, self.lr], y % 2)
        return self

    def predict(self, X):
        return self.lr_m.predict(X[:, self.lr]).astype(int) + 2 * self.tb_m.predict(X[:, self.tb]).astype(int)


def best3():
    """best2 plus 4 copies of every tap with real background noise and a
    different tap strength: the augmentation that helped (+2 points).
    Returns (features, extra copies, and the copies' raw windows)."""
    F, extra = best2()
    Wa = [augmented_W(s, noise=True, gain=True) for s in range(4)]
    return F, extra + [signed(w, **BEST1) for w in Wa], Wa


@experiment
def axes():
    F, extra, Wa = best3()
    Wx = [None] * len(BEST_SHIFTS) + Wa              # what each extra copy was made from
    L = late(W_ALL)
    Lx = [late(W_ALL, shift=s) for s in BEST_SHIFTS] + [late(w) for w in Wa]
    both = np.hstack([F, L])
    both_x = [np.hstack([E, Le]) for E, Le in zip(extra, Lx)]
    n = F.shape[1]
    tb, everything = slice(0, n), slice(None)
    svm = lambda: make_pipeline(StandardScaler(), SVC(C=3))
    mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0))
    run("axes", {"best3 (noise+gain aug)": (F, extra)}, ("svm", "mlp"))
    run("axes", {"best3 + late ringing": (both, both_x)}, [
        ("svm 4-way", svm), ("mlp 4-way", mlp),
        ("svm split: tb=early, lr=early+late", lambda: AxisSplit(svm, tb, everything)),
        ("svm split: tb=early, lr=late only", lambda: AxisSplit(svm, tb, slice(n, None))),
        ("mlp split: tb=early, lr=early+late", lambda: AxisSplit(mlp, tb, everything)),
    ])


def stretch(W, rates=np.geomspace(0.7, 1.43, 41), ms=36):
    """How stretched in time each tap is: the speed factor that best lines
    it up with its half's average tap (|correlation|, sign kept apart),
    found per tap. Right-side taps looked slower than left in torsion.png."""
    X = signed(W, **{**BEST1, "ms": 60, "loud": False, "norm": "peak"})           # 4 kHz
    n = ms * 4
    t = np.arange(n)
    early = X[:, : 10 * 4]
    # Up/down is easy (93%) and flips the shape: compare each tap with the
    # average of taps whose first 10 ms look like its own.
    sign = np.sign(early @ early.mean(0))
    templates = np.stack([(X[:, :n] * sign[:, None]).mean(0)])
    best_r = np.ones(len(X))
    best_c = np.full(len(X), -np.inf)
    for r in rates:
        Y = np.stack([np.interp(t * r, np.arange(X.shape[1]), x) for x in X])
        c = np.abs(Y @ templates[0]) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(templates[0]) + 1e-9)
        better = c > best_c
        best_c[better], best_r[better] = c[better], r
    return best_r, best_c


@experiment
def warp():
    r, c = stretch(W_ALL)
    left, right = Y_ALL % 2 == 0, Y_ALL % 2 == 1
    print("\n== warp: each tap's best time-stretch against the average tap (>1 = faster than average)")
    for half, rows in (("top", Y_ALL < 2), ("bottom", Y_ALL >= 2)):
        print(f"  {half:6}: left median {np.median(r[rows & left]):.3f}, right median {np.median(r[rows & right]):.3f}")
    F, extra = best2()
    feat = np.log(r)[:, None].astype(np.float32)
    run("warp", {"stretch factor alone": feat,
                 "best2 + stretch factor": (np.hstack([F, feat]), [np.hstack([E, feat]) for E in extra])},
        ("logistic", "svm", "mlp"))


def _fold_proba(make, F, train, test, extra, tta):
    X = np.concatenate([F[train]] + [E[train] for E in extra])
    y = np.concatenate([Y_ALL[train]] * (1 + len(extra)))
    m = make().fit(X, y)
    p = lambda A: m.proba(A[test]) if hasattr(m, "proba") else m.predict_proba(A[test])
    return test, np.mean([p(A) for A in (F,) + tuple(tta)], axis=0)


def oof(F, make, extra=(), tta=()):
    """Out-of-session probabilities for every tap (leave one session out),
    and for the random taps of the grid → random split. With `tta`, each
    test tap is also cut at those other offsets and the answers averaged."""
    jobs = [(SESSION != i, SESSION == i) for i in range(len(S))] + [(GRID, RAND)]
    res = Parallel(n_jobs=4)(delayed(_fold_proba)(make, F, tr, te, extra, tta) for tr, te in jobs)
    P = np.zeros((len(Y_ALL), 4))
    for te, p in res[:-1]:
        P[te] = p
    return P, res[-1][1]


def score(P, Pg):
    pred = P.argmax(1)
    per = [(pred[SESSION == i] == Y_ALL[SESSION == i]).mean() for i in range(len(S))]
    return (pred == Y_ALL).mean(), np.mean(per), (Pg.argmax(1) == Y_ALL[RAND]).mean(), pred


@experiment
def pool():
    """Every strong model's out-of-session probabilities, saved once
    (pool.npz), then: each alone, with test-time shifts, every ensemble of
    them, and accuracy when doubtful taps are refused."""
    from itertools import combinations
    F, extra, _ = best3()
    tta = [signed(W_ALL, **BEST1, shift=s) for s in (-12, 12)]
    makers = {
        "svm": lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0)),
        "extratrees": lambda: ExtraTreesClassifier(500, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(7, weights="distance")),
    }
    probs = {}
    print("\n== pool")
    for name, make in makers.items():
        for use_tta in (False, True):
            key = name + (" +tta" if use_tta else "")
            probs[key] = oof(F, make, extra, tta if use_tta else ())
            log("pool", "best3" + (" + test-time shifts" if use_tta else ""), name, score(*probs[key]))
    np.savez(os.path.join(HERE, "pool.npz"), **{k + "|P": v[0] for k, v in probs.items()},
             **{k + "|Pg": v[1] for k, v in probs.items()})

    names = [k for k in probs if k.endswith("+tta")]
    rows = []
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            P = np.mean([probs[k][0] for k in combo], axis=0)
            Pg = np.mean([probs[k][1] for k in combo], axis=0)
            rows.append((score(P, Pg), combo))
    rows.sort(key=lambda r: -r[0][0])
    print("\n  best ensembles (soft vote, all with test-time shifts):")
    for sc, combo in rows[:6]:
        log("pool", "ensemble best3 + tta", "+".join(k.split()[0] for k in combo), sc)

    sc, combo = rows[0]
    P = np.mean([probs[k][0] for k in combo], axis=0)
    conf, ok = P.max(1), P.argmax(1) == Y_ALL
    print(f"\n  refusing doubtful taps ({'+'.join(k.split()[0] for k in combo)}):")
    for keep in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        cut = np.quantile(conf, 1 - keep)
        m = conf >= cut
        print(f"    answer {keep:4.0%} of taps (confidence ≥ {cut:.2f}): {ok[m].mean():6.1%} right")


@experiment
def warmup():
    """A short calibration at the start of each session: the first k taps
    per quadrant of the held-out session join the training set (counted
    `weight` times), and the model is scored on that session's remaining
    taps only. Baseline (k=0) is scored on the same remaining taps."""
    F, extra, _ = best3()
    svm = lambda: make_pipeline(StandardScaler(), SVC(C=3))
    print("\n== warmup (svm, best3): first k taps per quadrant of the new session added to training")
    print("   sessions with ≥ k+2 taps in every quadrant; scored on the taps after the warm-up")
    for weight in (1, 5):
        for k in (1, 2, 3):
            base_ok, warm_ok, n = 0, 0, 0
            for i in range(len(S)):
                te_all = np.where(SESSION == i)[0]
                if min(np.sum(Y_ALL[te_all] == q) for q in range(4)) < k + 2:
                    continue
                warm = np.concatenate([te_all[Y_ALL[te_all] == q][:k] for q in range(4)])
                rest = np.setdiff1d(te_all, warm)
                tr = np.where(SESSION != i)[0]
                X0 = np.concatenate([F[tr]] + [E[tr] for E in extra])
                y0 = np.concatenate([Y_ALL[tr]] * (1 + len(extra)))
                Xw = np.concatenate([F[warm]] + [E[warm] for E in extra])
                yw = np.concatenate([Y_ALL[warm]] * (1 + len(extra)))
                base = svm().fit(X0, y0).predict(F[rest])
                with_w = svm().fit(np.concatenate([X0] + [Xw] * weight), np.concatenate([y0] + [yw] * weight)).predict(F[rest])
                base_ok += (base == Y_ALL[rest]).sum()
                warm_ok += (with_w == Y_ALL[rest]).sum()
                n += len(rest)
            print(f"  k={k} per quadrant, weight ×{weight}:  without {base_ok / n:6.1%}  →  with warm-up {warm_ok / n:6.1%}   ({n} taps scored)", flush=True)
            log("warmup", f"k={k}/quadrant ×{weight} (without: {base_ok / n:.3f})", "svm",
                (warm_ok / n, np.nan, np.nan, np.full(len(Y_ALL), -1)))


@experiment
def pool_cnn():
    """The CNN with best3's training copies (shifted + noise/gain), its
    out-of-session probabilities added to pool.npz, then every ensemble
    with it re-scored."""
    from itertools import combinations
    _, _, Wa = best3()
    F = signed(W_ALL, **WAVE)
    extra = [signed(W_ALL, **WAVE, shift=s) for s in BEST_SHIFTS] + [signed(w, **WAVE) for w in Wa]
    print("\n== pool_cnn")
    P, Pg = oof(F, lambda: TorchCNN(40), extra)
    log("pool_cnn", "wave + best3 copies", "cnn", score(P, Pg))
    path = os.path.join(HERE, "pool.npz")
    saved = dict(np.load(path))
    saved["cnn|P"], saved["cnn|Pg"] = P, Pg
    np.savez(path, **saved)
    probs = {k[:-2]: (saved[k], saved[k[:-2] + "|Pg"]) for k in saved if k.endswith("|P") and ("+tta" in k or k == "cnn|P")}
    rows = []
    for r in range(2, len(probs) + 1):
        for combo in combinations(probs, r):
            if "cnn" not in combo:
                continue
            rows.append((score(np.mean([probs[k][0] for k in combo], 0), np.mean([probs[k][1] for k in combo], 0)), combo))
    rows.sort(key=lambda r: -r[0][0])
    for sc, combo in rows[:6]:
        log("pool_cnn", "ensemble with cnn", "+".join(k.split()[0] for k in combo), sc)


@experiment
def cutoff():
    """The shipped ensemble's (svm + mlp + logistic + cnn) accuracy when it
    only answers its surest taps, from pool.npz (run `pool pool_cnn` first,
    on the current data). Pick ml.CUTOFF from this table."""
    p = np.load(os.path.join(HERE, "pool.npz"))
    if len(p["svm|P"]) != len(Y_ALL):
        sys.exit(f"pool.npz is from {len(p['svm|P'])} taps, the data now has {len(Y_ALL)}: run `research.py pool pool_cnn` first")
    P = np.mean([p[k + "|P"] for k in ("svm", "mlp", "logistic", "cnn")], axis=0)
    conf, ok = P.max(1), P.argmax(1) == Y_ALL
    print(f"\n== cutoff (svm+mlp+logistic+cnn, out-of-session, {len(Y_ALL)} taps; ml.CUTOFF is {ml.CUTOFF})")
    print(f"  {'answer':>7}  {'cutoff':>7}  {'right':>7}  {'refused taps were right':>24}")
    for keep in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6):
        c = np.quantile(conf, 1 - keep)
        m = conf >= c
        refused = f"{ok[~m].mean():.0%}" if (~m).any() else "–"
        print(f"  {keep:>7.0%}  {c:>7.3f}  {ok[m].mean():>7.1%}  {refused:>24}")


LAYOUT_RESULTS = os.path.join(HERE, "layout-results.tsv")


def _fold_layout(make, X, extra, y, train, test, proba):
    m = make().fit(np.concatenate([X[train]] + [E[train] for E in extra]), np.concatenate([y[train]] * (1 + len(extra))))
    return test, (m.predict_proba(X[test]) if proba else m.predict(X[test]))


def loso_layout(F, extra, y, keep, make, proba=False):
    """Leave one session out on the taps in `keep`, labelled `y` (a layout's
    sections). Returns predictions (or probabilities) for those taps; taps
    outside `keep` are never trained or tested on."""
    jobs = [((SESSION != i) & keep, (SESSION == i) & keep) for i in range(len(S))]
    jobs = [(tr, te) for tr, te in jobs if te.any()]
    res = Parallel(n_jobs=4)(delayed(_fold_layout)(make, F, extra, y, tr, te, proba) for tr, te in jobs)
    k = y[keep].max() + 1
    out = np.full((len(y), k), np.nan) if proba else np.full(len(y), -1)
    for te, p in res:
        out[te] = p
    return out


def log_layout(experiment, name, gap, layout, y, keep, pred):
    ok = pred[keep] == y[keep]
    per = [(ok[y[keep] == i].mean() if (y[keep] == i).any() else np.nan, int((y[keep] == i).sum())) for i in range(len(layout))]
    coverage = sum(w * h for _, _, w, h in layout)
    new = not os.path.exists(LAYOUT_RESULTS)
    with open(LAYOUT_RESULTS, "a") as f:
        if new:
            f.write("time\texperiment\tlayout\tgap\tsections\taccuracy\tworst_section\tper_section\ttaps\tcoverage\n")
        f.write(f"{time.strftime('%H:%M:%S')}\t{experiment}\t{name}\t{gap}\t{len(layout)}\t{ok.mean():.4f}\t"
                f"{np.nanmin([a for a, _ in per]):.4f}\t" + ",".join(f"{a:.3f}/{n}" for a, n in per)
                + f"\t{keep.sum()}\t{coverage:.3f}\n")
    print(f"  {name:14} gap {gap:4.0%}  {ok.mean():6.1%}  worst {np.nanmin([a for a, _ in per]):6.1%}  "
          f"coverage {coverage:4.0%}  | " + "  ".join(f"{a:4.0%} ({n})" for a, n in per), flush=True)


@experiment
def layouts():
    """docs/Gestures/Research plan.md #1: every tap relabelled by the section
    its `where` falls in, for row, column and grid layouts, scored leave
    one session out (svm on best2). → layout-results.tsv"""
    import gestures as g
    F, extra = best2()
    svm = models(("svm",))["svm"]
    print(f"\n== layouts ({len(Y_ALL)} taps, svm on best2; per section: accuracy (taps))")
    named = {f"{k} rows": g.rows(k) for k in range(2, 7)} | {f"{k} columns": g.columns(k) for k in range(2, 7)} \
        | {"grid 2x2": g.grid(2, 2), "grid 3x2": g.grid(3, 2), "grid 2x3": g.grid(2, 3)}
    for name, layout in named.items():
        y = g.section_of(WHERE, layout)
        keep = y >= 0
        log_layout("layouts", name, 0.0, layout, y, keep, loso_layout(F, extra, y, keep, svm))


@experiment
def dead_zones():
    """docs/Gestures/Research plan.md #2: a dead zone of width gap along every
    inner border. Trained and scored only on taps outside the dead zones, as
    if people aimed for the sections. → layout-results.tsv"""
    import gestures as g
    F, extra = best2()
    svm = models(("svm",))["svm"]
    print(f"\n== dead zones ({len(Y_ALL)} taps, svm on best2)")
    for name in ("top / bottom", "left / right", "quadrants", "3 rows", "5 rows"):
        for gap in (0.0, 0.05, 0.10, 0.15):
            layout = g.shrink(g.PRESETS[name], gap)
            y = g.section_of(WHERE, layout)
            keep = y >= 0
            log_layout("dead_zones", name, gap, layout, y, keep, loso_layout(F, extra, y, keep, svm))


@experiment
def double_sim():
    """docs/Gestures/Research plan.md #3: simulated double and triple taps.
    Out-of-session probabilities per tap, then taps of the same section and
    session grouped in twos and threes, probabilities multiplied."""
    import gestures as g
    F, extra = best2()
    svm = lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0))
    print(f"\n== double_sim ({len(Y_ALL)} taps, svm on best2 with probabilities)")
    for name in ("top / bottom", "left / right", "quadrants", "3 columns", "5 rows"):
        layout = g.PRESETS[name]
        y = g.section_of(WHERE, layout)
        keep = y >= 0
        P = loso_layout(F, extra, y, keep, svm, proba=True)
        rng = np.random.default_rng(0)
        line = f"  {name:14} single {np.mean(np.nanargmax(P[keep], 1) == y[keep]):6.1%}"
        for n in (2, 3):
            right = []
            for s in range(len(S)):
                for sec in range(len(layout)):
                    idx = rng.permutation(np.flatnonzero((SESSION == s) & (y == sec)))
                    for grp in idx[: len(idx) // n * n].reshape(-1, n):
                        right.append(np.argmax(np.log(P[grp] + 1e-9).sum(0)) == sec)
            line += f"  {'double' if n == 2 else 'triple'} {np.mean(right):6.1%} ({len(right)})"
        print(line, flush=True)


GATE_RESULTS = os.path.join(HERE, "gate-results.tsv")


@experiment
def gate():
    """Tap or not, on the early window. Taps: what share gets through, each
    scored by a gate that never saw its session. Noise: leave one recording
    out, and only the noises loud enough to fire the detector count. Every
    gate keeps the same share of taps (ml.GATE_KEEP), so the noise numbers
    compare directly. → gate-results.tsv"""
    T, tg, N, ng = ml.gate_data()
    loud = ml.would_fire(N)
    recs = list(dict.fromkeys(ng))
    PT, PN = ml.gate_parts(T), ml.gate_parts(N)
    feats = {"jagged": ("jagged",), "spectrum+jagged": ("spectrum", "jagged"),
             "shape+spectrum": ("shape", "spectrum"), "shape+spectrum+jagged": ("shape", "spectrum", "jagged")}
    makers = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
              "svm": ml.gate_model}
    print(f"\n== gate ({len(T)} taps · {len(N)} noise windows, {loud.sum()} loud enough to fire: "
          + ", ".join(f"{r.removeprefix('noise-')} {(loud & (ng == r)).sum()}" for r in recs) + ")")
    new = not os.path.exists(GATE_RESULTS)
    for label, parts in feats.items():
        FT = np.hstack([PT[p] for p in parts]); FN = np.hstack([PN[p] for p in parts])
        for name, make in makers.items():
            thr = np.quantile(ml.tap_scores(FT, tg, FN, make), 1 - ml.GATE_KEEP)
            rejected = []
            for r in recs:
                keep = ng != r
                m = make().fit(np.vstack([FT, FN[keep]]), np.r_[np.ones(len(FT)), np.zeros(keep.sum())])
                sel = (ng == r) & loud
                rejected.append((m.decision_function(FN[sel]) < thr).mean() if sel.any() else np.nan)
            n_loud = [(loud & (ng == r)).sum() for r in recs]
            overall = np.nansum(np.array(rejected) * n_loud) / sum(n_loud)
            with open(GATE_RESULTS, "a") as f:
                if new:
                    f.write("time\tfeatures\tmodel\ttaps_kept\toverall_rejected\tper_recording\ttaps\tloud_noise\n")
                    new = False
                f.write(f"{time.strftime('%H:%M:%S')}\t{label}\t{name}\t{ml.GATE_KEEP}\t{overall:.4f}\t"
                        + ",".join(f"{r}={x:.3f}" for r, x in zip(recs, rejected)) + f"\t{len(T)}\t{loud.sum()}\n")
            print(f"  {label:24} {name:9} loud noise rejected {overall:6.1%}  | "
                  + "  ".join(f"{r.removeprefix('noise-')[:15]} {x:4.0%}" for r, x in zip(recs, rejected)), flush=True)


@experiment
def errors():
    """Where do the mistakes happen? Accuracy by distance from the middle
    lines, and the predicted x,y against the true one. → errors.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    F, extra = best2()
    pred = exam(F, lambda: make_pipeline(StandardScaler(), SVC(C=3)), extra)[3]
    reg = lambda: ViaXY(make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, epsilon=0.05))))
    # Out-of-session x,y predictions.
    XY = np.zeros_like(WHERE)
    for i in range(len(S)):
        tr, te = SESSION != i, SESSION == i
        r = reg().reg.fit(np.concatenate([F[tr]] + [E[tr] for E in extra]), np.concatenate([WHERE[tr]] * (1 + len(extra))))
        XY[te] = r.predict(F[te])
    dx, dy = np.abs(WHERE[:, 0] - 0.5), np.abs(WHERE[:, 1] - 0.5)
    lr_ok = (pred % 2) == (Y_ALL % 2)
    tb_ok = (pred >= 2) == (Y_ALL >= 2)
    print("\nleft/right right, by distance from the vertical middle line (screen fraction):")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5)):
        m = (dx >= lo) & (dx < hi)
        print(f"  {lo:.1f}–{hi:.1f}  {lr_ok[m].mean():6.1%}  ({m.sum()} taps)")
    print("top/bottom right, by distance from the horizontal middle line:")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5)):
        m = (dy >= lo) & (dy < hi)
        print(f"  {lo:.1f}–{hi:.1f}  {tb_ok[m].mean():6.1%}  ({m.sum()} taps)")
    print(f"x,y regression: x error {np.abs(XY[:, 0] - WHERE[:, 0]).mean():.3f}, y error {np.abs(XY[:, 1] - WHERE[:, 1]).mean():.3f} "
          f"(screen fractions; guessing the middle gives {np.abs(WHERE - 0.5).mean(0).round(3)})")
    print("left/right right, per session: " + ", ".join(f"{S[i][0][-6:]} {lr_ok[SESSION == i].mean():.0%}" for i in range(len(S))))

    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
    for a, (k, title) in zip(ax[:2], ((0, "x"), (1, "y"))):
        a.scatter(WHERE[:, k], XY[:, k], s=8, c=SESSION, cmap="tab10")
        a.plot([0, 1], [0, 1], color="#888")
        a.set(title=f"true {title} vs heard {title} (out-of-session)", xlabel="true", ylabel="predicted")
    a = ax[2]
    a.scatter(WHERE[lr_ok, 0], WHERE[lr_ok, 1], s=8, color="#98c379", label="left/right right")
    a.scatter(WHERE[~lr_ok, 0], WHERE[~lr_ok, 1], s=14, color="#e06c75", label="left/right wrong")
    a.axvline(0.5, color="#888"); a.axhline(0.5, color="#888"); a.invert_yaxis(); a.legend()
    a.set(title="Where the left/right mistakes are")
    fig.tight_layout()
    fig.savefig(os.path.join(ml.DATA, "errors.png"), dpi=80)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--list" in args:
        print("experiments:", " ".join(EXPERIMENTS))
    elif "--best" in args:
        import csv
        rows = list(csv.DictReader(open(RESULTS), delimiter="\t"))
        n = int(args[args.index("--best") + 1]) if args[-1] != "--best" else 15
        for r in sorted(rows, key=lambda r: -float(r["loso_pooled"]))[:n]:
            print(f"{float(r['loso_pooled']):6.1%} {float(r['grid_random']):6.1%}  {r['experiment']:12} {r['variant']:44} {r['model']}")
    else:
        print(f"{len(S)} sessions, {len(Y_ALL)} taps · grid→random in {S[BAL][0]} ({GRID.sum()}→{RAND.sum()})")
        for name in args:
            EXPERIMENTS[name]()
