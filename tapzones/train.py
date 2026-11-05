"""Trains the models tapzones ships, from the recordings in DATA
(docs/training.md):

    tapzones train sections GESTURE   GESTURE's section model, for its layout
                                      in gestures.toml → sections-GESTURE.joblib
    tapzones train gate               tap or not (signed + jaggedness, svm) → gate.joblib
    tapzones train final              the overlay's ensemble (svm+mlp+logistic+cnn) → final.joblib
    tapzones train xy                 rough x,y position model → xy-svr.joblib
    tapzones train signed             the signed CNN on everything → signed-cnn.pt

How well each idea does is research/ (evaluate.py, experiments.py).
"""
import glob, os, sys
import numpy as np
from .paths import DATA
from .audio import SR, HOP, PRE_HOPS
from .features import BEST1, WAVE, BEST_SHIFTS, CUTOFF, signed, wave4k, tap_features, cnn

GATE_KEEP = 0.97                       # share of real taps, from sittings it never saw, the gate lets through
SECTION_TARGET = 0.90    # a layout's cutoff: the lowest that gets answered taps this right…
SECTION_MIN_ANSWER = 0.6 # …but never answering fewer than this share of taps

# ------------------------------------------------------------- the data

def session_paths():
    """Every labelled recording, one per file: the guided calibration, test
    rounds, collect and locate sessions. Run 1 (one spot, clipped) is left
    out. Locate sessions cover the centre lines and edges, which collect
    keeps clear of. confirmed.npz holds the misses the user said were taps
    ("That was a tap"), each placed at a spot inside the section picked."""
    paths = [os.path.join(DATA, "run2-calibration.npz")] + sorted(
        glob.glob(os.path.join(DATA, "test-*.npz")) + glob.glob(os.path.join(DATA, "collect-*.npz"))
        + glob.glob(os.path.join(DATA, "locate-*.npz"))) + [os.path.join(DATA, "confirmed.npz")]
    return [p for p in paths if os.path.exists(p)]


def signed_sessions():
    """[(name, raw mono windows, labels, where each tap came from)]: 'grid
    spot N', 'random', or '' for older sessions."""
    out = []
    for path in session_paths():
        d = np.load(path)
        y = d["labels"] if "labels" in d else d["truth"]
        src = d["source"] if "source" in d else np.full(len(y), "")
        out.append((os.path.basename(path).removesuffix(".npz"), d["windows"].mean(axis=2).astype(np.float32),
                    y.astype(int), src))
    return out


def tape_windows(raw_path, rng, ratio=2.0):
    """Cuts a continuous noise recording into tap-sized windows (20 ms before,
    150 ms after): one at every bump, with a far lower bar than the live
    detector, plus one random slice per second of plain background.
    Returns (windows as (n, 8400, 2), the prompt each came from)."""
    import json
    x = np.fromfile(raw_path, "<i2").astype(np.float32) / 32767
    meta = json.load(open(raw_path.removesuffix(".raw") + ".json"))
    starts = [(t * SR, p) for t, p in meta["prompts"]]
    prompt_at = lambda i: [p for s, p in starts if s <= i][-1] if starts else "noise"
    span, before = (PRE_HOPS + 1 + 30) * HOP, PRE_HOPS * HOP
    rms = np.sqrt((x[: len(x) // HOP * HOP].reshape(-1, HOP) ** 2).mean(axis=1))
    picks, last = [], -10**9
    for h in range(200, len(rms)):
        floor = np.median(rms[h - 200: h])
        if rms[h] > max(ratio * floor, 1e-4) and rms[h] > ratio * rms[h - 6: h - 2].mean() and h - last > 10:
            picks.append(h * HOP - before)
            last = h
    picks += list(rng.integers(0, len(x) - span, len(x) // SR))
    picks = [p for p in picks if 0 <= p and p + span <= len(x)]
    W = np.array([np.repeat(x[p: p + span, None], 2, axis=1) for p in picks])
    return W, np.array([prompt_at(p) for p in picks])


def noise_set(rng):
    """Every recorded non-tap: continuous tapes cut into windows, and the
    earlier threshold-caught sounds."""
    # Grouped by recording (what was happening doesn't matter: it's all noise).
    W, kind = [], []
    for raw in sorted(glob.glob(os.path.join(DATA, "noise-*.raw"))):
        w, _ = tape_windows(raw, rng)
        W.append(w); kind.append(np.full(len(w), os.path.basename(raw)))
    for p in sorted(glob.glob(os.path.join(DATA, "noise-*.npz"))):
        d = np.load(p)
        W.append(d["windows"]); kind.append(np.full(len(d["windows"]), os.path.basename(p)))
    return np.concatenate(W), np.concatenate(kind)

# ----------------------------------------------------------- augmentation

def background(windows):
    """Real mic noise: the quiet 15 ms before each tap, end to end."""
    return np.concatenate([w.mean(axis=1)[:720] for w in windows]).astype(np.float32)


def augment(x, rng, noise_bank):
    """One made-up variation of a tap: louder or softer (sometimes clipped),
    noisier, shifted, EQ'd, pitched, damped. Each effect is applied with
    some probability, so copies range from mild to heavy."""
    n = len(x)
    y = x.copy()
    if rng.random() < 0.7:   # tone of the tap: a smooth random EQ curve over log frequency
        spec = np.fft.rfft(y)
        f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 20)
        knots = np.log(np.geomspace(20, SR / 2, 6))
        gains_db = rng.uniform(-6, 6, 6)
        spec *= 10 ** (np.interp(np.log(f), knots, gains_db) / 20)
        y = np.fft.irfft(spec, n).astype(np.float32)
    if rng.random() < 0.5:   # lid stiffness: ±5% speed and pitch
        r = rng.uniform(0.95, 1.05)
        y = np.interp(np.arange(n) * r, np.arange(n), y, right=0).astype(np.float32)
    if rng.random() < 0.4:   # a finger left on the screen damps the ring
        y *= np.exp(-np.arange(n) / SR / rng.uniform(0.01, 0.08)).astype(np.float32)
    y = np.roll(y, rng.integers(-48, 49))                       # ±1 ms
    y *= 10 ** (rng.uniform(-15, 15) / 20)                       # louder / softer
    if rng.random() < 0.6:   # real background noise, louder or quieter than it was
        start = rng.integers(0, len(noise_bank) - n)
        y += noise_bank[start: start + n] * 10 ** (rng.uniform(-6, 12) / 20)
    if rng.random() < 0.4:   # white noise, 20–50 dB under the tap
        y += rng.normal(0, np.abs(y).max() * 10 ** (-rng.uniform(20, 50) / 20), n).astype(np.float32)
    if rng.random() < 0.15:  # too hard a tap: the mic clips
        y = np.clip(y, -1, 1) if np.abs(y).max() > 1 else np.clip(y, -np.abs(y).max() * 0.5, np.abs(y).max() * 0.5)
    return y.astype(np.float32)


def noise_gain(W, seed, bank):
    """One copy of every tap with real background noise (from `bank`, a
    long mono recording) and a different tap strength: the augmentation that
    helped in research (experiments.py augment_kinds). Stereo layout kept."""
    rng = np.random.default_rng(seed)
    mono = W.mean(axis=2)
    n = mono.shape[1]
    out = np.empty_like(mono)
    for k in range(len(mono)):
        s = rng.integers(0, len(bank) - n)
        out[k] = mono[k] * 10 ** (rng.uniform(-10, 10) / 20) + bank[s: s + n] * 10 ** (rng.uniform(-6, 12) / 20)
    return np.repeat(out[:, :, None], 2, axis=2).astype(np.float32)

# ------------------------------------------------------------- the CNN

def train_cnn(Ftr, ytr, seed, epochs=40, threads=4):
    import torch, torch.nn as nn
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cnn(Ftr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(Ftr)), max(1, len(Ftr) // 64)):
            # ±0.5 ms of slack in where the hit starts, never a sign flip.
            x = np.stack([np.roll(Ftr[i], rng.integers(-2, 3)) for i in b])
            loss = nn.functional.cross_entropy(model(torch.tensor(x[:, None])), torch.tensor(ytr[b]),
                                               label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def export_signed(copies=10, seeds=3):
    """Trains the signed CNN on every session (+ augmentation), `seeds`
    copies, and saves them for the overlay: DATA/signed-cnn.pt."""
    import torch
    S = signed_sessions()
    M = np.concatenate([m for _, m, _, _ in S])
    y = np.concatenate([y for _, _, y, _ in S])
    rng = np.random.default_rng(0)
    bank = np.concatenate([m[:720] for m in M])
    extra = np.array([augment(m, rng, bank) for m in M for _ in range(copies)])
    F = np.array([wave4k(m) for m in np.concatenate([M, extra])])
    Y = np.concatenate([y, np.repeat(y, copies)])
    states = []
    for seed in range(seeds):
        states.append(train_cnn(F, Y, seed).state_dict())
        print(f"  seed {seed} trained", flush=True)
    path = os.path.join(DATA, "signed-cnn.pt")
    torch.save({"states": states, "n": F.shape[1], "taps": len(y), "sessions": [n for n, *_ in S]}, path)
    print(f"saved {path}: {seeds} CNNs on {len(y)} taps (+{len(extra)} augmented)")

# --------------------------------------------- the overlay's quadrant models

def export_final(copies=4, cnn_seeds=3):
    """The model the overlay ships with: soft vote of SVM + MLP + logistic
    on best1 features and the CNN on the waveform, all trained on every
    session plus shifted and noise/gain copies. → DATA/final.joblib"""
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    W, y, names = [], [], []
    for path in session_paths():
        d = np.load(path)
        W.append(d["windows"].astype(np.float32))
        y.append((d["labels"] if "labels" in d else d["truth"]).astype(int))
        names.append(os.path.basename(path).removesuffix(".npz"))
    W, y = np.concatenate(W), np.concatenate(y)
    bank = np.concatenate([w.mean(axis=1)[:720] for w in W])
    noisy = [noise_gain(W, s, bank) for s in range(copies)]

    def rows(settings):
        return np.concatenate([signed(W, **settings, shift=s) for s in (0,) + BEST_SHIFTS]
                              + [signed(w, **settings) for w in noisy])
    Y = np.concatenate([y] * (1 + len(BEST_SHIFTS) + copies))
    X, Xw = rows(BEST1), rows(WAVE)
    sk = {
        "svm": make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0)),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
    }
    for name, m in sk.items():
        m.fit(X, Y)
        print(f"  {name} trained", flush=True)
    cnn_states = [train_cnn(Xw, Y, s).state_dict() for s in range(cnn_seeds)]
    print("  cnn trained", flush=True)
    path = os.path.join(DATA, "final.joblib")
    joblib.dump({"sklearn": sk, "cnn": cnn_states, "n": Xw.shape[1], "cutoff": CUTOFF,
                 "taps": len(y), "sessions": names}, path)
    print(f"saved {path}: svm + mlp + logistic + cnn on {len(y)} taps ({len(names)} sessions), "
          f"{len(Y)} training rows, cutoff {CUTOFF}")


def export_xy():
    """The rough-position model: SVR from the signed waveform to where on the
    screen (x, y fractions) the tap was, trained on every session plus
    shifted copies. → DATA/xy-svr.joblib"""
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR
    W, where, names = [], [], []
    for path in session_paths():
        d = np.load(path)
        W.append(d["windows"].astype(np.float32))
        where.append(np.asarray(d["where"], float))
        names.append(os.path.basename(path).removesuffix(".npz"))
    W, where = np.concatenate(W), np.concatenate(where)
    X = np.concatenate([signed(W, **BEST1, shift=s) for s in (0,) + BEST_SHIFTS])
    model = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, epsilon=0.05)))
    model.fit(X, np.concatenate([where] * (1 + len(BEST_SHIFTS))))
    path = os.path.join(DATA, "xy-svr.joblib")
    joblib.dump({"model": model, "taps": len(W), "sessions": names}, path)
    print(f"saved {path}: SVR on {len(W)} taps ({len(names)} sessions)")

# ------------------------------------------------- a layout's sections

def section_models():
    """The final ensemble's sklearn part (same settings), for any layout."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    return [make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
            make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0)),
            make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))]


def _section_fold(X, Y, copies, train, test, Xtest):
    rows = np.concatenate([np.flatnonzero(train) + k * len(train) for k in range(copies)])
    return test, np.mean([m.fit(X[rows], Y[rows]).predict_proba(Xtest) for m in section_models()], axis=0)


def export_sections(gesture, copies=4):
    """One gesture's section model, for the layout in gestures.toml: every
    recorded tap labelled by the section its `where` falls in (taps in dead
    zones left out), SVM + MLP + logistic on best1 plus shifted and
    noise/gain copies. Scored leave one session out first, which sets the
    cutoff and the accuracy the editor shows. → DATA/sections-<gesture>.joblib"""
    import joblib
    from joblib import Parallel, delayed
    from . import config, layout as L
    from .models import model_path
    cfg = config.load()
    layout = [s["rect"] for s in cfg[gesture]["sections"]]
    if bad := L.problems(cfg, gesture):
        sys.exit(f"{gesture}: " + "; ".join(bad))
    W, where, sess = [], [], []
    for i, path in enumerate(session_paths()):
        d = np.load(path)
        W.append(d["windows"].astype(np.float32)); where.append(np.asarray(d["where"], float))
        sess.append(np.full(len(d["windows"]), i))
    W, where, sess = np.concatenate(W), np.concatenate(where), np.concatenate(sess)
    y = L.section_of(where, layout)
    keep = y >= 0
    W, y, sess = W[keep], y[keep], sess[keep]
    counts = np.bincount(y, minlength=len(layout))
    names = [s["name"] for s in cfg[gesture]["sections"]]
    if len(layout) > 1 and counts.min() < L.MIN_TAPS:
        sys.exit("too few recorded taps in: " + ", ".join(f"{n} ({c})" for n, c in zip(names, counts) if c < L.MIN_TAPS)
                 + ". Record taps there (tapzones locate or collect) or make the section bigger.")
    path = model_path(gesture)
    if len(layout) == 1:     # anywhere: nothing to tell apart
        joblib.dump({"models": [], "layout": layout, "hash": L.layout_hash(layout), "cutoff": 0.0,
                     "accuracy": 1.0, "answered": 1.0, "answered_accuracy": 1.0, "per_section": [1.0], "taps": len(y)}, path)
        print(f"saved {path}: one section, every tap counts")
        return
    bank = np.concatenate([w.mean(axis=1)[:720] for w in W])
    noisy = [noise_gain(W, s, bank) for s in range(copies)]
    X = np.concatenate([signed(W, **BEST1, shift=s) for s in (0,) + BEST_SHIFTS] + [signed(w, **BEST1) for w in noisy])
    n_copies = 1 + len(BEST_SHIFTS) + copies
    Y = np.concatenate([y] * n_copies)
    # Out-of-session probabilities: the honest accuracy, and the cutoff.
    F = X[: len(y)]
    jobs = [(sess != s, sess == s) for s in np.unique(sess)]
    P = np.zeros((len(y), len(layout)))
    for te, p in Parallel(n_jobs=4)(delayed(_section_fold)(X, Y, n_copies, tr, te, F[te]) for tr, te in jobs
                                    if len(np.unique(y[tr])) == len(layout)):
        P[te] = p
    guess, conf = P.argmax(1), P.max(1)
    cutoff = 0.0
    for c in np.sort(np.unique(conf)):
        answered = conf >= c
        if answered.mean() < SECTION_MIN_ANSWER:
            break
        cutoff = c
        if (guess[answered] == y[answered]).mean() >= SECTION_TARGET:
            break
    answered = conf >= cutoff
    per = [float((guess[y == i] == i).mean()) for i in range(len(layout))]
    models = [m.fit(X, Y) for m in section_models()]
    joblib.dump({"models": models, "layout": layout, "hash": L.layout_hash(layout), "cutoff": float(cutoff),
                 "accuracy": float((guess == y).mean()), "answered": float(answered.mean()),
                 "answered_accuracy": float((guess[answered] == y[answered]).mean()), "per_section": per,
                 "taps": len(y)}, path)
    print(f"saved {path}: {len(layout)} sections, {len(y)} taps. Across sessions: {(guess == y).mean():.1%} right; "
          f"cutoff {cutoff:.2f} answers {answered.mean():.0%} of taps, {(guess[answered] == y[answered]).mean():.1%} of those right")
    print("  " + "  ".join(f"{n}: {a:.0%} ({c} taps)" for n, a, c in zip(names, per, counts)))

# --------------------------------------------------- tap or not: the gate

def would_fire(W):
    """Which windows the overlay's detector reacts to: the hit's 5 ms slice
    past 0.008 and 4× the sound just before (as audio.Mic, roughly). Only
    these noises matter: the overlay never hears the rest."""
    m = W.mean(axis=2)
    n = m.shape[1] // HOP * HOP
    h = np.sqrt((m[:, :n].reshape(len(m), -1, HOP) ** 2).mean(axis=2))
    before = h[:, : PRE_HOPS - 1].mean(axis=1) + 1e-6
    at = h[:, PRE_HOPS - 1: PRE_HOPS + 2].max(axis=1)
    return (at > 0.008) & (at > 4 * before)


def gate_model():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced"))


def gate_data():
    """(tap windows, their session, noise windows, their recording): every
    screen tap, and every not-a-tap from the noise recordings (tapzones
    noise) and the false alarms marked with N in free mode."""
    T, tg = [], []
    for path in session_paths():
        w = np.load(path)["windows"].astype(np.float32)
        T.append(w); tg += [os.path.basename(path)] * len(w)
    N, ng = noise_set(np.random.default_rng(0))
    return np.concatenate(T), np.array(tg), N.astype(np.float32), ng


def tap_scores(FT, tg, FN, make=gate_model):
    """Each tap's gate score from a gate that never saw that tap's session:
    what the threshold is set from, so it holds for a new sitting."""
    st = np.zeros(len(FT))
    for g in np.unique(tg):
        m = make().fit(np.vstack([FT[tg != g], FN]), np.r_[np.ones((tg != g).sum()), np.zeros(len(FN))])
        st[tg == g] = m.decision_function(FT[tg == g])
    return st


def export_gate():
    """The tap-or-not gate the service checks before answering: SVM on
    tap_features, taps vs every recorded noise, the threshold set to let
    GATE_KEEP of unseen sittings' taps through. → DATA/gate.joblib"""
    import joblib
    T, tg, N, ng = gate_data()
    FT, FN = tap_features(T), tap_features(N)
    threshold = float(np.quantile(tap_scores(FT, tg, FN), 1 - GATE_KEEP))
    model = gate_model().fit(np.vstack([FT, FN]), np.r_[np.ones(len(FT)), np.zeros(len(FN))])
    loud = would_fire(N)
    path = os.path.join(DATA, "gate.joblib")
    joblib.dump({"model": model, "threshold": threshold, "keep": GATE_KEEP, "taps": len(T),
                 "noise": len(N), "loud": int(loud.sum()), "recordings": sorted(set(ng))}, path)
    print(f"saved {path}: SVM on {len(T)} taps vs {len(N)} noises ({loud.sum()} loud enough to fire, "
          f"{len(set(ng))} recordings), threshold {threshold:.3f} keeps {GATE_KEEP:.0%} of taps from unseen sittings."
          f"\nFor how well it rejects noise it never heard: research/experiments.py gate")


JOBS = {"final": export_final, "gate": export_gate, "xy": export_xy, "signed": export_signed}
