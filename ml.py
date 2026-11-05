#!/usr/bin/env python3
"""Can a learned model hear which quadrant was tapped? Trains on the
calibration taps, scores on the test-round taps (points never trained on).

    .venv/bin/python ml.py                 across sessions (calibration, test
                                           rounds, collect sessions)
    .venv/bin/python ml.py --deep          the same, then the deep models
    .venv/bin/python ml.py --augment       quadrants: real taps vs real + augmented
    .venv/bin/python ml.py --gate          tap or not: screen taps vs noise sessions
    .venv/bin/python ml.py --signed        quadrants from the lid's push/pull waveform
    .venv/bin/python ml.py --export        train that CNN on everything → signed-cnn.pt
    .venv/bin/python ml.py --export-xy     rough x,y position model → xy-svr.joblib
    .venv/bin/python ml.py --export-final  the shipped ensemble (svm+mlp+logistic+cnn) → final.joblib
    .venv/bin/python ml.py --export-gate   the tap-or-not gate (signed + jaggedness, svm) → gate.joblib
    .venv/bin/python ml.py CAL.npz TEST.npz…   calibration → test only

Every model is scored the same way; deep models are averaged over seeds
because 20 test taps are a small, noisy exam (one tap = 5%).
"""
import glob, os, sys
import numpy as np

DATA = os.path.expanduser("~/.local/share/tapzones")
SR, HOP, PRE_HOPS = 48000, 240, 4
ZONES = ["top-left", "top-right", "bottom-left", "bottom-right"]

# ------------------------------------------------------------ preprocessing

def aligned(window, n=4096, before=48):
    """Mono, starting 1 ms before the loudest point of the hit."""
    x = window.mean(axis=1)
    p = int(np.argmax(np.abs(x[: (PRE_HOPS + 8) * HOP])))
    s = x[max(0, p - before): max(0, p - before) + n]
    return np.pad(s, (0, n - len(s))).astype(np.float32)


def mel_filters(n_mels=64, n_fft=512, fmin=50, fmax=20000):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], max(bins[i + 2], bins[i + 1] + 1)
        fb[i, a:b] = np.linspace(0, 1, b - a, endpoint=False)
        fb[i, b:c] = np.linspace(1, 0, c - b, endpoint=False)
    return fb

FB = mel_filters()

def logmel(x, n_fft=512, hop=112):
    """64 mel bands × 32 frames (2.3 ms apart): the tap as a picture."""
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:32]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    m = 10 * np.log10(FB @ spec.T + 1e-10)
    return np.clip(m - m.max(), -80, 0).astype(np.float32)   # loudness divided out


def load(cal_path, test_paths):
    c = np.load(cal_path)
    Xc = np.array([aligned(w) for w in c["windows"]])
    yc = c["labels"].astype(int)
    Xt, yt = [], []
    for p in test_paths:
        t = np.load(p)
        Xt += [aligned(w) for w in t["windows"]]
        yt += list(t["truth"].astype(int))
    return Xc, yc, np.array(Xt), np.array(yt)

# ------------------------------------------------------------ classic ML

def classic(Xc, yc, Xt, yt):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Summary of the log-mel picture: each band's mean, peak and decay
    # slope, plus the coarse picture itself.
    def feats(X):
        out = []
        for x in X:
            m = logmel(x)
            slope = np.polyfit(np.arange(m.shape[1]), m.T, 1)[0]
            coarse = m.reshape(16, 4, 8, 4).mean(axis=(1, 3)).ravel()
            out.append(np.concatenate([m.mean(1), m.max(1), slope, coarse]))
        return np.array(out)

    Fc, Ft = feats(Xc), feats(Xt)
    models = {
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=3)),
        "SVM on 20 PCA dims": make_pipeline(StandardScaler(), PCA(20), SVC(C=3)),
        "random forest": RandomForestClassifier(500, random_state=0),
        "extra trees": ExtraTreesClassifier(500, random_state=0),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05),
    }
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    rows = []
    for name, m in models.items():
        cv = cross_val_score(m, Fc, yc, cv=StratifiedKFold(6, shuffle=True, random_state=0)).mean()
        m.fit(Fc, yc)
        rows.append((name, cv, (m.predict(Ft) == yt).mean(), m.predict(Ft)))
    return rows

# ------------------------------------------------------------ deep learning

def deep(Xc, yc, Xt, yt, kind, seeds=5, epochs=300):
    import torch, torch.nn as nn
    torch.set_num_threads(4)

    def augment(x, rng):
        """A different tap every epoch: shifted, louder or softer, noisier."""
        x = np.roll(x, rng.integers(-24, 25))
        x = x * 10 ** (rng.uniform(-12, 12) / 20)
        return x + rng.normal(0, 10 ** rng.uniform(-4, -2.5), x.shape).astype(np.float32)

    def encode(x, rng=None):
        if kind == "cnn":
            m = logmel(x)
            if rng is not None:   # SpecAugment: hide a band and a moment
                f, t = rng.integers(0, 56), rng.integers(0, 28)
                m[f: f + rng.integers(0, 8)] = -80
                m[:, t: t + rng.integers(0, 4)] = -80
            return (m / 40 + 1)[None]
        w = x[:1440]                                  # first 30 ms of raw sound
        return (w / (np.abs(w).max() + 1e-9))[None]

    def net():
        if kind == "cnn":
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 4))
        return nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 4))

    test_in = torch.tensor(np.array([encode(x) for x in Xt]))
    scores, votes = [], np.zeros((len(Xt), 4))
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = net()
        opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-3)
        y = torch.tensor(yc)
        for _ in range(epochs):
            model.train()
            batch = torch.tensor(np.array([encode(augment(x, rng), rng) for x in Xc]))
            loss = nn.functional.cross_entropy(model(batch), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(test_in), 1).numpy()
        votes += p
        scores.append((p.argmax(1) == yt).mean())
    return np.mean(scores), np.std(scores), votes.argmax(1)


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


def far_away(x, rng, noise_bank):
    """A tap-like sound made distant: high frequencies gone, a room's echo
    tail, much quieter, sitting in the real background. Labelled not-a-tap,
    so the gate learns that tap frequencies from across the room aren't one."""
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    cutoff = rng.uniform(300, 2500)
    spec /= np.sqrt(1 + (f / cutoff) ** (2 * rng.integers(2, 5)))
    y = np.fft.irfft(spec, n)
    # Room: exponentially decaying noise as the impulse response.
    rt = rng.uniform(0.08, 0.4)
    ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / (rt / 6.9))
    ir[0] = 1 / rng.uniform(0.2, 1)
    wet = np.fft.irfft(np.fft.rfft(y) * np.fft.rfft(ir / np.abs(ir).sum() * 8), n)
    mix = rng.uniform(0.3, 0.8)
    y = (1 - mix) * y + mix * wet
    y *= 10 ** (-rng.uniform(15, 40) / 20)
    start = rng.integers(0, len(noise_bank) - n)
    return (y + noise_bank[start: start + n]).astype(np.float32)


def gate_features(X):
    """For tap vs not: the tap's shape (as for quadrants) plus what was
    divided out there, how loud it was and how sharp its attack."""
    shape = summary_features(X)
    extra = []
    for x in X:
        a = np.abs(x)
        peak = a.max() + 1e-9
        p = int(a.argmax())
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        total = spec.sum() + 1e-12
        rise = p - int(np.argmax(a[: p + 1] > 0.1 * peak))       # samples from 10% to peak
        extra.append([
            20 * np.log10(peak), 10 * np.log10((x ** 2).mean() + 1e-12),
            20 * np.log10(peak / (np.sqrt((x ** 2).mean()) + 1e-9)),    # crest factor
            (spec[f > 4000].sum() / total), (spec[f < 300].sum() / total),
            (f * spec).sum() / total / 1000,                            # centroid, kHz
            rise / SR * 1000,
        ])
    return np.hstack([shape, np.array(extra)])


def augmented(X, y, copies, rng, noise_bank):
    """The real taps plus `copies` made-up variations of each."""
    extra = [augment(x, rng, noise_bank) for x in X for _ in range(copies)]
    return np.concatenate([X, np.array(extra)]), np.concatenate([y, np.repeat(y, copies)])

# ------------------------------------------------------- across sessions

def session_paths():
    """Every labelled recording, one per file: the guided calibration, test
    rounds, collect and locate sessions. Run 1 (one spot, clipped) is left
    out. Locate sessions cover the centre lines and edges, which collect
    keeps clear of."""
    paths = [os.path.join(DATA, "run2-calibration.npz")] + sorted(
        glob.glob(os.path.join(DATA, "test-*.npz")) + glob.glob(os.path.join(DATA, "collect-*.npz"))
        + glob.glob(os.path.join(DATA, "locate-*.npz")))
    return [p for p in paths if os.path.exists(p)]


def sessions():
    out = []
    for path in session_paths():
        d = np.load(path)
        y = d["labels"] if "labels" in d else d["truth"]
        X = np.array([aligned(w) for w in d["windows"]])
        out.append((os.path.basename(path).removesuffix(".npz"), X, y.astype(int)))
    return out


def summary_features(X):
    out = []
    for x in X:
        m = logmel(x)
        slope = np.polyfit(np.arange(m.shape[1]), m.T, 1)[0]
        coarse = m.reshape(16, 4, 8, 4).mean(axis=(1, 3)).ravel()
        out.append(np.concatenate([m.mean(1), m.max(1), slope, coarse]))
    return np.array(out)


def across_sessions():
    """The honest question: trained on other sessions, does it hear a session
    it has never seen? And does that improve as sessions are added?"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier

    S = sessions()
    if len(S) < 2:
        sys.exit("need at least 2 sessions: run taps.py collect")
    F = [summary_features(X) for _, X, _ in S]
    makers = {
        "kNN (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
        "extra trees": lambda: ExtraTreesClassifier(400, random_state=0, n_jobs=-1),
    }
    print("sessions: " + ", ".join(f"{n} ({len(y)})" for n, _, y in S))
    print(f"total {sum(len(y) for *_, y in S)} taps · chance 25%\n")

    print("Leave one session out (score on the session left out):")
    print(f"{'model':22}" + "".join(f"{n[:14]:>16}" for n, *_ in S) + f"{'mean':>8}")
    for name, make in makers.items():
        accs = []
        for i in range(len(S)):
            tr = [j for j in range(len(S)) if j != i]
            m = make().fit(np.vstack([F[j] for j in tr]), np.concatenate([S[j][2] for j in tr]))
            accs.append((m.predict(F[i]) == S[i][2]).mean())
        print(f"{name:22}" + "".join(f"{a:>16.0%}" for a in accs) + f"{np.mean(accs):>8.0%}")

    # Learning curve: for each held-out session, train on 1, 2, … of the
    # others (in random orders), and average.
    print("\nLearning curve (SVM): score on an unseen session vs sessions trained on")
    rng = np.random.default_rng(0)
    for k in range(1, len(S)):
        accs = []
        for i in range(len(S)):
            others = [j for j in range(len(S)) if j != i]
            for _ in range(5):
                pick = rng.choice(others, k, replace=False)
                m = makers["SVM (RBF)"]().fit(np.vstack([F[j] for j in pick]), np.concatenate([S[j][2] for j in pick]))
                accs.append((m.predict(F[i]) == S[i][2]).mean())
        taps = np.mean([len(S[j][2]) for j in range(len(S))]) * k
        bar = "█" * int(np.mean(accs) * 40)
        print(f"  {k} session(s), ~{taps:4.0f} taps   {np.mean(accs):4.0%}  {bar}")


def with_and_without(copies=20):
    """Same models, same honest split (leave one session out), trained on
    the real taps alone vs the real taps plus augmented copies. Augmented
    copies are only ever trained on, never scored."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier

    S = sessions()
    raw = [np.load(p)["windows"] for p in
           [os.path.join(DATA, "run2-calibration.npz")] + sorted(glob.glob(os.path.join(DATA, "test-*.npz")) + glob.glob(os.path.join(DATA, "collect-*.npz")))]
    makers = {
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
        "extra trees": lambda: ExtraTreesClassifier(400, random_state=0, n_jobs=-1),
    }
    print("sessions: " + ", ".join(f"{n} ({len(y)})" for n, _, y in S))
    print(f"augmentation: {copies} made-up copies of every training tap · chance 25%\n")
    print(f"{'model':22}{'':12}" + "".join(f"{n[:14]:>16}" for n, *_ in S) + f"{'mean':>8}")
    F_test = [summary_features(X) for _, X, _ in S]
    results = {}
    for aug in (False, True):
        for name, make in makers.items():
            accs = []
            for i in range(len(S)):
                tr = [j for j in range(len(S)) if j != i]
                X = np.concatenate([S[j][1] for j in tr])
                y = np.concatenate([S[j][2] for j in tr])
                if aug:
                    rng = np.random.default_rng(i)
                    X, y = augmented(X, y, copies, rng, background(np.concatenate([raw[j] for j in tr])))
                m = make().fit(summary_features(X), y)
                accs.append((m.predict(F_test[i]) == S[i][2]).mean())
            results[(name, aug)] = accs
            print(f"{name:22}{'augmented' if aug else 'real only':12}" + "".join(f"{a:>16.0%}" for a in accs) + f"{np.mean(accs):>8.0%}", flush=True)
    return results


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


def gate():
    """Tap or not? Screen taps from every session vs recorded noises (and,
    in training only, augmented taps and far-away fakes). Scored two ways:
    random folds, and leaving out a whole kind of noise at a time."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold

    S = sessions()
    taps = np.concatenate([X for _, X, _ in S])
    if not glob.glob(os.path.join(DATA, "noise-*")):
        sys.exit("no noise sessions yet: run taps.py noise")
    noise_w, kind = noise_set(np.random.default_rng(0))
    noise = np.array([aligned(w) for w in noise_w])
    raw_windows = np.concatenate([np.load(p)["windows"] for p in
        [os.path.join(DATA, "run2-calibration.npz")] + sorted(glob.glob(os.path.join(DATA, "collect-*.npz")))])
    bank = background(raw_windows)
    print(f"{len(taps)} screen taps · {len(noise)} recorded non-taps: "
          + ", ".join(f"{k} {np.sum(kind == k)}" for k in dict.fromkeys(kind)) + "\n")

    def fit(tap_X, noise_X, seed):
        rng = np.random.default_rng(seed)
        ta, _ = augmented(tap_X, np.zeros(len(tap_X), int), 3, rng, bank)
        na, _ = augmented(noise_X, np.zeros(len(noise_X), int), 3, rng, bank)
        fakes = np.array([far_away(tap_X[i], rng, bank) for i in rng.integers(0, len(tap_X), len(tap_X))])
        X = np.concatenate([ta, na, fakes])
        y = np.concatenate([np.ones(len(ta)), np.zeros(len(na) + len(fakes))])
        return ExtraTreesClassifier(400, class_weight="balanced", random_state=seed, n_jobs=-1).fit(gate_features(X), y)

    Ft, Fn = gate_features(taps), gate_features(noise)

    # 1. Random folds: taps and noises split together.
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc_t, acc_n = np.zeros(len(taps)), np.zeros(len(noise))
    for fold, ((tr_t, te_t), (tr_n, te_n)) in enumerate(zip(cv.split(taps, np.zeros(len(taps))),
                                                         cv.split(noise, np.zeros(len(noise))))):
        m = fit(taps[tr_t], noise[tr_n], fold)
        acc_t[te_t] = m.predict(Ft[te_t])
        acc_n[te_n] = m.predict(Fn[te_n])
    print("Random 5-fold:")
    print(f"  screen taps accepted   {acc_t.mean():.0%}")
    print(f"  non-taps rejected      {1 - acc_n.mean():.0%}")
    for k in dict.fromkeys(kind):
        print(f"    {k:42} {1 - acc_n[kind == k].mean():.0%} rejected")

    # 2. A whole kind of sound never heard in training.
    print("\nLeave one noise recording out (never heard that recording):")
    for i, k in enumerate(dict.fromkeys(kind)):
        m = fit(taps, noise[kind != k], 100 + i)
        print(f"    {k:42} {1 - m.predict(Fn[kind == k]).mean():.0%} rejected")

    # 3. Far-away fakes made from taps it never trained on.
    rng = np.random.default_rng(7)
    half = len(taps) // 2
    m = fit(taps[:half], noise, 7)
    fakes = np.array([far_away(x, rng, bank) for x in taps[half:]])
    print(f"\nFar-away versions of unseen taps: {1 - m.predict(gate_features(fakes)).mean():.0%} rejected")


# ------------------------------------------- the lid's push or pull, signed

def onset(mono, frac=0.4):
    """First sample past `frac` of the peak: where the hit starts. Steadier
    than the peak itself, which can land on the first or second swing."""
    a = np.abs(mono[: (PRE_HOPS + 8) * HOP])
    return int(np.argmax(a > frac * a.max()))


def wave4k(mono, ms=40, before=2):
    """The tap as the lid moves: from `before` ms ahead of the hit, averaged
    down to 4 kHz (the sound is almost all under 300 Hz), scaled to a peak
    of 1 with the sign kept. Spectra drop the sign; this is the point."""
    n = ms * SR // 1000
    i = max(0, onset(mono) - before * SR // 1000)
    seg = np.pad(mono[i: i + n], (0, max(0, n - len(mono[i: i + n]))))
    w = seg.reshape(-1, 12).mean(1)
    return (w / (np.abs(w).max() + 1e-9)).astype(np.float32)


def channel(W, chan):
    if chan == "mono":
        return W.mean(axis=2)
    if chan == "diff":
        return W[..., 0] - W[..., 1]
    return W[..., 0 if chan == "left" else 1]


def starts(mono, frac):
    """Where each tap's hit starts (mono is (taps, samples)): the first
    sample past `frac` of its peak, or the peak itself."""
    head = np.abs(mono[:, : (PRE_HOPS + 8) * HOP])
    if frac == "peak":
        return head.argmax(1)
    return np.argmax(head > frac * head.max(1, keepdims=True), axis=1)


def lowpass(X, cutoff):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff, fs=SR, output="sos")
    return sosfiltfilt(sos, X, axis=1).astype(np.float32)


def signed(W, ms=40, before=2, frac=0.4, dec=12, norm="peak", lp="block", chan="mono", dc=False, loud=False, shift=0):
    """The taps' waveforms (W is (taps, samples, 2)), sign kept, cut at the
    onset. Knobs: length, lead-in, onset rule, sample rate after decimation,
    low-pass method, normalisation, channel, baseline removal, a loudness
    feature, and a shift of the cut (samples) for training copies."""
    mono = W.mean(axis=2)
    x = channel(W, chan)
    if lp != "block":
        x = lowpass(x, lp)
    i0 = np.maximum(starts(mono, frac) - before * SR // 1000 + shift, 0)
    n = ms * SR // 1000
    x = np.pad(x, ((0, 0), (0, n)))
    seg = np.stack([x[k, i: i + n] for k, i in enumerate(i0)])
    if dc:
        seg = seg - np.stack([x[k, max(0, i - 240): i].mean() if i > 0 else 0 for k, i in enumerate(i0)])[:, None]
    seg = seg.reshape(len(seg), -1, dec).mean(2) if lp == "block" else seg[:, ::dec]
    scale = {"peak": np.abs(seg).max(1), "rms": np.sqrt((seg ** 2).mean(1)), "none": np.full(len(seg), 1e-2)}[norm]
    out = seg / (scale[:, None] + 1e-9)
    if loud:
        out = np.hstack([out, np.log10(np.abs(seg).max(1, keepdims=True) + 1e-9)])
    return out.astype(np.float32)


# The best settings from research.py, and the shifted training copies.
BEST1 = dict(ms=40, frac=0.2, dc=True, loud=True, norm="rms")
WAVE = dict(ms=40, frac=0.2, dc=True, norm="rms")        # best1 without the loudness column, for the CNN
BEST_SHIFTS = (-24, -12, 12, 24)
CUTOFF = 0.64      # final ensemble: answers ~80% of taps, ~89% of those right (docs/research.md)


def noise_gain(W, seed, bank):
    """One copy of every tap with real background noise (from `bank`, a
    long mono recording) and a different tap strength: the augmentation that
    helped in research.py (augment_kinds). Stereo layout kept."""
    rng = np.random.default_rng(seed)
    mono = W.mean(axis=2)
    n = mono.shape[1]
    out = np.empty_like(mono)
    for k in range(len(mono)):
        s = rng.integers(0, len(bank) - n)
        out[k] = mono[k] * 10 ** (rng.uniform(-10, 10) / 20) + bank[s: s + n] * 10 ** (rng.uniform(-6, 12) / 20)
    return np.repeat(out[:, :, None], 2, axis=2).astype(np.float32)


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

SECTION_TARGET = 0.90    # a layout's cutoff: the lowest that gets answered taps this right…
SECTION_MIN_ANSWER = 0.6 # …but never answering fewer than this share of taps


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
    import gestures
    cfg = gestures.load()
    layout = [s["rect"] for s in cfg[gesture]["sections"]]
    if bad := gestures.problems(cfg, gesture):
        sys.exit(f"{gesture}: " + "; ".join(bad))
    W, where, sess = [], [], []
    for i, path in enumerate(session_paths()):
        d = np.load(path)
        W.append(d["windows"].astype(np.float32)); where.append(np.asarray(d["where"], float))
        sess.append(np.full(len(d["windows"]), i))
    W, where, sess = np.concatenate(W), np.concatenate(where), np.concatenate(sess)
    y = gestures.section_of(where, layout)
    keep = y >= 0
    W, y, sess = W[keep], y[keep], sess[keep]
    counts = np.bincount(y, minlength=len(layout))
    names = [s["name"] for s in cfg[gesture]["sections"]]
    if len(layout) > 1 and counts.min() < gestures.MIN_TAPS:
        sys.exit("too few recorded taps in: " + ", ".join(f"{n} ({c})" for n, c in zip(names, counts) if c < gestures.MIN_TAPS)
                 + ". Record taps there (taps.py locate or collect) or make the section bigger.")
    path = gestures.model_path(gesture)
    if len(layout) == 1:     # anywhere: nothing to tell apart
        joblib.dump({"models": [], "layout": layout, "hash": gestures.layout_hash(layout), "cutoff": 0.0,
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
    joblib.dump({"models": models, "layout": layout, "hash": gestures.layout_hash(layout), "cutoff": float(cutoff),
                 "accuracy": float((guess == y).mean()), "answered": float(answered.mean()),
                 "answered_accuracy": float((guess[answered] == y[answered]).mean()), "per_section": per,
                 "taps": len(y)}, path)
    print(f"saved {path}: {len(layout)} sections, {len(y)} taps. Across sessions: {(guess == y).mean():.1%} right; "
          f"cutoff {cutoff:.2f} answers {answered.mean():.0%} of taps, {(guess[answered] == y[answered]).mean():.1%} of those right")
    print("  " + "  ".join(f"{n}: {a:.0%} ({c} taps)" for n, a, c in zip(names, per, counts)))

# --------------------------------------------------- tap or not: the gate

EARLY_N = (PRE_HOPS + 1 + 9) * HOP    # what the overlay has when it answers (taps.EARLY_HOPS = 9)
GATE_KEEP = 0.97                       # share of real taps, from sittings it never saw, the gate lets through


def would_fire(W):
    """Which windows the overlay's detector reacts to: the hit's 5 ms slice
    past 0.008 and 4× the sound just before (as taps.Mic, roughly). Only
    these noises matter: the overlay never hears the rest."""
    m = W.mean(axis=2)
    n = m.shape[1] // HOP * HOP
    h = np.sqrt((m[:, :n].reshape(len(m), -1, HOP) ** 2).mean(axis=2))
    before = h[:, : PRE_HOPS - 1].mean(axis=1) + 1e-6
    at = h[:, PRE_HOPS - 1: PRE_HOPS + 2].max(axis=1)
    return (at > 0.008) & (at > 4 * before)


def gate_parts(W):
    """Tap-or-not features from the early window, in three groups:
    shape: the signed waveform (best1), as the quadrant model sees it;
    spectrum: loudness, brightness, zero crossings, decay;
    jagged: how much fast wiggle rides on the swing. Table knocks that get
    past the detector are mostly smooth like a tap, but many carry fast
    swings a screen tap never has (docs/research.md, the gate)."""
    from scipy.signal import butter, sosfiltfilt
    W = W[:, :EARLY_N]
    mono = W.mean(axis=2)
    f = np.fft.rfftfreq(mono.shape[1], 1 / SR)
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.shape[1]), axis=1)) ** 2
    total = spec.sum(axis=1) + 1e-12
    wave = signed(W, **WAVE)
    half = wave.shape[1] // 2
    spectrum = np.column_stack([
        np.log10(np.abs(mono).max(axis=1) + 1e-9),
        spec[:, f > 1000].sum(axis=1) / total,
        spec[:, f > 4000].sum(axis=1) / total,
        np.log10((spec * f).sum(axis=1) / total + 1),
        (np.diff(np.sign(wave), axis=1) != 0).sum(axis=1),
        np.log10(np.sqrt((wave[:, half:] ** 2).mean(1)) / (np.sqrt((wave[:, :half] ** 2).mean(1)) + 1e-9) + 1e-6),
    ])
    # 40 ms at the full rate from 2 ms before the onset, pre-onset level removed.
    n = 40 * SR // 1000
    i0 = np.maximum(starts(mono, 0.2) - 2 * SR // 1000, 0)
    x = np.pad(mono, ((0, 0), (0, n)))
    seg = np.stack([x[k, i: i + n] - x[k, max(0, i - 240): i + 1].mean() for k, i in enumerate(i0)])
    rms = np.sqrt((seg ** 2).mean(axis=1)) + 1e-9
    resid = [np.log10(np.sqrt(((seg - sosfiltfilt(butter(4, hz, fs=SR, output="sos"), seg, axis=1)) ** 2).mean(1)) / rms + 1e-6)
             for hz in (300, 600, 1200)]
    d4 = seg.reshape(len(seg), -1, 12).mean(axis=2)
    jagged = np.column_stack(resid + [
        (np.diff(np.sign(np.diff(d4, axis=1)), axis=1) != 0).sum(axis=1),
        (np.diff(np.sign(d4), axis=1) != 0).sum(axis=1),
        np.log10(np.abs(np.diff(seg, axis=1)).sum(axis=1) / (np.abs(seg).max(axis=1) + 1e-9) + 1e-6),
    ])
    return {"shape": signed(W, **BEST1), "spectrum": spectrum, "jagged": jagged}


def tap_features(W):
    p = gate_parts(W)
    return np.hstack([p["shape"], p["spectrum"], p["jagged"]]).astype(np.float32)


def gate_model():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced"))


def gate_data():
    """(tap windows, their session, noise windows, their recording): every
    screen tap, and every not-a-tap from the noise recordings (taps.py
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
    """The tap-or-not gate the overlay checks before answering: SVM on
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
          f"\nFor how well it rejects noise it never heard: research.py gate")


def signed_sessions():
    """Like sessions(), but the raw mono windows and where each tap came
    from ('grid spot N', 'random', or '' for older sessions)."""
    out = []
    for path in session_paths():
        d = np.load(path)
        y = d["labels"] if "labels" in d else d["truth"]
        src = d["source"] if "source" in d else np.full(len(y), "")
        out.append((os.path.basename(path).removesuffix(".npz"), d["windows"].mean(axis=2).astype(np.float32),
                    y.astype(int), src))
    return out


def cnn(n=160):
    """A small 1D CNN on the 4 kHz waveform (n samples). It ends in Flatten,
    not global pooling, so *when* each swing happens stays part of the answer."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * (n // 8), 4))


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


def signed_cnn(Ftr, ytr, Fte, seeds=3, epochs=40):
    import torch
    votes = np.zeros((len(Fte), 4))
    for seed in range(seeds):
        with torch.no_grad():
            votes += torch.softmax(train_cnn(Ftr, ytr, seed, epochs)(torch.tensor(Fte[:, None])), 1).numpy()
    return votes.argmax(1)


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


def signed_eval(copies=10):
    """Quadrants from the signed 4 kHz waveform, on the same two exams as
    before: grid → random within the balanced session, and leave one
    session out. Each model with the real taps only and with augmentation."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    S = signed_sessions()
    feats = lambda M: np.array([wave4k(m) for m in M])
    makers = {
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
        "kNN (k=7)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    }

    def exam(Mtr, ytr, Mte, yte, seed):
        """{(model, augmented?): predictions on the test taps}."""
        rng = np.random.default_rng(seed)
        bank = np.concatenate([m[:720] for m in Mtr])
        extra = np.array([augment(m, rng, bank) for m in Mtr for _ in range(copies)])
        sets = {False: (feats(Mtr), ytr),
                True: (np.concatenate([feats(Mtr), feats(extra)]), np.concatenate([ytr, np.repeat(ytr, copies)]))}
        Fte, out = feats(Mte), {}
        for aug, (F, y) in sets.items():
            for name, make in makers.items():
                out[(name, aug)] = make().fit(F, y).predict(Fte)
            out[("1D CNN", aug)] = signed_cnn(F, y, Fte)
        return out

    rows = [(name, aug) for aug in (False, True) for name in list(makers) + ["1D CNN"]]
    print("sessions: " + ", ".join(f"{n} ({len(y)})" for n, _, y, _ in S))
    print(f"features: 40 ms at 4 kHz from the onset, sign kept · augmentation: {copies} copies · chance 25%\n")

    # 1. Grid → random inside the balanced session.
    balanced = [s for s in S if (s[3] == "random").any() and (s[3] != "random").any()]
    grid_scores = {}
    if balanced:
        name, M, y, src = balanced[-1]
        g, r = src != "random", src == "random"
        pred = exam(M[g], y[g], M[r], y[r], 0)
        grid_scores = {k: (p == y[r]).mean() for k, p in pred.items()}
        best = max(grid_scores, key=grid_scores.get)
        print(f"Grid → random in {name}: best {best[0]}{' + augmented' if best[1] else ''}, confusion "
              f"(rows = where you tapped):")
        confusion = np.zeros((4, 4), int)
        for a, b in zip(y[r], pred[best]):
            confusion[a, b] += 1
        for k in range(4):
            print(f"  {ZONES[k]:13}" + "".join(f"{v:5}" for v in confusion[k]))
        print()

    # 2. Leave one session out.
    loso = {k: [] for k in rows}
    for i in range(len(S)):
        tr = [j for j in range(len(S)) if j != i]
        pred = exam(np.concatenate([S[j][1] for j in tr]), np.concatenate([S[j][2] for j in tr]),
                    S[i][1], S[i][2], i)
        for k in rows:
            loso[k].append((pred[k] == S[i][2]).mean())
        print(f"  … left out {S[i][0]}", flush=True)

    print(f"\n{'model':22}{'':11}{'grid→rand':>10}" + "".join(f"{n[:14]:>16}" for n, *_ in S) + f"{'mean':>8}")
    for k in rows:
        print(f"{k[0]:22}{'augmented' if k[1] else 'real only':11}{grid_scores.get(k, np.nan):>10.0%}"
              + "".join(f"{a:>16.0%}" for a in loso[k]) + f"{np.mean(loso[k]):>8.0%}")


def main():
    import argparse, gestures
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    jobs = p.add_mutually_exclusive_group()
    for flag, fn, what in (("--gate", gate, "tap or not, old spectral features: screen taps vs noise sessions"),
                           ("--signed", signed_eval, "quadrants from the lid's push/pull waveform"),
                           ("--augment", with_and_without, "quadrants: real taps vs real + augmented"),
                           ("--export", export_signed, "train the signed CNN on everything → signed-cnn.pt"),
                           ("--export-xy", export_xy, "rough x,y position model → xy-svr.joblib"),
                           ("--export-final", export_final, "the shipped ensemble (svm+mlp+logistic+cnn) → final.joblib"),
                           ("--export-gate", export_gate, "the tap-or-not gate (signed + jaggedness, svm) → gate.joblib")):
        jobs.add_argument(flag, dest="job", action="store_const", const=fn, help=what)
    p.add_argument("--export-sections", metavar="GESTURE", choices=gestures.GESTURES,
                   help="train GESTURE's section model for its layout in gestures.toml → sections-GESTURE.joblib")
    p.add_argument("--sessions", action="store_true", help="across sessions (the default with no arguments)")
    p.add_argument("--deep", action="store_true", help="calibration → test, with the deep models")
    p.add_argument("files", nargs="*", metavar="CAL.npz TEST.npz", help="calibration → test only")
    args = p.parse_args()
    if args.export_sections:
        export_sections(args.export_sections)
        return
    if args.job:
        args.job()
        return
    if args.sessions or not (args.deep or args.files):
        across_sessions()
        if not args.deep:
            return
    cal = args.files[0] if args.files else os.path.join(DATA, "calibration.npz")
    tests = args.files[1:] or sorted(glob.glob(os.path.join(DATA, "test-*.npz")))
    Xc, yc, Xt, yt = load(cal, tests)
    print(f"train: {len(Xc)} calibration taps · test: {len(Xt)} taps at random points · chance 25%\n")
    print(f"{'model':28} {'calib (6-fold)':>15} {'test':>6}")
    for name, cv, acc, _ in classic(Xc, yc, Xt, yt):
        print(f"{name:28} {cv:>15.0%} {acc:>6.0%}")
    for kind, label in (("cnn", "CNN on log-mel picture"), ("wave", "1D CNN on raw waveform")):
        mean, sd, pred = deep(Xc, yc, Xt, yt, kind)
        print(f"{label:28} {'':>15} {mean:>6.0%}  ±{sd:.0%} over 5 seeds")
        confusion = np.zeros((4, 4), int)
        for a, b in zip(yt, pred):
            confusion[a, b] += 1
        print(f"{'':28} ensemble {np.mean(pred == yt):.0%}; per quadrant right: "
              + ", ".join(f"{ZONES[k]} {confusion[k, k]}/{confusion[k].sum()}" for k in range(4)))


if __name__ == "__main__":
    main()
