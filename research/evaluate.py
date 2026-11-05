#!/usr/bin/env python3
"""The first round of research (docs/research.md): can a learned model hear
which quadrant was tapped? Every model is scored the same way; deep models
are averaged over seeds because 20 test taps are a small, noisy exam (one
tap = 5%). The shipped models are trained by `tapzones train`.

    python research/evaluate.py                 across sessions (calibration,
                                                test rounds, collect sessions)
    python research/evaluate.py --deep          the same, then the deep models
    python research/evaluate.py --augment       quadrants: real taps vs real + augmented
    python research/evaluate.py --gate          tap or not, old spectral features
    python research/evaluate.py --signed        quadrants from the lid's push/pull waveform
    python research/evaluate.py CAL.npz TEST.npz…   calibration → test only
"""
import glob, os, sys
import numpy as np
from tapzones.paths import DATA
from tapzones.audio import SR
from tapzones.features import ZONES, aligned, logmel, summary_features, wave4k
from tapzones.train import session_paths, background, augment, noise_set, signed_sessions, train_cnn

# ------------------------------------------------------------ preprocessing

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

    Fc, Ft = summary_features(Xc), summary_features(Xt)
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

    def jitter(x, rng):
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
            batch = torch.tensor(np.array([encode(jitter(x, rng), rng) for x in Xc]))
            loss = nn.functional.cross_entropy(model(batch), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(test_in), 1).numpy()
        votes += p
        scores.append((p.argmax(1) == yt).mean())
    return np.mean(scores), np.std(scores), votes.argmax(1)

# ----------------------------------------------------------- augmentation

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

def sessions():
    out = []
    for path in session_paths():
        d = np.load(path)
        y = d["labels"] if "labels" in d else d["truth"]
        X = np.array([aligned(w) for w in d["windows"]])
        out.append((os.path.basename(path).removesuffix(".npz"), X, y.astype(int)))
    return out


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
        sys.exit("need at least 2 sessions: run tapzones collect")
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


def gate():
    """Tap or not? Screen taps from every session vs recorded noises (and,
    in training only, augmented taps and far-away fakes). Scored two ways:
    random folds, and leaving out a whole kind of noise at a time."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold

    S = sessions()
    taps = np.concatenate([X for _, X, _ in S])
    if not glob.glob(os.path.join(DATA, "noise-*")):
        sys.exit("no noise sessions yet: run tapzones noise")
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

def signed_cnn(Ftr, ytr, Fte, seeds=3, epochs=40):
    import torch
    votes = np.zeros((len(Fte), 4))
    for seed in range(seeds):
        with torch.no_grad():
            votes += torch.softmax(train_cnn(Ftr, ytr, seed, epochs)(torch.tensor(Fte[:, None])), 1).numpy()
    return votes.argmax(1)


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
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    jobs = p.add_mutually_exclusive_group()
    for flag, fn, what in (("--gate", gate, "tap or not, old spectral features: screen taps vs noise sessions"),
                           ("--signed", signed_eval, "quadrants from the lid's push/pull waveform"),
                           ("--augment", with_and_without, "quadrants: real taps vs real + augmented")):
        jobs.add_argument(flag, dest="job", action="store_const", const=fn, help=what)
    p.add_argument("--sessions", action="store_true", help="across sessions (the default with no arguments)")
    p.add_argument("--deep", action="store_true", help="calibration → test, with the deep models")
    p.add_argument("files", nargs="*", metavar="CAL.npz TEST.npz", help="calibration → test only")
    args = p.parse_args()
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
