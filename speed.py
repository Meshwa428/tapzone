#!/usr/bin/env python3
"""How long one tap takes to classify, per model, trained the way it would
ship (every session, best3 training copies). Times a single tap at a time,
as live, median of many runs.

    .venv/bin/python speed.py
"""
import time
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import ml
import research as R

torch.set_num_threads(1)


def timed(fn, runs=200):
    fn()                                  # warm up
    t = []
    for _ in range(runs):
        a = time.perf_counter()
        fn()
        t.append(time.perf_counter() - a)
    return np.median(t) * 1000, np.percentile(t, 95) * 1000


F, extra, Wa = R.best3()
X = np.concatenate([F] + extra)
y = np.concatenate([R.Y_ALL] * (1 + len(extra)))
Fw = ml.signed(R.W_ALL, **R.WAVE)
Xw = np.concatenate([Fw] + [ml.signed(R.W_ALL, **R.WAVE, shift=s) for s in R.BEST_SHIFTS]
                    + [ml.signed(w, **R.WAVE) for w in Wa])
window = R.W_ALL[123:124]                 # one live tap, (1, 8400, 2)
print(f"trained on {len(X)} rows ({len(R.Y_ALL)} taps + copies); timing one tap\n")

ms, p95 = timed(lambda: ml.signed(window, **ml.BEST1))
print(f"  {'features (signed, best1)':34} {ms:6.2f} ms   (95th pct {p95:.2f})")
f1 = ml.signed(window, **ml.BEST1)
fw = ml.signed(window, **R.WAVE)

models = {
    "svm (probability)": make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
    "mlp": make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=2000, random_state=0)),
    "extratrees 500": ExtraTreesClassifier(500, random_state=0),
    "extratrees 100": ExtraTreesClassifier(100, random_state=0),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
}
total = 0.0
for name, m in models.items():
    a = time.perf_counter()
    m.fit(X, y)
    fit_s = time.perf_counter() - a
    ms, p95 = timed(lambda: m.predict_proba(f1))
    extra_info = f"  ({m[-1].n_support_.sum()} support vectors)" if name.startswith("svm") else ""
    print(f"  {name:34} {ms:6.2f} ms   (95th pct {p95:.2f})   trained in {fit_s:.0f} s{extra_info}")
    if name != "extratrees 100":
        total += ms

net = ml.train_cnn(Xw, y, 0, threads=4)
torch.set_num_threads(1)
x = torch.tensor(fw[:, None])
with torch.no_grad():
    ms, p95 = timed(lambda: net(x))
print(f"  {'cnn (one copy)':34} {ms:6.2f} ms   (95th pct {p95:.2f})")
total += ms
print(f"\n  5-model ensemble, model time only: ~{total:.1f} ms  (+ features)")
