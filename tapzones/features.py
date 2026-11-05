"""What the models hear in a tap: the features computed from a tap's window,
live (the service, the overlay) and in training (train.py) alike.

A window is (samples, 2) or a stack of them (taps, samples, 2), 48 kHz
stereo starting 20 ms before the hit (audio.Mic)."""
import numpy as np
from .audio import SR, HOP, PRE_HOPS, EARLY_HOPS

ZONES = ["top-left", "top-right", "bottom-left", "bottom-right"]
EARLY_N = (PRE_HOPS + 1 + EARLY_HOPS) * HOP    # what the service has when it answers

# ------------------------------------------------------------ log-mel

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


def summary_features(X):
    """Each band's mean, peak and decay slope, plus the coarse picture, for
    aligned taps X: the overlay's older SVM recipe."""
    out = []
    for x in X:
        m = logmel(x)
        slope = np.polyfit(np.arange(m.shape[1]), m.T, 1)[0]
        coarse = m.reshape(16, 4, 8, 4).mean(axis=(1, 3)).ravel()
        out.append(np.concatenate([m.mean(1), m.max(1), slope, coarse]))
    return np.array(out)

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


# The best settings from research/experiments.py, and the shifted training copies.
BEST1 = dict(ms=40, frac=0.2, dc=True, loud=True, norm="rms")
WAVE = dict(ms=40, frac=0.2, dc=True, norm="rms")        # best1 without the loudness column, for the CNN
BEST_SHIFTS = (-24, -12, 12, 24)
CUTOFF = 0.64      # final ensemble: answers ~80% of taps, ~89% of those right (docs/research.md)

# --------------------------------------------------- tap or not: the gate

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

# ------------------------------------------------------------- the CNN

def cnn(n=160):
    """A small 1D CNN on the 4 kHz waveform (n samples). It ends in Flatten,
    not global pooling, so *when* each swing happens stays part of the answer."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * (n // 8), 4))
