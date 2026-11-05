#!/usr/bin/env python3
"""tapzones: tap a quadrant of the (non-touch) screen, the microphone hears
where. A full-screen overlay to calibrate, see, and test it.

    taps.py            calibrate (15 taps per quadrant), then review
    taps.py test       a test round against the saved calibration: tap 20
                       random points, see which quadrant each was heard as
    taps.py test --train collect-…[,…]   the test round guessed by an SVM
                       trained on those sessions (run with .venv/bin/python)
    taps.py test --signed   the test round guessed by the signed CNN
                       (ml.py --export makes it)
    taps.py test / taps.py free   with final.joblib present (ml.py
                       --export-final): the final ensemble, answering 45 ms
                       after the hit and ignoring taps it isn't sure of;
                       --knn for the calibration kNN instead
    taps.py locate     tap anywhere you're shown; the rough x,y model
                       (ml.py --export-xy) draws where it thinks you tapped,
                       with the running x and y error in pixels
    taps.py collect [PER_SPOT] [PER_ZONE]   training data, balanced: the 3×3
                       grid in every quadrant, PER_SPOT taps per spot (default
                       5), then PER_ZONE well-spaced random points per quadrant
                       (default 30). Saved as it goes to collect-<time>.npz
    taps.py run        the background service, no UI: hears gestures and runs
                       the commands in ~/.config/tapzones/gestures.toml (engine.py)
    taps.py status | pause | resume | toggle   talk to the running service
    taps.py flash      the GTK flash feedback, for desktops without the
                       Omarchy plugin (flash.py)
    taps.py layout     edit the sections of each gesture, see how well the
                       mic can tell them apart, save and train (editor.py;
                       on Omarchy the plugin's editor does this in the shell)
    taps.py noise      records everything the mic hears, no threshold, while
                       you make sounds that are NOT screen taps (prompted one
                       kind at a time): noise-<time>.raw + .json

Keys: Esc quit · Backspace undo last tap · R recalibrate · T test round ·
      F free tapping · V calibration results · N (free tapping) the last
      sound wasn't a tap: saved as noise for the gate (ml.py --export-gate)
"""
import collections, os, subprocess, sys, threading, time

if sys.argv[1:2] in (["run"], ["status"], ["pause"], ["resume"], ["toggle"]):
    import engine                 # headless: no GTK, no layer-shell restart
    sys.exit(engine.main(sys.argv[1:]))

# Layer shell only works when loaded before libwayland: restart once with it
# preloaded.
LAYER_LIB = "/usr/lib/libgtk4-layer-shell.so"
if LAYER_LIB not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = LAYER_LIB
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gtk, GLib, Gdk, Gtk4LayerShell as Layer
import cairo

from mic import SOURCE, SR, HOP, PRE_HOPS, POST_HOPS, EARLY_HOPS, DATA, LOCK, hold_overlay_lock, overlay_open, Mic
FFT = 4096                     # ~85 ms of sound per fingerprint
# Calibration walks a 3×3 grid of spots in each quadrant (snake order),
# two taps per spot, so the whole quadrant is learned, not one point.
GRID = (0.25, 0.5, 0.75)
SPOTS = [(GRID[c if r % 2 == 0 else 2 - c], GRID[r]) for r in range(3) for c in range(3)]
PER_SPOT = 2
TAPS = len(SPOTS) * PER_SPOT   # calibration taps per quadrant
TEST_PER_ZONE = 5              # random points per quadrant in a test round
LOCATE_TRAIL = 25              # locate mode: how many past taps stay drawn
# A collect session: first the calibration grid with several taps per spot,
# then random points; every quadrant gets the same number of each.
COLLECT_PER_SPOT = 5           # grid phase: taps per spot (9 spots × 4 quadrants)
COLLECT_PER_ZONE = 30          # random phase: points per quadrant
MIN_GAP_PX = 55                # random points stay this far from each other and the grid
NOT_A_TAP = 4                  # label for everything that isn't a screen tap
# Noise session: sounds that must never count as a tap, one prompt at a time.
NOISE_PROMPTS = [
    ("Knock and tap on your table", "near the laptop and far from it"),
    ("Tap the keyboard area and palm rest", "anywhere on the laptop that isn't the screen"),
    ("Type normally", "a sentence or two"),
    ("Move your chair, shift around", "lean back, roll, bump the desk"),
    ("Set things down", "phone, mug, bottle, a book"),
    ("Clap, snap, click a pen", "near and far"),
    ("Tap and click the touchpad", "and the mouse"),
    ("Talk, cough, laugh", "whatever you normally do"),
    ("Anything else around you", "doors, drawers, the desk itself"),
]
PER_PROMPT = 15                # sounds per prompt before moving on (Space skips)
ZONES = ["top-left", "top-right", "bottom-left", "bottom-right"]
ZONE_RGB = [(0.88, 0.42, 0.46), (0.90, 0.75, 0.48), (0.60, 0.76, 0.47), (0.38, 0.69, 0.94)]

# ------------------------------------------------------------------ theme

def theme():
    """Background, foreground and accent of the current Omarchy theme."""
    c = {"background": "#0c0b0c", "foreground": "#fafcfb", "accent": "#b59790"}
    try:
        for line in open(os.path.expanduser("~/.local/state/omarchy/current/theme/colors.toml")):
            k, _, v = line.partition("=")
            if k.strip() in c:
                c[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    rgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return rgb(c["background"]), rgb(c["foreground"]), rgb(c["accent"])

BG, FG, ACCENT = theme()

# --------------------------------------------------------------- features

FREQS = np.fft.rfftfreq(FFT, 1 / SR)
EDGES = np.geomspace(80, 20000, 41)
BANDS = [(FREQS >= lo) & (FREQS < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
WINDOW = np.hanning(FFT)

def fingerprint(win):
    """A tap's sound as (feature vector, spectrum curve in dB, envelope).
    Loudness is divided out: only the tap's colour and decay count, so a
    soft and a hard tap on the same spot look alike."""
    mono = win.mean(axis=1)
    p = int(np.argmax(np.abs(mono[: (PRE_HOPS + 8) * HOP])))
    seg = mono[max(0, p - 96): max(0, p - 96) + FFT]
    seg = np.pad(seg, (0, FFT - len(seg)))
    spec = np.abs(np.fft.rfft(seg * WINDOW))
    db = 20 * np.log10(np.array([spec[b].mean() for b in BANDS]) + 1e-9)
    curve = np.clip(db - db.max(), -60, 0)
    # Decay: energy in 12 slices of 4 ms after the peak, relative to the first.
    tail = mono[p: p + 12 * 192]
    tail = np.pad(tail, (0, 12 * 192 - len(tail)))
    env = 10 * np.log10((tail.reshape(12, 192) ** 2).mean(axis=1) + 1e-12)
    env = np.clip(env - env[0], -60, 10)
    feat = np.concatenate([db - db.mean(), env * 0.5])
    return feat, curve, env

class Model:
    """k nearest neighbours over the calibration taps."""
    def __init__(self, feats, labels):
        self.X, self.y = np.array(feats), np.array(labels)
        self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-6
        self.Xs = (self.X - self.mu) / self.sd

    def predict(self, feat, skip=None, k=5):
        d = np.linalg.norm(self.Xs - (feat - self.mu) / self.sd, axis=1)
        if skip is not None:
            d[skip] = np.inf
        near = self.y[np.argsort(d)[:k]]
        votes = np.bincount(near, minlength=len(ZONES))
        # Mean distance to each quadrant's taps, for the display.
        per_zone = [d[self.y == z][np.isfinite(d[self.y == z])].mean() for z in range(len(ZONES))]
        return int(votes.argmax()), votes.max() / k, per_zone

    def leave_one_out(self):
        confusion = np.zeros((len(ZONES), len(ZONES)), int)
        for i, true in enumerate(self.y):
            guess, _, _ = self.predict(self.X[i], skip=i)
            confusion[true, guess] += 1
        return np.trace(confusion) / confusion.sum(), confusion

class Learned:
    """ml.py's best recipe (SVM on log-mel summary features), trained on the
    named sessions in DATA, e.g. collect-20261104-193438. Needs scikit-learn:
    run with .venv-gui/bin/python."""
    def __init__(self, names):
        import ml
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        X, y = [], []
        for name in names:
            d = np.load(os.path.join(DATA, name + ".npz"))
            y += list(d["labels"] if "labels" in d else d["truth"])
            X += [ml.aligned(w) for w in d["windows"]]
        self.ml = ml
        self.clf = make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0))
        self.clf.fit(ml.summary_features(np.array(X)), np.array(y, int))
        self.label = f"SVM trained on {' + '.join(names)} ({len(y)} taps)"

    def predict(self, window):
        p = self.clf.predict_proba(self.ml.summary_features([self.ml.aligned(window)]))[0]
        return int(p.argmax()), float(p.max()), None


class Signed:
    """The CNN that hears the lid's push or pull (ml.py --export): its
    trained copies, probabilities averaged."""
    def __init__(self):
        import ml, torch
        self.ml, self.torch = ml, torch
        d = torch.load(os.path.join(DATA, "signed-cnn.pt"), weights_only=True)
        self.nets = []
        for state in d["states"]:
            net = ml.cnn(d["n"])
            net.load_state_dict(state)
            self.nets.append(net.eval())
        self.label = f"signed CNN on {d['taps']} taps ({len(d['sessions'])} sessions)"

    def predict(self, window):
        x = self.torch.tensor(self.ml.wave4k(window.mean(axis=1)))[None, None]
        with self.torch.no_grad():
            p = self.torch.stack([self.torch.softmax(net(x), 1)[0] for net in self.nets]).mean(0).numpy()
        return int(p.argmax()), float(p.max()), None

class Final:
    """The shipped model (ml.py --export-final): SVM + MLP + logistic on the
    signed features and the CNN on the waveform, probabilities averaged.
    It answers from the first 45 ms (Mic.on_early) and refuses taps it is
    less sure of than its cutoff."""
    early = True

    def __init__(self):
        import ml, joblib, torch
        self.ml, self.torch = ml, torch
        torch.set_num_threads(1)
        d = joblib.load(os.path.join(DATA, "final.joblib"))   # our own file, written by ml.py --export-final
        self.sk = list(d["sklearn"].values())
        self.nets = []
        for state in d["cnn"]:
            net = ml.cnn(d["n"])
            net.load_state_dict(state)
            self.nets.append(net.eval())
        self.cutoff = d["cutoff"]
        self.label = f"final ensemble on {d['taps']} taps ({len(d['sessions'])} sessions)"
        # Tap or not, asked first (ml.py --export-gate).
        import gestures
        self.gate = gestures.Gate()
        if self.gate.d:
            self.label += f" + gate ({self.gate.d['noise']} noises)"
        # The first call pays for torch's and sklearn's one-time setup (~30 ms):
        # spend it now, not on the first real tap.
        self.predict(np.zeros(((PRE_HOPS + 1 + EARLY_HOPS) * HOP, 2), np.float32))

    def proba(self, window):
        W = window[None]
        f = self.ml.signed(W, **self.ml.BEST1)
        p = [m.predict_proba(f)[0] for m in self.sk]
        x = self.torch.tensor(self.ml.signed(W, **self.ml.WAVE)[:, None])
        with self.torch.no_grad():
            p.append(np.mean([self.torch.softmax(n(x), 1)[0].numpy() for n in self.nets], axis=0))
        return np.mean(p, axis=0)

    def predict(self, window):
        p = self.proba(window)
        return int(p.argmax()), float(p.max()), None

    def is_tap(self, window):
        """False when the gate says this sound wasn't a screen tap (a knock
        on the table, typing…). Always True without a gate."""
        return self.gate.is_tap(window)


class Locator:
    """The rough-position model (ml.py --export-xy): where on the screen a
    tap was, as (x, y) fractions."""
    def __init__(self):
        import ml, joblib
        self.ml = ml
        d = joblib.load(os.path.join(DATA, "xy-svr.joblib"))   # our own file, written by ml.py --export-xy
        self.model = d["model"]
        self.label = f"x,y SVR on {d['taps']} taps ({len(d['sessions'])} sessions)"

    def predict(self, window):
        F = self.ml.signed(window[None], **self.ml.BEST1)
        return tuple(float(v) for v in self.model.predict(F)[0])

# --------------------------------------------------------------------- UI

class App:
    def __init__(self, app, mode, train=None):
        self.learned = (Final() if train == ["final"] else Signed() if train == ["signed"]
                        else Learned(train) if train else None)
        self.decision = None             # the early answer for the tap in flight (Final)
        self.refused = 0                 # taps the model wasn't sure enough to answer, this round
        self.gated = 0                   # sounds the gate called not a tap, this round
        self.last_gated = False          # was the last sound turned away by the gate?
        self.last_ms = None              # how long after the hit the last answer came
        self.false_alarms = []           # free mode: sounds marked with N as not a tap
        self.noise_file = time.strftime("noise-free-%Y%m%d-%H%M%S.npz")
        self.mode = mode                 # calibrate | review | test | free | collect
        self.zone = 0
        self.taps = [[] for _ in ZONES]  # per zone: (window, feat, curve, env)
        self.where = [[] for _ in ZONES] # per zone: spot of each tap, as screen fractions
        self.warn = ("", 0.0)            # (message, time)
        self.model, self.accuracy, self.confusion = None, None, None
        self.last = None                 # last tap: (window, curve, env, guess, conf, per_zone)
        self.flash = (None, 0.0)         # (zone, time)
        self.pause_until = 0.0
        self.error = None
        self.targets = []                # test round: points to tap, as screen fractions
        self.trials = []                 # test round: dicts of what happened at each
        self.collected = []              # collect session: dicts like trials, no guess
        self.plan = []                   # collect session: (point, phase) for every tap, in order
        self.collect_goal = 0
        self.collect_file = time.strftime("collect-%Y%m%d-%H%M%S.npz")
        if mode == "test":
            if self.learned or self.load():
                self.start_test()
            else:
                self.mode = "calibrate"
        elif mode == "free":
            if not self.learned and not self.load():
                self.mode = "calibrate"
        elif mode.startswith("collect"):
            self.mode = "collect"
            counts = [int(n) for n in mode.split(":")[1:]]
            per_spot = counts[0] if counts else COLLECT_PER_SPOT
            per_zone = counts[1] if len(counts) > 1 else COLLECT_PER_ZONE
            self.plan = self.collect_plan(per_spot, per_zone)
            self.collect_goal = len(self.plan)
        elif mode == "locate":
            self.locator = Locator()
            self.located = []            # dicts: window, where (target), guess (x, y), at
            self.locate_target = self.anywhere()
            self.locate_file = time.strftime("locate-%Y%m%d-%H%M%S.npz")
        elif mode == "noise":
            # Everything the mic hears, no threshold: noise-<time>.raw (16-bit
            # mono, 48 kHz) and noise-<time>.json (which prompt began when).
            os.makedirs(DATA, exist_ok=True)
            self.tape_name = time.strftime("noise-%Y%m%d-%H%M%S")
            self.prompt = 0
            self.prompt_log = []        # (seconds into the tape, prompt)
            self.tape_start = None

        hold_overlay_lock()
        self.win = Gtk.ApplicationWindow(application=app)
        Layer.init_for_window(self.win)
        Layer.set_layer(self.win, Layer.Layer.OVERLAY)
        Layer.set_namespace(self.win, "tapzones")
        for edge in (Layer.Edge.TOP, Layer.Edge.BOTTOM, Layer.Edge.LEFT, Layer.Edge.RIGHT):
            Layer.set_anchor(self.win, edge, True)
        Layer.set_exclusive_zone(self.win, -1)
        Layer.set_keyboard_mode(self.win, Layer.KeyboardMode.EXCLUSIVE)
        css = Gtk.CssProvider()
        css.load_from_data(b"window { background: transparent; }")
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, 800)

        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self.draw)
        self.win.set_child(self.area)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.key)
        self.win.add_controller(keys)

        early = (lambda w, t: GLib.idle_add(self.early, w, t)) if getattr(self.learned, "early", False) else None
        self.mic = Mic(lambda w: GLib.idle_add(self.tap, w), early)
        if self.mode == "noise":
            self.mic.tape = open(os.path.join(DATA, self.tape_name + ".raw"), "wb", buffering=0)
            self.tape_start = time.time()
            self.mark_prompt()
        self.mic.start()
        GLib.timeout_add(33, lambda: self.area.queue_draw() or True)
        self.win.present()

    # ---------------------------------------------------------- data

    def load(self):
        try:
            d = np.load(os.path.join(DATA, "calibration.npz"))
        except OSError:
            return False
        where = d["where"] if "where" in d else [(np.nan, np.nan)] * len(d["labels"])
        for w, z, pos in zip(d["windows"], d["labels"], where):
            self.taps[int(z)].append((w, *fingerprint(w)))
            self.where[int(z)].append(tuple(pos))
        self.train()
        return True

    def save_taps(self, name):
        """Writes every tap so far; called after each one, so nothing is
        lost if the overlay dies."""
        os.makedirs(DATA, exist_ok=True)
        windows = [t[0] for zone in self.taps for t in zone]
        labels = [z for z, zone in enumerate(self.taps) for _ in zone]
        where = [pos for zone in self.where for pos in zone]
        if windows:
            np.savez_compressed(os.path.join(DATA, name), windows=np.array(windows), labels=np.array(labels),
                                where=np.array(where, dtype=float))

    def save(self):
        self.save_taps("calibration.npz")
        with open(os.path.join(DATA, "report.txt"), "w") as f:
            f.write(f"accuracy (leave-one-out): {self.accuracy:.1%}\n")
            f.write("confusion (row = tapped, column = guessed):\n")
            f.write("              " + " ".join(f"{z[:12]:>12}" for z in ZONES) + "\n")
            for z, row in zip(ZONES, self.confusion):
                f.write(f"{z:>12}  " + " ".join(f"{n:>12}" for n in row) + "\n")

    def start_test(self):
        """A round of random points the calibration never saw: the same
        number in each quadrant, shuffled, a little clear of the dividers
        so each point has an unambiguous right answer."""
        rng = np.random.default_rng()
        points = []
        for z in range(len(ZONES)):
            for _ in range(TEST_PER_ZONE):
                u, v = rng.uniform(0.06, 0.94, 2)
                points.append(((z % 2 + u) / 2, (z // 2 + v) / 2))
        rng.shuffle(points)
        self.targets = [tuple(p) for p in points]
        self.trials = []
        self.refused = self.gated = 0
        self.mode, self.last = "test", None

    def save_trials(self):
        os.makedirs(DATA, exist_ok=True)
        name = time.strftime("test-%Y%m%d-%H%M%S.npz")
        t = self.trials
        np.savez_compressed(os.path.join(DATA, name),
                            model=np.array(self.learned.label if self.learned else "kNN on calibration"),
                            windows=np.array([x["window"] for x in t]),
                            where=np.array([x["where"] for x in t], dtype=float),
                            truth=np.array([x["truth"] for x in t]),
                            guess=np.array([x["guess"] for x in t]),
                            conf=np.array([x["conf"] for x in t]),
                            ms=np.array([x.get("ms", np.nan) for x in t], dtype=float),
                            refused=np.array(self.refused), gated=np.array(self.gated))
        right = sum(x["truth"] == x["guess"] for x in t)
        print(f"{name}: {right}/{len(t)} right, {self.refused} refused, {self.gated} called not a tap", flush=True)

    def mark_prompt(self):
        """Notes, by sample position on the tape, when the current prompt began."""
        import json
        at = self.mic.tape.tell() // 2 / SR if self.mic.tape else 0.0
        self.prompt_log.append((at, NOISE_PROMPTS[self.prompt][0]))
        with open(os.path.join(DATA, self.tape_name + ".json"), "w") as f:
            json.dump({"sr": SR, "format": "s16le mono", "prompts": self.prompt_log}, f, indent=1)

    def prompt_seconds(self):
        """Seconds recorded under each prompt so far."""
        now = self.mic.tape.tell() // 2 / SR if self.mic.tape else 0.0
        out = {}
        for (t, p), nxt in zip(self.prompt_log, self.prompt_log[1:] + [(now, None)]):
            out[p] = out.get(p, 0) + nxt[0] - t
        return out

    def collect_plan(self, per_spot, per_zone, w=1366, h=768):
        """Every tap of a collect session, in order.

        Grid phase: spot by spot, cycling through the four quadrants
        (top-left spot 1, top-right spot 1, …), `per_spot` taps each, so no
        quadrant is tapped only early or only late in the session.
        Random phase: `per_zone` points per quadrant, each at least
        MIN_GAP_PX from the others and the grid, served in rounds of one
        per quadrant (shuffled), so the counts stay equal throughout."""
        rng = np.random.default_rng()
        plan = []
        for i in range(len(SPOTS)):
            for z in range(len(ZONES)):
                plan += [(self.spot_xy(z, i), f"grid spot {i + 1}")] * per_spot
        far = lambda p, q: np.hypot((p[0] - q[0]) * w, (p[1] - q[1]) * h) >= gap
        points = []
        for z in range(len(ZONES)):
            gap = MIN_GAP_PX
            mine = []
            taken = [self.spot_xy(z, i) for i in range(len(SPOTS))]
            tries = 0
            while len(mine) < per_zone:
                u, v = rng.uniform(0.06, 0.94, 2)
                p = ((z % 2 + u) / 2, (z // 2 + v) / 2)
                if all(far(p, q) for q in taken + mine):
                    mine.append(p)
                tries += 1
                if tries % 5000 == 0:   # too crowded: relax the spacing a little
                    gap *= 0.9
            points.append(mine)
        for r in range(per_zone):
            for z in rng.permutation(len(ZONES)):
                plan.append((points[z][r], "random"))
        return plan

    @property
    def collect_target(self):
        i = len(self.collected)
        return self.plan[i][0] if i < len(self.plan) else None

    @staticmethod
    def random_point():
        """Anywhere on screen, a little clear of the dividers."""
        rng = np.random.default_rng()
        z = int(rng.integers(0, 4))
        u, v = rng.uniform(0.06, 0.94, 2)
        return ((z % 2 + u) / 2, (z // 2 + v) / 2)

    @staticmethod
    def anywhere():
        """A point anywhere on screen, dividers included: the x,y test
        wants every distance from the middle lines."""
        return tuple(np.random.default_rng().uniform(0.04, 0.96, 2))

    def save_located(self):
        if not self.located:
            return
        os.makedirs(DATA, exist_ok=True)
        c = self.located
        np.savez_compressed(os.path.join(DATA, self.locate_file),
                            model=np.array(self.locator.label),
                            windows=np.array([x["window"] for x in c]),
                            where=np.array([x["where"] for x in c], dtype=float),
                            guess_xy=np.array([x["guess"] for x in c], dtype=float),
                            truth=np.array([self.zone_of(x["where"]) for x in c]))

    def save_collected(self):
        if not self.collected:
            return
        os.makedirs(DATA, exist_ok=True)
        c = self.collected
        np.savez_compressed(os.path.join(DATA, self.collect_file),
                            windows=np.array([x["window"] for x in c]),
                            where=np.array([x["where"] for x in c], dtype=float),
                            truth=np.array([x["truth"] for x in c]),
                            source=np.array([x.get("source", "screen tap") for x in c]))

    @staticmethod
    def zone_of(pos):
        return int(pos[0] >= 0.5) + 2 * int(pos[1] >= 0.5)

    def train(self):
        feats = [t[1] for zone in self.taps for t in zone]
        labels = [z for z, zone in enumerate(self.taps) for _ in zone]
        self.model = Model(feats, labels)
        self.accuracy, self.confusion = self.model.leave_one_out()

    # -------------------------------------------------------- events

    def tap(self, window):
        try:
            self.handle_tap(window)
        except Exception as e:  # show it, don't silently stay stuck
            self.error = f"{type(e).__name__}: {e}"
            print(self.error, file=sys.stderr, flush=True)

    def early(self, window, heard_at):
        """The final model's answer, 45 ms after the hit: shown at once.
        The full window arrives at `tap` later, for saving."""
        try:
            if time.time() < self.pause_until or self.mode not in ("test", "free"):
                return
            if not self.learned.is_tap(window):
                ms = (time.perf_counter() - heard_at) * 1000
                self.decision = {"guess": None, "conf": None, "ms": ms, "refused": True, "gated": True}
                self.last_ms = ms
                self.warn = ("Not a screen tap: ignored", time.time())
                return
            guess, conf, _ = self.learned.predict(window)
            ms = (time.perf_counter() - heard_at) * 1000
            refused = conf < self.learned.cutoff
            self.decision = {"guess": guess, "conf": conf, "ms": ms, "refused": refused}
            self.last_ms = ms
            if refused:
                self.warn = (f"Not sure ({conf:.0%}): tap again", time.time())
            else:
                self.flash = (guess, time.time())
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(self.error, file=sys.stderr, flush=True)

    def answer(self, window, feat):
        """(guess, confidence, per-zone, ms after the hit) for this tap: the
        early decision when the model makes one, else computed now. guess is
        None when the model refused or the gate called it not a tap
        (self.last_gated says which)."""
        self.last_gated = False
        if getattr(self.learned, "early", False):
            d, self.decision = self.decision, None
            if d is None:
                return None, None, None, None
            self.last_gated = d.get("gated", False)
            return (None if d["refused"] else d["guess"]), d["conf"], None, d["ms"]
        guess, conf, per_zone = self.learned.predict(window) if self.learned else self.model.predict(feat)
        return guess, conf, per_zone, None

    def handle_tap(self, window):
        if time.time() < self.pause_until:
            return
        feat, curve, env = fingerprint(window)
        if np.abs(window).max() >= 0.99:
            self.warn = ("That tap clipped the mic: tap a little softer", time.time())
        if self.mode == "calibrate":
            self.where[self.zone].append(self.spot_xy(self.zone, len(self.taps[self.zone]) // PER_SPOT))
            self.taps[self.zone].append((window, feat, curve, env))
            self.save_taps("progress.npz")
            self.last = (window, curve, env, self.zone, None, None)
            self.flash = (self.zone, time.time())
            if len(self.taps[self.zone]) >= TAPS:
                self.pause_until = time.time() + 1.2
                if self.zone + 1 < len(ZONES):
                    self.zone += 1
                else:
                    self.train()
                    self.save()
                    self.mode = "review"
        elif self.mode == "test" and (self.model or self.learned) and len(self.trials) < len(self.targets):
            guess, conf, per_zone, ms = self.answer(window, feat)
            self.last = (window, curve, env, guess, conf, per_zone)
            if guess is None:          # refused or not a tap: same point again
                self.gated += self.last_gated
                self.refused += conf is not None
                return
            pos = self.targets[len(self.trials)]
            self.trials.append({"window": window, "where": pos, "truth": self.zone_of(pos),
                                "guess": guess, "conf": conf, "ms": ms, "at": time.time()})
            if ms is None:             # an early answer has flashed already
                self.flash = (guess, time.time())
            if len(self.trials) == len(self.targets):
                self.save_trials()
            else:
                self.pause_until = time.time() + 0.5
        elif self.mode == "collect" and len(self.collected) < self.collect_goal:
            pos, phase = self.plan[len(self.collected)]
            self.collected.append({"window": window, "where": pos, "truth": self.zone_of(pos),
                                   "source": phase, "at": time.time()})
            self.last = (window, curve, env, None, None, None)
            # Every 10 taps, and at the end: an Esc or a crash keeps nearly all.
            if len(self.collected) % 10 == 0 or len(self.collected) == self.collect_goal:
                self.save_collected()
        elif self.mode == "locate":
            guess = self.locator.predict(window)
            self.located.append({"window": window, "where": self.locate_target, "guess": guess, "at": time.time()})
            self.last = (window, curve, env, self.zone_of(guess), None, None)
            self.locate_target = self.anywhere()
            self.pause_until = time.time() + 0.5
            self.save_located()
        elif self.mode == "noise":
            self.last = (window, curve, env, None, None, None)   # just for the scope
        elif self.mode == "free" and (self.model or self.learned):
            guess, conf, per_zone, ms = self.answer(window, feat)
            self.last = (window, curve, env, guess, conf, per_zone)
            if guess is not None and ms is None:
                self.flash = (guess, time.time())

    def mark_noise(self):
        """N in free mode: the last sound wasn't a screen tap. Kept with this
        run's other false alarms as noise-free-<time>.npz, which the next
        ml.py --export-gate trains the gate on."""
        self.false_alarms.append(self.last[0])
        os.makedirs(DATA, exist_ok=True)
        np.savez_compressed(os.path.join(DATA, self.noise_file), windows=np.array(self.false_alarms, np.float32))
        self.warn = (f"Saved as not a tap ({len(self.false_alarms)} this run)", time.time())
        self.last = None

    def key(self, _c, keyval, _code, _state):
        # The keyboard is loud too: don't count the next 0.4 s as a tap.
        # (Except when typing is the very noise being recorded.)
        if self.mode != "noise":
            self.mic.mute_until = time.time() + 0.4
        name = Gdk.keyval_name(keyval)
        if self.mode == "noise" and name != "Escape":
            return True   # typing is part of the noise; only Esc does anything
        if name == "Escape":
            if self.mode == "collect":
                self.save_collected()
            if self.mic.tape:
                self.mic.tape, tape = None, self.mic.tape
                tape.close()
            self.mic.stop()
            self.win.get_application().quit()
        elif name == "BackSpace" and self.mode == "collect" and self.collected:
            self.collected.pop()
        elif name == "BackSpace" and self.mode == "locate" and self.located:
            self.locate_target = self.located.pop()["where"]
            self.save_located()
        elif name == "BackSpace" and self.mode == "calibrate":
            if not self.taps[self.zone] and self.zone > 0:
                self.zone -= 1
            if self.taps[self.zone]:
                self.taps[self.zone].pop()
                self.where[self.zone].pop()
        elif name in ("r", "R"):
            self.taps = [[] for _ in ZONES]
            self.where = [[] for _ in ZONES]
            self.zone, self.mode, self.last = 0, "calibrate", None
        elif name in ("t", "T") and (self.model or self.learned):
            self.start_test()
        elif name in ("f", "F") and (self.model or self.learned):
            self.mode, self.last = "free", None
        elif name in ("n", "N") and self.mode == "free" and self.last:
            self.mark_noise()
        elif name in ("v", "V") and self.model:
            self.mode = "review"
        return True

    # --------------------------------------------------------- drawing

    def draw(self, _area, cr, w, h):
        cr.set_source_rgb(*BG)
        cr.paint()
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        qw, qh = w / 2, h / 2
        for z in range(len(ZONES)):
            self.quadrant(cr, z, (z % 2) * qw, (z // 2) * qh, qw, qh)
        # The cross between quadrants.
        cr.set_source_rgba(*FG, 0.12)
        cr.set_line_width(1)
        cr.move_to(qw, 0); cr.line_to(qw, h); cr.move_to(0, qh); cr.line_to(w, qh)
        cr.stroke()
        testing = self.mode == "test" and len(self.trials) < len(self.targets)
        if self.mode == "calibrate":
            self.spots(cr, w, h)
        elif self.mode == "test":
            self.test_marks(cr, w, h)
        elif self.mode == "collect":
            self.collect_marks(cr, w, h)
        elif self.mode == "noise":
            self.noise_prompt(cr, w, h)
        elif self.mode == "locate":
            self.locate_marks(cr, w, h)
        collecting = self.mode == "collect" and len(self.collected) < self.collect_goal
        if self.mode == "noise":
            self.scope(cr, w / 2 - 300, h * 0.62, 600, 140)
        elif self.mode == "review":
            self.review(cr, w, h)
        elif self.mode == "test" and not testing:
            self.test_summary(cr, w, h)
        elif self.mode == "calibrate" or testing or collecting or self.mode == "locate":
            # Keep the panel out of the half being tapped.
            if self.mode == "calibrate":
                top = self.zone < 2
            elif self.mode == "locate":
                top = self.locate_target[1] < 0.5
            elif collecting:
                top = self.collect_target[1] < 0.5
            else:
                top = self.targets[len(self.trials)][1] < 0.5
            sy = h * (0.75 if top else 0.25)
            self.scope(cr, w / 2 - 300, sy - 70, 600, 140)
        else:
            self.scope(cr, w / 2 - 300, h / 2 - 70, 600, 140)
        msg, at = self.warn
        if msg and time.time() - at < 2.5:
            self.text(cr, msg, w / 2, h - 46, 15, ZONE_RGB[1], center=True)
        if self.error:
            self.text(cr, self.error, w / 2, h - 20, 13, ZONE_RGB[0], center=True)

    def quadrant(self, cr, z, x, y, w, h):
        rgb = ZONE_RGB[z]
        target = self.mode == "calibrate" and z == self.zone
        # Flash where the tap landed (calibration) or was guessed (test).
        fz, ft = self.flash
        age = time.time() - ft
        if fz == z and age < 0.6:
            cr.set_source_rgba(*(ACCENT if self.mode == "calibrate" else rgb), 0.35 * (1 - age / 0.6))
            cr.rectangle(x, y, w, h)
            cr.fill()
        if target:
            cr.set_source_rgba(*ACCENT, 0.9)
            cr.set_line_width(3)
            cr.rectangle(x + 6, y + 6, w - 12, h - 12)
            cr.stroke()

        # Corner label: name and count.
        tx = x + 28 if z % 2 == 0 else x + w - 28
        ty = y + 44 if z < 2 else y + h - 28
        n = len(self.taps[z])
        self.text(cr, ZONES[z].upper(), tx, ty, 18, rgb, right=z % 2 == 1)
        sub = f"{n}/{TAPS} taps" if self.mode == "calibrate" else f"{n} taps"
        if self.mode == "collect":
            mine = sum(t["truth"] == z for t in self.collected)
            sub = f"{mine}/{self.collect_goal // len(ZONES)} taps"
        self.text(cr, sub, tx, ty + (22 if z < 2 else -26), 13, FG, 0.6, right=z % 2 == 1)
        # This quadrant's fingerprints: every tap thin, the average bold.
        px, pw = x + 40, w - 80
        py = y + (90 if z < 2 else 70)
        ph = h * 0.32
        if z < 2:
            py = y + h - ph - 110
        # Under the spots being tapped, the chart steps back.
        cr.push_group()
        self.spectrum_plot(cr, z, px, py, pw, ph)
        cr.pop_group_to_source()
        cr.paint_with_alpha(0.35 if target or self.mode in ("test", "collect", "locate") else 1.0)

    @staticmethod
    def spot_xy(z, i):
        """Spot i of quadrant z, as fractions of the whole screen."""
        fx, fy = SPOTS[min(i, len(SPOTS) - 1)]
        return ((z % 2 + fx) / 2, (z // 2 + fy) / 2)

    def spots(self, cr, w, h):
        """The target spots of the quadrant being calibrated: done ones
        filled, the current one a pulsing ring with a crosshair."""
        z = self.zone
        done = len(self.taps[z])
        current = done // PER_SPOT
        rgb = ZONE_RGB[z]
        for i in range(len(SPOTS)):
            fx, fy = self.spot_xy(z, i)
            sx, sy = fx * w, fy * h
            cr.set_source_rgb(*BG)
            cr.arc(sx, sy, 15, 0, 2 * np.pi)
            cr.fill()
            if i < current:
                cr.set_source_rgba(*rgb, 0.9)
                cr.arc(sx, sy, 7, 0, 2 * np.pi)
                cr.fill()
            elif i > current:
                cr.set_source_rgba(*FG, 0.3)
                cr.set_line_width(1.5)
                cr.arc(sx, sy, 7, 0, 2 * np.pi)
                cr.stroke()
        if current < len(SPOTS):
            fx, fy = self.spot_xy(z, current)
            self.ring(cr, fx * w, fy * h, w, f"tap here ×{PER_SPOT - done % PER_SPOT}")

    def ring(self, cr, sx, sy, w, label):
        """The pulsing target: where to tap next."""
        pulse = 0.5 + 0.5 * np.sin(time.time() * 6)
        cr.set_source_rgba(*ACCENT, 0.25 + 0.2 * pulse)
        cr.arc(sx, sy, 26 + 6 * pulse, 0, 2 * np.pi)
        cr.fill()
        cr.set_source_rgb(*ACCENT)
        cr.set_line_width(2.5)
        cr.arc(sx, sy, 16, 0, 2 * np.pi)
        cr.stroke()
        cr.move_to(sx - 24, sy); cr.line_to(sx + 24, sy)
        cr.move_to(sx, sy - 24); cr.line_to(sx, sy + 24)
        cr.stroke()
        right_side = sx < w - 220
        self.text(cr, label, sx + (38 if right_side else -38), sy + 5, 15, ACCENT, right=not right_side)

    def test_marks(self, cr, w, h):
        """Test round: every point tapped so far as ✓ (right quadrant) or ✕
        (with what it was taken for), and the next point as the ring."""
        good, bad = (0.55, 0.85, 0.5), (0.95, 0.35, 0.35)
        for t in self.trials:
            sx, sy = t["where"][0] * w, t["where"][1] * h
            ok = t["guess"] == t["truth"]
            cr.set_source_rgb(*BG)
            cr.arc(sx, sy, 13, 0, 2 * np.pi)
            cr.fill()
            cr.set_source_rgb(*(good if ok else bad))
            cr.set_line_width(3)
            if ok:
                cr.move_to(sx - 7, sy); cr.line_to(sx - 2, sy + 6); cr.line_to(sx + 8, sy - 7)
            else:
                cr.move_to(sx - 7, sy - 7); cr.line_to(sx + 7, sy + 7)
                cr.move_to(sx + 7, sy - 7); cr.line_to(sx - 7, sy + 7)
            cr.stroke()
            if not ok:
                self.text(cr, f"heard {ZONES[t['guess']]}", sx + 16, sy + 5, 12, ZONE_RGB[t["guess"]])
            # The newest result pops for a moment.
            age = time.time() - t["at"]
            if age < 0.8:
                cr.set_source_rgba(*(good if ok else bad), 0.5 * (1 - age / 0.8))
                cr.arc(sx, sy, 14 + 40 * age, 0, 2 * np.pi)
                cr.set_line_width(3)
                cr.stroke()
        i = len(self.trials)
        if i < len(self.targets):
            fx, fy = self.targets[i]
            self.ring(cr, fx * w, fy * h, w, f"tap here  {i + 1}/{len(self.targets)}")
            right = sum(t["guess"] == t["truth"] for t in self.trials)
            refused = f"  ·  {self.refused} not sure (tapped again)" if self.refused else ""
            self.text(cr, f"{right}/{i} right so far{refused}", w / 2, 28, 15, FG, 0.8, center=True)

    def collect_marks(self, cr, w, h):
        """Collect session: past taps as small dots, the next point as the
        ring, and progress with pace at the top."""
        for t in self.collected[-80:]:
            cr.set_source_rgba(*ZONE_RGB[t["truth"]], 0.6)
            cr.arc(t["where"][0] * w, t["where"][1] * h, 4, 0, 2 * np.pi)
            cr.fill()
        n = len(self.collected)
        if n < self.collect_goal:
            fx, fy = self.collect_target
            phase = self.plan[n][1]
            if phase == "random":
                label = "tap here"
            else:
                # Taps left at this spot: the run of identical plan entries ahead.
                left_here = 1
                while n + left_here < len(self.plan) and self.plan[n + left_here] == self.plan[n]:
                    left_here += 1
                label = f"tap here ×{left_here}"
                for i in range(len(SPOTS)):     # the whole grid, faint
                    for z in range(len(ZONES)):
                        px, py = self.spot_xy(z, i)
                        cr.set_source_rgba(*FG, 0.18)
                        cr.arc(px * w, py * h, 5, 0, 2 * np.pi)
                        cr.fill()
            self.ring(cr, fx * w, fy * h, w, label)
            recent = [t["at"] for t in self.collected[-10:]]
            pace = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0
            left = (self.collect_goal - n) / pace / 60 if pace else 0
            stage = "1 · grid" if phase != "random" else "2 · random points"
            status = f"phase {stage}  ·  {n}/{self.collect_goal} taps" + (f"  ·  ~{left:.0f} min left" if pace else "")
            self.text(cr, status, w / 2, 28, 15, FG, 0.8, center=True)
        else:
            self.text(cr, f"Done: {n} taps saved to {self.collect_file}", w / 2, h / 2 - 10, 20, ACCENT, center=True)
            self.text(cr, "Esc to close", w / 2, h / 2 + 22, 14, FG, 0.7, center=True)

    def locate_marks(self, cr, w, h):
        """Locate mode: for each recent tap, where you tapped (hollow ring),
        where the model heard it (dot in the colour of that quadrant), and a
        line between; older ones fade. Running errors in pixels on top."""
        recent = self.located[-LOCATE_TRAIL:]
        for k, t in enumerate(recent):
            fade = 0.25 + 0.75 * (k + 1) / len(recent)
            tx, ty = t["where"][0] * w, t["where"][1] * h
            gx, gy = t["guess"][0] * w, t["guess"][1] * h
            rgb = ZONE_RGB[self.zone_of(t["guess"])]
            cr.set_source_rgba(*FG, 0.35 * fade)
            cr.set_line_width(1.5)
            cr.move_to(tx, ty); cr.line_to(gx, gy)
            cr.stroke()
            cr.set_source_rgba(*FG, 0.8 * fade)
            cr.arc(tx, ty, 7, 0, 2 * np.pi)
            cr.stroke()
            cr.set_source_rgba(*rgb, fade)
            cr.arc(gx, gy, 6, 0, 2 * np.pi)
            cr.fill()
        if self.located:
            last = self.located[-1]
            age = time.time() - last["at"]
            if age < 0.8:      # the newest guess pops
                cr.set_source_rgba(*ACCENT, 0.6 * (1 - age / 0.8))
                cr.arc(last["guess"][0] * w, last["guess"][1] * h, 8 + 40 * age, 0, 2 * np.pi)
                cr.set_line_width(3)
                cr.stroke()
        self.ring(cr, self.locate_target[0] * w, self.locate_target[1] * h, w, "tap here")

        # Running numbers, in pixels of this screen.
        n = len(self.located)
        if n:
            T = np.array([t["where"] for t in self.located]) * (w, h)
            G = np.array([t["guess"] for t in self.located]) * (w, h)
            dx, dy = np.abs(G[:, 0] - T[:, 0]), np.abs(G[:, 1] - T[:, 1])
            dist = np.hypot(dx, dy)
            right = sum(self.zone_of(t["guess"]) == self.zone_of(t["where"]) for t in self.located)
            ldx, ldy = G[-1] - T[-1]
            line1 = (f"{n} taps  ·  x error {dx.mean():.0f} px (median {np.median(dx):.0f})  ·  "
                     f"y error {dy.mean():.0f} px (median {np.median(dy):.0f})  ·  miss {np.median(dist):.0f} px median")
            line2 = (f"last tap: x {ldx:+.0f} px, y {ldy:+.0f} px  ·  quadrant right {right}/{n}  ·  "
                     f"screen {w}×{h}  ·  Backspace undo · Esc quit")
        else:
            line1 = "Tap the ring: the dot is where the model hears it, the line is the miss"
            line2 = f"{self.locator.label}  ·  Esc quit"
        self.text(cr, line1, w / 2, 28, 15, FG, 0.9, center=True)
        self.text(cr, line2, w / 2, 50, 12, FG, 0.6, center=True)

    def noise_prompt(self, cr, w, h):
        """Noise session: what to do now, how many sounds it caught, and the
        list of prompts with their counts."""
        cr.set_source_rgba(*BG, 0.9)
        cr.rectangle(0, 0, w, h)
        cr.fill()
        total = self.mic.tape.tell() / 2 / SR if self.mic.tape else 0.0
        dot = 0.5 + 0.5 * np.sin(time.time() * 4)
        cr.set_source_rgba(*ZONE_RGB[0], 0.5 + 0.5 * dot)
        cr.arc(w / 2 - 16, h * 0.13 - 9, 9, 0, 2 * np.pi)
        cr.fill()
        self.text(cr, f"{int(total // 60)}:{total % 60:04.1f}", w / 2 + 2, h * 0.13, 26, FG, center=False)
        self.text(cr, "Recording everything: make any noise that isn't a screen tap", w / 2, h * 0.13 + 40, 18, ACCENT, center=True)
        self.text(cr, "Ideas:", w / 2 - 250, h * 0.3, 13, FG, 0.5)
        for i, (t, d) in enumerate(NOISE_PROMPTS):
            self.text(cr, f"·  {t}", w / 2 - 250, h * 0.3 + 26 + 24 * i, 14, FG, 0.85)
            self.text(cr, d, w / 2 + 120, h * 0.3 + 26 + 24 * i, 12, FG, 0.45)

    def test_summary(self, cr, w, h):
        """End of a test round: score, per quadrant, and what was mixed up."""
        t = self.trials
        right = sum(x["guess"] == x["truth"] for x in t)
        pw, ph = 600, 335
        x, y = (w - pw) / 2, (h - ph) / 2
        cr.set_source_rgba(*BG, 0.97)
        self.rounded(cr, x, y, pw, ph, 12)
        cr.fill_preserve()
        cr.set_source_rgba(*FG, 0.2)
        cr.stroke()
        self.text(cr, "Test round: random points it never saw", x + 24, y + 36, 16, FG)
        self.text(cr, self.learned.label if self.learned else "kNN on the calibration taps", x + 24, y + 56, 11, FG, 0.5)
        self.text(cr, f"{right}/{len(t)}", x + 24, y + 96, 44, ACCENT if right / len(t) >= 0.85 else ZONE_RGB[0])
        self.text(cr, f"right  ({right / len(t):.0%})", x + 170, y + 96, 16, FG, 0.7)
        ms = [r["ms"] for r in t if r.get("ms") is not None]
        extra = [f"{self.refused} not sure, tapped again"] if getattr(self.learned, "early", False) else []
        if self.gated:
            extra.append(f"{self.gated} heard as not a tap, tapped again")
        if ms:
            extra.append(f"answered {np.median(ms):.0f} ms after the hit (median)")
        # One note per line: side by side they ran past the panel's edge.
        for i, note in enumerate(extra):
            self.text(cr, note, x + 170, y + 116 + 16 * i, 12, FG, 0.6)
        confusion = np.zeros((4, 4), int)
        for r in t:
            confusion[r["truth"], r["guess"]] += 1
        cy = y + 175
        self.text(cr, "tapped ↓   heard →", x + 24, cy, 12, FG, 0.5)
        for j, z in enumerate(ZONES):
            self.text(cr, z, x + 200 + 95 * j, cy, 11, ZONE_RGB[j])
        for i, row in enumerate(confusion):
            self.text(cr, ZONES[i], x + 24, cy + 22 + 20 * i, 12, ZONE_RGB[i])
            for j, n in enumerate(row):
                self.text(cr, str(n), x + 200 + 95 * j, cy + 22 + 20 * i, 12, FG, 1 if i == j else (0.8 if n else 0.25))
        self.text(cr, "T another round  ·  F free tapping  ·  V calibration  ·  R recalibrate  ·  Esc",
                  w / 2, y + ph + 30, 13, FG, 0.8, center=True)

    def spectrum_plot(self, cr, z, x, y, w, h):
        rgb = ZONE_RGB[z]
        cr.set_source_rgba(*FG, 0.06)
        cr.rectangle(x, y, w, h)
        cr.fill()
        lx = lambda f: x + w * np.log(f / EDGES[0]) / np.log(EDGES[-1] / EDGES[0])
        for f, label in ((100, "100"), (1000, "1k"), (10000, "10k")):
            cr.set_source_rgba(*FG, 0.15)
            cr.move_to(lx(f), y); cr.line_to(lx(f), y + h); cr.stroke()
            self.text(cr, label, lx(f) + 3, y + h - 4, 10, FG, 0.4)
        self.text(cr, "Hz", x + w - 18, y + h - 4, 10, FG, 0.4)
        curves = [t[2] for t in self.taps[z]]
        line = lambda c: [(lx(f), y + h * (-v / 60)) for f, v in zip(CENTERS, c)]
        for c in curves:
            self.polyline(cr, line(c), rgb, 0.25, 1)
        if curves:
            self.polyline(cr, line(np.mean(curves, axis=0)), rgb, 1, 2.5)
        # In test mode, the new tap in white over every quadrant's average,
        # with how far it is from that quadrant's taps.
        if self.mode in ("test", "free") and self.last:
            self.polyline(cr, line(self.last[1]), FG, 0.9, 1.5)
            per_zone = self.last[5]
            if per_zone:
                best = int(np.argmin(per_zone))
                self.text(cr, f"distance {per_zone[z]:.1f}", x + 6, y + 14, 12,
                          rgb if z == best else FG, 1 if z == best else 0.5)

    def scope(self, cr, x, y, w, h):
        """Centre panel: 2 s of loudness with the tap threshold, and the
        last tap's waveform."""
        cr.set_source_rgba(*BG, 0.96)
        self.rounded(cr, x, y, w, h, 10)
        cr.fill_preserve()
        cr.set_source_rgba(*FG, 0.15)
        cr.set_line_width(1)
        cr.stroke()

        lw = w * 0.58
        # Copies: the mic thread keeps appending while we draw.
        levels = np.array(list(self.mic.levels))
        marks = list(self.mic.marks)
        to_y = lambda v: y + h - 18 - (h - 36) * np.clip((20 * np.log10(v + 1e-9) + 70) / 70, 0, 1)
        n = len(levels)
        for i, v in enumerate(levels):
            cr.set_source_rgba(*ACCENT, 0.8)
            cr.rectangle(x + 10 + (lw - 20) * i / n, to_y(v), max(1, (lw - 20) / n), y + h - 18 - to_y(v))
            cr.fill()
        ty = to_y(self.mic.threshold)
        cr.set_source_rgba(*FG, 0.5)
        cr.set_dash([4, 4])
        cr.move_to(x + 10, ty); cr.line_to(x + lw - 10, ty); cr.stroke()
        cr.set_dash([])
        for m in marks:
            i = n - (self.mic.count - m)
            if 0 <= i < n:
                mx = x + 10 + (lw - 20) * i / n
                cr.set_source_rgba(*FG, 0.9)
                cr.move_to(mx, y + 18); cr.line_to(mx, y + h - 18); cr.stroke()
        self.text(cr, "mic, last 2 s", x + 10, y + 14, 11, FG, 0.5)
        self.text(cr, "tap threshold", x + lw - 10, ty - 4, 10, FG, 0.5, right=True)

        # The last tap, 170 ms of it.
        wx, ww = x + lw + 10, w - lw - 20
        self.text(cr, "last tap", wx, y + 14, 11, FG, 0.5)
        if self.last:
            wave = self.last[0].mean(axis=1)
            peak = np.abs(wave).max() + 1e-9
            step = max(1, len(wave) // int(ww))
            pts = [(wx + ww * i / len(wave), y + h / 2 - (h / 2 - 22) * wave[i] / peak) for i in range(0, len(wave), step)]
            guess = self.last[3]
            self.polyline(cr, pts, ZONE_RGB[guess] if guess is not None else FG, 0.9, 1)
            if self.mode in ("test", "free") and self.last[4] is not None:
                conf = self.last[4]
                when = f"  ·  {self.last_ms:.0f} ms after the hit" if self.last_ms is not None and self.learned and getattr(self.learned, "early", False) else ""
                if guess is None:
                    self.text(cr, f"not sure ({conf:.0%}): ignored{when}", wx, y + h - 6, 13, FG, 0.7)
                else:
                    self.text(cr, f"→ {ZONES[guess]}  {conf:.0%}{when}", wx, y + h - 6, 13, ZONE_RGB[guess])
            elif self.mode in ("test", "free") and self.last_gated:
                self.text(cr, "not a screen tap: ignored", wx, y + h - 6, 13, FG, 0.7)

        hint = {"calibrate": "Tap the ringed spot  ·  Backspace undo  ·  Esc quit",
                "test": "Tap the ringed point  ·  T restart round  ·  Esc quit",
                "free": "Tap anywhere  ·  N that wasn't a tap  ·  T test round  ·  Esc quit",
                "collect": "Tap the ring  ·  Backspace undo  ·  Esc save and quit",
                "noise": "Don't tap the screen  ·  Esc stop",
                "locate": "Tap the ring  ·  Backspace undo  ·  Esc quit"}[self.mode]
        self.text(cr, hint, x + w / 2, y + h + 26, 13, FG, 0.7, center=True)

    def review(self, cr, w, h):
        """After calibration: all four averages on one plot, and the score."""
        pw, ph = 720, 400
        x, y = (w - pw) / 2, (h - ph) / 2
        cr.set_source_rgba(*BG, 0.97)
        self.rounded(cr, x, y, pw, ph, 12)
        cr.fill_preserve()
        cr.set_source_rgba(*FG, 0.2)
        cr.stroke()
        self.text(cr, "How each quadrant sounds", x + 24, y + 36, 18, FG)
        px, py, plw, plh = x + 24, y + 56, 420, 220
        cr.set_source_rgba(*FG, 0.06)
        cr.rectangle(px, py, plw, plh)
        cr.fill()
        lx = lambda f: px + plw * np.log(f / EDGES[0]) / np.log(EDGES[-1] / EDGES[0])
        for f, label in ((100, "100"), (1000, "1k"), (10000, "10k Hz")):
            cr.set_source_rgba(*FG, 0.15)
            cr.move_to(lx(f), py); cr.line_to(lx(f), py + plh); cr.stroke()
            self.text(cr, label, lx(f) + 3, py + plh - 4, 10, FG, 0.4)
        for z in range(len(ZONES)):
            curves = [t[2] for t in self.taps[z]]
            if curves:
                pts = [(lx(f), py + plh * (-v / 60)) for f, v in zip(CENTERS, np.mean(curves, axis=0))]
                self.polyline(cr, pts, ZONE_RGB[z], 1, 2.5)
            self.text(cr, "━ " + ZONES[z], px + plw + 24, py + 20 + 24 * z, 14, ZONE_RGB[z])

        good = self.accuracy >= 0.85
        self.text(cr, f"{self.accuracy:.0%}", px + plw + 24, py + 170, 44, ACCENT if good else ZONE_RGB[0])
        self.text(cr, "of taps recognised", px + plw + 24, py + 194, 12, FG, 0.6)
        total = sum(len(t) for t in self.taps)
        self.text(cr, f"(each tap judged by the other {total - 1})", px + plw + 24, py + 212, 11, FG, 0.4)

        # Confusion matrix: rows are where you tapped, columns the guess.
        cy = py + plh + 34
        self.text(cr, "tapped ↓   guessed →", px, cy, 12, FG, 0.5)
        for j, z in enumerate(ZONES):
            self.text(cr, z, px + 190 + 120 * j, cy, 12, ZONE_RGB[j])
        for i, row in enumerate(self.confusion):
            self.text(cr, ZONES[i], px, cy + 20 + 18 * i, 12, ZONE_RGB[i])
            for j, n in enumerate(row):
                self.text(cr, str(n), px + 190 + 120 * j, cy + 20 + 18 * i, 12, FG, 1 if i == j else (0.8 if n else 0.25))
        self.text(cr, "T test round  ·  F free tapping  ·  R recalibrate  ·  Esc quit", x + pw / 2, y + ph + 30, 14, FG, 0.8, center=True)

    # --------------------------------------------------------- helpers

    @staticmethod
    def text(cr, s, x, y, size, rgb, alpha=1.0, right=False, center=False):
        cr.set_font_size(size)
        ext = cr.text_extents(s)
        if right:
            x -= ext.x_advance
        elif center:
            x -= ext.x_advance / 2
        cr.set_source_rgba(*rgb, alpha)
        cr.move_to(x, y)
        cr.show_text(s)

    @staticmethod
    def polyline(cr, pts, rgb, alpha, width):
        if not pts:
            return
        cr.set_source_rgba(*rgb, alpha)
        cr.set_line_width(width)
        cr.move_to(*pts[0])
        for p in pts[1:]:
            cr.line_to(*p)
        cr.stroke()

    @staticmethod
    def rounded(cr, x, y, w, h, r):
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -np.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, np.pi / 2)
        cr.arc(x + r, y + h - r, r, np.pi / 2, np.pi)
        cr.arc(x + r, y + r, r, np.pi, 3 * np.pi / 2)
        cr.close_path()


def main():
    args = sys.argv[1:]
    if args[:1] == ["flash"]:
        import flash
        return flash.main()
    if args[:1] == ["layout"]:
        import editor
        return editor.main()
    train = None
    if "--train" in args:   # --train collect-…,test-…  (session names in DATA)
        i = args.index("--train")
        train = args[i + 1].split(",")
        del args[i: i + 2]
    if "--signed" in args:
        train = ["signed"]
        args.remove("--signed")
    knn = "--knn" in args
    if knn:
        args.remove("--knn")
    mode = "calibrate"
    if args[:1] in (["test"], ["free"]):
        mode = args[0]
        # The final ensemble unless another model was asked for.
        if train is None and not knn and os.path.exists(os.path.join(DATA, "final.joblib")):
            train = ["final"]
    elif args[:1] == ["noise"]:
        mode = "noise"
    elif args[:1] == ["locate"]:
        mode = "locate"
    elif args[:1] == ["collect"]:
        mode = "collect" + "".join(f":{int(a)}" for a in args[1:3])
    app = Gtk.Application(application_id="tapzones.overlay")
    holder = []
    app.connect("activate", lambda a: holder.append(App(a, mode, train)))
    app.run([])


if __name__ == "__main__":
    main()
