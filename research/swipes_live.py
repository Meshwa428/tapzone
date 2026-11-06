#!/usr/bin/env python3
"""The swipe detector, live (research/swipe_detector.py trains it): every
0.1 s it asks "was there a swipe in the last second?" and flashes SWIPE
when the answer passes the threshold, then which way: ← or →, from how the
sound changed as the finger travelled (research/swipe_direction.py).

    python research/swipes_live.py

Free mode: swipe, tap, type, knock, and watch what fires. G starts a
scored round: random cues (swipe left / swipe right / tap / do nothing),
each scored a hit, a miss or a false alarm, and each hit's direction right
or wrong. → research/results/swipes/live-<time>.json.
The service ignores the mic while this is open (the overlay lock).

Keys: G scored round · Space pause the round · Esc quit
"""
import collections, json, os, random, subprocess, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import joblib
from tapzones.gtk import Gtk, GLib, Gdk, Layer, BG, FG, ACCENT, text, rounded
from tapzones.audio import SOURCE, hold_overlay_lock
from swipe_detector import HERE, MODEL, SR, FRAME, HOP, WIN, Floor, frame_db, window_features
from swipe_direction import DIRECTION, bounds
from swipe_features import NFFT, slopes

STEP = 10                     # frames between answers: 0.1 s
HOLD_S = 1.0                  # after a swipe fires, it can't fire again for this long
SETTLE_S = 3.0                # the floor needs a few seconds of the room first
AFTER = 50                    # frames to wait after a swipe fires, so all of it is heard
KEEP_S = 4.0                  # the raw sound kept, for reading a swipe's direction
ROUND = {"left": 6, "right": 6, "tap": 9, "nothing": 9}
READY_S, GO_S, REST_S = 1.2, 2.5, 1.0
LATE_S = 0.8                  # a detection this long after "now" ends still counts


class Listener(threading.Thread):
    """The mic → relative dB frames → the model, every STEP frames. `p` is
    the latest swipe probability, `level` the loudness above the room,
    `fired` the times a swipe was detected, `ways` which way each went:
    ("left" | "right", how sure), or None until it has been heard out."""
    def __init__(self, model, threshold, way):
        super().__init__(daemon=True)
        self.model, self.threshold, self.way = model, threshold, way
        self.floor = Floor()
        self.frames = collections.deque(maxlen=WIN + AFTER + WIN)
        self.ways = []
        self.due = None               # frames until the last swipe's direction is read
        # Frame k starts at sample k * HOP of the stream; `audio` holds the
        # last KEEP_S of it, ending at sample `heard`.
        self.n_frames, self.heard = 0, 0
        self.audio = np.zeros(0, np.float32)
        self.p, self.level, self.started = 0.0, 0.0, time.time()
        self.history = collections.deque(maxlen=300)      # (p, level) per answer: the last 30 s
        self.fired = []
        self.proc = subprocess.Popen(
            ["parec", "-d", SOURCE, "--format=s16le", f"--rate={SR}", "--channels=1", "--latency-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self):
        tail, since = np.zeros(0, np.float32), 0
        while chunk := self.proc.stdout.read(HOP * 2 * 5):
            new = np.frombuffer(chunk, "<i2").astype(np.float32) / 32768
            self.audio = np.concatenate([self.audio, new])[-int(KEEP_S * SR):]
            self.heard += len(new)
            tail = np.concatenate([tail, new])
            n = 1 + (len(tail) - FRAME) // HOP if len(tail) >= FRAME else 0
            if not n:
                continue
            rel = self.floor.feed(frame_db(tail[: FRAME + (n - 1) * HOP]))
            tail = tail[n * HOP:]
            for row in rel:
                self.frames.append(row)
                self.n_frames += 1
                self.level = float(row[-1])
                since += 1
                if self.due is not None:
                    self.due -= 1
                    if self.due <= 0:
                        self.due = None
                        self.read_way()
                if since >= STEP and len(self.frames) >= WIN:
                    since = 0
                    self.answer()

    def answer(self):
        now = time.time()
        if now - self.started < SETTLE_S:
            return
        last = np.array(self.frames)[-WIN:]
        self.p = float(self.model.predict_proba(window_features(last)[None])[0, 1])
        self.history.append((self.p, float(last[-STEP:, -1].max())))
        if self.p > self.threshold and (not self.fired or now - self.fired[-1] > HOLD_S):
            self.fired.append(now)
            self.ways.append(None)
            self.due = AFTER

    def read_way(self):
        """The swipe as in training: the loudest second of what was heard
        around it, from its first frame over the room to its last, then
        each band's trend over that stretch of the raw sound."""
        rel = np.array(self.frames)
        power = 10 ** (np.clip(rel[:, -1], 0, None) / 10)
        s = int(np.argmax(np.convolve(power, np.ones(WIN), "valid")))
        f, l = bounds(rel[s: s + WIN])
        first = self.n_frames - len(rel) + s       # the stream's frame number at rel[s]
        start = self.heard - len(self.audio)        # the stream's sample number at audio[0]
        a, b = (first + f) * HOP - start, (first + l) * HOP + NFFT - start
        mono = self.audio[max(0, a): max(0, b)]
        if len(mono) < NFFT:
            self.ways[-1] = ("unknown", 0.0)
            return
        p_left = float(self.way.predict_proba(slopes(mono)[None])[0, 1])
        self.ways[-1] = ("left", p_left) if p_left >= 0.5 else ("right", 1 - p_left)

    def stop(self):
        self.proc.kill()


class App:
    def __init__(self, app):
        saved = joblib.load(MODEL)                  # our own files, from research/results
        self.threshold = saved["threshold"]
        self.ear = Listener(saved["model"], self.threshold, joblib.load(DIRECTION)["model"])
        self.ear.start()
        self.round = None             # the scored round: {"plan", "i", "phase", "since", "done"}
        self.summary = None

        hold_overlay_lock()
        self.win = Gtk.ApplicationWindow(application=app)
        Layer.init_for_window(self.win)
        Layer.set_layer(self.win, Layer.Layer.OVERLAY)
        Layer.set_namespace(self.win, "tapzones")
        for edge in (Layer.Edge.TOP, Layer.Edge.BOTTOM, Layer.Edge.LEFT, Layer.Edge.RIGHT):
            Layer.set_anchor(self.win, edge, True)
        Layer.set_exclusive_zone(self.win, -1)
        Layer.set_keyboard_mode(self.win, Layer.KeyboardMode.EXCLUSIVE)
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self.draw)
        self.win.set_child(self.area)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.key)
        self.win.add_controller(keys)
        GLib.timeout_add(33, self.tick)
        self.win.present()

    # ---------------------------------------------------------- the scored round

    def start_round(self):
        plan = [k for k, n in ROUND.items() for _ in range(n)]
        random.shuffle(plan)
        self.round = {"plan": plan, "i": 0, "phase": "ready", "since": time.time(), "done": []}
        self.summary = None

    def tick(self):
        r = self.round
        if r and r["phase"] != "paused":
            t = time.time() - r["since"]
            if r["phase"] == "ready" and t >= READY_S:
                r["go"] = time.time()
                r["phase"], r["since"] = "go", time.time()
            elif r["phase"] == "go" and t >= GO_S + LATE_S and self.ear.due is None:
                kind = r["plan"][r["i"]]
                hits = [i for i, f in enumerate(self.ear.fired) if r["go"] <= f <= r["go"] + GO_S + LATE_S]
                swipe = kind in ("left", "right")
                outcome = ("hit" if hits else "miss") if swipe else ("false alarm" if hits else "quiet")
                way = self.ear.ways[hits[0]] if hits else None
                r["done"].append({"kind": kind, "outcome": outcome,
                                  "after_s": round(self.ear.fired[hits[0]] - r["go"], 2) if hits else None,
                                  "heard_way": way[0] if way else None,
                                  "sure": round(way[1], 2) if way else None,
                                  "way_right": (way[0] == kind) if way and swipe else None})
                r["phase"], r["since"] = "rest", time.time()
            elif r["phase"] == "rest" and t >= REST_S - LATE_S:
                r["i"] += 1
                if r["i"] < len(r["plan"]):
                    r["phase"], r["since"] = "ready", time.time()
                else:
                    self.finish_round()
        self.area.queue_draw()
        return True

    def finish_round(self):
        done = self.round["done"]
        swipes = [d for d in done if d["kind"] in ("left", "right")]
        caught = [d for d in swipes if d["outcome"] == "hit"]
        rate = lambda kind: (sum(d["outcome"] == "false alarm" for d in done if d["kind"] == kind),
                             sum(d["kind"] == kind for d in done))
        self.summary = {"swipes caught": (len(caught), len(swipes)),
                        "right way": (sum(bool(d["way_right"]) for d in caught), len(caught)),
                        "taps fired": rate("tap"), "nothing fired": rate("nothing")}
        os.makedirs(HERE, exist_ok=True)
        path = os.path.join(HERE, time.strftime("live-%Y%m%d-%H%M%S.json"))
        with open(path, "w") as f:
            json.dump({"threshold": self.threshold, "timing": {"ready": READY_S, "go": GO_S, "late": LATE_S},
                       "summary": {k: list(v) for k, v in self.summary.items()}, "trials": done}, f, indent=1)
        self.round = None

    def key(self, _c, keyval, _code, _state):
        name = Gdk.keyval_name(keyval)
        if name == "Escape":
            self.ear.stop()
            self.win.close()
        elif name in ("g", "G") and not self.round:
            self.start_round()
        elif name == "space" and self.round:
            r = self.round
            if r["phase"] == "paused":
                r["phase"], r["since"] = "ready", time.time()
            elif r["phase"] in ("ready", "rest"):
                if r["phase"] == "rest":
                    r["i"] += 1
                if r["i"] < len(r["plan"]):
                    r["phase"] = "paused"
                else:
                    self.finish_round()
        return True

    # ---------------------------------------------------------- drawing

    def draw(self, _area, cr, w, h):
        cr.set_source_rgba(*BG, 0.94)
        cr.paint()
        now = time.time()
        text(cr, "Swipe detector · live", 40, 60, 20, FG, 0.7)
        text(cr, "G scored round · Space pause · Esc quit", w - 40, 60, 14, FG, 0.45, right=True)
        settling = now - self.ear.started < SETTLE_S

        # The answer: SWIPE lights up when one is detected, and its arrow
        # once the whole swipe has been heard.
        recent = self.ear.fired and now - self.ear.fired[-1] < 2.0
        way = self.ear.ways[-1] if recent else None
        cx, cy = w / 2, h * 0.36
        rounded(cr, cx - 190, cy - 70, 380, 140, 24)
        cr.set_source_rgba(*(ACCENT if recent else FG), 0.95 if recent else 0.08)
        cr.fill()
        label = ("listening to the room…" if settling else "no swipe" if not recent
                 else "SWIPE …" if way is None else f"SWIPE {dict(left='←', right='→').get(way[0], '?')}")
        text(cr, label, cx, cy + 16, 46 if recent else 30, BG if recent else FG, 1.0 if recent else 0.5, center=True)
        if way:
            text(cr, f"{way[0]} · {way[1]:.0%} sure", cx, cy + 52, 16, BG, 0.8, center=True)
        text(cr, f"{len(self.ear.fired)} swipes heard", cx, cy + 110, 16, FG, 0.55, center=True)

        # The last 30 s: probability (line) against the threshold (dashed).
        gx, gy, gw, gh = w * 0.1, h * 0.58, w * 0.8, h * 0.16
        cr.set_source_rgba(*FG, 0.06)
        cr.rectangle(gx, gy, gw, gh)
        cr.fill()
        cr.set_source_rgba(*FG, 0.4)
        cr.set_dash([6, 6])
        cr.set_line_width(1)
        cr.move_to(gx, gy + gh * (1 - self.threshold))
        cr.line_to(gx + gw, gy + gh * (1 - self.threshold))
        cr.stroke()
        cr.set_dash([])
        hist = list(self.ear.history)
        if hist:
            cr.set_source_rgba(*ACCENT, 0.95)
            cr.set_line_width(2)
            for i, (p, _) in enumerate(hist):
                x = gx + gw * (i + 300 - len(hist)) / 300
                (cr.line_to if i else cr.move_to)(x, gy + gh * (1 - p))
            cr.stroke()
        text(cr, f"swipe probability · now {self.ear.p:.2f} · fires above {self.threshold:.2f}", gx, gy - 10, 14, FG, 0.6)
        text(cr, f"loudness {self.ear.level:+.0f} dB over the room", gx + gw, gy - 10, 14, FG, 0.6, right=True)

        # The scored round's cue, or its result.
        if self.round:
            r = self.round
            kind = r["plan"][min(r["i"], len(r["plan"]) - 1)]
            what = {"left": "swipe ← right to left", "right": "swipe → left to right",
                    "tap": "tap the screen once", "nothing": "do nothing"}[kind]
            live = r["phase"] == "go"
            cue = ("Paused · Space to go on" if r["phase"] == "paused"
                   else f"now: {what}" if live else f"get ready: {what}")
            text(cr, cue, cx, h * 0.86, 28, ACCENT if live else FG, 1.0 if live else 0.5, center=True)
            text(cr, f"{len(r['done'])} / {len(r['plan'])}", cx, h * 0.86 + 34, 15, FG, 0.5, center=True)
        elif self.summary:
            parts = [f"{k}: {a} of {n}" for k, (a, n) in self.summary.items()]
            text(cr, "  ·  ".join(parts), cx, h * 0.86, 22, FG, center=True)
            text(cr, "saved · G for another round", cx, h * 0.86 + 32, 15, FG, 0.5, center=True)
        else:
            text(cr, "Swipe, tap, type, knock: see what fires. G for a scored round.", cx, h * 0.86, 18, FG, 0.55, center=True)


def main():
    if not os.path.exists(MODEL):
        sys.exit(f"no model yet: python research/swipe_detector.py (it writes {MODEL})")
    app = Gtk.Application(application_id="tapzones.swipes_live")
    holder = []
    app.connect("activate", lambda a: holder.append(App(a)))
    app.run([])


if __name__ == "__main__":
    main()
