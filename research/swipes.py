#!/usr/bin/env python3
"""Records left and right swipes across the screen for the swipe
experiment: which way did the finger go? Swipes only by default: the taps
to compare with are already recorded (987 in DATA). The screen isn't a
touchscreen, so nothing says when a swipe happened: each trial is cued
("get ready", then "swipe now") and the samples its "now" window began and
ended at are logged.

    python research/swipes.py [SWIPES_PER_SIDE [TAPS]]     (default 40 0)

Writes research/results/swipes/swipes-<time>.raw (everything the mic
hears, 16-bit stereo at 48 kHz) and swipes-<time>.json (each trial: kind,
row, its "now" window in samples).
The service ignores the mic while this is open (the overlay lock).

Keys: Space start / pause · Backspace redo the last trial · Esc save and quit
"""
import json, os, random, subprocess, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tapzones.gtk import Gtk, GLib, Gdk, Layer, BG, FG, ACCENT, text
from tapzones.audio import SOURCE, SR, HOP, hold_overlay_lock

# Research data stays with the research, not in the app's data folder.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "swipes")
QUIET_S = 5.0                 # first: stay still, the room's own sound
READY_S, GO_S, REST_S = 1.0, 1.8, 0.7
GUIDE_S = 0.8                 # the dot along the arrow: a pace, not a rule
ROWS = (0.3, 0.5, 0.7)        # where the arrow is, as a share of the height
SPAN = (0.15, 0.85)           # the arrow runs across this share of the width


class Tape(threading.Thread):
    """Every sample from the mic to `path`; `n` is how many so far, the
    clock the trials are logged in."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.f = open(path, "wb", buffering=0)
        self.n = 0
        self.proc = subprocess.Popen(
            ["parec", "-d", SOURCE, "--format=s16le", f"--rate={SR}", "--channels=2", "--latency-msec=10"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self):
        while chunk := self.proc.stdout.read(HOP * 4):
            self.f.write(chunk)
            self.n += len(chunk) // 4

    def stop(self):
        self.proc.kill()
        self.f.close()


class App:
    def __init__(self, app, per_side, taps):
        rng = random.Random()
        self.plan = [{"kind": k, "row": rng.choice(ROWS), "at": rng.uniform(*SPAN)}
                     for k in ["left"] * per_side + ["right"] * per_side + ["tap"] * taps]
        rng.shuffle(self.plan)
        self.plan.insert(0, {"kind": "quiet", "row": 0.5, "at": 0.5})
        self.done = []               # trials as logged: kind, row, at, go [start, end], bad
        self.i = 0                   # the trial in self.plan now under way
        self.phase = "intro"         # intro | ready | go | rest | paused | saved
        self.since = time.time()
        os.makedirs(OUT, exist_ok=True)
        self.name = os.path.join(OUT, time.strftime("swipes-%Y%m%d-%H%M%S"))
        self.tape = None

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
        GLib.timeout_add(20, self.tick)
        self.win.present()

    # ---------------------------------------------------------- trials

    def tick(self):
        t = time.time() - self.since
        trial = self.plan[self.i] if self.i < len(self.plan) else None
        if self.phase == "ready" and t >= READY_S:
            self.go_start = self.tape.n
            self.set("go")
        elif self.phase == "go" and t >= (QUIET_S if trial["kind"] == "quiet" else GO_S):
            self.done.append(dict(trial, go=[self.go_start, self.tape.n], bad=False))
            self.save()
            self.set("rest")
        elif self.phase == "rest" and t >= REST_S:
            self.i += 1
            if self.i < len(self.plan):
                self.set("ready")
            else:
                self.finish()
        self.area.queue_draw()
        return self.phase != "closed"

    def set(self, phase):
        self.phase, self.since = phase, time.time()

    def save(self):
        with open(self.name + ".json.part", "w") as f:
            json.dump({"sr": SR, "channels": 2, "format": "s16le", "source": SOURCE,
                       "screen": [self.area.get_width(), self.area.get_height()],
                       "timing": {"ready": READY_S, "go": GO_S, "rest": REST_S, "guide": GUIDE_S},
                       "trials": self.done}, f, indent=1)
        os.replace(self.name + ".json.part", self.name + ".json")

    def finish(self):
        self.save()
        self.tape.stop()
        self.set("saved")
        GLib.timeout_add(2500, self.close)

    def close(self):
        self.phase = "closed"
        self.win.close()

    def key(self, _c, keyval, _code, _state):
        name = Gdk.keyval_name(keyval)
        if name == "Escape":
            if self.tape:
                self.finish()
            else:
                self.close()
        elif name == "space" and self.phase == "intro":
            self.tape = Tape(self.name + ".raw")
            self.tape.start()
            self.set("ready")
        elif name == "space" and self.phase == "paused":
            self.set("ready")
        elif name == "space" and self.phase in ("ready", "rest"):
            # Pausing mid-swipe would cut a trial short: only between them.
            if self.phase == "rest":
                self.i += 1
            if self.i < len(self.plan):
                self.set("paused")
            else:
                self.finish()
        elif name == "BackSpace" and self.done and not self.done[-1]["bad"] and self.done[-1]["kind"] != "quiet":
            # That one went wrong: keep it marked, and do it again at the end.
            self.done[-1]["bad"] = True
            last = self.done[-1]
            self.plan.append({k: last[k] for k in ("kind", "row", "at")})
            self.save()
        return True

    # ---------------------------------------------------------- drawing

    def draw(self, _area, cr, w, h):
        cr.set_source_rgba(*BG, 0.94)
        cr.paint()
        good = sum(not t["bad"] and t["kind"] != "quiet" for t in self.done)
        total = len(self.plan) - 1 - sum(t["bad"] for t in self.done)
        text(cr, f"Swipe experiment · {good} / {total}", 40, 60, 20, FG, 0.7)
        text(cr, "Space pause · Backspace redo the last one · Esc save and quit", w - 40, 60, 14, FG, 0.45, right=True)
        if self.phase == "intro":
            text(cr, "Swipe along each arrow with one finger, touching the screen", w / 2, h * 0.42, 26, FG, center=True)
            text(cr, "When it says \"now\".", w / 2, h * 0.42 + 40, 18, FG, 0.6, center=True)
            text(cr, "Space to start", w / 2, h * 0.62, 22, ACCENT, center=True)
            return
        if self.phase == "saved":
            text(cr, f"Saved {good} trials", w / 2, h / 2, 30, ACCENT, center=True)
            return
        if self.phase == "paused":
            text(cr, "Paused · Space to go on", w / 2, h / 2, 26, FG, center=True)
            return
        trial = self.plan[min(self.i, len(self.plan) - 1)]
        if trial["kind"] == "quiet":
            left = max(0.0, QUIET_S - (time.time() - self.since)) if self.phase == "go" else QUIET_S
            text(cr, "Don't touch anything", w / 2, h / 2, 30, FG, center=True)
            text(cr, f"{left:.0f} s", w / 2, h / 2 + 44, 20, FG, 0.6, center=True)
            return
        live = self.phase == "go"
        alpha = 1.0 if live else 0.3
        y = trial["row"] * h
        x0, x1 = SPAN[0] * w, SPAN[1] * w
        if trial["kind"] == "tap":
            x = trial["at"] * w
            cr.set_source_rgba(*ACCENT, alpha)
            cr.arc(x, y, 18, 0, 6.2832)
            cr.fill()
            label = "tap now" if live else "get ready: tap"
        else:
            a, b = (x1, x0) if trial["kind"] == "left" else (x0, x1)
            cr.set_source_rgba(*ACCENT, alpha)
            cr.set_line_width(6)
            cr.move_to(a, y)
            cr.line_to(b, y)
            cr.stroke()
            d = 1 if b > a else -1
            cr.move_to(b, y)
            cr.line_to(b - d * 28, y - 18)
            cr.line_to(b - d * 28, y + 18)
            cr.close_path()
            cr.fill()
            if live:
                f = min(1.0, (time.time() - self.since) / GUIDE_S)
                cr.set_source_rgba(*FG, 0.9)
                cr.arc(a + (b - a) * f, y, 10, 0, 6.2832)
                cr.fill()
            arrow = "←" if trial["kind"] == "left" else "→"
            label = f"swipe {arrow} now" if live else f"get ready: swipe {trial['kind']}"
        text(cr, label, w / 2, y - 50, 24, FG, alpha, center=True)


def main():
    per_side = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    taps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    app = Gtk.Application(application_id="tapzones.swipes")
    holder = []
    app.connect("activate", lambda a: holder.append(App(a, per_side, taps)))
    app.run([])


if __name__ == "__main__":
    main()
