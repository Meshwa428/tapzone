#!/usr/bin/env python3
"""The layout editor (docs/Gestures/Sections.md): `taps.py layout`.

A full-screen overlay. Pick a gesture, start from a preset or draw sections
by dragging, name them and give each a command. Every section shows how
well the mic tells it apart, estimated live from the taps recorded so far.
Save writes ~/.config/tapzones/gestures.toml and trains the section model
the background service (taps.py run) uses.

Mouse: drag on empty screen draws a section · drag inside one moves it ·
       drag its edge resizes it · right-click deletes it
Keys:  1–5 gesture · P / Shift+P preset · E gesture on/off · F feedback
       Delete remove section · U undo · S save and train · Esc quit
"""
import copy, os, subprocess, sys, threading, time
import numpy as np
import taps                       # restarts the process with layer-shell preloaded, once
from taps import Gtk, GLib, Gdk, Layer, cairo, ACCENT, FG, BG, ZONE_RGB, DATA
import gestures

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = list(gestures.PRESETS)
EDGE_PX = 10                      # how close to an edge a drag resizes instead of moving
SNAP = 0.012                      # edges snap to the screen and to other sections within this
GOOD, MID, BAD = (0.55, 0.85, 0.5), (0.92, 0.75, 0.42), (0.95, 0.38, 0.38)


def grade(a):
    return GOOD if a >= 0.85 else MID if a >= 0.7 else BAD


class Editor:
    def __init__(self, app):
        taps.hold_overlay_lock()
        self.cfg = gestures.load()
        self.saved = copy.deepcopy(self.cfg)
        self.trained = self.model_ok("single")   # the saved single-tap layout's model, if trained
        self.gesture = "single"
        self.selected = None              # index of the selected section
        self.history = []                 # earlier configs, for undo
        self.drag = None
        self.warn = ("", 0.0)
        self.quit_armed = False
        self.taps = None                  # (features, where, session) of every recorded tap
        self.estimate = {}                # layout hash → (overall, per section)
        self.pending = None               # layout waiting for an estimate
        self.training = None              # (Popen, gesture, log path)
        self.status = "loading recorded taps…"
        threading.Thread(target=self.load_taps, daemon=True).start()

        self.win = Gtk.ApplicationWindow(application=app)
        Layer.init_for_window(self.win)
        Layer.set_layer(self.win, Layer.Layer.OVERLAY)
        Layer.set_namespace(self.win, "tapzones")
        for edge in (Layer.Edge.TOP, Layer.Edge.BOTTOM, Layer.Edge.LEFT, Layer.Edge.RIGHT):
            Layer.set_anchor(self.win, edge, True)
        Layer.set_exclusive_zone(self.win, -1)
        Layer.set_keyboard_mode(self.win, Layer.KeyboardMode.EXCLUSIVE)
        hexa = lambda c, a: f"rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {a})"
        css = Gtk.CssProvider()
        css.load_from_data(f"""
            window {{ background: transparent; }}
            .panel {{ background: {hexa(BG, 0.96)}; border: 1px solid {hexa(FG, 0.2)}; border-radius: 12px; padding: 10px 14px; }}
            .panel label {{ color: {hexa(FG, 0.6)}; font-family: monospace; }}
            .panel entry {{ font-family: monospace; }}
        """.encode())
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, 800)

        self.area = Gtk.DrawingArea(focusable=True)
        self.area.set_draw_func(self.draw)
        drag = Gtk.GestureDrag(button=1)
        drag.connect("drag-begin", self.drag_begin)
        drag.connect("drag-update", self.drag_update)
        drag.connect("drag-end", self.drag_end)
        self.area.add_controller(drag)
        right = Gtk.GestureClick(button=3)
        right.connect("pressed", lambda g, n, x, y: self.delete(self.hit(x, y)[0]))
        self.area.add_controller(right)

        # The selected section's name and command, along the bottom.
        self.name = Gtk.Entry(width_chars=22, placeholder_text="name")
        self.command = Gtk.Entry(width_chars=56, placeholder_text="shell command, e.g. omarchy launch browser")
        self.syncing = False
        for entry, field in ((self.name, "name"), (self.command, "command")):
            entry.connect("changed", self.edited, field)
            entry.connect("activate", lambda e: self.area.grab_focus())
        self.panel = Gtk.Box(spacing=10, halign=Gtk.Align.CENTER, valign=Gtk.Align.END, margin_bottom=56)
        self.panel.add_css_class("panel")
        for w in (Gtk.Label(label="Name"), self.name, Gtk.Label(label="Command"), self.command):
            self.panel.append(w)
        self.panel.set_visible(False)
        overlay = Gtk.Overlay()
        overlay.set_child(self.area)
        overlay.add_overlay(self.panel)
        self.win.set_child(overlay)

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.key)
        self.win.add_controller(keys)
        GLib.timeout_add(33, self.tick)
        self.win.present()
        self.area.grab_focus()

    # ------------------------------------------------------------ state

    @property
    def sections(self):
        return self.cfg[self.gesture]["sections"]

    def layout(self):
        return [s["rect"] for s in self.sections]

    def remember(self):
        self.history.append((copy.deepcopy(self.cfg), self.gesture, self.selected))
        del self.history[:-50]
        self.quit_armed = False

    def changed(self):
        self.pending = self.layout()

    def say(self, msg):
        self.warn = (msg, time.time())

    def select(self, i):
        self.selected = i
        self.syncing = True
        if i is None:
            self.panel.set_visible(False)
        else:
            self.name.set_text(self.sections[i]["name"])
            self.command.set_text(self.sections[i]["command"])
            self.panel.set_visible(True)
        self.syncing = False

    def edited(self, entry, field):
        if self.syncing or self.selected is None:
            return
        self.sections[self.selected][field] = entry.get_text()
        self.quit_armed = False

    # -------------------------------------------------- the estimate

    def load_taps(self):
        """Every recorded tap's cell probabilities (gestures.estimate_data)."""
        self.taps = gestures.estimate_data(lambda msg: setattr(self, "status", msg))
        self.status = f"rough estimates from {len(self.taps[1])} recorded taps (usually a few points low; Save shows the real number)"
        self.pending = self.layout()

    def run_estimate(self, layout):
        key = gestures.layout_hash(layout)
        if key not in self.estimate and layout:
            self.estimate[key] = gestures.quick_estimate(*self.taps, layout)

    # ---------------------------------------------------------- mouse

    def frac(self, x, y):
        return x / self.area.get_width(), y / self.area.get_height()

    def hit(self, x, y):
        """(section index, edges) under the pointer. edges is a set of
        'l', 'r', 't', 'b' when on a border (resize), else empty (move)."""
        w, h = self.area.get_width(), self.area.get_height()
        order = ([self.selected] if self.selected is not None else []) + [i for i in range(len(self.sections)) if i != self.selected]
        for i in order:
            rx, ry, rw, rh = self.sections[i]["rect"]
            x0, y0, x1, y1 = rx * w, ry * h, (rx + rw) * w, (ry + rh) * h
            if x0 - EDGE_PX <= x <= x1 + EDGE_PX and y0 - EDGE_PX <= y <= y1 + EDGE_PX:
                edges = {e for e, near in (("l", abs(x - x0) < EDGE_PX), ("r", abs(x - x1) < EDGE_PX),
                                           ("t", abs(y - y0) < EDGE_PX), ("b", abs(y - y1) < EDGE_PX)) if near}
                if edges or (x0 <= x <= x1 and y0 <= y <= y1):
                    return i, edges
        return None, set()

    def snap(self, v, axis, skip):
        """v rounded to half a percent, pulled onto the screen edge or any
        other section's edge within SNAP."""
        others = [0.0, 1.0]
        for j, s in enumerate(self.sections):
            if j != skip:
                p, n = (s["rect"][0], s["rect"][2]) if axis == 0 else (s["rect"][1], s["rect"][3])
                others += [p, p + n]
        near = min(others, key=lambda o: abs(o - v))
        return near if abs(near - v) < SNAP else min(1.0, max(0.0, round(v / 0.005) * 0.005))

    def drag_begin(self, g, x, y):
        i, edges = self.hit(x, y)
        self.remember()
        if i is None:
            self.drag = {"mode": "draw", "start": self.frac(x, y), "index": None}
            self.select(None)
        else:
            self.select(i)
            self.drag = {"mode": "resize" if edges else "move", "edges": edges, "index": i,
                         "start": self.frac(x, y), "orig": list(self.sections[i]["rect"])}
        self.area.grab_focus()

    def drag_update(self, g, dx, dy):
        d = self.drag
        if not d:
            return
        fx, fy = d["start"][0] + dx / self.area.get_width(), d["start"][1] + dy / self.area.get_height()
        if d["mode"] == "draw":
            if d["index"] is None:
                if abs(dx) < 8 and abs(dy) < 8:
                    return
                self.sections.append({"name": f"Section {len(self.sections) + 1}", "rect": [0, 0, 0, 0], "command": ""})
                d["index"] = len(self.sections) - 1
                self.select(d["index"])
            i = d["index"]
            x0, x1 = sorted((self.snap(d["start"][0], 0, i), self.snap(fx, 0, i)))
            y0, y1 = sorted((self.snap(d["start"][1], 1, i), self.snap(fy, 1, i)))
            self.sections[i]["rect"] = [x0, y0, x1 - x0, y1 - y0]
        elif d["mode"] == "move":
            ox, oy, w, h = d["orig"]
            nx = min(max(ox + fx - d["start"][0], 0), 1 - w)
            ny = min(max(oy + fy - d["start"][1], 0), 1 - h)
            sx, sy = self.snap(nx, 0, d["index"]), self.snap(ny, 1, d["index"])
            ex, ey = self.snap(nx + w, 0, d["index"]) - w, self.snap(ny + h, 1, d["index"]) - h
            nx = sx if abs(sx - nx) <= abs(ex - nx) else ex   # snap whichever side is closer
            ny = sy if abs(sy - ny) <= abs(ey - ny) else ey
            self.sections[d["index"]]["rect"] = [min(max(nx, 0), 1 - w), min(max(ny, 0), 1 - h), w, h]
        else:
            x0, y0, w, h = d["orig"]
            x1, y1 = x0 + w, y0 + h
            i = d["index"]
            if "l" in d["edges"]: x0 = min(self.snap(fx, 0, i), x1 - gestures.MIN_SIDE)
            if "r" in d["edges"]: x1 = max(self.snap(fx, 0, i), x0 + gestures.MIN_SIDE)
            if "t" in d["edges"]: y0 = min(self.snap(fy, 1, i), y1 - gestures.MIN_SIDE)
            if "b" in d["edges"]: y1 = max(self.snap(fy, 1, i), y0 + gestures.MIN_SIDE)
            self.sections[i]["rect"] = [max(x0, 0), max(y0, 0), min(x1, 1) - max(x0, 0), min(y1, 1) - max(y0, 0)]

    def drag_end(self, g, dx, dy):
        d, self.drag = self.drag, None
        if not d or d["index"] is None:
            self.history.pop()            # a click on empty screen: nothing changed
            return
        i = d["index"]
        bad = [p for p in gestures.problems(self.cfg, self.gesture) if "overlap" in p or "smaller" in p]
        if bad:
            self.undo()
            self.say(bad[0])
            return
        if d["mode"] != "draw" and d["orig"] == self.sections[i]["rect"]:
            self.history.pop()            # just selected it
        if len(self.sections) > self.cfg["max_sections"]:
            self.say(f"{len(self.sections)} sections: more than the recommended {self.cfg['max_sections']}. "
                     "Raise max_sections in gestures.toml to save it")
        self.changed()

    def delete(self, i):
        if i is None:
            return
        self.remember()
        del self.sections[i]
        self.select(None)
        self.changed()

    def undo(self):
        if self.history:
            self.cfg, self.gesture, sel = self.history.pop()
            self.select(sel if sel is not None and sel < len(self.sections) else None)
            self.changed()

    # ----------------------------------------------------------- keys

    def key(self, _c, keyval, _code, state):
        name = Gdk.keyval_name(keyval)
        typing = self.name.has_focus() or self.command.has_focus()
        if typing:
            if name == "Escape":
                self.area.grab_focus()
                return True
            return False
        shift = bool(state & Gdk.ModifierType.SHIFT_MASK)
        if name == "Escape":
            if self.cfg != self.saved and not self.quit_armed:
                self.quit_armed = True
                self.say("unsaved changes: S saves, Esc again quits without saving")
            else:
                self.win.get_application().quit()
        elif name in [str(k + 1) for k in range(len(gestures.GESTURES))]:
            self.gesture = gestures.GESTURES[int(name) - 1]
            self.select(None)
            self.changed()
        elif name.lower() == "p":
            self.remember()
            cur = next((k for k, v in gestures.PRESETS.items() if [list(r) for r in v] == self.layout()), None)
            k = (PRESETS.index(cur) + (-1 if shift else 1)) % len(PRESETS) if cur else 0
            old = self.sections
            self.cfg[self.gesture]["sections"] = [
                {"name": n, "rect": list(r), "command": old[j]["command"] if j < len(old) else ""}
                for j, (n, r) in enumerate(zip(gestures.PRESET_NAMES[PRESETS[k]], gestures.PRESETS[PRESETS[k]]))]
            self.select(None)
            self.changed()
        elif name.lower() == "e":
            self.remember()
            self.cfg[self.gesture]["enabled"] = not self.cfg[self.gesture]["enabled"]
        elif name.lower() == "f":
            self.remember()
            modes = gestures.FEEDBACK
            self.cfg["feedback"] = modes[(modes.index(self.cfg["feedback"]) + 1) % len(modes)]
        elif name in ("Delete", "BackSpace"):
            self.delete(self.selected)
        elif name.lower() == "u":
            self.undo()
        elif name.lower() == "s":
            self.save()
        return True

    # ----------------------------------------------------- save, train

    def save(self):
        for g in gestures.GESTURES:
            if bad := gestures.problems(self.cfg, g):
                self.gesture = g
                self.say(f"not saved: {gestures.TITLES[g]}: {bad[0]}")
                return
        gestures.save(self.cfg)
        self.saved = copy.deepcopy(self.cfg)
        self.trained = self.model_ok("single")
        self.quit_armed = False
        self.say(f"saved {gestures.CONFIG}")
        g = "single"   # the only gesture the service hears so far
        if self.cfg[g]["enabled"] and not self.model_ok(g):
            if self.training:
                self.training[0].kill()
            log = os.path.join(DATA, f"train-{g}.log")
            proc = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(HERE, "ml.py"), "--export-sections", g],
                                    stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
            self.training = (proc, g, log, time.time())

    def model_ok(self, g):
        import joblib
        path = gestures.model_path(g)
        if not os.path.exists(path):
            return None
        d = joblib.load(path)   # our own file, written by ml.py --export-sections
        return d if d["hash"] == gestures.layout_hash([s["rect"] for s in self.saved[g]["sections"]]) else None

    def tick(self):
        if self.pending is not None and self.taps is not None:
            layout, self.pending = self.pending, None
            self.run_estimate(layout)     # milliseconds: live while dragging
        if self.drag:
            self.pending = self.layout()
        if self.training and self.training[0].poll() is not None:
            proc, g, log, _ = self.training
            self.training = None
            lines = open(log).read().strip().splitlines()
            self.say("trained: the service uses it now" if proc.returncode == 0 else f"training failed: {lines[-1] if lines else '?'}")
            self.trained = self.model_ok("single")
        self.area.queue_draw()
        return True

    # ---------------------------------------------------------- drawing

    def draw(self, _area, cr, w, h):
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_source_rgba(*BG, 0.97)      # dead zones stay dark
        cr.paint()
        est = self.estimate.get(gestures.layout_hash(self.layout()))
        enabled = self.cfg[self.gesture]["enabled"]
        for i, s in enumerate(self.sections):
            x, y, rw, rh = s["rect"][0] * w, s["rect"][1] * h, s["rect"][2] * w, s["rect"][3] * h
            rgb = ZONE_RGB[i % len(ZONE_RGB)]
            taps.App.rounded(cr, x + 3, y + 3, rw - 6, rh - 6, 10)
            cr.set_source_rgba(*rgb, 0.14 if enabled else 0.05)
            cr.fill_preserve()
            cr.set_source_rgba(*(ACCENT if i == self.selected else rgb), 0.95 if i == self.selected else 0.5)
            cr.set_line_width(3 if i == self.selected else 1.5)
            cr.stroke()
            cx, cy = x + rw / 2, y + rh / 2
            taps.App.text(cr, s["name"], cx, cy - 8, 18, FG, 0.95 if enabled else 0.4, center=True)
            taps.App.text(cr, s["command"] or "no command", cx, cy + 14, 11, FG, 0.55 if s["command"] else 0.3, center=True)
            if est and i < len(est[1]):
                a, n = est[1][i]
                if np.isnan(a):
                    taps.App.text(cr, f"only {n} recorded taps here", cx, cy + 40, 13, BAD, 0.9, center=True)
                else:
                    taps.App.text(cr, f"≈ {a:.0%}", cx, cy + 44, 22, grade(a), 0.95, center=True)
                    taps.App.text(cr, f"{n} taps", cx, cy + 62, 10, FG, 0.4, center=True)
        self.header(cr, w, h, est)

    def header(self, cr, w, h, est):
        pw, ph = 760, 128
        x, y = (w - pw) / 2, 24
        taps.App.rounded(cr, x, y, pw, ph, 12)
        cr.set_source_rgba(*BG, 0.96)
        cr.fill_preserve()
        cr.set_source_rgba(*FG, 0.2)
        cr.set_line_width(1)
        cr.stroke()
        # Gesture tabs.
        tx = x + 20
        for k, g in enumerate(gestures.GESTURES):
            on = self.cfg[g]["enabled"]
            label = f"{k + 1} {gestures.TITLES[g]}{'' if on else ' (off)'}"
            self.text_tab(cr, label, tx, y + 30, g == self.gesture)
            cr.set_font_size(12)
            tx += cr.text_extents(label).x_advance + 22
        g = self.gesture
        n = len(self.sections)
        cur = next((k for k, v in gestures.PRESETS.items() if [list(r) for r in v] == self.layout()), "custom")
        line = f"{cur} · {n}/{self.cfg['max_sections']} sections · feedback: {self.cfg['feedback']}"
        if est and not np.isnan(est[0]):
            line += f" · estimate ≈ {est[0]:.0%}"
        taps.App.text(cr, line, x + 20, y + 58, 13, FG, 0.85)
        # What the service will do with it.
        if g not in gestures.DETECTED:
            state, rgb = "the service can't hear this gesture yet: layout is saved for later (docs: Roadmap)", MID
        elif not self.cfg[g]["enabled"]:
            state, rgb = "off: E turns it on", MID
        elif self.training:
            state, rgb = f"training the section model… {time.time() - self.training[3]:.0f} s (about 3–5 min)", MID
        elif self.cfg != self.saved:
            state, rgb = "unsaved: S saves and trains", MID
        elif d := self.trained:
            state, rgb = (f"trained: {d['accuracy']:.0%} right across sessions; answers {d['answered']:.0%} of taps, "
                          f"{d['answered_accuracy']:.0%} of those right"), GOOD
        else:
            state, rgb = "saved, but not trained for this layout: S trains it", MID
        taps.App.text(cr, state, x + 20, y + 80, 12, rgb, 0.95)
        msg, at = self.warn
        if msg and time.time() - at < 6:
            taps.App.text(cr, msg, x + 20, y + 102, 12, BAD if "not" in msg or "fail" in msg or "more than" in msg else FG, 0.95)
        else:
            taps.App.text(cr, self.status, x + 20, y + 102, 11, FG, 0.45)
        taps.App.text(cr, "drag: draw / move / resize · right-click: delete · P preset · E on/off · F feedback · "
                          "U undo · S save · Esc quit", w / 2, h - 20, 12, FG, 0.7, center=True)

    def text_tab(self, cr, label, x, y, active):
        cr.set_font_size(12)
        if active:
            ext = cr.text_extents(label)
            taps.App.rounded(cr, x - 8, y - 16, ext.x_advance + 16, 24, 6)
            cr.set_source_rgba(*ACCENT, 0.25)
            cr.fill()
        taps.App.text(cr, label, x, y, 12, ACCENT if active else FG, 1 if active else 0.6)


def main():
    app = Gtk.Application(application_id="tapzones.editor")
    holder = []
    app.connect("activate", lambda a: holder.append(Editor(a)))
    app.run([])


if __name__ == "__main__":
    main()
