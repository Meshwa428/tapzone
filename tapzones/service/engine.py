"""The tapzones service (docs/Gestures/Gesture system.md), with no UI: it
listens to the mic and runs the command of the section a gesture lands in.
`tapzones run`, or the systemd unit tapzones.service.

Reads ~/.config/tapzones/gestures.toml and the section model trained for it,
and reloads both when they change. Single taps only so far
(docs/Gestures/Roadmap.md).

Front-ends talk to it over a Unix socket, $XDG_RUNTIME_DIR/tapzones.sock,
one JSON object per line each way:

    → {"cmd": "hello", "ui": true}      a UI that draws the flash feedback
    → {"cmd": "status"} · {"cmd": "pause" | "resume" | "toggle"}
    → {"cmd": "editing", "on": true}    an open layout editor: stay quiet
    → {"cmd": "save", "config": {...}}  validate, write, train what needs it
    → {"cmd": "train", "gesture": "single"}
    → {"cmd": "estimate"}               the editor's live-estimate data
    → {"cmd": "mistake"}                the last tap that ran a command wasn't a
                                        tap: keep its sound as noise for the gate
    → {"cmd": "noise", "on": true}      record what isn't a tap: every sound the
                                        detector catches is kept, nothing runs
    → {"cmd": "confirm", "section": 2}  the last miss (a sound the gate turned
                                        away, or a tap it wasn't sure where) was
                                        a tap in section 2: run it, keep it
    → {"cmd": "calibrate"}              open the calibration (tapzones setup)
    → {"cmd": "set", "sound": false}    change feedback or sound; nothing retrains
    ← {"event": "state", ...} · "tap" · "unsure" · "ignored" · "saved"
      · "trained" · "estimate" · "marked" · "confirmed" · "noise" · "error"

Marked taps and recorded noise go to DATA/noise-*.npz, confirmed taps to
DATA/confirmed.npz (a session like the recorded ones), and the gate trains
on both. The service retrains the gate (tapzones train gate) a little after
either changes, and reloads it when that's done. Section models pick
confirmed taps up the next time a layout trains. New recorded sessions
(a calibration, `tapzones collect`) train the single-tap model and then the
gate as soon as the overlay that recorded them closes.

The Omarchy plugin (manifest.json, omarchy/), the GTK flash (gtk/flash.py)
and `tapzones status` (client.py) are such front-ends. With feedback =
"flash" and no UI connected, the service sends a notification instead, so
feedback never silently vanishes.
"""
import json, os, queue, signal, socket, subprocess, sys, threading, time
from contextlib import nullcontext
import numpy as np
from .. import audio, config, layout, sound
from ..models import Gate, SectionModel, model_path, GATE_PATH
from ..paths import CONFIG, DATA, STATE, PAUSED, SOCKET
from ..train import session_paths, noise_paths

MARKED = os.path.join(DATA, "noise-marked.npz")   # taps the user said weren't taps
CONFIRMED = os.path.join(DATA, "confirmed.npz")    # misses the user said were taps
SETUP_LOG = os.path.join(DATA, "setup.log")         # the calibration overlay's output
GATE_AFTER = 20           # s after the last new noise, the gate retrains
RECORD_LIMIT = 15 * 60     # s: a noise recording left running stops by itself


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Client:
    def __init__(self, conn):
        self.conn, self.ui, self.editing = conn, False, False
        self.lock = threading.Lock()

    def send(self, obj):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            self.conn.sendall(data)


class Engine:
    def __init__(self):
        self.q = queue.Queue()                 # everything the main loop handles, in order
        self.clients = []
        self.training = None                   # (Popen, job, log path, started); job: a gesture or "gate"
        self.estimate = None                   # (P, where) once prepared
        self.preparing = False
        self.paused = os.path.exists(PAUSED)
        self.warned = None
        self.last = None                       # the newest tap that ran a command: {at, name, window}
                                               # (window None until the full 150 ms arrives)
        self.missed = None                     # the newest sound that ran nothing: {at, why, window}
        self.awaiting = None                   # last or missed, until its full window arrives
        self.recording = None                  # {path, windows, since, client} while recording noise
        self.gate_due = None                   # when noise arrived that the gate hasn't learned yet
        self.marked = marked_count()
        self.confirmed = marked_count(CONFIRMED)
        self.calibrating = None                # the calibration overlay this service opened (Popen)
        self.progress_seen = None              # the training progress last broadcast
        self.recorded = recordings()           # the recorded sessions, as last trained for
        self.calibration = calibration()
        self.reload()

    # ------------------------------------------------ config and model

    def stamp(self):
        return tuple(os.path.getmtime(p) if os.path.exists(p) else 0
                     for p in (CONFIG, model_path("single"), GATE_PATH))

    def reload(self):
        self.seen = self.stamp()
        self.gate = Gate()
        self.cfg = config.load()
        single = self.cfg["single"]
        self.sections = single["sections"] if single["enabled"] else []
        self.model = SectionModel("single", [s["rect"] for s in self.sections]) if self.sections else None
        if self.model and not self.model.ok and not self.calibration["taps"]:
            why = "not calibrated yet: click the tap icon in the bar to calibrate this laptop"
            log(why)
            if self.warned != why:
                self.notify("tapzones", why)
                self.warned = why
        elif self.model and not self.model.ok:
            why = f"single taps are off: {self.model.why}. Open the layout editor and save to train it."
            log(why)
            if self.warned != why and not self.training:
                self.notify("tapzones", why)
                self.warned = why
        elif self.model:
            log(f"single tap: {len(self.sections)} sections, cutoff {self.model.cutoff:.2f}, "
                f"{self.model.d['accuracy']:.0%} right across sessions")

    def state(self):
        m = self.model
        model = ({"ok": True, "accuracy": m.d["accuracy"], "answered": m.d["answered"],
                  "answered_accuracy": m.d["answered_accuracy"], "cutoff": m.cutoff, "taps": m.d.get("taps", 0),
                  "sittings": m.d.get("sittings", 2),
                  "per_section": [None if a != a else float(a) for a in m.d.get("per_section", [])]} if m and m.ok
                 else {"ok": False, "why": "not calibrated yet" if not self.calibration["taps"]
                       else m.why if m else "single taps are off"})
        last = self.last if self.last and self.last["window"] is not None else None
        missed = self.missed if self.missed and self.missed["window"] is not None else None
        return {"event": "state", "paused": self.paused, "editing": self.quiet(),
                "gate": bool(self.gate.d), "config": self.cfg, "model": model,
                "noise": {"learned": self.gate.d["noise"] if self.gate.d else 0, "marked": self.marked,
                          "confirmed": self.confirmed, "retrain_due": self.gate_due is not None},
                "last": {"name": last["name"], "at": last["at"]} if last else None,
                "missed": {"why": missed["why"], "at": missed["at"]} if missed else None,
                # What the panel corrects: the newer of the two.
                "newest": {k: v for k, v in self.newest().items() if k != "window"} if self.newest() else None,
                "recording": {"since": self.recording["since"], "count": len(self.recording["windows"])}
                             if self.recording else None,
                "training": dict(zip(("progress", "stage"), training_progress(self.training[2])),
                                 gesture=self.training[1], job=self.training[1], since=self.training[3])
                            if self.training else None,
                "calibration": dict(self.calibration, running=self.calibrating is not None),
                "estimate": "ready" if self.estimate else "preparing" if self.preparing else "none",
                "detected": sorted(config.DETECTED), "socket": SOCKET}

    # ----------------------------------------------------------- loop

    def run(self):
        self.listen()
        # The early window answers; the full one (150 ms) is what the gate
        # trains on, kept in case the tap turns out to be a mistake.
        self.mic = audio.Mic(lambda w: self.q.put(("full", w)), lambda w, t: self.q.put(("tap", w, t)))
        self.mic.start()
        log(f"listening on {audio.SOURCE}: gate {'on' if self.gate.d else 'off'}, "
            f"feedback {self.cfg['feedback']}{', paused' if self.paused else ''}; socket {SOCKET}")
        while True:
            try:
                item = self.q.get(timeout=1)
            except queue.Empty:
                item = ("tick",)
            try:
                getattr(self, "on_" + item[0])(*item[1:])
            except Exception as e:             # keep listening whatever one event does
                log(f"{item[0]}: {type(e).__name__}: {e}")
            if not self.mic.is_alive():
                log("the mic stopped (parec exited): restarting the service")
                return 1

    def on_tick(self):
        if self.stamp() != self.seen:
            self.reload()
            self.broadcast(self.state())
        if self.training and self.training[0].poll() is None:
            p = training_progress(self.training[2])
            if p != self.progress_seen:            # the bar moved: tell the front-ends
                self.progress_seen = p
                self.broadcast(self.state())
        if self.training and self.training[0].poll() is not None:
            proc, g, path, _ = self.training
            self.training = None
            lines = open(path).read().strip().splitlines()
            ok = proc.returncode == 0
            msg = next((l for l in reversed(lines) if l.startswith("saved ")), "") if ok else (lines[-1] if lines else "?")
            log(f"training {g}: {'done' if ok else 'failed'}: {msg}")
            if g != "gate":                  # the noise filter retrains on its own, quietly
                self.sound("trained" if ok else "failed")
            self.reload()
            self.broadcast({"event": "trained", "gesture": g, "ok": ok, "message": msg})
            self.broadcast(self.state())
        if self.recording and time.time() - self.recording["since"] > RECORD_LIMIT:
            self.stop_recording()
        if self.calibrating and self.calibrating.poll() is not None:
            proc, self.calibrating = self.calibrating, None
            if proc.returncode != 0:              # never opened, or crashed: say why
                try:
                    with open(SETUP_LOG) as f:
                        lines = f.read().strip().splitlines()
                except OSError:
                    lines = []
                why = lines[-1] if lines else f"exit code {proc.returncode}"
                log(f"calibration failed: {why} (log: {SETUP_LOG})")
                self.sound("failed")
                self.notify("tapzones: calibration didn't open", why)
                self.broadcast({"event": "error", "cmd": "calibrate", "message": why})
            self.broadcast(self.state())
        if not self.calibrating and not audio.overlay_open() and recordings() != self.recorded:
            self.recorded_new()
        if self.gate_due and not self.training and not self.recording and time.time() - self.gate_due > GATE_AFTER:
            self.gate_due = None
            if session_paths() and noise_paths():
                self.train("gate")
            else:
                log("gate: training waits for recorded taps and noise")
            self.broadcast(self.state())

    def recorded_new(self):
        """A calibration (or `tapzones collect`) added taps: train the
        single-tap model on them, and the gate right after it."""
        self.recorded = recordings()
        self.calibration = calibration()
        log(f"new recordings: {self.calibration['taps']} taps from {self.calibration['sittings']} sittings")
        if self.cfg["single"]["enabled"]:
            self.train("single")
        self.gate_due = time.time() - GATE_AFTER - 1     # due now: runs once the single model is done
        self.reload()
        self.broadcast(self.state())

    # ----------------------------------------------------------- taps

    def quiet(self):
        return audio.overlay_open() or any(c.editing for c in self.clients)

    def on_tap(self, window, heard_at):
        if self.recording or self.paused or self.quiet() or not self.model or not self.model.ok:
            return
        if not self.gate.is_tap(window):
            log("not a tap: ignored")
            self.miss("ignored")
            self.broadcast({"event": "ignored", "why": "gate"})
            return
        if len(self.sections) == 1:
            i, conf = 0, 1.0
        else:
            p = self.model.proba(window)
            i, conf = int(p.argmax()), float(p.max())
        ms = (time.perf_counter() - heard_at) * 1000
        s = self.sections[i]
        if conf < self.model.cutoff:
            log(f"not sure ({conf:.0%}, best {s['name']}), {ms:.0f} ms")
            self.miss("unsure")
            self.broadcast({"event": "unsure", "gesture": "single", "best": i, "name": s["name"],
                            "conf": conf, "ms": ms, "feedback": self.cfg["feedback"]})
            return                                # silent: the "tap again" note is enough
        log(f"{s['name']} ({conf:.0%}), {ms:.0f} ms: {s['command'] or '(no command)'}")
        if s["command"].strip():
            subprocess.Popen(["sh", "-c", s["command"]], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.last = self.awaiting = {"at": time.time(), "name": s["name"], "index": i, "window": None}
        self.sound("tap")
        fb = self.cfg["feedback"]
        self.broadcast({"event": "tap", "gesture": "single", "index": i, "name": s["name"], "rect": s["rect"],
                        "command": s["command"], "conf": conf, "ms": ms, "feedback": fb})
        if fb == "notify" or (fb == "flash" and not any(c.ui for c in self.clients)):
            self.notify(s["name"], s["command"] or "single tap")

    def on_full(self, window):
        """The whole window of a sound the detector caught, 150 ms after it."""
        if self.recording:
            r = self.recording
            r["windows"].append(window)
            if len(r["windows"]) % 10 == 0:          # don't lose much if the service stops
                save_noise(r["path"], r["windows"])
            self.broadcast({"event": "noise", "count": len(r["windows"])})
        elif self.awaiting:
            self.awaiting["window"] = window       # the sound of the tap (or miss) just heard
            self.awaiting = None
            self.broadcast(self.state())           # it can be marked now: tell the panel

    def miss(self, why):
        """A sound that ran nothing: the gate turned it away ("ignored"), or
        the model wasn't sure where ("unsure"). The newest can be confirmed."""
        self.missed = self.awaiting = {"at": time.time(), "why": why, "window": None}

    def newest(self):
        """The sound a correction is about: the last tap that ran a command or
        the last miss, whichever came later, once its whole window is in."""
        heard = [r for r in (self.last, self.missed) if r and r["window"] is not None]
        return max(heard, key=lambda r: r["at"]) if heard else None

    def forget(self, r):
        """A sound is corrected once: then it's no longer the one to correct."""
        if r is self.last:
            self.last = None
        if r is self.missed:
            self.missed = None

    # ------------------------------------------------- what isn't a tap

    def mistake(self, client):
        """The newest sound wasn't a tap (one that ran a command, or a miss it
        wasn't sure about): keep it as noise, once: marking it clears it."""
        t = self.newest()
        if not t:
            self.sound("nothing")
            return client.send({"event": "error", "cmd": "mistake",
                                "message": "nothing to mark: no sound since the last one marked"})
        self.forget(t)
        t = dict(t, name=t.get("name", "missed sound"))
        with np.load(MARKED) if os.path.exists(MARKED) else nullcontext() as old:
            windows = [t["window"]] if old is None else list(old["windows"]) + [t["window"]]
            times = [t["at"]] if old is None else list(old["at"]) + [t["at"]]
            names = [t["name"]] if old is None else list(old["name"]) + [t["name"]]
        save_noise(MARKED, windows, at=np.array(times), name=np.array(names))
        self.marked = len(windows)
        ago = time.time() - t["at"]
        log(f"marked as not a tap: {t['name']} from {ago:.0f} s ago ({len(windows)} marked in all)")
        self.gate_due = time.time()
        self.sound("marked")
        self.broadcast({"event": "marked", "name": t["name"], "ago": ago, "total": len(windows)})

    # ---------------------------------------------- what was a tap after all

    def confirm(self, client, i):
        """The newest sound was a tap in section i: a miss (turned away, or
        not sure where), or a tap that ran the wrong section. Run section i's
        command now and keep the sound as a tap there, once (it clears it)."""
        m = self.newest()
        if not m:
            self.sound("nothing")
            return client.send({"event": "error", "cmd": "confirm",
                                "message": "nothing to confirm: no sound since the last one corrected"})
        if not isinstance(i, int) or not 0 <= i < len(self.sections):
            return client.send({"event": "error", "cmd": "confirm", "message": f"no section {i!r}"})
        if m.get("index") == i:
            return client.send({"event": "error", "cmd": "confirm", "message": "that section already ran"})
        self.forget(m)
        was = m.get("name")                      # the section that wrongly ran, or None for a miss
        s = self.sections[i]
        # A spot in the section, for training: section models relabel taps by
        # where they landed (train.export_sections), and only the section is known.
        where = layout.inside_point([x["rect"] for x in self.sections], i)
        with np.load(CONFIRMED) if os.path.exists(CONFIRMED) else nullcontext() as old:
            add = lambda key, v: [v] if old is None else list(old[key]) + [v]
            windows, wheres = add("windows", m["window"]), add("where", where)
            truth, times, names = (add("truth", int(where[1] >= .5) * 2 + int(where[0] >= .5)),
                                   add("at", m["at"]), add("name", s["name"]))
        save_noise(CONFIRMED, windows, where=np.array(wheres), truth=np.array(truth),
                   at=np.array(times), name=np.array(names))
        self.confirmed = len(windows)
        log(f"confirmed as a tap: {s['name']} ({'not ' + was if was else m['why']}, "
            f"{time.time() - m['at']:.0f} s ago, {len(windows)} confirmed in all): {s['command'] or '(no command)'}")
        if s["command"].strip():
            subprocess.Popen(["sh", "-c", s["command"]], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not was:
            self.gate_due = time.time()          # the gate turned it away (or saw it): it learns
        self.sound("tap")
        self.broadcast({"event": "confirmed", "name": s["name"], "was": was, "total": len(windows)})

    def start_recording(self, client):
        if self.recording:
            return
        path = os.path.join(DATA, time.strftime("noise-rec-%Y%m%d-%H%M%S.npz"))
        self.recording = {"path": path, "windows": [], "since": time.time(), "client": client}
        log(f"recording what isn't a tap → {path}")

    def stop_recording(self):
        r, self.recording = self.recording, None
        if r["windows"]:
            save_noise(r["path"], r["windows"])
            self.gate_due = time.time()
        log(f"noise recording stopped: {len(r['windows'])} sounds kept")
        self.broadcast({"event": "noise", "count": len(r["windows"]), "done": True})

    def sound(self, moment):
        if self.cfg.get("sound", True):
            sound.play(moment)

    @staticmethod
    def notify(title, body):
        subprocess.Popen(["notify-send", "-a", "tapzones", "-t", "1500", title, body],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -------------------------------------------------------- commands

    def on_cmd(self, client, msg):
        cmd = msg.get("cmd")
        if cmd == "hello":
            client.ui = bool(msg.get("ui"))
        elif cmd in ("pause", "resume", "toggle"):
            self.set_paused(not self.paused if cmd == "toggle" else cmd == "pause")
        elif cmd == "editing":
            client.editing = bool(msg.get("on"))
        elif cmd == "save":
            return self.save(client, msg.get("config") or {})
        elif cmd == "train":
            self.train(msg.get("gesture", "single"))
        elif cmd == "estimate":
            return self.send_estimate(client)
        elif cmd == "mistake":
            self.mistake(client)
        elif cmd == "confirm":
            self.confirm(client, msg.get("section"))
        elif cmd == "noise":
            if msg.get("on"):
                self.start_recording(client)
            elif self.recording:
                self.stop_recording()
        elif cmd == "calibrate":
            self.calibrate()
        elif cmd == "set":
            return self.set_options(client, msg)
        elif cmd != "status":
            return client.send({"event": "error", "message": f"unknown command {cmd!r}"})
        self.broadcast(self.state()) if cmd != "status" else client.send(self.state())

    def set_paused(self, on):
        self.paused = on
        os.makedirs(STATE, exist_ok=True)
        if on:
            open(PAUSED, "w").close()
        elif os.path.exists(PAUSED):
            os.remove(PAUSED)
        log("paused" if on else "resumed")
        self.sound("pause" if on else "resume")

    def calibrate(self):
        """Opens the calibration overlay (tapzones setup), once. While it's
        open the service stays quiet; when it closes, recorded_new trains."""
        if self.calibrating or audio.overlay_open():
            return
        if self.recording:
            self.stop_recording()
        os.makedirs(DATA, exist_ok=True)
        self.calibrating = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "tapzones", "setup"],
                                            stdout=open(SETUP_LOG, "w"),
                                            stderr=subprocess.STDOUT, start_new_session=True)
        log("calibration opened")

    def set_options(self, client, msg):
        """Settings that only change how a tap is answered: no retraining."""
        cfg = dict(self.cfg)
        if "feedback" in msg:
            if msg["feedback"] not in config.FEEDBACK:
                return client.send({"event": "error", "message": f"feedback must be one of {', '.join(config.FEEDBACK)}"})
            cfg["feedback"] = msg["feedback"]
        if "sound" in msg:
            cfg["sound"] = bool(msg["sound"])
        config.save(cfg)
        self.reload()
        self.broadcast(self.state())

    def save(self, client, cfg):
        try:
            cfg = self.checked(cfg)
        except ValueError as e:
            return client.send({"event": "error", "message": f"not saved: {e}"})
        config.save(cfg)
        self.reload()
        self.broadcast({"event": "saved", "path": CONFIG})
        if cfg["single"]["enabled"] and not (self.model and self.model.ok) and self.calibration["taps"]:
            self.train("single")
        elif self.training and self.training[1] == "single":
            # The saved layout already has a model (say, an edit was undone and
            # saved again): the run in progress is for a layout nobody wants,
            # and would replace the good model when it finished.
            self.stop_training()
            log("training single: stopped, the saved layout already has a model")
        self.broadcast(self.state())

    @staticmethod
    def checked(cfg):
        """The config a client sent, in config.load's shape, or ValueError."""
        out = config.default_config()
        out["feedback"] = cfg.get("feedback", out["feedback"])
        out["sound"] = bool(cfg.get("sound", out["sound"]))
        out["max_sections"] = int(cfg.get("max_sections", out["max_sections"]))
        if out["feedback"] not in config.FEEDBACK:
            raise ValueError(f"feedback must be one of {', '.join(config.FEEDBACK)}")
        for g in config.GESTURES:
            if g in cfg:
                out[g] = {"enabled": bool(cfg[g].get("enabled")),
                          "sections": [config.section(s.get("name") or f"Section {i + 1}", s["rect"],
                                                      s.get("command", ""), s.get("over"))
                                       for i, s in enumerate(cfg[g].get("sections", []))]}
            if bad := layout.problems(out, g):
                raise ValueError(f"{config.TITLES[g]}: {bad[0]}")
        return out

    def stop_training(self):
        """Stop the training run and its joblib workers (its own session)."""
        proc, job = self.training[:2]
        self.training = None
        if job == "gate":
            self.gate_due = time.time()         # still owed: run it after this one
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait()

    def train(self, g):
        if self.training:
            self.stop_training()
        path = os.path.join(DATA, f"train-{g}.log")
        proc = subprocess.Popen(train_command(g), stdout=open(path, "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.training = (proc, g, path, time.time())
        log(f"training {g} (log: {path})")

    def send_estimate(self, client):
        if self.estimate:
            P, where = self.estimate
            return client.send({"event": "estimate", "cells": layout.CELLS, "min_taps": layout.MIN_TAPS,
                                "P": [[round(float(v), 4) for v in row] for row in P],
                                "where": [[round(float(v), 4) for v in w] for w in where]})
        if not self.preparing:
            self.preparing = True
            threading.Thread(target=self.prepare_estimate, daemon=True).start()
        self.broadcast(self.state())

    def prepare_estimate(self):
        try:
            self.estimate = layout.estimate_data(log)
        finally:
            self.preparing = False
        self.q.put(("estimate_ready",))

    def on_estimate_ready(self):
        for c in list(self.clients):
            if c.editing:
                self.send_estimate(c)
        self.broadcast(self.state())

    # --------------------------------------------------------- socket

    def listen(self):
        if not SOCKET:
            sys.exit("tapzones needs $XDG_RUNTIME_DIR: a directory only you can open (systemd's user session sets it)")
        if os.path.exists(SOCKET):
            try:
                socket.socket(socket.AF_UNIX).connect(SOCKET)
                sys.exit(f"tapzones is already running ({SOCKET})")
            except OSError:
                os.remove(SOCKET)                # left over from a crash
        self.server = socket.socket(socket.AF_UNIX)
        old = os.umask(0o077)                    # owner-only from the moment it exists
        try:
            self.server.bind(SOCKET)
        finally:
            os.umask(old)
        self.server.listen()
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            conn, _ = self.server.accept()
            c = Client(conn)
            self.clients.append(c)
            threading.Thread(target=self.read, args=(c,), daemon=True).start()
            self.q.put(("cmd", c, {"cmd": "status"}))

    def read(self, c):
        buf = b""
        try:
            while chunk := c.conn.recv(65536):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            self.q.put(("cmd", c, json.loads(line)))
                        except json.JSONDecodeError:
                            c.send({"event": "error", "message": "not JSON"})
        except OSError:
            pass
        self.q.put(("gone", c))

    def on_gone(self, c):
        if c in self.clients:
            self.clients.remove(c)
            c.conn.close()
            if self.recording and self.recording["client"] is c:
                self.stop_recording()          # the panel that started it went away
            if c.editing or not self.recording:
                self.broadcast(self.state())

    def broadcast(self, obj):
        for c in list(self.clients):
            try:
                c.send(obj)
            except OSError:
                self.on_gone(c)


def train_command(job):
    """Trains a section model, or the gate, in a separate process (it takes
    a minute or more): the same Python, so the same install."""
    what = ["gate"] if job == "gate" else ["sections", job]
    return [sys.executable, "-W", "ignore", "-m", "tapzones", "train", *what]


def training_progress(log_path):
    """(fraction done, stage) from a training log's newest `progress` line
    (train.progress), or (0, "starting") before the first."""
    try:
        with open(log_path) as f:
            lines = [l for l in f.read().splitlines() if l.startswith("progress ")]
    except OSError:
        lines = []
    if not lines:
        return 0.0, "starting"
    _, frac, stage = lines[-1].split(" ", 2)
    return float(frac), stage


def recordings():
    """The recorded tap sessions (not confirmed.npz, which only trains a
    layout the next time it trains), with their change times."""
    return tuple((p, os.path.getmtime(p)) for p in session_paths() if p != CONFIRMED)


def calibration():
    """How much this device has been calibrated: recorded taps, sittings, and
    when the newest was recorded."""
    paths = [p for p, _ in recordings()]
    taps = 0
    for p in paths:
        try:
            with np.load(p) as d:
                taps += len(d["truth"] if "truth" in d else d["labels"])
        except (OSError, KeyError, ValueError):
            pass
    return {"taps": taps, "sittings": len(paths), "last": max((os.path.getmtime(p) for p in paths), default=None)}


def marked_count(path=None):
    """How many taps MARKED (or CONFIRMED) holds."""
    try:
        with np.load(path or MARKED) as d:
            return len(d["at"])
    except (OSError, KeyError):
        return 0


def save_noise(path, windows, **extra):
    """Windows the gate should learn aren't taps, as noise-*.npz (train.noise_set),
    or confirmed taps (CONFIRMED). Written aside first, under a name that glob
    doesn't pick up half-written."""
    os.makedirs(DATA, exist_ok=True)
    tmp = os.path.join(DATA, ".saving-" + os.path.basename(path))
    np.savez(tmp, windows=np.array(windows, np.float32), **extra)
    os.replace(tmp, path)


def run():
    engine = Engine()
    try:
        return engine.run()
    finally:
        if getattr(engine, "server", None) and os.path.exists(SOCKET):
            os.remove(SOCKET)            # only a socket this process bound
