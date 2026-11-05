#!/usr/bin/env python3
"""The tapzones service (docs/Gestures/Gesture system.md), with no UI: it
listens to the mic and runs the command of the section a gesture lands in.

    taps.py run                 the service (or the systemd unit tapzones.service)
    taps.py status              what it's doing, as JSON
    taps.py pause | resume | toggle

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
    ← {"event": "state", ...} · "tap" · "unsure" · "ignored" · "saved"
      · "trained" · "estimate" · "error"

The Omarchy plugin (manifest.json, omarchy/) and the GTK flash (flash.py)
are such front-ends. With feedback = "flash" and no UI connected, the
service sends a notification instead, so feedback never silently vanishes.
"""
import json, os, queue, socket, subprocess, sys, threading, time
import gestures, mic

HERE = os.path.dirname(os.path.abspath(__file__))
SOCKET = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/tapzones-{os.getuid()}", "tapzones.sock")
STATE = os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "tapzones")
PAUSED = os.path.join(STATE, "paused")       # present while paused; survives restarts


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
        self.training = None                   # (Popen, gesture, log path, started)
        self.estimate = None                   # (P, where) once prepared
        self.preparing = False
        self.paused = os.path.exists(PAUSED)
        self.warned = None
        self.gate = gestures.Gate()
        self.reload()

    # ------------------------------------------------ config and model

    def stamp(self):
        return tuple(os.path.getmtime(p) if os.path.exists(p) else 0
                     for p in (gestures.CONFIG, gestures.model_path("single")))

    def reload(self):
        self.seen = self.stamp()
        self.cfg = gestures.load()
        single = self.cfg["single"]
        self.sections = single["sections"] if single["enabled"] else []
        self.model = gestures.SectionModel("single", [s["rect"] for s in self.sections]) if self.sections else None
        if self.model and not self.model.ok:
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
                  "answered_accuracy": m.d["answered_accuracy"], "cutoff": m.cutoff,
                  "per_section": [None if a != a else float(a) for a in m.d.get("per_section", [])]} if m and m.ok
                 else {"ok": False, "why": m.why if m else "single taps are off"})
        return {"event": "state", "paused": self.paused, "editing": self.quiet(),
                "gate": bool(self.gate.d), "config": self.cfg, "model": model,
                "training": {"gesture": self.training[1], "since": self.training[3]} if self.training else None,
                "estimate": "ready" if self.estimate else "preparing" if self.preparing else "none",
                "detected": sorted(gestures.DETECTED), "socket": SOCKET}

    # ----------------------------------------------------------- loop

    def run(self):
        self.listen()
        self.mic = mic.Mic(lambda w: None, lambda w, t: self.q.put(("tap", w, t)))
        self.mic.start()
        log(f"listening on {mic.SOURCE}: gate {'on' if self.gate.d else 'off'}, "
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
        if self.training and self.training[0].poll() is not None:
            proc, g, path, _ = self.training
            self.training = None
            lines = open(path).read().strip().splitlines()
            ok = proc.returncode == 0
            msg = next((l for l in reversed(lines) if l.startswith("saved ")), "") if ok else (lines[-1] if lines else "?")
            log(f"training {g}: {'done' if ok else 'failed'}: {msg}")
            self.reload()
            self.broadcast({"event": "trained", "gesture": g, "ok": ok, "message": msg})
            self.broadcast(self.state())

    # ----------------------------------------------------------- taps

    def quiet(self):
        return mic.overlay_open() or any(c.editing for c in self.clients)

    def on_tap(self, window, heard_at):
        if self.paused or self.quiet() or not self.model or not self.model.ok:
            return
        if not self.gate.is_tap(window):
            log("not a tap: ignored")
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
            self.broadcast({"event": "unsure", "gesture": "single", "best": i, "name": s["name"],
                            "conf": conf, "ms": ms, "feedback": self.cfg["feedback"]})
            return
        log(f"{s['name']} ({conf:.0%}), {ms:.0f} ms: {s['command'] or '(no command)'}")
        if s["command"].strip():
            subprocess.Popen(["sh", "-c", s["command"]], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fb = self.cfg["feedback"]
        self.broadcast({"event": "tap", "gesture": "single", "index": i, "name": s["name"], "rect": s["rect"],
                        "command": s["command"], "conf": conf, "ms": ms, "feedback": fb})
        if fb == "notify" or (fb == "flash" and not any(c.ui for c in self.clients)):
            self.notify(s["name"], s["command"] or "single tap")

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

    def save(self, client, cfg):
        try:
            cfg = self.checked(cfg)
        except ValueError as e:
            return client.send({"event": "error", "message": f"not saved: {e}"})
        gestures.save(cfg)
        self.reload()
        self.broadcast({"event": "saved", "path": gestures.CONFIG})
        if cfg["single"]["enabled"] and not (self.model and self.model.ok):
            self.train("single")
        self.broadcast(self.state())

    @staticmethod
    def checked(cfg):
        """The config a client sent, in gestures' shape, or ValueError."""
        out = gestures.default_config()
        out["feedback"] = cfg.get("feedback", out["feedback"])
        out["max_sections"] = int(cfg.get("max_sections", out["max_sections"]))
        if out["feedback"] not in gestures.FEEDBACK:
            raise ValueError(f"feedback must be one of {', '.join(gestures.FEEDBACK)}")
        for g in gestures.GESTURES:
            if g in cfg:
                out[g] = {"enabled": bool(cfg[g].get("enabled")),
                          "sections": [{"name": str(s.get("name") or f"Section {i + 1}"),
                                        "rect": [float(v) for v in s["rect"]][:4],
                                        "command": str(s.get("command", ""))}
                                       for i, s in enumerate(cfg[g].get("sections", []))]}
            if bad := gestures.problems(out, g):
                raise ValueError(f"{gestures.TITLES[g]}: {bad[0]}")
        return out

    def train(self, g):
        if self.training:
            self.training[0].kill()
        path = os.path.join(gestures.DATA, f"train-{g}.log")
        proc = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(HERE, "ml.py"), "--export-sections", g],
                                stdout=open(path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        self.training = (proc, g, path, time.time())
        log(f"training {g} (log: {path})")

    def send_estimate(self, client):
        if self.estimate:
            P, where = self.estimate
            return client.send({"event": "estimate", "cells": gestures.CELLS, "min_taps": gestures.MIN_TAPS,
                                "P": [[round(float(v), 4) for v in row] for row in P],
                                "where": [[round(float(v), 4) for v in w] for w in where]})
        if not self.preparing:
            self.preparing = True
            threading.Thread(target=self.prepare_estimate, daemon=True).start()
        self.broadcast(self.state())

    def prepare_estimate(self):
        try:
            self.estimate = gestures.estimate_data(log)
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
        os.makedirs(os.path.dirname(SOCKET), exist_ok=True)
        if os.path.exists(SOCKET):
            try:
                socket.socket(socket.AF_UNIX).connect(SOCKET)
                sys.exit(f"tapzones is already running ({SOCKET})")
            except OSError:
                os.remove(SOCKET)                # left over from a crash
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(SOCKET)
        os.chmod(SOCKET, 0o600)
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
            if c.editing:
                self.broadcast(self.state())

    def broadcast(self, obj):
        for c in list(self.clients):
            try:
                c.send(obj)
            except OSError:
                self.on_gone(c)


def client(cmd):
    """`taps.py status|pause|resume|toggle`: one command to a running service."""
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(SOCKET)
    except OSError:
        sys.exit(f"tapzones isn't running (no {SOCKET}). Start it: systemctl --user start tapzones")
    f = s.makefile("rw")
    f.write(json.dumps({"cmd": cmd}) + "\n")
    f.flush()
    for line in f:
        msg = json.loads(line)
        if msg.get("event") == "state":
            m = msg["model"]
            print(json.dumps({k: msg[k] for k in ("paused", "editing", "gate", "training", "estimate")} |
                             {"model": {k: v for k, v in m.items() if k != "per_section"}}, indent=2))
            return 0


def main(argv=()):
    import argparse
    p = argparse.ArgumentParser(prog="taps.py", description="the tapzones service")
    p.add_argument("cmd", choices=["run", "status", "pause", "resume", "toggle"])
    a = p.parse_args(argv)
    if a.cmd != "run":
        return client(a.cmd)
    engine = Engine()
    try:
        return engine.run()
    finally:
        if getattr(engine, "server", None) and os.path.exists(SOCKET):
            os.remove(SOCKET)            # only a socket this process bound


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
