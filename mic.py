"""The microphone side of tapzones, with no UI: the tap detector the overlay
(taps.py) and the headless service (engine.py) share.

TAPZONES_SOURCE picks the PulseAudio/PipeWire source to listen to; the
default is this laptop's built-in mic, which every recorded tap came from.
"""
import collections, os, subprocess, threading, time
import numpy as np

SOURCE = os.environ.get("TAPZONES_SOURCE", "alsa_input.pci-0000_00_1f.3.analog-stereo")
SR = 48000
HOP = 240                      # 5 ms analysis step
PRE_HOPS, POST_HOPS = 4, 30    # keep 20 ms before a tap and 150 ms after
EARLY_HOPS = 9                 # the final model answers 45 ms after the hit: all it needs (docs/latency.md)
DATA = os.path.expanduser("~/.local/share/tapzones")

# While the overlay is open the background service stays quiet, so
# calibrating, testing or editing a layout never runs commands.
LOCK = os.path.join(DATA, "overlay.pid")


def hold_overlay_lock():
    os.makedirs(DATA, exist_ok=True)
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.exists(LOCK) and open(LOCK).read() == str(os.getpid()) and os.remove(LOCK))


def overlay_open():
    try:
        os.kill(int(open(LOCK).read()), 0)
        return True
    except (OSError, ValueError):
        return False


class Mic(threading.Thread):
    """Reads the mic 5 ms at a time, keeps 2 s of levels for the scope, and
    hands every tap (a sudden jump in loudness) to `on_tap` with the full
    window. With `on_early`, the first 45 ms after the hit also go there
    straight away (with the moment the tap was heard), so a model can
    answer without waiting for the rest."""
    def __init__(self, on_tap, on_early=None):
        super().__init__(daemon=True)
        self.on_tap = on_tap
        self.on_early = on_early
        self.heard_at = 0.0
        self.levels = collections.deque([1e-4] * 400, maxlen=400)   # rms per hop
        self.marks = collections.deque(maxlen=20)                    # hop counter of each tap
        self.hops = collections.deque(maxlen=PRE_HOPS + 1 + POST_HOPS)
        self.count = 0
        self.threshold = 0.0
        self.mute_until = 0.0
        self.tape = None    # an open file: when set, every sample goes to it (16-bit mono)
        self.proc = subprocess.Popen(
            ["parec", "-d", SOURCE, "--format=float32le", f"--rate={SR}", "--channels=2", "--latency-msec=10"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self):
        size = HOP * 2 * 4
        pending, last_tap = -1, -10**9
        while True:
            raw = self.proc.stdout.read(size)
            if not raw or len(raw) < size:
                return
            x = np.frombuffer(raw, np.float32).reshape(-1, 2)
            self.hops.append(x)
            if self.tape:
                self.tape.write((np.clip(x.mean(axis=1), -1, 1) * 32767).astype("<i2").tobytes())
            rms = float(np.sqrt((x.mean(1) ** 2).mean()))
            history = list(self.levels)
            floor = float(np.median(history[-200:]))
            before = float(np.mean(history[-6:-2]))
            self.levels.append(rms)
            self.count += 1
            self.threshold = max(6 * floor, 0.008)
            if pending > 0:
                pending -= 1
                early = PRE_HOPS + 1 + EARLY_HOPS
                if self.on_early and pending == POST_HOPS - EARLY_HOPS and len(self.hops) >= early:
                    self.on_early(np.concatenate(list(self.hops)[-early:]), self.heard_at)
                # Every tap the same length: skip one heard before the
                # buffer filled.
                if pending == 0 and len(self.hops) == self.hops.maxlen:
                    self.on_tap(np.concatenate(self.hops))
                continue
            if (rms > self.threshold and rms > 4 * before and self.count - last_tap > 60
                    and time.time() > self.mute_until):
                last_tap = self.count
                self.marks.append(self.count)
                self.heard_at = time.perf_counter()
                pending = POST_HOPS

    def stop(self):
        self.proc.kill()
