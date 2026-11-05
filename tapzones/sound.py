"""Short sounds for what the service does: a tap, "tap again", pause… From
SND01 "sine" by Yasuhiro Tsuchiya (https://snd.dev, sounds/CREDITS.md).

Played by whichever player the system has (PipeWire, PulseAudio or ALSA),
without waiting. The laptop's own mic doesn't take them for taps: 28 plays
caught nothing (2026-11-06)."""
import os, random, shutil, subprocess

HERE = os.path.join(os.path.dirname(__file__), "sounds")

# What each moment plays. "tap" has five takes of the same sound, played at
# random so a sound heard on every tap doesn't grate (as SND intends).
SOUNDS = {
    "tap": ["tap_01", "tap_02", "tap_03", "tap_04", "tap_05"],
    "marked": ["select"],           # "that wasn't a tap" kept
    "nothing": ["disabled"],        # nothing to mark
    "resume": ["toggle_on"],
    "pause": ["toggle_off"],
    "trained": ["notification"],    # a layout finished training
    "failed": ["caution"],
}

_player = None


def player():
    """The command that plays a WAV file, or [] if there's none."""
    global _player
    if _player is None:
        _player = next(([p] for p in ("pw-play", "paplay", "aplay") if shutil.which(p)), [])
        if _player == ["aplay"]:
            _player = ["aplay", "-q"]
    return _player


def play(moment):
    path = os.path.join(HERE, random.choice(SOUNDS[moment]) + ".wav")
    if player() and os.path.exists(path):
        subprocess.Popen(player() + [path], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
