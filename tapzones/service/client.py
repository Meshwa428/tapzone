"""`tapzones status | pause | resume | toggle`: one command to the running
service, over its socket (engine.py has the protocol)."""
import json, socket, sys
from ..paths import SOCKET


def connect():
    """A socket connected to the service. OSError when it isn't running,
    or when there is no safe place for the socket (paths.runtime_dir)."""
    if not SOCKET:
        raise OSError("$XDG_RUNTIME_DIR is unset or not private to you")
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(SOCKET)
    except OSError:
        s.close()
        raise
    return s


def send(cmd):
    if not SOCKET:
        sys.exit("tapzones needs $XDG_RUNTIME_DIR: a directory only you can open (your systemd user session sets it)")
    try:
        f = connect().makefile("rw")
    except OSError as e:
        sys.exit(f"tapzones isn't running ({SOCKET or e}). Start it: systemctl --user start tapzones")
    f.write(json.dumps({"cmd": cmd}) + "\n")
    f.flush()
    for line in f:
        msg = json.loads(line)
        if msg.get("event") == "state":
            m = msg["model"]
            print(json.dumps({k: msg[k] for k in ("paused", "editing", "gate", "training", "estimate")} |
                             {"model": {k: v for k, v in m.items() if k != "per_section"}}, indent=2))
            return 0
