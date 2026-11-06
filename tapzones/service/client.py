"""`tapzones status | pause | resume | toggle | mistake | confirm SECTION`:
one command to the running service, over its socket (engine.py has the
protocol)."""
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


def section_index(sections, which):
    """The index of the section named `which` (any case), or numbered (from 1)."""
    names = [s["name"].lower() for s in sections]
    if which.lower() in names:
        return names.index(which.lower())
    if which.isdigit() and 1 <= int(which) <= len(sections):
        return int(which) - 1
    sys.exit(f"no section {which!r}: " + ", ".join(f"{i + 1} {s['name']}" for i, s in enumerate(sections)))


def send(cmd, section=None):
    if not SOCKET:
        sys.exit("tapzones needs $XDG_RUNTIME_DIR: a directory only you can open (your systemd user session sets it)")
    try:
        f = connect().makefile("rw")
    except OSError as e:
        sys.exit(f"tapzones isn't running ({SOCKET or e}). Start it: systemctl --user start tapzones")
    # The service greets every connection with its state, before it reads
    # the command: a command that changes the state answers with a second one.
    # confirm reads the greeting first, for the section names.
    if cmd == "confirm":
        greeting = json.loads(f.readline())
        f.write(json.dumps({"cmd": cmd, "section": section_index(greeting["config"]["single"]["sections"], section)}) + "\n")
    else:
        f.write(json.dumps({"cmd": cmd}) + "\n")
    f.flush()
    states = 0
    for line in f:
        msg = json.loads(line)
        event = msg.get("event")
        if event == "error":
            print(msg["message"], file=sys.stderr)
            return 1
        if event == "marked":
            print(f"marked {msg['name']!r} from {msg['ago']:.0f} s ago as not a tap "
                  f"({msg['total']} marked in all); the noise filter retrains shortly")
            return 0
        if event == "confirmed":
            what = f"the last tap (it ran {msg['was']!r})" if msg.get("was") else "the last miss"
            print(f"kept {what} as a tap in {msg['name']!r} and ran it ({msg['total']} corrected in all)")
            return 0
        if event == "state" and cmd not in ("mistake", "confirm"):
            states += 1
            if cmd == "status" or states == 2:
                m = msg["model"]
                print(json.dumps({k: msg[k] for k in ("paused", "editing", "gate", "noise", "training", "estimate")} |
                                 {"model": {k: v for k, v in m.items() if k != "per_section"}}, indent=2))
                return 0
