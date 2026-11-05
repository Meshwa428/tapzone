"""Where tapzones keeps things: the config, the recordings and models, the
pause flag, and the service's socket."""
import os, stat

CONFIG = os.path.expanduser("~/.config/tapzones/gestures.toml")
DATA = os.path.expanduser("~/.local/share/tapzones")
STATE = os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "tapzones")
PAUSED = os.path.join(STATE, "paused")       # present while paused; survives restarts

# While the overlay is open the background service stays quiet, so
# calibrating, testing or editing a layout never runs commands.
LOCK = os.path.join(DATA, "overlay.pid")


def runtime_dir():
    """$XDG_RUNTIME_DIR if it's safe to put the socket in: a real directory
    (not a symlink) that only this user can get into. Anyone who can reach
    the socket can save section commands, which run as this user, so there
    is no fallback to a shared place like /tmp. None when unsafe or unset."""
    d = os.environ.get("XDG_RUNTIME_DIR")
    if not d:
        return None
    try:
        st = os.lstat(d)
    except OSError:
        return None
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        return None
    return d


SOCKET = os.path.join(runtime_dir(), "tapzones.sock") if runtime_dir() else ""
