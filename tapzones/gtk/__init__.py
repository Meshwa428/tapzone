"""The GTK front-ends (overlay, layout editor, flash) and what they share:
GTK 4 with layer shell, the Omarchy theme's colours, and drawing helpers.

Importing this restarts the process once with gtk4-layer-shell preloaded:
layer shell only works when loaded before libwayland."""
import os, sys

LAYER_LIB = "/usr/lib/libgtk4-layer-shell.so"
if LAYER_LIB not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = LAYER_LIB
    os.execv(sys.executable, sys.orig_argv)    # the same command line: `python -m tapzones …` or the script

import numpy as np
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gtk, GLib, Gdk, Gtk4LayerShell as Layer  # noqa: F401  (the front-ends import these from here)
import cairo  # noqa: F401

ZONE_RGB = [(0.88, 0.42, 0.46), (0.90, 0.75, 0.48), (0.60, 0.76, 0.47), (0.38, 0.69, 0.94),
            (0.77, 0.61, 0.91), (0.44, 0.82, 0.77)]


def theme():
    """Background, foreground and accent of the current Omarchy theme."""
    c = {"background": "#0c0b0c", "foreground": "#fafcfb", "accent": "#b59790"}
    try:
        for line in open(os.path.expanduser("~/.local/state/omarchy/current/theme/colors.toml")):
            k, _, v = line.partition("=")
            if k.strip() in c:
                c[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    rgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return rgb(c["background"]), rgb(c["foreground"]), rgb(c["accent"])

BG, FG, ACCENT = theme()


def text(cr, s, x, y, size, rgb, alpha=1.0, right=False, center=False):
    cr.set_font_size(size)
    ext = cr.text_extents(s)
    if right:
        x -= ext.x_advance
    elif center:
        x -= ext.x_advance / 2
    cr.set_source_rgba(*rgb, alpha)
    cr.move_to(x, y)
    cr.show_text(s)


def rounded(cr, x, y, w, h, r):
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -np.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, np.pi / 2)
    cr.arc(x + r, y + h - r, r, np.pi / 2, np.pi)
    cr.arc(x + r, y + r, r, np.pi, 3 * np.pi / 2)
    cr.close_path()
