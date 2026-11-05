#!/usr/bin/env python3
"""The flash feedback for desktops without the Omarchy plugin: a GTK
front-end of the service (engine.py), `taps.py flash`.

Connects to the service's socket and, with feedback = "flash", briefly draws
the section a tap landed in, or "not sure: tap again". Reconnects when the
service restarts. On Omarchy the plugin (omarchy/Service.qml) does this
instead, in the theme's colours.
"""
import json, socket, threading, time
import taps                       # restarts the process with layer-shell preloaded, once
from taps import Gtk, GLib, Gdk, Layer, cairo, ACCENT, FG
import engine

FLASH_S = 0.45                    # how long a flash stays on screen


class Flash:
    """A click-through layer over the whole screen, mapped only while a
    flash plays: the section that fired, or "not sure"."""
    def __init__(self, app):
        app.hold()                # no window is open most of the time: keep running anyway
        self.win = Gtk.ApplicationWindow(application=app)
        Layer.init_for_window(self.win)
        Layer.set_layer(self.win, Layer.Layer.OVERLAY)
        Layer.set_namespace(self.win, "tapzones-flash")
        for edge in (Layer.Edge.TOP, Layer.Edge.BOTTOM, Layer.Edge.LEFT, Layer.Edge.RIGHT):
            Layer.set_anchor(self.win, edge, True)
        Layer.set_exclusive_zone(self.win, -1)
        Layer.set_keyboard_mode(self.win, Layer.KeyboardMode.NONE)
        css = Gtk.CssProvider()
        css.load_from_data(b"window { background: transparent; }")
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, 800)
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self.draw)
        self.win.set_child(self.area)
        # Clicks go straight through to whatever is underneath.
        self.win.connect("map", lambda w: w.get_surface().set_input_region(cairo.Region()))
        self.what, self.since, self.ticking = None, 0.0, False
        threading.Thread(target=self.listen, daemon=True).start()

    def listen(self):
        """Events from the service, forever: reconnects every 2 s while it's down."""
        while True:
            try:
                s = socket.socket(socket.AF_UNIX)
                s.connect(engine.SOCKET)
                f = s.makefile("rw")
                f.write(json.dumps({"cmd": "hello", "ui": True}) + "\n")
                f.flush()
                for line in f:
                    GLib.idle_add(self.heard, json.loads(line))
            except OSError:
                pass
            time.sleep(2)

    def heard(self, msg):
        if msg.get("feedback") != "flash":
            return
        if msg["event"] == "tap":
            self.show(msg["rect"], msg["name"])
        elif msg["event"] == "unsure":
            self.show(None, "not sure: tap again")

    def show(self, rect, label):
        """rect: the section (screen fractions), or None for "not sure"."""
        self.what, self.since = (rect, label), time.time()
        self.win.set_visible(True)
        if not self.ticking:
            self.ticking = True
            GLib.timeout_add(16, self.tick)

    def tick(self):
        if time.time() - self.since > FLASH_S:
            self.win.set_visible(False)
            self.ticking = False
            return False
        self.area.queue_draw()
        return True

    def draw(self, _area, cr, w, h):
        if not self.what:
            return
        rect, label = self.what
        a = max(0.0, 1 - (time.time() - self.since) / FLASH_S)
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        if rect is None:
            taps.App.text(cr, label, w / 2, h / 2, 18, FG, 0.8 * a, center=True)
            return
        x, y, rw, rh = rect[0] * w, rect[1] * h, rect[2] * w, rect[3] * h
        taps.App.rounded(cr, x + 6, y + 6, rw - 12, rh - 12, 14)
        cr.set_source_rgba(*ACCENT, 0.16 * a)
        cr.fill_preserve()
        cr.set_source_rgba(*ACCENT, 0.85 * a)
        cr.set_line_width(3)
        cr.stroke()
        taps.App.text(cr, label, x + rw / 2, y + rh / 2, 22, FG, a, center=True)


def main():
    app = Gtk.Application(application_id="tapzones.flash")
    holder = []
    app.connect("activate", lambda a: holder.append(Flash(a)))
    app.run([])


if __name__ == "__main__":
    main()
