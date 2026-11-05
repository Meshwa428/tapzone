"""The config: ~/.config/tapzones/gestures.toml (docs/Gestures/Configuration.md).
Per gesture, whether it's on and its sections: a name, a rectangle (layout.py)
and a shell command."""
import os, tomllib
from .paths import CONFIG
from .layout import PRESETS

GESTURES = ["single", "double", "triple", "two_finger", "three_finger"]
TITLES = {"single": "Single tap", "double": "Double tap", "triple": "Triple tap",
          "two_finger": "2-finger tap", "three_finger": "3-finger tap"}
DETECTED = {"single"}          # gestures the service can hear so far (docs/Gestures/Roadmap.md)
FEEDBACK = ("flash", "notify", "none")


def default_config():
    """Today's behaviour: quadrants for single taps, no commands yet. The
    other gestures start off, with the defaults from docs/Gestures."""
    sections = lambda layout, names: [{"name": n, "rect": list(r), "command": ""} for n, r in zip(names, layout)]
    return {
        "feedback": "flash",
        "max_sections": 5,
        "single": {"enabled": True, "sections": sections(PRESETS["quadrants"],
                                                         ["Top left", "Top right", "Bottom left", "Bottom right"])},
        "double": {"enabled": False, "sections": sections(PRESETS["top / bottom"], ["Top", "Bottom"])},
        "triple": {"enabled": False, "sections": sections(PRESETS["whole screen"], ["Anywhere"])},
        "two_finger": {"enabled": False, "sections": sections(PRESETS["whole screen"], ["Anywhere"])},
        "three_finger": {"enabled": False, "sections": sections(PRESETS["whole screen"], ["Anywhere"])},
    }


def load(path=CONFIG):
    """The config, or the defaults where the file has nothing."""
    cfg = default_config()
    if not os.path.exists(path):
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg["feedback"] = raw.get("feedback", cfg["feedback"])
    cfg["max_sections"] = int(raw.get("max_sections", cfg["max_sections"]))
    for g in GESTURES:
        if g in raw:
            cfg[g] = {"enabled": bool(raw[g].get("enabled", False)),
                      "sections": [section(s.get("name", f"Section {i + 1}"), s["rect"], s.get("command", ""), s.get("over"))
                                   for i, s in enumerate(raw[g].get("section", []))]}
    return cfg


def section(name, rect, command, over=False):
    """A section as the config holds it; `over` (it sits over the others,
    layout.py) is only there when it's set."""
    s = {"name": str(name), "rect": [float(v) for v in rect][:4], "command": str(command)}
    if over:
        s["over"] = True
    return s


def save(cfg, path=CONFIG):
    """Writes the config as TOML (the format in docs/Gestures/Configuration.md)."""
    q = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    out = ["# tapzones gestures: edit with `tapzones layout`, or by hand.",
           "# rect = [x, y, width, height] as fractions of the screen from the top-left.",
           f"feedback = {q(cfg['feedback'])}          # flash | notify | none",
           f"max_sections = {cfg['max_sections']}", ""]
    for g in GESTURES:
        out += [f"[{g}]", f"enabled = {'true' if cfg[g]['enabled'] else 'false'}"]
        for s in cfg[g]["sections"]:
            out += [f"[[{g}.section]]", f"name = {q(s['name'])}",
                    "rect = [" + ", ".join(f"{v:.4f}" for v in s["rect"]) + "]", f"command = {q(s['command'])}"]
            if s.get("over"):
                out.append("over = true          # sits over the others and takes the area it covers")
        out.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(out))
    os.replace(path + ".tmp", path)
