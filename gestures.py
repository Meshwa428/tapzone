"""Sections and layouts (docs/Gestures/Sections.md): a layout is a list of
rectangles on the screen, as fractions (x, y, width, height) from the
top-left, one per section. Parts of the screen no section covers are dead
zones: a tap there does nothing.

Also the config (~/.config/tapzones/gestures.toml, docs/Gestures/Configuration.md)
and the models the background service runs: the gate and a layout's
section model."""
import hashlib, os, tomllib
import numpy as np

CONFIG = os.path.expanduser("~/.config/tapzones/gestures.toml")
DATA = os.path.expanduser("~/.local/share/tapzones")
GESTURES = ["single", "double", "triple", "two_finger", "three_finger"]
TITLES = {"single": "Single tap", "double": "Double tap", "triple": "Triple tap",
          "two_finger": "2-finger tap", "three_finger": "3-finger tap"}
DETECTED = {"single"}          # gestures the service can hear so far (docs/Gestures/Roadmap.md)
FEEDBACK = ("flash", "notify", "none")
MIN_SIDE = 0.05                # a section smaller than this is below the mic's typical error
MIN_TAPS = 10                  # recorded taps a section needs before it can be trained


def rows(k):
    """k bands across the full width, top to bottom."""
    return [(0.0, i / k, 1.0, 1 / k) for i in range(k)]


def columns(k):
    """k bands down the full height, left to right."""
    return [(i / k, 0.0, 1 / k, 1.0) for i in range(k)]


def grid(cols, rows_):
    """cols × rows_ cells, row by row from the top-left."""
    return [(c / cols, r / rows_, 1 / cols, 1 / rows_) for r in range(rows_) for c in range(cols)]


# Quadrants in the order the quadrant model and the recorded `truth` use:
# top-left, top-right, bottom-left, bottom-right.
PRESETS = {
    "whole screen": rows(1),
    "top / bottom": rows(2),
    "left / right": columns(2),
    "3 rows": rows(3),
    "quadrants": grid(2, 2),
    "5 rows": rows(5),
    "3 columns": columns(3),
}


PRESET_NAMES = {
    "whole screen": ["Anywhere"],
    "top / bottom": ["Top", "Bottom"],
    "left / right": ["Left", "Right"],
    "3 rows": ["Top", "Middle", "Bottom"],
    "quadrants": ["Top left", "Top right", "Bottom left", "Bottom right"],
    "5 rows": [f"Row {i + 1}" for i in range(5)],
    "3 columns": ["Left", "Middle", "Right"],
}


def section_of(where, layout):
    """For points (n, 2) as screen fractions: the index of the section each
    falls in, or -1 for a dead zone."""
    where = np.clip(np.asarray(where, float), 0, 1 - 1e-9)
    out = np.full(len(where), -1)
    for i, (x, y, w, h) in enumerate(layout):
        inside = (where[:, 0] >= x) & (where[:, 0] < x + w) & (where[:, 1] >= y) & (where[:, 1] < y + h)
        out[inside & (out < 0)] = i
    return out


def layout_hash(layout):
    """Identifies a layout's geometry: a section model is only valid for
    the layout it was trained on. Names and commands don't count."""
    return hashlib.sha1(repr([tuple(round(v, 4) for v in r) for r in layout]).encode()).hexdigest()[:12]

# ------------------------------------------------------------------ config

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
                      "sections": [{"name": s.get("name", f"Section {i + 1}"), "rect": [float(v) for v in s["rect"]],
                                    "command": s.get("command", "")} for i, s in enumerate(raw[g].get("section", []))]}
    return cfg


def save(cfg, path=CONFIG):
    """Writes the config as TOML (the format in docs/Gestures/Configuration.md)."""
    q = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    out = ["# tapzones gestures: edit with `taps.py layout`, or by hand.",
           "# rect = [x, y, width, height] as fractions of the screen from the top-left.",
           f"feedback = {q(cfg['feedback'])}          # flash | notify | none",
           f"max_sections = {cfg['max_sections']}", ""]
    for g in GESTURES:
        out += [f"[{g}]", f"enabled = {'true' if cfg[g]['enabled'] else 'false'}"]
        for s in cfg[g]["sections"]:
            out += [f"[[{g}.section]]", f"name = {q(s['name'])}",
                    "rect = [" + ", ".join(f"{v:.4f}" for v in s["rect"]) + "]", f"command = {q(s['command'])}"]
        out.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(out))
    os.replace(path + ".tmp", path)


def problems(cfg, gesture):
    """What's wrong with a gesture's layout, as messages (empty: fine)."""
    g = cfg[gesture]
    rects = [s["rect"] for s in g["sections"]]
    out = []
    if g["enabled"] and not rects:
        out.append("add a section, or turn the gesture off")
    if len(rects) > cfg["max_sections"]:
        out.append(f"{len(rects)} sections: more than max_sections ({cfg['max_sections']})")
    for i, (x, y, w, h) in enumerate(rects):
        if x < -1e-6 or y < -1e-6 or x + w > 1 + 1e-6 or y + h > 1 + 1e-6:
            out.append(f"{g['sections'][i]['name']} goes off the screen")
        if w < MIN_SIDE or h < MIN_SIDE:
            out.append(f"{g['sections'][i]['name']} is smaller than {MIN_SIDE:.0%} of the screen")
        for j in range(i):
            a, b = rects[j], rects[i]
            if a[0] < b[0] + b[2] - 1e-6 and b[0] < a[0] + a[2] - 1e-6 and a[1] < b[1] + b[3] - 1e-6 and b[1] < a[1] + a[3] - 1e-6:
                out.append(f"{g['sections'][j]['name']} and {g['sections'][i]['name']} overlap")
    return out


def model_path(gesture):
    return os.path.join(DATA, f"sections-{gesture}.joblib")

# ------------------------------------------------------- runtime models

class Gate:
    """Tap or not (ml.py --export-gate). Without gate.joblib, every sound
    counts as a tap."""
    def __init__(self):
        import joblib, ml
        self.ml = ml
        path = os.path.join(DATA, "gate.joblib")
        self.d = joblib.load(path) if os.path.exists(path) else None   # our own file
        if self.d:
            self.is_tap(np.zeros((ml.EARLY_N, 2), np.float32))   # one-time setup now, not on the first tap

    def is_tap(self, window):
        if not self.d:
            return True
        score = self.d["model"].decision_function(self.ml.tap_features(window[None]))[0]
        return bool(score >= self.d["threshold"])


class SectionModel:
    """One gesture's section model (ml.py --export-sections GESTURE): which
    section a tap was in, from the early window. `ok` is False when there is
    no model, or it was trained for a different layout."""
    def __init__(self, gesture, layout):
        import joblib, ml
        self.ml, self.ok, self.why = ml, False, ""
        path = model_path(gesture)
        if not os.path.exists(path):
            self.why = "no model for this layout yet"
            return
        self.d = joblib.load(path)                                      # our own file
        if self.d["hash"] != layout_hash(layout):
            self.why = "the model is for a different layout"
            return
        self.ok, self.cutoff = True, self.d["cutoff"]
        self.proba(np.zeros((ml.EARLY_N, 2), np.float32))

    def proba(self, window):
        f = self.ml.signed(window[None], **self.ml.BEST1)
        return np.mean([m.predict_proba(f)[0] for m in self.d["models"]], axis=0)

# ------------------------------------------------------- the estimate

CELLS = 6    # the estimate's grid: CELLS × CELLS cells


def cell_probs(F, where, session):
    """Once per data set (cached by the editor): every tap's probability of
    being in each cell of a CELLS × CELLS grid, from a model that never saw
    that tap's session (logistic regression on best1). Any layout's section
    probabilities are then sums of cells, so its estimate is instant."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    cell = section_of(where, grid(CELLS, CELLS))
    P = np.zeros((len(F), CELLS * CELLS))
    for s in np.unique(session):
        tr, te = session != s, session == s
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(F[tr], cell[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(F[te])
    return P


def estimate_data(status=lambda msg: None):
    """(P, where) for quick_estimate: cell_probs over every recorded session,
    a few minutes the first time, then cached in DATA/estimate-cells.npz
    until the recordings change. `status` hears what's going on."""
    import ml
    paths = ml.session_paths()
    cache = os.path.join(DATA, "estimate-cells.npz")
    key = np.array([f"{os.path.basename(p)}:{os.path.getsize(p)}" for p in paths])
    if os.path.exists(cache):
        c = np.load(cache)
        if c["key"].shape == key.shape and (c["key"] == key).all():
            return c["P"], c["where"]
    status("preparing estimates from the recorded taps (once, a few minutes)…")
    W, where, sess = [], [], []
    for i, path in enumerate(paths):
        d = np.load(path)
        W.append(d["windows"].astype(np.float32))
        where.append(np.asarray(d["where"], float))
        sess.append(np.full(len(d["windows"]), i))
    where = np.concatenate(where)
    P = cell_probs(ml.signed(np.concatenate(W), **ml.BEST1), where, np.concatenate(sess))
    np.savez(cache, P=P, where=where, key=key)
    return P, where


def quick_estimate(P, where, layout):
    """The editor's live estimate (docs/Gestures/Sections.md), from
    cell_probs: each section's probability is the sum of the cells whose
    centre it covers. Returns (overall, [(accuracy, taps) per section]);
    accuracy is nan for a section with fewer than MIN_TAPS recorded taps."""
    y = section_of(where, layout)
    keep = y >= 0
    per_n = [int((y == i).sum()) for i in range(len(layout))]
    if len(layout) < 2:
        return 1.0, [(1.0, int(keep.sum()))]
    centres = np.array([(x + w / 2, y_ + h / 2) for x, y_, w, h in grid(CELLS, CELLS)])
    owner = section_of(centres, layout)
    M = np.zeros((CELLS * CELLS, len(layout)))
    M[owner >= 0, owner[owner >= 0]] = 1
    pred = (P @ M).argmax(1)
    ok = pred == y
    per = [(ok[y == i].mean() if n >= MIN_TAPS else np.nan, n) for i, n in enumerate(per_n)]
    return (ok[keep].mean() if keep.any() else np.nan), per


def shrink(layout, gap):
    """The layout with a dead zone of width `gap` (screen fraction) along every
    border between two sections. Screen edges keep no margin."""
    out = []
    for x, y, w, h in layout:
        left = gap / 2 if x > 0 else 0
        top = gap / 2 if y > 0 else 0
        right = gap / 2 if x + w < 1 - 1e-9 else 0
        bottom = gap / 2 if y + h < 1 - 1e-9 else 0
        out.append((x + left, y + top, w - left - right, h - top - bottom))
    return out
