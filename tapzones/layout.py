"""Sections and layouts (docs/Gestures/Sections.md): a layout is a list of
rectangles on the screen, as fractions (x, y, width, height) from the
top-left, one per section. Parts of the screen no section covers are dead
zones: a tap there does nothing. Sections don't overlap, except one marked
`over` (listed first): it sits over the others and takes the area it
covers, since a tap belongs to the first section it falls in.

Also the editor's live estimate of how well the mic tells a layout's
sections apart. omarchy/Layout.js mirrors the geometry for the plugin's
editor (tests/test_layout.py keeps the two in step)."""
import hashlib, itertools, os, re
import numpy as np
from .paths import DATA

MIN_SIDE = 0.05               # a section smaller than this is below the mic's typical error
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


# ------------------------------------------------------- the presets
#
# Layouts come only from presets (the editors can't draw sections). For 1–5
# sections the presets are every layout straight cuts can make: cut the
# screen into equal rows or columns, then cut any band the other way, and so
# on. Written "1 / 2" (rows, top to bottom: one section over two side by
# side) or "1 | 2" (columns, left to right: one beside two stacked). Plus
# corners + centre (a section over the middle of the quadrants) and a few of 6.

COUNTS = (1, 2, 3, 4, 5, 6)


def compositions(n):
    """Ordered ways to write n as a sum of two or more positive parts."""
    for first in range(1, n):
        yield (first, n - first)
        for tail in compositions(n - first):
            yield (first,) + tail


def cuts(n, box, across):
    """(rects, label) for every layout of n sections filling box whose first
    cut is into rows (across) or columns, each later cut the other way."""
    x, y, w, h = box
    for parts in compositions(n):
        k = len(parts)
        bands = [(x, y + i * h / k, w, h / k) if across else (x + i * w / k, y, w / k, h) for i in range(k)]
        options = [list(fill(p, b, not across)) for p, b in zip(parts, bands)]
        for combo in itertools.product(*options):
            yield [r for c in combo for r in c[0]], (" / " if across else " | ").join(c[1] for c in combo)


def fill(n, box, across):
    """Like cuts, for a band inside a layout: its label is 1, a plain count
    when it's only cut once, or the cuts in brackets."""
    if n == 1:
        yield [box], "1"
        return
    for rects, label in cuts(n, box, across):
        yield rects, str(n) if label == (" / " if across else " | ").join(["1"] * n) else f"({label})"


def reading_order(rects):
    return sorted(rects, key=lambda r: (round(r[1], 6), round(r[0], 6)))


def section_names(rects):
    """Names from where each section is: Top / Middle / Bottom and Left /
    Centre / Right, a word left out when it spans the screen that way."""
    if all(w > 1 - 1e-9 for _, _, w, _ in rects) and len(rects) > 3:
        return [f"Row {i + 1}" for i in range(len(rects))]
    if all(h > 1 - 1e-9 for _, _, _, h in rects) and len(rects) > 3:
        return [f"Column {i + 1}" for i in range(len(rects))]
    word = lambda c, low, mid, high: low if c < 1 / 3 - 1e-9 else high if c > 2 / 3 + 1e-9 else mid
    names = []
    for x, y, w, h in rects:
        v = "" if h > 1 - 1e-9 else word(y + h / 2, "Top", "Middle", "Bottom")
        s = "" if w > 1 - 1e-9 else word(x + w / 2, "Left", "Centre", "Right")
        names.append(f"{v} {s.lower()}" if v and s else v or s or "Anywhere")
    for name in sorted(set(names)):                # the same place twice: number them
        same = [i for i, m in enumerate(names) if m == name]
        if len(same) > 1:
            for j, i in enumerate(same):
                names[i] = f"{name} {j + 1}"
    return names


def friendly(rects):
    """A plain name for the layouts people know, or None."""
    n = len(rects)
    if n == 1:
        return "whole screen"
    if all(w > 1 - 1e-9 for _, _, w, _ in rects):
        return "top / bottom" if n == 2 else f"{n} rows"
    if all(h > 1 - 1e-9 for _, _, _, h in rects):
        return "left / right" if n == 2 else f"{n} columns"
    if n == 4 and all(abs(w - 0.5) < 1e-9 and abs(h - 0.5) < 1e-9 for _, _, w, h in rects):
        return "quadrants"
    return None


# What most people pick for each count: listed first, and where the editor
# lands when the count changes.
DEFAULTS = {1: "whole screen", 2: "top / bottom", 3: "3 rows", 4: "quadrants",
            5: "corners + centre", 6: "3 × 2 grid"}


def complexity(label):
    """(how deeply the cuts nest, how many cuts) from a layout's label: plain
    rows or columns are (0, 1), "1 / 2" is (0, 2), "1 / (1 | 2)" is (1, 3)."""
    depth = deepest = 0
    for ch in label:
        depth += (ch == "(") - (ch == ")")
        deepest = max(deepest, depth)
    return deepest, 1 + label.count("(") + sum(int(t) > 1 for t in re.findall(r"\d+", label))


def corners_and_centre():
    """The quadrants with a section over the middle (sections listed first
    sit over the others): four L-shaped corners round a centre."""
    return [(0.3, 0.3, 0.4, 0.4)] + grid(2, 2), ["Centre", "Top left", "Top right", "Bottom left", "Bottom right"]


def catalogue(n):
    """[(name, rects, section names, over)] of every preset with n sections,
    the usual one first, then the simplest; `over` is how many of the first
    sections sit over the others."""
    if n == 6:
        return [(friendly(r) or label, r, section_names(r), 0) for r, label in
                ((grid(3, 2), "3 × 2 grid"), (grid(2, 3), "2 × 3 grid"), (rows(6), None), (columns(6), None))]
    found, seen = [], set()
    whole = [([(0.0, 0.0, 1.0, 1.0)], "1")]
    for across in (True, False):
        for rects, label in (cuts(n, (0.0, 0.0, 1.0, 1.0), across) if n > 1 else whole):
            rects = reading_order(rects)
            key = tuple(tuple(round(v, 6) for v in r) for r in rects)
            if key not in seen:
                seen.add(key)
                found.append((friendly(rects) or label, rects, section_names(rects), 0, complexity(label)))
    if n == 5:
        found.append(("corners + centre", *corners_and_centre(), 1, (0, 0)))
    order = sorted(range(len(found)), key=lambda i: (found[i][0] != DEFAULTS[n], found[i][4], i))
    return [found[i][:4] for i in order]


# name → rects, in order of section count, the usual first. Quadrants are in
# the order the quadrant model and the recorded `truth` use: top-left,
# top-right, bottom-left, bottom-right.
PRESETS, PRESET_NAMES, PRESET_OVER, BY_COUNT = {}, {}, {}, {}
for _n in COUNTS:
    BY_COUNT[_n] = []
    for _name, _rects, _names, _over in catalogue(_n):
        PRESETS[_name], PRESET_NAMES[_name], PRESET_OVER[_name] = _rects, _names, _over
        BY_COUNT[_n].append(_name)


def preset_of(layout):
    """The preset a layout is, or "custom" (moved or resized)."""
    return next((m for m in BY_COUNT.get(len(layout), [])
                 if np.allclose(PRESETS[m], layout, atol=1e-4)), "custom")


def apply_preset(sections, name):
    """The sections of preset `name`; commands are kept by position, so
    trying presets doesn't lose them."""
    out = []
    for j, (n, r) in enumerate(zip(PRESET_NAMES[name], PRESETS[name])):
        out.append({"name": n, "rect": list(r), "command": sections[j]["command"] if j < len(sections) else ""})
        if j < PRESET_OVER[name]:
            out[-1]["over"] = True
    return out


def section_of(where, layout):
    """For points (n, 2) as screen fractions: the index of the section each
    falls in, or -1 for a dead zone."""
    where = np.clip(np.asarray(where, float), 0, 1 - 1e-9)
    out = np.full(len(where), -1)
    for i, (x, y, w, h) in enumerate(layout):
        inside = (where[:, 0] >= x) & (where[:, 0] < x + w) & (where[:, 1] >= y) & (where[:, 1] < y + h)
        out[inside & (out < 0)] = i
    return out


def inside_point(layout, i):
    """A point that section_of puts in section i: its centre, or, when a
    section over it covers that, the spot of a fine grid over it furthest
    from that section's edges (a corner around a centre section, say)."""
    x, y, w, h = layout[i]
    centre = [x + w / 2, y + h / 2]
    if section_of([centre], layout)[0] == i:
        return centre
    g = np.linspace(0.05, 0.95, 19)
    pts = np.array([[x + a * w, y + b * h] for a in g for b in g])
    ok = pts[section_of(pts, layout) == i]
    if not len(ok):
        return centre
    edge = np.minimum.reduce([ok[:, 0] - x, x + w - ok[:, 0], ok[:, 1] - y, y + h - ok[:, 1]])
    return ok[edge.argmax()].tolist()


def layout_hash(layout):
    """Identifies a layout's geometry: a section model is only valid for
    the layout it was trained on. Names and commands don't count."""
    return hashlib.sha1(repr([tuple(round(v, 4) for v in r) for r in layout]).encode()).hexdigest()[:12]


def problems(cfg, gesture):
    """What's wrong with a gesture's layout in the config, as messages
    (empty: fine). More sections than max_sections isn't wrong: it's the
    recommended limit, and the editors only warn."""
    g = cfg[gesture]
    rects = [s["rect"] for s in g["sections"]]
    over = [bool(s.get("over")) for s in g["sections"]]
    out = []
    if g["enabled"] and not rects:
        out.append("add a section, or turn the gesture off")
    for i, (x, y, w, h) in enumerate(rects):
        if x < -1e-6 or y < -1e-6 or x + w > 1 + 1e-6 or y + h > 1 + 1e-6:
            out.append(f"{g['sections'][i]['name']} goes off the screen")
        if w < MIN_SIDE or h < MIN_SIDE:
            out.append(f"{g['sections'][i]['name']} is smaller than {MIN_SIDE:.0%} of the screen")
        if over[i] and not all(over[:i]):
            out.append(f"{g['sections'][i]['name']} sits over the others, so it must come first")
        for j in range(i):
            if over[i] or over[j]:        # one sits over the other and takes that area
                continue
            a, b = rects[j], rects[i]
            if a[0] < b[0] + b[2] - 1e-6 and b[0] < a[0] + a[2] - 1e-6 and a[1] < b[1] + b[3] - 1e-6 and b[1] < a[1] + a[3] - 1e-6:
                out.append(f"{g['sections'][j]['name']} and {g['sections'][i]['name']} overlap")
    return out

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
    from .train import session_paths
    from .features import signed, BEST1
    paths = session_paths()
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
    P = cell_probs(signed(np.concatenate(W), **BEST1), where, np.concatenate(sess))
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
