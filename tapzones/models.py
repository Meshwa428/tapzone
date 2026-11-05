"""The trained models the service runs on every tap: the gate (tap or not)
and a gesture's section model. Both are trained by train.py."""
import os
import numpy as np
from .paths import DATA
from .features import EARLY_N, BEST1, signed, tap_features
from .layout import layout_hash


GATE_PATH = os.path.join(DATA, "gate.joblib")


def model_path(gesture):
    return os.path.join(DATA, f"sections-{gesture}.joblib")


class Gate:
    """Tap or not (tapzones train gate). Without gate.joblib, every sound
    counts as a tap."""
    def __init__(self):
        import joblib
        self.d = joblib.load(GATE_PATH) if os.path.exists(GATE_PATH) else None   # our own file
        if self.d:
            self.is_tap(np.zeros((EARLY_N, 2), np.float32))   # one-time setup now, not on the first tap

    def is_tap(self, window):
        if not self.d:
            return True
        score = self.d["model"].decision_function(tap_features(window[None]))[0]
        return bool(score >= self.d["threshold"])


class SectionModel:
    """One gesture's section model (tapzones train sections GESTURE): which
    section a tap was in, from the early window. `ok` is False when there is
    no model, or it was trained for a different layout."""
    def __init__(self, gesture, layout):
        import joblib
        self.ok, self.why = False, ""
        path = model_path(gesture)
        if not os.path.exists(path):
            self.why = "no model for this layout yet"
            return
        self.d = joblib.load(path)                                      # our own file
        if self.d["hash"] != layout_hash(layout):
            self.why = "the model is for a different layout"
            return
        self.ok, self.cutoff = True, self.d["cutoff"]
        self.proba(np.zeros((EARLY_N, 2), np.float32))

    def proba(self, window):
        f = signed(window[None], **BEST1)
        return np.mean([m.predict_proba(f)[0] for m in self.d["models"]], axis=0)
