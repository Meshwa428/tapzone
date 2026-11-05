"""omarchy/Layout.js must agree with tapzones/layout.py and config.py: the
plugin's editor validates and estimates layouts itself, and the service
then trains and runs them with the Python rules."""
import json, os, shutil, subprocess, unittest
import numpy as np
from tapzones import config, layout as L

LAYOUT_JS = os.path.join(os.path.dirname(__file__), "..", "omarchy", "Layout.js")

# Loads Layout.js (a QML JS library) in node and answers the questions on stdin.
HARNESS = r"""
const fs = require("fs")
const src = fs.readFileSync(process.argv[1], "utf8").replace(/^\.pragma library.*$/m, "")
const L = new Function(src + "; return {GESTURES, TITLES, FEEDBACK, MIN_SIDE, PRESETS, PRESET_NAMES, PRESET_OVER, BY_COUNT," +
                       " sectionOf, problems, quickEstimate}")()
const q = JSON.parse(fs.readFileSync(0, "utf8"))
const nan = v => (Number.isNaN(v) ? null : v)
console.log(JSON.stringify({
  constants: {GESTURES: L.GESTURES, TITLES: L.TITLES, FEEDBACK: L.FEEDBACK, MIN_SIDE: L.MIN_SIDE,
              PRESETS: L.PRESETS, PRESET_NAMES: L.PRESET_NAMES, PRESET_OVER: L.PRESET_OVER, BY_COUNT: L.BY_COUNT},
  sections: q.layouts.map(lay => q.points.map(p => L.sectionOf(p[0], p[1], lay))),
  problems: q.configs.map(c => L.problems(c, "single")),
  estimates: q.layouts.map(lay => {
    const e = L.quickEstimate(q.est, lay)
    return {overall: nan(e.overall), per: e.per.map(s => [nan(s.acc), s.n])}
  }),
}))
"""


def config_with(rects, max_sections=5, enabled=True, over=()):
    """Single taps with these rects; the sections at the indices in `over`
    sit over the others."""
    cfg = config.default_config()
    cfg["max_sections"] = max_sections
    cfg["single"] = {"enabled": enabled,
                     "sections": [config.section(f"S{i}", r, "", i in over) for i, r in enumerate(rects)]}
    return cfg


@unittest.skipUnless(shutil.which("node"), "node isn't installed")
class LayoutParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.layouts = [[list(r) for r in v] for v in L.PRESETS.values()] + [
            [list(r) for r in L.shrink(L.PRESETS["quadrants"], 0.1)],
            [[0.1, 0.1, 0.3, 0.2], [0.5, 0.5, 0.4, 0.4], [0.0, 0.7, 0.3, 0.3]],
            [list(r) for r in L.rows(6)],    # over the recommended 5: allowed
        ]
        cls.points = np.vstack([rng.random((300, 2)), [[0, 0], [1, 1], [0.5, 0.5], [-0.1, 1.2]]])
        n = 400
        cls.P = rng.dirichlet(np.ones(L.CELLS ** 2), n)
        cls.where = rng.random((n, 2))
        overs = [range(L.PRESET_OVER[n]) for n in L.PRESETS] + [()] * 3
        cls.configs = [config_with(l, over=o) for l, o in zip(cls.layouts, overs)] + [
            config_with([]),                                              # enabled, no sections
            config_with([[0, 0, 0.6, 1], [0.5, 0, 0.5, 1]]),              # overlap
            config_with([[0, 0, 0.5, 1], [0.5, 0, 0.5, 1], [0.3, 0.3, 0.4, 0.4]], over=(2,)),   # over, not first
            config_with([[0.9, 0, 0.2, 1]]),                              # off the screen
            config_with([[0, 0, 0.02, 1]]),                               # too small
        ]
        query = {"layouts": cls.layouts, "points": cls.points.tolist(), "configs": cls.configs,
                 "est": {"cells": L.CELLS, "min_taps": L.MIN_TAPS, "P": cls.P.tolist(), "where": cls.where.tolist()}}
        out = subprocess.run(["node", "-e", HARNESS, os.path.abspath(LAYOUT_JS)], input=json.dumps(query),
                             capture_output=True, text=True, check=True).stdout
        cls.js = json.loads(out)

    def test_constants(self):
        c = self.js["constants"]
        self.assertEqual(c["GESTURES"], config.GESTURES)
        self.assertEqual(c["TITLES"], config.TITLES)
        self.assertEqual(c["FEEDBACK"], list(config.FEEDBACK))
        self.assertEqual(c["MIN_SIDE"], L.MIN_SIDE)
        self.assertEqual(c["PRESET_NAMES"], L.PRESET_NAMES)
        self.assertEqual(set(c["PRESETS"]), set(L.PRESETS))
        for name, rects in L.PRESETS.items():
            np.testing.assert_allclose(c["PRESETS"][name], rects, err_msg=name)
        self.assertEqual(c["BY_COUNT"], {str(n): [p[0] for p in L.catalogue(n)] for n in L.COUNTS})
        self.assertEqual(c["PRESET_OVER"], L.PRESET_OVER)

    def test_section_of(self):
        for lay, js in zip(self.layouts, self.js["sections"]):
            self.assertEqual(js, L.section_of(self.points, lay).tolist(), lay)

    def test_problems(self):
        for cfg, js in zip(self.configs, self.js["problems"]):
            self.assertEqual(js, L.problems(cfg, "single"))
        self.assertFalse(any(self.js["problems"][:len(self.layouts)]), "the good layouts have no problem")
        self.assertTrue(all(self.js["problems"][len(self.layouts):]), "every broken config has a problem")

    def test_quick_estimate(self):
        for lay, js in zip(self.layouts, self.js["estimates"]):
            overall, per = L.quick_estimate(self.P, self.where, lay)
            self.assertAlmostEqual(js["overall"], overall, msg=lay)
            for (acc_js, n_js), (acc, n) in zip(js["per"], per):
                self.assertEqual(n_js, n)
                if np.isnan(acc):
                    self.assertIsNone(acc_js)
                else:
                    self.assertAlmostEqual(acc_js, acc)


class Presets(unittest.TestCase):
    def test_counts(self):
        # 1–5: every layout straight cuts make (the large Schröder numbers,
        # less the 2×2 grids two cut orders make alike), plus corners + centre.
        self.assertEqual([len(L.catalogue(n)) for n in L.COUNTS], [1, 2, 6, 21, 91, 4])

    def test_usual_first_then_simplest(self):
        for n in L.COUNTS:
            self.assertEqual(L.BY_COUNT[n][0], L.DEFAULTS[n])
        self.assertEqual(L.BY_COUNT[3], ["3 rows", "3 columns", "1 / 2", "2 / 1", "1 | 2", "2 | 1"])
        self.assertEqual(L.BY_COUNT[4][:3], ["quadrants", "4 rows", "4 columns"])
        ranks = [L.complexity(n) for n in L.BY_COUNT[5][3:]]      # after corners + centre, 5 rows, 5 columns
        self.assertEqual(ranks, sorted(ranks), "simplest first")

    def test_each_is_a_good_layout(self):
        grid_points = np.array([(x, y) for x in np.linspace(0.005, 0.995, 100) for y in np.linspace(0.005, 0.995, 100)])
        for name, rects in L.PRESETS.items():
            sections = L.apply_preset([], name)
            cfg = config.default_config()
            cfg["single"] = {"enabled": True, "sections": sections}
            self.assertEqual(L.problems(cfg, "single"), [], name)
            hit = L.section_of(grid_points, rects)
            self.assertTrue((hit >= 0).all(), f"{name} covers the screen")
            self.assertEqual(set(hit), set(range(len(rects))), f"{name}: every section can be hit")
            self.assertEqual(len(set(L.PRESET_NAMES[name])), len(rects), f"{name}: section names differ")

    def test_corners_and_centre(self):
        rects = L.PRESETS["corners + centre"]
        names = L.PRESET_NAMES["corners + centre"]
        hit = lambda x, y: names[L.section_of(np.array([[x, y]]), rects)[0]]
        self.assertEqual(hit(0.5, 0.5), "Centre")
        self.assertEqual(hit(0.4, 0.2), "Top left")        # beside the centre: the corner's L
        self.assertEqual(hit(0.2, 0.4), "Top left")
        self.assertEqual(hit(0.8, 0.8), "Bottom right")
        self.assertEqual([s.get("over", False) for s in L.apply_preset([], "corners + centre")],
                         [True, False, False, False, False])

    def test_over_survives_save_and_load(self):
        import tempfile
        cfg = config.default_config()
        cfg["single"]["sections"] = L.apply_preset(cfg["single"]["sections"], "corners + centre")
        with tempfile.TemporaryDirectory() as d:
            config.save(cfg, os.path.join(d, "g.toml"))
            self.assertEqual(config.load(os.path.join(d, "g.toml")), cfg)

    def test_known_layouts_kept(self):
        # The trained quadrant model and the research results use these.
        self.assertEqual(L.PRESETS["quadrants"], L.grid(2, 2))
        for name in ("whole screen", "top / bottom", "left / right", "3 rows", "5 rows", "3 columns"):
            self.assertIn(name, L.PRESETS)


if __name__ == "__main__":
    unittest.main()
