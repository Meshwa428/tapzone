"""What isn't a tap: marking the last tap as a mistake, and recording noise,
both end up where the gate trains from (train.noise_set). Runs the service's
handlers without a mic, in a throwaway data folder."""
import os, tempfile, time, unittest
from unittest import mock
import numpy as np
from tapzones import audio, config, models, train
from tapzones.features import EARLY_N
from tapzones.service import engine

FULL = (4 + 1 + 30) * 240          # a whole detector window, in samples


class FakeModel:
    ok, cutoff, d = True, 0.0, {"accuracy": 1, "answered": 1, "answered_accuracy": 1}

    def proba(self, window):
        return np.array([0.9, 0.1])


class FakeClient:
    ui = editing = False

    def __init__(self):
        self.got = []

    def send(self, obj):
        self.got.append(obj)


class Noise(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        patches = [mock.patch.object(engine, "DATA", d), mock.patch.object(train, "DATA", d),
                   mock.patch.object(engine, "MARKED", os.path.join(d, "noise-marked.npz")),
                   mock.patch.object(engine, "CONFIRMED", os.path.join(d, "confirmed.npz")),
                   mock.patch.object(engine, "GATE_PATH", os.path.join(d, "gate.joblib")),
                   mock.patch.object(models, "GATE_PATH", os.path.join(d, "gate.joblib")),
                   mock.patch.object(audio, "overlay_open", lambda: False),
                   mock.patch.object(engine, "CONFIG", os.path.join(d, "gestures.toml")),
                   mock.patch.object(engine.config, "load", config.default_config),
                   mock.patch.object(engine, "PAUSED", os.path.join(d, "paused")),
                   mock.patch.object(engine, "log", lambda msg: None),
                   mock.patch.object(engine.sound, "play", lambda moment: self.played.append(moment)),
                   mock.patch.object(engine.Engine, "notify", staticmethod(lambda title, body: None))]
        self.played = []
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.e = engine.Engine()
        self.e.model = FakeModel()
        self.e.sections = [config.section("Top", [0, 0, 1, .5], ""), config.section("Bottom", [0, .5, 1, .5], "")]
        self.client = FakeClient()
        self.e.clients.append(self.client)

    def tearDown(self):
        self.dir.cleanup()

    def tap(self, level=0.1):
        self.e.on_tap(np.full((EARLY_N, 2), level, np.float32), time.perf_counter())
        self.e.on_full(np.full((FULL, 2), level, np.float32))

    def events(self, name):
        return [m for m in self.client.got if m.get("event") == name]

    def test_each_moment_has_its_sound(self):
        self.tap()
        self.e.on_cmd(self.client, {"cmd": "mistake"})
        self.e.on_cmd(self.client, {"cmd": "mistake"})        # nothing left to mark
        self.e.on_cmd(self.client, {"cmd": "pause"})
        self.e.on_cmd(self.client, {"cmd": "resume"})
        self.assertEqual(self.played, ["tap", "marked", "nothing", "pause", "resume"])

    def test_sound_off_plays_nothing(self):
        self.e.cfg["sound"] = False
        self.tap()
        self.e.on_cmd(self.client, {"cmd": "pause"})
        self.assertEqual(self.played, [])

    def test_mistake_keeps_the_last_tap_as_noise(self):
        self.tap(0.1)
        self.tap(0.2)
        self.assertEqual(self.e.state()["last"]["name"], "Top")
        self.e.on_cmd(self.client, {"cmd": "mistake"})
        with np.load(engine.MARKED) as d:
            self.assertEqual(d["windows"].shape, (1, FULL, 2))
            self.assertAlmostEqual(float(d["windows"][0, 0, 0]), 0.2, places=5)   # the newest
            self.assertEqual(list(d["name"]), ["Top"])
        self.assertEqual(self.events("marked")[-1]["total"], 1)
        self.assertIsNotNone(self.e.gate_due)
        self.assertIsNone(self.e.state()["last"])
        self.e.on_cmd(self.client, {"cmd": "mistake"})         # only the newest: nothing older
        self.assertEqual(self.e.marked, 1)
        self.assertIn("nothing to mark", self.events("error")[-1]["message"])
        self.tap(0.3)                                           # a new tap can be marked again
        self.e.on_cmd(self.client, {"cmd": "mistake"})
        self.assertEqual(self.e.marked, 2)

    def test_unsure_sound_doesnt_replace_the_last_tap(self):
        self.tap(0.1)
        self.e.on_full(np.full((FULL, 2), 0.9, np.float32))     # a sound that ran nothing
        self.e.on_cmd(self.client, {"cmd": "mistake"})
        with np.load(engine.MARKED) as d:
            self.assertAlmostEqual(float(d["windows"][0, 0, 0]), 0.1, places=5)

    def test_confirm_keeps_a_turned_away_sound_as_a_tap_there(self):
        with mock.patch.object(self.e.gate, "is_tap", lambda w: False):
            self.tap(0.4)
        self.assertEqual(self.e.state()["missed"]["why"], "ignored")
        self.assertIsNone(self.e.last)                           # nothing ran
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 1})
        with np.load(engine.CONFIRMED) as d:
            self.assertAlmostEqual(float(d["windows"][0, 0, 0]), 0.4, places=5)
            self.assertEqual(list(d["name"]), ["Bottom"])
            self.assertEqual(engine.layout.section_of(d["where"], [s["rect"] for s in self.e.sections])[0], 1)
        self.assertEqual(self.events("confirmed")[-1]["total"], 1)
        self.assertIn(engine.CONFIRMED, train.session_paths())   # the gate and section models train on it
        self.assertIsNotNone(self.e.gate_due)
        self.assertEqual(self.played[-1], "tap")
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 0})   # only once
        self.assertIn("nothing to confirm", self.events("error")[-1]["message"])

    def test_confirm_takes_an_unsure_tap(self):
        self.e.model.cutoff = 0.95                               # 90% isn't sure enough
        self.tap(0.5)
        self.assertEqual(self.e.state()["missed"]["why"], "unsure")
        self.assertEqual(self.played, [])                        # "tap again" is silent
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 0})
        self.assertEqual(self.e.confirmed, 1)
        self.assertEqual(self.events("confirmed")[-1]["name"], "Top")

    def test_confirm_needs_a_real_section(self):
        with mock.patch.object(self.e.gate, "is_tap", lambda w: False):
            self.tap()
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 5})
        self.assertIn("no section", self.events("error")[-1]["message"])
        self.assertIsNotNone(self.e.state()["missed"])           # still there to confirm

    def test_inside_point_avoids_a_section_over_it(self):
        corners = [[0.25, 0.25, 0.5, 0.5], [0, 0, 1, 1]]         # a centre over the whole screen
        p = engine.layout.inside_point(corners, 1)
        self.assertEqual(engine.layout.section_of([p], corners)[0], 1)

    def test_recording_keeps_every_sound_and_runs_nothing(self):
        self.e.on_cmd(self.client, {"cmd": "noise", "on": True})
        for _ in range(3):
            self.tap()
        self.assertEqual(self.e.state()["recording"]["count"], 3)
        self.assertIsNone(self.e.last)                           # nothing ran
        self.e.on_cmd(self.client, {"cmd": "noise", "on": False})
        self.assertTrue(self.events("noise")[-1]["done"])
        self.assertIsNone(self.e.recording)
        N, kind = train.noise_set(np.random.default_rng(0))
        self.assertEqual(len(N), 3)
        self.assertTrue(all(k.startswith("noise-rec-") for k in kind))

    def test_recording_stops_when_its_panel_goes(self):
        self.e.on_cmd(self.client, {"cmd": "noise", "on": True})
        self.tap()
        self.client.conn = mock.Mock()
        self.e.on_gone(self.client)
        self.assertIsNone(self.e.recording)
        self.assertEqual(len([f for f in os.listdir(self.dir.name) if f.startswith("noise-rec-")]), 1)

    def test_gate_waits_for_recorded_taps(self):
        self.tap()
        self.e.on_cmd(self.client, {"cmd": "mistake"})
        self.e.gate_due -= engine.GATE_AFTER + 1
        with mock.patch.object(self.e, "train") as start:
            self.e.on_tick()
            start.assert_not_called()                            # no taps recorded yet
        self.assertIsNone(self.e.gate_due)


if __name__ == "__main__":
    unittest.main()
