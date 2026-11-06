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

    def test_confirm_moves_a_tap_that_ran_the_wrong_section(self):
        self.tap(0.6)                                            # heard as Top
        self.assertEqual(self.e.state()["newest"]["name"], "Top")
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 0})
        self.assertIn("already ran", self.events("error")[-1]["message"])
        self.e.on_cmd(self.client, {"cmd": "confirm", "section": 1})
        self.assertEqual(self.events("confirmed")[-1]["was"], "Top")
        self.assertEqual(self.events("confirmed")[-1]["name"], "Bottom")
        with np.load(engine.CONFIRMED) as d:
            self.assertEqual(list(d["name"]), ["Bottom"])
        self.assertIsNone(self.e.gate_due)                       # the gate was right: nothing to learn
        self.assertIsNone(self.e.state()["newest"])

    def test_corrections_go_to_the_newest_sound(self):
        self.tap(0.1)                                            # ran Top
        with mock.patch.object(self.e.gate, "is_tap", lambda w: False):
            self.tap(0.7)                                        # then a miss
        self.assertEqual(self.e.state()["newest"]["why"], "ignored")
        self.e.on_cmd(self.client, {"cmd": "mistake"})           # the miss, not the older tap
        with np.load(engine.MARKED) as d:
            self.assertAlmostEqual(float(d["windows"][0, 0, 0]), 0.7, places=5)
        self.assertEqual(self.e.state()["newest"]["name"], "Top")   # the tap is next

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


class Setup(unittest.TestCase):
    """A new device: nothing recorded until the calibration overlay closes
    with a session, which trains the single-tap model, then the gate."""
    setUp, tearDown, events = Noise.setUp, Noise.tearDown, Noise.events

    def record(self, name, n=40):
        rng = np.random.default_rng(0)
        np.savez(os.path.join(self.dir.name, name), windows=np.zeros((n, FULL, 2), np.float32),
                 where=rng.uniform(0, 1, (n, 2)), truth=np.zeros(n, int))

    def test_uncalibrated_says_so_and_saving_trains_nothing(self):
        st = self.e.state()
        self.assertEqual(st["calibration"]["taps"], 0)
        self.e.model = engine.SectionModel("single", [s["rect"] for s in self.e.sections])
        self.assertEqual(self.e.state()["model"]["why"], "not calibrated yet")
        with mock.patch.object(engine.config, "save"), mock.patch.object(self.e, "train") as start:
            self.e.on_cmd(self.client, {"cmd": "save", "config": config.default_config()})
            start.assert_not_called()                            # nothing to train on

    def test_a_new_session_trains_the_model_then_the_gate(self):
        self.record("collect-20261106-140000.npz")
        with mock.patch.object(self.e, "train") as start:
            self.e.on_tick()
        self.assertEqual(start.call_args_list[0], mock.call("single"))
        self.assertEqual(self.e.state()["calibration"]["taps"], 40)
        self.assertEqual(self.e.state()["calibration"]["sittings"], 1)
        self.assertIsNone(self.e.gate_due)                       # no noise recorded: the gate waits
        with mock.patch.object(self.e, "train") as start:
            self.e.on_tick()
            start.assert_not_called()                            # only once per new session

    def test_nothing_trains_while_the_overlay_is_open(self):
        self.record("collect-20261106-140000.npz")
        with mock.patch.object(audio, "overlay_open", lambda: True), mock.patch.object(self.e, "train") as start:
            self.e.on_tick()
            start.assert_not_called()

    def test_a_calibration_that_crashes_says_why(self):
        log = os.path.join(self.dir.name, "setup.log")
        with open(log, "w") as f:
            f.write("Traceback …\nModuleNotFoundError: No module named 'gi'\n")
        self.e.calibrating = mock.Mock(returncode=1, poll=lambda: 1)
        with mock.patch.object(engine, "SETUP_LOG", log):
            self.e.on_tick()
        self.assertEqual(self.events("error")[-1], {"event": "error", "cmd": "calibrate",
                                                    "message": "ModuleNotFoundError: No module named 'gi'"})
        self.assertEqual(self.played, ["failed"])
        self.assertFalse(self.e.state()["calibration"]["running"])

    def test_set_changes_sound_without_training(self):
        with mock.patch.object(engine.config, "save") as save, mock.patch.object(self.e, "train") as start:
            self.e.on_cmd(self.client, {"cmd": "set", "sound": False, "feedback": "notify"})
            start.assert_not_called()
        saved = save.call_args[0][0]
        self.assertEqual((saved["sound"], saved["feedback"]), (False, "notify"))
        self.e.on_cmd(self.client, {"cmd": "set", "feedback": "loud"})
        self.assertIn("feedback must be", self.events("error")[-1]["message"])

    def test_training_reports_its_progress(self):
        log = os.path.join(self.dir.name, "train-single.log")
        self.assertEqual(engine.training_progress(log), (0.0, "starting"))
        with open(log, "w") as f:
            f.write("progress 0.050 testing on each sitting\nnoise\nprogress 0.420 testing on each sitting (3 of 8)\n")
        self.assertEqual(engine.training_progress(log), (0.42, "testing on each sitting (3 of 8)"))

    def test_one_sitting_is_scored_by_its_halves(self):
        self.assertEqual(list(train.sittings(["a"] * 4)), [0, 0, 1, 1])
        self.assertEqual(list(train.sittings(["a", "a", "b"])), [0, 0, 1])

    def test_corrections_are_never_a_sitting_of_their_own(self):
        # Four corrections held out alone would leave the rest scored by a
        # model that only saw them: every section but one read 0%.
        self.assertEqual(list(train.sittings(["a"] * 4 + ["confirmed.npz"] * 2)), [0, 0, 1, 1, -1, -1])
        self.assertEqual(list(train.sittings(["a", "b", "confirmed.npz"])), [0, 1, -1])


if __name__ == "__main__":
    unittest.main()
