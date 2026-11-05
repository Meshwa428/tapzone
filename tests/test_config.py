"""gestures.toml survives a save and load, whatever the commands contain, and
the service checks what a front-end sends before saving it."""
import os, tempfile, unittest
from tapzones import config
from tapzones.service.engine import Engine


class RoundTrip(unittest.TestCase):
    def test_save_load(self):
        cfg = config.default_config()
        cfg["feedback"] = "notify"
        cfg["max_sections"] = 7
        cfg["double"]["enabled"] = True
        cfg["single"]["sections"][0]["command"] = r'notify-send "hi \"there\"" C:\path; echo $HOME'
        cfg["single"]["sections"][1]["name"] = 'Top "right"'
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "gestures.toml")
            config.save(cfg, path)
            self.assertEqual(config.load(path), cfg)

    def test_missing_file_is_default(self):
        self.assertEqual(config.load("/nonexistent/gestures.toml"), config.default_config())


class Checked(unittest.TestCase):
    """Engine.checked: the config a client sent over the socket."""
    def test_accepts_a_good_config(self):
        cfg = config.default_config()
        self.assertEqual(Engine.checked(cfg), cfg)

    def test_fills_in_missing_gestures(self):
        self.assertEqual(Engine.checked({"feedback": "none"})["single"], config.default_config()["single"])

    def test_rejects(self):
        bad_feedback = config.default_config() | {"feedback": "loud"}
        overlap = config.default_config()
        overlap["single"]["sections"][1]["rect"] = [0.25, 0, 0.5, 0.5]
        for cfg in (bad_feedback, overlap):
            with self.assertRaises(ValueError):
                Engine.checked(cfg)

    def test_coerces_types(self):
        cfg = config.default_config()
        cfg["single"]["sections"][0] |= {"name": 5, "command": 7, "rect": ["0", "0", "0.5", "0.5", "9"]}
        s = Engine.checked(cfg)["single"]["sections"][0]
        self.assertEqual((s["name"], s["command"], s["rect"]), ("5", "7", [0.0, 0.0, 0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
