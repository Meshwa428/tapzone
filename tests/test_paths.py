"""The socket only goes in a runtime directory no one else can get into:
whoever reaches it can save section commands, which run as you."""
import os, tempfile, unittest
from unittest import mock
from tapzones import paths


class RuntimeDir(unittest.TestCase):
    def check(self, d):
        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": d}):
            return paths.runtime_dir()

    def test_private_dir(self):
        with tempfile.TemporaryDirectory() as d:
            os.chmod(d, 0o700)
            self.assertEqual(self.check(d), d)

    def test_shared_dir(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in (0o755, 0o770, 0o1777):
                os.chmod(d, mode)
                self.assertIsNone(self.check(d), oct(mode))

    def test_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            os.chmod(d, 0o700)
            link = os.path.join(d, "link")
            os.symlink(d, link)
            self.assertIsNone(self.check(link))

    def test_unset_or_missing(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertIsNone(paths.runtime_dir())
        self.assertIsNone(self.check("/nonexistent/run"))


if __name__ == "__main__":
    unittest.main()
