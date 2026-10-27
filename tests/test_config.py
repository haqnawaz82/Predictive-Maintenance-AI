import runpy
from pathlib import Path
import unittest

CONFIG_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "predictive_maintenance"
    / "config.py"
)


class TestConfig(unittest.TestCase):
    def test_project_paths_exist(self):
        config = runpy.run_path(str(CONFIG_FILE))

        self.assertTrue(config["PROJECT_ROOT"].is_dir())
        self.assertTrue(config["RAW_DATA_DIR"].is_dir())
        self.assertTrue(config["PROCESSED_DATA_DIR"].is_dir())
        self.assertTrue(config["MODEL_DIR"].is_dir())
        self.assertTrue(config["REPORT_DIR"].is_dir())

    def test_random_seed_is_fixed(self):
        config = runpy.run_path(str(CONFIG_FILE))
        self.assertEqual(config["RANDOM_SEED"], 42)


if __name__ == "__main__":
    unittest.main()