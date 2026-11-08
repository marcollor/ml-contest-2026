"""CLI requests must exit before training, downloads, or registry mutations."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_help_and_invalid_commands_stop_without_outputs(self):
        for script in ("train.py", "prepare_data.py", "model_registry.py"):
            for arguments, expected_code in ((["--help"], 0),
                                             (["--unknown-option"], 2),
                                             (["unknown-command"], 2)):
                with self.subTest(script=script, arguments=arguments):
                    with tempfile.TemporaryDirectory() as temporary:
                        result = subprocess.run(
                            [sys.executable, str(ROOT / "src" / script), *arguments],
                            cwd=temporary, capture_output=True, text=True, timeout=60,
                        )
                        self.assertEqual(result.returncode, expected_code, result.stderr)
                        output = result.stdout if expected_code == 0 else result.stderr
                        self.assertIn("usage:", output)
                        self.assertIn("--help", output)
                        self.assertIn("--models-dir" if script == "model_registry.py" else "--repo", output)
                        self.assertNotIn("Training configuration:", result.stdout)
                        self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_option_abbreviations_are_rejected(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src/model_registry.py"), "refresh", "--crit", "accuracy"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)
        self.assertIn("--models-dir", result.stderr)


if __name__ == "__main__":
    unittest.main()
