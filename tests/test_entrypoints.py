import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OfflineEntrypointTests(unittest.TestCase):
    def test_cli_validates_every_example_without_flight_dependencies(self):
        for example in (ROOT / "examples").glob("*.json"):
            with self.subTest(example=example.name):
                result = subprocess.run(
                    [sys.executable, "-m", "mission_control", "validate", str(example)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], "validated")

    def test_cli_emits_launch_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "mission_control", "validate", "examples/reach_orbit.json", "--launch-options"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["target_ap"], 120_000)

    def test_cli_rejects_invalid_command_without_traceback(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"task_id":"bad","task":"land","target":{},"unexpected":true}')
            result = subprocess.run(
                [sys.executable, "-m", "mission_control", "validate", str(path)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Mission rejected", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_all_python_sources_compile(self):
        for folder in ("mission_control", "scripts", "tests"):
            for path in (ROOT / folder).glob("*.py"):
                with self.subTest(path=path.name):
                    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
