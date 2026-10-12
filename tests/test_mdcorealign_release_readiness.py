"""Release-readiness tests for the user-facing MDCoreAlign package."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MDCoreAlignReleaseReadinessTest(unittest.TestCase):
    """Validate package metadata, CLI help, and the toy quickstart."""

    def _env(self) -> dict[str, str]:
        """Return an environment that can import the local src package."""

        env = os.environ.copy()
        if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") == "product":
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def test_import_mdcorealign_namespace(self) -> None:
        """The release namespace should import successfully."""

        completed = subprocess.run(
            [sys.executable, "-c", "import mdcorealign; print(mdcorealign.__version__)"],
            cwd=ROOT,
            env=self._env(),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("1.0.0", completed.stdout)

    def test_pyproject_metadata_and_console_script(self) -> None:
        """Package metadata should expose the user-facing CLI."""

        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(data["project"]["name"], "mdcorealign")
        self.assertEqual(data["project"]["version"], "1.0.0")
        self.assertEqual(data["project"]["scripts"]["mdcorealign"], "mdcorealign.cli:main")
        self.assertIn("dev", data["project"]["optional-dependencies"])
        self.assertIn("external", data["project"]["optional-dependencies"])

    def test_cli_help_commands(self) -> None:
        """All required CLI help commands should run without external tools."""

        commands = [
            [sys.executable, "-m", "mdcorealign", "--help"],
            [sys.executable, "-m", "mdcorealign", "align", "--help"],
            [sys.executable, "-m", "mdcorealign", "flexibility", "--help"],
            [sys.executable, "-m", "mdcorealign", "diagnostics", "--help"],
            [sys.executable, "-m", "mdcorealign", "audit-preprocessing", "--help"],
            [sys.executable, "-m", "mdcorealign", "local-deformation", "--help"],
            [sys.executable, "-m", "mdcorealign", "apply", "--help"],
        ]
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=self._env(),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("usage:", completed.stdout)

    def test_toy_quickstart_outputs(self) -> None:
        """The bundled toy quickstart should produce the expected files."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "toy_output"
            subprocess.run(
                ["bash", str(ROOT / "examples" / "toy_md" / "run_quickstart.sh"), str(output_dir)],
                cwd=ROOT,
                env=self._env(),
                check=True,
                capture_output=True,
                text=True,
            )
            expected = [
                "residue_profile.tsv",
                "frame_metrics.tsv",
                "transforms.npz",
                "flexible_segments.tsv",
                "summary.json",
            ]
            for name in expected:
                path = output_dir / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 0, name)
            with (output_dir / "residue_profile.tsv").open(newline="") as handle:
                columns = csv.DictReader(handle, delimiter="\t").fieldnames or []
            self.assertIn("stable_weight", columns)
            self.assertIn("mobility_score", columns)
            self.assertTrue((output_dir / "diagnostics").is_dir())

    def test_readme_has_no_local_absolute_paths(self) -> None:
        """The user README should not contain local machine paths."""

        text = (ROOT / "README.md").read_text()
        forbidden = ["/" + "storage/", "/" + "home/", "MDAnalysis" + "_data"]
        for pattern in forbidden:
            self.assertNotIn(pattern, text)


if __name__ == "__main__":
    unittest.main()
