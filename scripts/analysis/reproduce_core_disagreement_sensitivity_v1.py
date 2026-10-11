"""Reproduce the bounded two-case analysis in a fresh sibling timestamp directory."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.atlas_strict_single_thread_cpu_v1.common import sha256, thread_environment, write_json

ROOT = Path(__file__).resolve().parents[2]


def reproduce(source: Path, destination: Path | None) -> None:
    source = source.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination.resolve() if destination else source.parent / stamp
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    config = json.loads((source / "analysis_config.json").read_text())
    manifest = json.loads((source / "run_manifest.json").read_text())
    if sha256(source / "analysis_config.json") != manifest["configuration_sha256"]:
        raise ValueError("Frozen configuration changed")
    for path, expected in manifest["implementation_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Implementation changed: {path}")
    before = json.loads((source / "validation/protected_before.json").read_text())
    for path, expected in before["biological_snapshot"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen biological source changed: {path}")
    destination.mkdir(parents=True)
    for folder in ["cases", "figures", "validation"]:
        (destination / folder).mkdir()
    config["run_id"] = destination.name
    config["run_directory"] = str(destination)
    for case in config["cases"]:
        case_dir = destination / "cases" / case["case_id"]
        case_dir.mkdir()
        case["selection_coords_cache"] = str(case_dir / "selection_coords.npy")
        case["evaluation_coords_cache"] = str(case_dir / "evaluation_coords.npy")
    write_json(destination / "analysis_config.json", config)
    (destination / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(
        source / "validation/protected_before.json",
        destination / "validation/protected_before.json",
    )
    for name in [
        "supervisor_reply_en.txt",
        "manuscript_sensitivity_insert.md",
        "diagnostic_interpretation.md",
        "reproduce.sh",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    manifest.update(
        {
            "run_id": destination.name,
            "configuration_sha256": sha256(destination / "analysis_config.json"),
            "reproduced_from": str(source),
            "manuscript_policy": "Original human-reviewed prose copied; recomputed tables are authoritative if any discrepancy arises.",
        }
    )
    write_json(destination / "run_manifest.json", manifest)
    stages = [
        (
            "sets",
            [
                "scripts.analysis.core_disagreement_sensitivity_v1",
                "--run",
                str(destination),
                "--stage",
                "sets",
            ],
        ),
        (
            "sampling",
            [
                "scripts.analysis.core_disagreement_sensitivity_sampling_v1",
                "--config",
                str(destination / "analysis_config.json"),
            ],
        ),
        (
            "geometry",
            [
                "scripts.analysis.core_disagreement_sensitivity_v1",
                "--run",
                str(destination),
                "--stage",
                "geometry",
            ],
        ),
        (
            "rmsf_ADK",
            [
                "scripts.analysis.core_disagreement_rmsf_v1",
                "--run",
                str(destination),
                "--case",
                "ADK",
            ],
        ),
        (
            "rmsf_3nci_A",
            [
                "scripts.analysis.core_disagreement_rmsf_v1",
                "--run",
                str(destination),
                "--case",
                "3nci_A",
            ],
        ),
        (
            "figures",
            ["scripts.analysis.plot_core_disagreement_sensitivity_v1", "--run", str(destination)],
        ),
        (
            "report",
            ["scripts.analysis.report_core_disagreement_sensitivity_v1", "--run", str(destination)],
        ),
    ]
    statuses = []
    for name, args in stages:
        with (destination / "validation" / f"{name}.log").open("w") as log:
            process = subprocess.run(
                [sys.executable, "-m", *args],
                cwd=ROOT,
                env=thread_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        statuses.append({"stage": name, "returncode": process.returncode})
        write_json(destination / "validation/reproduction_status.json", statuses)
        print(name, process.returncode, flush=True)
    original = json.loads((ROOT / before["original_protection_list"]).read_text())
    strict = json.loads((ROOT / before["source_protection_record"]).read_text())[
        "additional_strict_snapshot"
    ]
    mismatches = {
        name: [path for path, h in hashes.items() if sha256(ROOT / path) != h]
        for name, hashes in [
            ("original", original),
            ("strict", strict),
            ("biological", before["biological_snapshot"]),
            ("inputs", before["original_input_hashes"]),
        ]
    }
    write_json(destination / "validation/protected_after.json", mismatches)
    print(destination, flush=True)
    if any(s["returncode"] for s in statuses) or any(mismatches.values()):
        raise SystemExit("Some stages/checks failed; outputs and diagnostics retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    reproduce(args.source_run, args.destination)
