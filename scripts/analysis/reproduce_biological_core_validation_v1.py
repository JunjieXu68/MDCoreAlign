"""Reproduce numerical/figure artifacts in a new directory from frozen inputs."""

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
    if destination is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = source / ("reproduction_" + stamp)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    config = json.loads((source / "analysis_config.json").read_text())
    manifest = json.loads((source / "run_manifest.json").read_text())
    if sha256(source / "analysis_config.json") != manifest["configuration_sha256"]:
        raise ValueError("Frozen source configuration hash mismatch")
    if sha256(source / "literature_annotations.tsv") != config["annotation_sha256"]:
        raise ValueError("Frozen literature annotation hash mismatch")
    for relative_path, expected_hash in manifest["reused_code_sha256"].items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise ValueError(f"Production/reused implementation changed: {relative_path}")
    for relative_path, expected_hash in manifest.get("analysis_script_sha256", {}).items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise ValueError(f"Analysis implementation changed: {relative_path}")
    for case in config["cases"]:
        for key in ["coords_path", "test_coords_path"]:
            if key in case:
                path = Path(case[key])
                expected = manifest["analysis_coordinate_sha256"][str(path)]
                if sha256(path) != expected:
                    raise ValueError(f"Frozen coordinate input changed: {path}")
    destination.mkdir(parents=True)
    (destination / "validation").mkdir()
    for name in ["sources"]:
        shutil.copytree(source / name, destination / name)
    for name in [
        "case_inventory.tsv",
        "residue_mapping.tsv",
        "literature_annotations.tsv",
        "task_scope.json",
        "references.tsv",
        "references.bib",
        "manuscript_insert.md",
        "reproduce.sh",
    ]:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    for case in config["cases"]:
        case_dest = destination / "cases" / case["case_id"]
        case_dest.mkdir(parents=True)
        for name in ["reference_full_protein.pdb", "reference_structure_manifest.json"]:
            shutil.copy2(source / "cases" / case["case_id"] / name, case_dest / name)
    config["run_directory"] = str(destination)
    write_json(destination / "analysis_config.json", config)
    (destination / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest.update(
        {
            "reproduced_from": str(source),
            "configuration_sha256": sha256(destination / "analysis_config.json"),
            "new_result_directory": str(destination),
            "reproduction_scheduling": "sequential cases, one numerical thread, inherited CPU affinity; no timing benchmark",
            "prose_policy": "Source manuscript draft copied for review; numerical and figure artifacts recomputed from frozen inputs.",
        }
    )
    write_json(destination / "run_manifest.json", manifest)
    # Validate exactly the inherited old protection snapshot, without editing it.
    protected = json.loads((source / "validation/protected_before.json").read_text())
    shutil.copy2(
        source / "validation/protected_before.json",
        destination / "validation/protected_before.json",
    )
    statuses = []
    env = thread_environment()
    for case in config["cases"]:
        cid = case["case_id"]
        cmd = [
            sys.executable,
            "-m",
            "scripts.analysis.biological_core_validation_v1",
            "--config",
            str(destination / "analysis_config.json"),
            "--case",
            cid,
        ]
        with (destination / "validation" / f"{cid}_analysis.log").open("w") as log:
            proc = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        statuses.append({"case_id": cid, "returncode": proc.returncode})
        print(cid, proc.returncode, flush=True)
        if not proc.returncode:
            rendered = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.analysis.plot_biological_core_validation_v1",
                    "--run",
                    str(destination),
                    "--case",
                    cid,
                ],
                cwd=ROOT,
                env=env,
                check=False,
            )
            statuses[-1]["plot_returncode"] = rendered.returncode
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.analysis.report_biological_core_validation_v1",
            "--run",
            str(destination),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    write_json(destination / "validation/reproduction_status.json", statuses)
    original = json.loads((ROOT / protected["original_list"]).read_text())
    mismatch = [p for p, h in original.items() if sha256(ROOT / p) != h]
    extra = [p for p, h in protected["additional_strict_snapshot"].items() if sha256(ROOT / p) != h]
    write_json(
        destination / "validation/protected_after.json",
        {
            "original_count": len(original),
            "original_mismatches": mismatch,
            "additional_strict_count": len(protected["additional_strict_snapshot"]),
            "additional_mismatches": extra,
        },
    )
    print(destination, flush=True)
    if mismatch or extra or any(s["returncode"] or s.get("plot_returncode", 0) for s in statuses):
        raise SystemExit("Some checks/cases failed; completed artifacts and evidence retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    reproduce(args.source_run, args.destination)
