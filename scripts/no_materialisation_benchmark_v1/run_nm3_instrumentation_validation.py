#!/usr/bin/env python3
"""Validate the separately named timing-only MDLovoFit build."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import subprocess
import tempfile

from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    STOCK,
    TIMED,
    parse_timing,
    run_external,
    sha256,
    stock_command,
    timed_command,
    write_json,
    write_text,
    write_tsv,
)
from scripts.no_materialisation_benchmark_v1.run_nm1_verification import (
    LOCKED_CASES,
    METRICS,
    _input,
)


DEST = OUT / "04_instrumented_kernel"
ORIGINAL = Path("/home/jjxu/Downloads/MDLovoFit/src/mdlovofit.f90")
INSTRUMENTED = DEST / "source/mdlovofit_timed.f90"
EXPECTED_TIMERS = (
    "process_setup",
    "command_line_parsing",
    "PDB_parsing",
    "initial_reference_setup",
    "LOVO_search",
    "final_fixed_subset_fit",
    "RMSD_reduction",
    "stdout_formatting",
    "coordinate_preparation",
    "total_program",
)


def _compiler_version() -> str:
    return subprocess.run(
        ["/usr/bin/f95", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    original_lines = ORIGINAL.read_text().splitlines(keepends=True)
    timed_lines = INSTRUMENTED.read_text().splitlines(keepends=True)
    patch = "".join(
        difflib.unified_diff(
            original_lines,
            timed_lines,
            fromfile="official_MDLovoFit_20.1.1/src/mdlovofit.f90",
            tofile="timing_only/mdlovofit_timed.f90",
        )
    )
    write_text(DEST / "instrumentation_patch.diff", patch)
    compile_command = (
        "/usr/bin/f95 04_instrumented_kernel/source/common.f90 "
        "04_instrumented_kernel/source/lib.f90 "
        "04_instrumented_kernel/source/mdlovofit_timed.f90 "
        "-o 04_instrumented_kernel/bin/mdlovofit_timed"
    )
    write_text(
        DEST / "build_manifest.txt",
        "\n".join(
            (
                "MDLovoFit timing-only build manifest",
                f"compiler: {_compiler_version()}",
                "compiler_flags: none (matches the discovered stock CMake build)",
                f"compile_command_repository_relative: {compile_command}",
                f"stock_binary_sha256: {sha256(STOCK)}",
                f"timed_binary_sha256: {sha256(TIMED)}",
                f"official_main_source_sha256: {sha256(ORIGINAL)}",
                f"timed_main_source_sha256: {sha256(INSTRUMENTED)}",
                "allowed_change: monotonic system_clock instrumentation and stderr NM_TIMING records only",
            )
        ),
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nm3-timed-") as temporary:
        root = Path(temporary)
        for case in LOCKED_CASES:
            case_root = root / case["case"]
            case_root.mkdir()
            input_path, frames, residues, provenance = _input(case, case_root)
            stock = run_external(stock_command(input_path), cwd=case_root, timeout=1800)
            timed = run_external(timed_command(input_path), cwd=case_root, timeout=1800)
            timing = parse_timing(str(timed.get("stderr", "")))
            differences: list[float] = []
            metrics_identical = True
            for metric in METRICS:
                left = stock.get(metric)
                right = timed.get(metric)
                if isinstance(left, float) and isinstance(right, float):
                    difference = abs(left - right)
                    same = difference <= 5e-5
                else:
                    difference = float(left != right)
                    same = left == right
                differences.append(difference)
                metrics_identical &= same
            stdout_byte_identical = stock.get("stdout") == timed.get("stdout")
            timer_complete = all(name in timing and timing[name] >= 0.0 for name in EXPECTED_TIMERS)
            total_bounds_ok = (
                timer_complete
                and timing["total_program"] >= timing["LOVO_search"]
                and timing["total_program"] >= (
                    timing["LOVO_search"]
                    + timing["final_fixed_subset_fit"]
                    + timing["RMSD_reduction"]
                )
            )
            passed = (
                stock["status"] == timed["status"] == "success"
                and stock["return_code"] == timed["return_code"] == 0
                and metrics_identical
                and stdout_byte_identical
                and timer_complete
                and total_bounds_ok
            )
            rows.append(
                {
                    **case,
                    "n_frames": frames,
                    "n_residues": residues,
                    "input_provenance": provenance,
                    "stock_return_code": stock["return_code"],
                    "timed_return_code": timed["return_code"],
                    "stdout_byte_identical": stdout_byte_identical,
                    "maximum_printed_numeric_difference": max(differences),
                    "timing_regions_complete": timer_complete,
                    "timing_total_bounds_ok": total_bounds_ok,
                    "stock_wall_seconds": stock["wall_seconds"],
                    "timed_wall_seconds": timed["wall_seconds"],
                    **{f"timed_{name}_seconds": timing.get(name, "") for name in EXPECTED_TIMERS},
                    "numerical_identity_pass": passed,
                }
            )
    write_tsv(DEST / "numerical_identity.tsv", rows)
    passed = all(bool(row["numerical_identity_pass"]) for row in rows)
    write_json(
        DEST / "nm3_status.json",
        {
            "status": "pass" if passed else "fail",
            "cases": len(rows),
            "stdout_byte_identical_all": all(bool(row["stdout_byte_identical"]) for row in rows),
            "timing_regions_complete_all": all(bool(row["timing_regions_complete"]) for row in rows),
        },
    )
    write_text(
        DEST / "instrumentation_validation.md",
        f"""# Timing-only MDLovoFit Validation

Status: **{'PASS' if passed else 'FAIL'}** across {len(rows)} locked cases.

The separately named `mdlovofit_timed` binary was compiled with the discovered
stock build's compiler and empty optimisation-flag set. Its only source changes
are `system_clock` measurements and `NM_TIMING` records written to stderr. Stock
and instrumented stdout are byte-identical, parsed native values and exit codes
match, and all registered timing regions are finite. `coordinate_preparation`
is reported separately because MDLovoFit 20.1.1 still rotates coordinates in
its final pass even when `-t` is absent; it is excluded from C2 core compute.
""",
    )
    if not passed:
        raise RuntimeError("NM3 instrumentation identity gate failed")
    print(json.dumps({"status": "pass", "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
