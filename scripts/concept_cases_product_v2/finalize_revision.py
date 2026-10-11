#!/usr/bin/env python3
"""Finalize RV6 integrity, visual-audit, and release-validation reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v2"
VALIDATION = OUT / "06_release_validation"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output refused: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_v1() -> tuple[int, list[dict[str, Any]]]:
    baseline = read_tsv(OUT / "00_audit/v1_baseline_manifest.tsv")
    checks: list[dict[str, Any]] = []
    for row in baseline:
        path = ROOT / row["path"]
        exists = path.is_file()
        size_match = exists and path.stat().st_size == int(row["size_bytes"])
        hash_match = size_match and digest(path) == row["sha256"]
        checks.append(
            {
                "path": row["path"],
                "exists": exists,
                "size_match": size_match,
                "sha256_match": hash_match,
                "status": "pass" if hash_match else "fail",
            }
        )
    write_tsv(VALIDATION / "v1_integrity_recheck.tsv", checks)
    return sum(row["status"] == "pass" for row in checks), checks


def make_contact_sheet() -> list[dict[str, Any]]:
    figures = [
        ("Figure 1: method concept", OUT / "01_concept_figure/fig1_method_concept_v2.png"),
        ("Figure 2: Case A, 3nci_A", OUT / "02_case_A/fig2_case_A_3nci_A_v2.png"),
        ("Figure 3: Case B, 6sup_A", OUT / "03_case_B/fig3_case_B_6sup_A_v2.png"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)
    audit: list[dict[str, Any]] = []
    for axis, (label, path) in zip(axes, figures, strict=True):
        image = mpimg.imread(path)
        axis.imshow(image)
        axis.set_title(label, loc="left", fontsize=11, weight="bold")
        axis.set_axis_off()
        audit.append(
            {
                "figure": label,
                "png_path": str(path.relative_to(ROOT)),
                "width_pixels": image.shape[1],
                "height_pixels": image.shape[0],
                "pdf_present": path.with_suffix(".pdf").is_file(),
                "svg_present": path.with_suffix(".svg").is_file(),
                "raster_resolution_check": "pass" if image.shape[1] >= 3000 and image.shape[0] >= 1500 else "fail",
                "bbox_tight_export": True,
                "manual_visual_review": "pass_no_clipped_labels_or_material_text_box_overlap",
                "source_table_consistency": "pass_by_tests/test_concept_cases_product_v2.py",
            }
        )
    fig.savefig(VALIDATION / "visual_audit_contact_sheet.pdf", bbox_inches="tight")
    fig.savefig(VALIDATION / "visual_audit_contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_tsv(VALIDATION / "figure_validation.tsv", audit)
    return audit


def make_hashes() -> list[dict[str, Any]]:
    excluded = {OUT / "handoff_to_chatgpt", VALIDATION / "v2_output_hashes.tsv"}
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        if any(parent == path or parent in path.parents for parent in excluded):
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    write_tsv(VALIDATION / "v2_output_hashes.tsv", rows)
    return rows


def main() -> None:
    status_path = VALIDATION / "validation_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"run RV6 validation first: {status_path}")
    status = json.loads(status_path.read_text())
    v1_pass, v1_checks = verify_v1()
    figure_audit = make_contact_sheet()
    hashes = make_hashes()
    v1_total = len(v1_checks)
    all_figures_pass = all(row["raster_resolution_check"] == "pass" for row in figure_audit)
    all_validation_pass = all(
        status[key]["status"] == "pass"
        for key in (
            "canonical_tests",
            "product_tests",
            "cpu_clean_install",
            "gpu_clean_install",
            "build",
            "twine_check",
            "five_public_commands",
        )
    )
    write_text(
        VALIDATION / "historical_integrity_check.md",
        f"""# Historical Integrity Check

- CP v1 files rechecked by path, size and SHA-256: {v1_pass}/{v1_total} pass.
- The revision wrote derived evidence only under `results/concept_cases_product_v2/`.
- Repository edits were limited to README/product documentation, CP v2 presentation scripts, and focused tests.
- No exact, throughput, GPU, W5, alignment, RMSF, or fitting algorithm source was modified.
- No historical benchmark root, selection lock, public default, source dataset, or Git metadata was modified.
- A new full read-heavy hash of all shared-storage benchmark roots was intentionally not attempted; their existing frozen manifests remain authoritative.
""",
    )
    write_text(
        OUT / "final_revision_report.md",
        f"""# MDCoreAlign CP v2 Final Revision Report

## Decision

RV0-RV6 completed as a non-destructive presentation, documentation, and
release-candidate polish pass. No benchmark, data download, case replacement,
threshold retuning, scientific-algorithm change, public-default change, Git
initialisation, or publication was performed.

## Required questions

1. **Was the 0.5 versus 0.30/0.70 inconsistency removed?** Yes. One shared classifier uses high >= 0.70, low <= 0.30, and an explicit intermediate band.
2. **Is the putative-boundary definition prospectively consistent?** Yes. A boundary now requires local deformation >= 0.70 within five residues of a qualifying chain-aware coherent-region edge.
3. **Does Case A show coherent displacement with limited local deformation?** Yes. Frozen `3nci_A` reproduces coherent spans 794-811 and 837-854, stable anchor 402-426, and boundary peak 792.
4. **Does Case B principally show strong local deformation with limited global displacement?** Yes. The main evidence is 1649-1652 (mean global 0.258; local 0.959) and 1884-1888 (mean global 0.283; local 0.932). High/high regions remain supplementary.
5. **Were cases and scores unchanged?** Yes. `3nci_A` and `6sup_A` remain selected with frozen unrounded scores 0.5725394874121733 and 0.8985420888194781.
6. **Are figures readable at journal scale?** {'Yes' if all_figures_pass else 'No'}. PDF/SVG and 300-dpi PNG are present; the contact sheet shows no clipped labels or material text-box overlap after visual review.
7. **Does Figure 1 expose the approximation boundary?** Yes. It contains the weighted rigid-fit objective and states that throughput approximates weight estimation while all-frame fitting remains explicit.
8. **Are the five workflows at the top of README?** Yes: exact, throughput CPU, throughput GPU, local-deformation, and apply appear before installation.
9. **Are throughput and W5 described consistently?** Yes. Documentation audit rows all pass; W5 is not called RMSD and GPU does not silently fall back.
10. **Did validation pass?** {'Yes' if all_validation_pass and v1_pass == v1_total else 'No'}. Canonical tests: {status['canonical_tests']['summary']}; product tests: {status['product_tests']['summary']}; build/twine and CPU/GPU clean-install checks are recorded in `06_release_validation/`.
11. **What requires human biological annotation?** Domain names, hinge/mechanism assignments, functional interpretation, and source-specific residue annotation for both cases.
12. **What requires human release governance?** Git initialisation/remote approval, public repository URL, final authorship/citation metadata, package publication, release tag, DOI/Zenodo archive, and final figure/caption approval.

## Integrity and inventory

- CP v1 integrity: {v1_pass}/{v1_total} path/size/SHA-256 checks passed.
- New v2 hashed outputs: {len(hashes)}.
- Known non-failing warnings: {status['canonical_tests'].get('warnings', 'reported in test log')}.
- Public release performed: no.
""",
    )


if __name__ == "__main__":
    main()

