"""Regression checks for the non-destructive CP v2 presentation revision."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.image as mpimg

from scripts.concept_cases_product_v2.case_interpretation import (
    HIGH_SCORE,
    LOW_SCORE,
    classify_scores,
    detect_regions,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results/concept_cases_product_v1"
V2 = ROOT / "results/concept_cases_product_v2"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_locked_thresholds_and_intermediate_class() -> None:
    assert HIGH_SCORE == 0.70
    assert LOW_SCORE == 0.30
    assert classify_scores(0.50, 0.20, 0.20) == "intermediate"
    assert classify_scores(0.20, 0.50, 0.20) == "intermediate"
    assert classify_scores(0.70, 0.30, 0.20) == "coherent_displacement_candidate"
    assert classify_scores(0.30, 0.70, 0.20) == "local_flexibility_candidate"
    assert classify_scores(0.70, 0.70, 0.90) == "high_global_high_local"


def test_boundary_requires_a_qualifying_coherent_region_and_respects_chain() -> None:
    rows = []
    for index in range(1, 15):
        rows.append(
            {
                "chain": "A",
                "residue_id": str(index),
                "residue_index": str(index),
                "global_displacement_score": "0.8" if 3 <= index <= 10 else "0.2",
                "local_deformation_score": "0.2",
                "exact_stable_weight": "0.2",
            }
        )
    rows[1]["local_deformation_score"] = "0.9"
    regions = detect_regions(rows)
    boundary = [region for region in regions if region.region_type == "putative_boundary"]
    assert len(boundary) == 1
    assert boundary[0].start_residue_index == 2
    assert boundary[0].qualifying_coherent_region == "coherent_1"

    no_coherent = [dict(row, global_displacement_score="0.2") for row in rows]
    assert not [region for region in detect_regions(no_coherent) if region.region_type == "putative_boundary"]

    cross_chain = [dict(row) for row in rows]
    cross_chain[1]["chain"] = "B"
    assert not [region for region in detect_regions(cross_chain) if region.region_type == "putative_boundary"]


def test_scatter_and_track_share_one_classifier_and_no_unregistered_split() -> None:
    source = (ROOT / "scripts/concept_cases_product_v2/generate_revision.py").read_text()
    classifier = (ROOT / "scripts/concept_cases_product_v2/case_interpretation.py").read_text()
    assert 'row["v2_primary_class"]' in source
    forbidden = [r"axvline\(0\.5", r"axhline\(0\.5", r">=\s*0\.5", r"<=\s*0\.5"]
    assert not any(re.search(pattern, source) for pattern in forbidden)
    assert not any(re.search(pattern, classifier) for pattern in forbidden)


def test_locked_case_identity_scores_and_metrics_are_unchanged() -> None:
    decision = json.loads((V1 / "02_case_selection/selection_decision.json").read_text())
    assert decision["case_A"] == "3nci_A"
    assert decision["case_B"] == "6sup_A"
    candidates = {row["system"]: row for row in _rows(V1 / "02_case_selection/case_candidate_table.tsv")}
    assert float(candidates["3nci_A"]["case_A_score"]) == 0.5725394874121733
    assert float(candidates["6sup_A"]["case_B_score"]) == 0.8985420888194781

    for role, directory in (("case_A", "02_case_A"), ("case_B", "03_case_B")):
        old = _rows(V1 / f"03_case_studies/{role}/residue_decomposition.tsv")
        new = _rows(V2 / f"{directory}/{role}_classification_track_v2.tsv")
        assert len(old) == len(new)
        for old_row, new_row in zip(old, new, strict=True):
            for field in (
                "exact_stable_weight",
                "throughput_stable_weight",
                "robust_RMSF_A",
                "ordinary_RMSF_A",
                "global_displacement_score",
                "local_deformation_score",
            ):
                assert old_row[field] == new_row[field]


def test_case_a_registered_regions_reproduce() -> None:
    rows = _rows(V2 / "02_case_A/case_A_key_regions_v2.tsv")
    spans = {(row["region_type"], int(row["start_residue_index"]), int(row["end_residue_index"])) for row in rows}
    assert ("stable_anchor", 402, 426) in spans
    assert ("coherently_displaced", 794, 811) in spans
    assert ("coherently_displaced", 837, 854) in spans
    assert ("putative_boundary", 792, 792) in spans


def test_case_b_principal_low_global_high_local_regions_reproduce() -> None:
    rows = _rows(V2 / "03_case_B/case_B_key_regions_v2.tsv")
    by_start = {int(row["start_residue_index"]): row for row in rows}
    for start, end in ((1649, 1652), (1884, 1888)):
        row = by_start[start]
        assert int(row["end_residue_index"]) == end
        assert float(row["mean_global_displacement_score"]) <= LOW_SCORE
        assert float(row["mean_local_deformation_score"]) >= HIGH_SCORE
    complete = _rows(V2 / "03_case_B/case_B_region_summary_v2.tsv")
    assert any(row["region_type"] == "high_global_high_local" for row in complete)


def test_v1_path_size_and_hash_snapshot_is_still_valid() -> None:
    manifest = _rows(V2 / "00_audit/v1_baseline_manifest.tsv")
    assert manifest
    for row in manifest:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["size_bytes"])
        assert _digest(path) == row["sha256"]


def test_main_figures_and_captions_are_consistent() -> None:
    figures = [
        V2 / "01_concept_figure/fig1_method_concept_v2",
        V2 / "02_case_A/fig2_case_A_3nci_A_v2",
        V2 / "03_case_B/fig3_case_B_6sup_A_v2",
    ]
    for stem in figures:
        for suffix in (".pdf", ".svg", ".png"):
            path = stem.with_suffix(suffix)
            assert path.is_file() and path.stat().st_size > 0
        raster = mpimg.imread(stem.with_suffix(".png"))
        assert raster.shape[0] >= 1500
        assert raster.shape[1] >= 3000

    captions = (V2 / "04_manuscript_assets/figure_captions_v2.md").read_text()
    assert "0.30" in captions and "0.70" in captions
    assert "402-426" in captions and "794-811" in captions and "837-854" in captions
    assert "1649-1652" in captions and "1884-1888" in captions
    assert "approximates weight estimation" in captions


def test_readme_first_screen_and_document_consistency() -> None:
    readme = (ROOT / "README.md").read_text()
    install = readme.index("## Installation")
    for command in (
        "--mode exact",
        "--mode throughput --backend cpu",
        "--mode throughput --backend gpu",
        "mdcorealign local-deformation",
        "mdcorealign apply",
    ):
        assert readme.index(command) < install
    audit = _rows(V2 / "05_product_docs/documentation_consistency_audit.tsv")
    assert audit and all(row["status"] == "pass" for row in audit)
