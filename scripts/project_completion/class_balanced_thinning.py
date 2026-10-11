#!/usr/bin/env python3
"""Create class-balanced mdCATH thinning analyses for MDCoreAlign claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260625


def _paired_success(df: pd.DataFrame) -> pd.DataFrame:
    """Return paired MDCoreAlign-MDLovoFit success rows."""

    mask = df["mdcorealign_status"].eq("success") & df["mdlovofit_status"].eq("success")
    out = df.loc[mask].copy()
    out["cath_class"] = out["cath_class"].astype("Int64")
    return out.dropna(subset=["cath_class"])


def _diverse_select(group: pd.DataFrame, n: int, *, rng: np.random.Generator) -> pd.DataFrame:
    """Greedily select a deterministic diverse subset within one CATH class."""

    if len(group) <= n:
        return group.copy()
    shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    selected: list[int] = []
    seen_arch: set[object] = set()
    seen_topology: set[object] = set()
    seen_hsf: set[object] = set()
    for idx, row in shuffled.iterrows():
        novelty = (
            row.get("cath_architecture") not in seen_arch
            or row.get("cath_topology") not in seen_topology
            or row.get("cath_homologous_superfamily") not in seen_hsf
        )
        if novelty or len(selected) + (len(shuffled) - len(selected)) <= n:
            selected.append(idx)
            seen_arch.add(row.get("cath_architecture"))
            seen_topology.add(row.get("cath_topology"))
            seen_hsf.add(row.get("cath_homologous_superfamily"))
        if len(selected) == n:
            break
    if len(selected) < n:
        remaining = [idx for idx in shuffled.index if idx not in set(selected)]
        selected.extend(remaining[: n - len(selected)])
    return group.loc[selected].copy()


def _make_subset(df: pd.DataFrame, n_per_class: int, *, rng: np.random.Generator) -> pd.DataFrame:
    """Select n_per_class paired-success domains from each CATH class."""

    parts = []
    for _, group in df.groupby("cath_class", sort=True):
        parts.append(_diverse_select(group, n_per_class, rng=rng))
    return pd.concat(parts, ignore_index=True)


def _summary_row(name: str, subset: pd.DataFrame, n_per_class: int, notes: str) -> dict[str, object]:
    """Summarize a thinned subset."""

    speedup = subset["mdlovofit_runtime_sec"] / subset["mdcorealign_runtime_sec"]
    return {
        "subset_name": name,
        "n_total": int(len(subset)),
        "n_per_class": int(n_per_class),
        "speedup_vs_MDLovoFit": float(np.nanmedian(speedup)),
        "percent_MDCoreAlign_faster": float(np.nanmean(subset["mdcorealign_faster_than_mdlovofit"].astype(float)) * 100.0),
        "percent_MDCoreAlign_lower_RMSD": float(
            np.nanmean(subset["mdcorealign_lower_RMSD_than_mdlovofit"].astype(float)) * 100.0
        ),
        "median_RMSD_delta": float(np.nanmedian(subset["RMSD_delta_mdcorealign_minus_mdlovofit"])),
        "median_runtime_delta": float(np.nanmedian(subset["runtime_delta_mdcorealign_minus_mdlovofit"])),
        "notes": notes,
    }


def run(input_path: Path, output_dir: Path) -> None:
    """Create the class-balanced thinning output tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = _paired_success(pd.read_csv(input_path, sep="\t"))
    rng = np.random.default_rng(SEED)
    counts = df.groupby("cath_class").size()
    max_equal = int(counts.min())
    n_40 = min(40, max_equal)

    subset_40 = _make_subset(df, n_40, rng=rng)
    subset_max = _make_subset(df, max_equal, rng=rng)

    subset_40.to_csv(output_dir / "balanced_thinning_40_per_class.tsv", sep="\t", index=False)
    subset_max.to_csv(output_dir / "balanced_thinning_max_equal_class.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            _summary_row(
                "balanced_40_per_class",
                subset_40,
                n_40,
                "Uses 40 per class when all classes have at least 40 paired-success domains; otherwise uses the maximum feasible equal count.",
            ),
            _summary_row(
                "balanced_max_equal_class",
                subset_max,
                max_equal,
                "Uses the maximum equal per-class count available among paired MDCoreAlign-MDLovoFit success rows.",
            ),
        ]
    )
    summary.to_csv(output_dir / "balanced_thinning_summary.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/mdcath_benchmark/all_domain_canonical/aggregate/all_domain_canonical_per_domain_summary.tsv",
    )
    parser.add_argument("--output-dir", default="results/project_completion/tables")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
