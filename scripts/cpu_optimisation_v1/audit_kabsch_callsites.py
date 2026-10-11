#!/usr/bin/env python3
"""Inventory fast-Kabsch call sites and their current output requirements."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/cpu_optimisation_v1/03_all_frame_engine/kabsch_callsite_inventory.tsv"
FUNCTIONS = {
    "weighted_kabsch_single",
    "weighted_kabsch_batch",
    "weighted_kabsch_variable_weights_batch",
    "apply_transforms_batch",
}


def classification(function: str, context: str) -> tuple[str, str]:
    if function == "apply_transforms_batch":
        return "aligned-coordinate consumer", "explicit transform application/materialisation"
    lower = context.lower()
    if "rmsf" in lower or "residual" in lower or ".aligned_coords" in context:
        return "residual/RMSF", "uses aligned coordinates for residue-level reduction"
    if ".rmsd" in context or "weighted_rmsd" in lower:
        return "scalar-metrics-only", "uses scalar fit metrics"
    return "transform-only", "no aligned-coordinate access detected in local assignment block"


def main() -> None:
    rows: list[dict[str, object]] = []
    for base in (ROOT / "src/mdsuperpose", ROOT / "src/mdcorealign", ROOT / "tests", ROOT / "scripts"):
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or path == Path(__file__):
                continue
            source = path.read_text(errors="replace")
            lines = source.splitlines()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name not in FUNCTIONS:
                    continue
                start = max(0, node.lineno - 2)
                stop = min(len(lines), (getattr(node, "end_lineno", node.lineno) + 14))
                context = "\n".join(lines[start:stop])
                contract, reason = classification(name, context)
                compute_metrics = "not_explicit"
                for keyword in node.keywords:
                    if keyword.arg == "compute_metrics":
                        compute_metrics = ast.unparse(keyword.value)
                rows.append(
                    {
                        "relative_path": path.relative_to(ROOT).as_posix(),
                        "line": node.lineno,
                        "function": name,
                        "current_compute_metrics": compute_metrics,
                        "consumer_contract": contract,
                        "classification_reason": reason,
                        "migration_status": "audit_before_O2",
                    }
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} call sites")


if __name__ == "__main__":
    main()
