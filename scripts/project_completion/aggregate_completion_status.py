#!/usr/bin/env python3
"""Aggregate completion-pipeline phase status into a TSV table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STATUS_JSON = Path("results/project_completion/status/current_status.json")
OUT = Path("results/project_completion/status/phase_status_summary.tsv")


def main() -> None:
    if not STATUS_JSON.exists():
        raise SystemExit(f"Status file not found: {STATUS_JSON}")
    data = json.loads(STATUS_JSON.read_text())
    rows = []
    for phase, entry in data.get("phases", {}).items():
        rows.append(
            {
                "phase": phase,
                "status": entry.get("status"),
                "started_at": entry.get("started_at"),
                "ended_at": entry.get("ended_at"),
                "message": entry.get("message"),
                "n_outputs": len(entry.get("outputs", [])),
                "n_warnings": len(entry.get("warnings", [])),
                "n_errors": len(entry.get("errors", [])),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
