"""Run repeated formal structure-pair benchmarks including external THESEUS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import benchmark_result_to_record, benchmark_results_to_rows, summarize_benchmark_rows, write_benchmark_csv, write_benchmark_json
from benchmark.real_structure_pair_benchmark import benchmark_pair_methods


PAIR_DATASETS = {
    "ifabp_first_last_ca": "ifabp_first_last_ca_pair",
    "adk_first_last_ca": "adk_first_last_ca_pair",
    "nhaa_first_last_ca": "nhaa_equilibrium_first_last_ca_pair",
    "yiip_short_first_last_ca": "yiip_equilibrium_short_first_last_ca_pair",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the formal THESEUS structure-pair run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=sorted(PAIR_DATASETS),
        default=sorted(PAIR_DATASETS),
    )
    parser.add_argument("--fraction", type=float, default=0.60)
    parser.add_argument("--rule", type=str, default="huber")
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "higher_tier_external_baseline_theseus_structure_pairs_formal",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/dev/shm/mdca_external_baseline_theseus_structure_pairs_formal"),
    )
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    parser.add_argument("--theseus-executable", type=Path, default=None)
    parser.add_argument("--theseus-config", type=Path, default=None)
    return parser.parse_args()


def build_note(summary_rows: list[dict[str, object]]) -> str:
    """Build a concise markdown note for the formal THESEUS structure-pair benchmark."""

    lines = [
        "# Formal THESEUS Structure-Pair Summary",
        "",
        "This benchmark compares:",
        "",
        "- ordinary alignment pair",
        "- soft robust alignment pair",
        "- external MDLovoFit pair",
        "- external THESEUS pair",
        "",
        "Key summary rows:",
        "",
    ]
    for row in summary_rows:
        dataset_name = row["dataset_name"]
        method_name = row["method_name"]
        runtime = row.get("runtime_seconds_mean")
        rmsd = row.get("unified_posthoc_rmsd_mean")
        if runtime is not None and rmsd is not None:
            lines.append(f"- {dataset_name} / {method_name}: runtime={runtime:.4f}s, unified_posthoc_rmsd={rmsd:.4f}")
        else:
            lines.append(f"- {dataset_name} / {method_name}")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- unified post-hoc RMSD remains the primary cross-method metric",
            "- MDLovoFit-native and THESEUS-native metrics are reported separately when available",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run formal repeated structure-pair benchmarks including THESEUS."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    for pair_name in args.pairs:
        dataset_name = PAIR_DATASETS[pair_name]
        pair_output_dir = args.output_dir / pair_name
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        pair_rows: list[dict[str, object]] = []
        pair_records: list[dict[str, object]] = []

        for repeat_index in range(args.repeats):
            namespace = argparse.Namespace(
                named_pair=pair_name,
                example=False,
                mobile_pdb=None,
                reference_pdb=None,
                mobile_npy=None,
                reference_npy=None,
                dataset_name=dataset_name,
                fraction=args.fraction,
                rule=args.rule,
                delta=args.delta,
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                output_dir=pair_output_dir,
                workdir=args.workdir / pair_name / f"repeat{repeat_index}",
                mdlovofit_executable=args.mdlovofit_executable,
                mdlovofit_config=args.mdlovofit_config,
                theseus_executable=args.theseus_executable,
                theseus_config=args.theseus_config,
            )
            results, preprocessing_steps = benchmark_pair_methods(namespace)
            rows = benchmark_results_to_rows(
                results,
                benchmark_mode="external_baseline_theseus_structure_pair",
                dataset_name=dataset_name,
                repeat_index=repeat_index,
                preprocessing_steps=preprocessing_steps,
            )
            pair_rows.extend(rows)
            pair_records.append(
                {
                    "benchmark_mode": "external_baseline_theseus_structure_pair",
                    "dataset_name": dataset_name,
                    "repeat_index": repeat_index,
                    "preprocessing_steps": preprocessing_steps,
                    "results": [benchmark_result_to_record(result) for result in results],
                }
            )

        pair_summary = summarize_benchmark_rows(pair_rows)
        write_benchmark_csv(pair_output_dir / f"{dataset_name}_theseus_pair_formal_runs.csv", pair_rows)
        write_benchmark_csv(pair_output_dir / f"{dataset_name}_theseus_pair_formal_summary.csv", pair_summary)
        write_benchmark_json(
            pair_output_dir / f"{dataset_name}_theseus_pair_formal_runs.json",
            {
                "benchmark_mode": "external_baseline_theseus_structure_pair",
                "dataset_name": dataset_name,
                "fraction": args.fraction,
                "repeats": args.repeats,
                "runs": pair_records,
                "summary_rows": pair_summary,
            },
        )

        all_rows.extend(pair_rows)
        all_records.extend(pair_records)

    combined_summary = summarize_benchmark_rows(all_rows)
    write_benchmark_csv(args.output_dir / "combined_theseus_structure_pair_formal_runs.csv", all_rows)
    write_benchmark_csv(args.output_dir / "combined_theseus_structure_pair_formal_summary.csv", combined_summary)
    write_benchmark_json(
        args.output_dir / "combined_theseus_structure_pair_formal_runs.json",
        {
            "benchmark_mode": "external_baseline_theseus_structure_pair",
            "pairs": args.pairs,
            "fraction": args.fraction,
            "repeats": args.repeats,
            "runs": all_records,
            "summary_rows": combined_summary,
        },
    )
    (args.output_dir / "theseus_structure_pair_formal_note.md").write_text(build_note(combined_summary))

    print("Formal THESEUS structure-pair benchmark complete")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
