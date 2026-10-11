"""Profile the current mdsuperpose implementation and report bottlenecks."""

from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
from dataclasses import dataclass
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mdsuperpose.benchmark as benchmark_module
import mdsuperpose.robust as robust_module
from mdsuperpose import Trajectory, benchmark_soft_weight_robust_alignment

from mdstablecore.trajectory_io import load_mdanalysis_trajectory


NAMED_DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
    },
    "adk_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
    },
}


@dataclass(slots=True)
class ComponentStat:
    """Aggregate timing information for one profiled component."""

    total_seconds: float = 0.0
    calls: int = 0

    def add(self, elapsed_seconds: float) -> None:
        """Accumulate one timing sample."""

        self.total_seconds += float(elapsed_seconds)
        self.calls += 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the profile run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-dataset", choices=sorted(NAMED_DATASETS), default="ifabp")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--rule", type=str, default="huber")
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--sequence-continuity-strength", type=float, default=0.0)
    parser.add_argument("--sequence-continuity-passes", type=int, default=1)
    parser.add_argument("--contact-consistency-strength", type=float, default=0.0)
    parser.add_argument("--contact-consistency-passes", type=int, default=1)
    parser.add_argument("--contact-distance-cutoff", type=float, default=8.0)
    parser.add_argument("--contact-min-sequence-separation", type=int, default=2)
    parser.add_argument("--contact-max-neighbors", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_profiles")
    return parser.parse_args()


def load_named_trajectory(args: argparse.Namespace) -> Trajectory:
    """Load one named real trajectory for profiling."""

    dataset = NAMED_DATASETS[args.named_dataset]
    coords = load_mdanalysis_trajectory(
        topology_path=Path(dataset["topology"]),
        trajectory_path=Path(dataset["trajectory"]),
        selection=str(dataset["selection"]),
        stride=args.stride,
        max_frames=args.max_frames,
    ).coords
    return Trajectory(coords=coords)


@contextmanager
def instrument_components(component_stats: dict[str, ComponentStat]):
    """Temporarily instrument key implementation components."""

    originals = {
        "alignment_solver": robust_module.solve_weighted_rigid_alignment,
        "residual_computation": robust_module.residual_distances,
        "weight_update_dispatch": robust_module.update_irls_weights,
        "weight_update_normalize": robust_module.normalize_weights,
        "sequence_regularization": robust_module.apply_sequence_continuity_regularization,
        "contact_regularization": robust_module.apply_contact_consistency_regularization,
        "contact_graph_build": robust_module.build_contact_graph,
        "hard_core_extraction": benchmark_module.select_core_by_target_fraction,
    }

    def wrap(label: str, fn):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            component_stats[label].add(time.perf_counter() - start)
            return result

        return wrapped

    robust_module.solve_weighted_rigid_alignment = wrap("alignment_solver", originals["alignment_solver"])
    robust_module.residual_distances = wrap("residual_computation", originals["residual_computation"])
    robust_module.update_irls_weights = wrap("weight_update", originals["weight_update_dispatch"])
    robust_module.normalize_weights = wrap("weight_update", originals["weight_update_normalize"])
    robust_module.apply_sequence_continuity_regularization = wrap(
        "regularization",
        originals["sequence_regularization"],
    )
    robust_module.apply_contact_consistency_regularization = wrap(
        "regularization",
        originals["contact_regularization"],
    )
    robust_module.build_contact_graph = wrap("regularization", originals["contact_graph_build"])
    benchmark_module.select_core_by_target_fraction = wrap("hard_core_extraction", originals["hard_core_extraction"])

    try:
        yield
    finally:
        robust_module.solve_weighted_rigid_alignment = originals["alignment_solver"]
        robust_module.residual_distances = originals["residual_computation"]
        robust_module.update_irls_weights = originals["weight_update_dispatch"]
        robust_module.normalize_weights = originals["weight_update_normalize"]
        robust_module.apply_sequence_continuity_regularization = originals["sequence_regularization"]
        robust_module.apply_contact_consistency_regularization = originals["contact_regularization"]
        robust_module.build_contact_graph = originals["contact_graph_build"]
        benchmark_module.select_core_by_target_fraction = originals["hard_core_extraction"]


def build_profile_payload(
    *,
    args: argparse.Namespace,
    trajectory: Trajectory,
    component_stats: dict[str, ComponentStat],
    total_runtime_seconds: float,
    benchmark_result,
    cprofile_text: str,
) -> dict[str, object]:
    """Build a JSON-serializable profiling payload."""

    measured_sum = sum(stat.total_seconds for stat in component_stats.values())
    other_seconds = max(0.0, total_runtime_seconds - measured_sum)
    payload = {
        "dataset": args.named_dataset,
        "n_frames": trajectory.n_frames,
        "n_atoms": trajectory.n_atoms,
        "configuration": {
            "fraction": args.fraction,
            "rule": args.rule,
            "delta": args.delta,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "sequence_continuity_strength": args.sequence_continuity_strength,
            "sequence_continuity_passes": args.sequence_continuity_passes,
            "contact_consistency_strength": args.contact_consistency_strength,
            "contact_consistency_passes": args.contact_consistency_passes,
            "contact_distance_cutoff": args.contact_distance_cutoff,
            "contact_min_sequence_separation": args.contact_min_sequence_separation,
            "contact_max_neighbors": args.contact_max_neighbors,
            "stride": args.stride,
            "max_frames": args.max_frames,
        },
        "total_runtime_seconds": total_runtime_seconds,
        "benchmark_result": {
            "method_name": benchmark_result.method_name,
            "runtime_seconds": benchmark_result.runtime_seconds,
            "weighted_rmsd": benchmark_result.weighted_rmsd,
            "rmsd": benchmark_result.rmsd,
            "iteration_count": benchmark_result.iteration_count,
            "core_size": benchmark_result.core_size,
            "core_fraction": benchmark_result.core_fraction,
        },
        "component_breakdown": {
            name: {
                "seconds": stat.total_seconds,
                "calls": stat.calls,
                "fraction_of_total": 0.0 if total_runtime_seconds <= 0.0 else stat.total_seconds / total_runtime_seconds,
            }
            for name, stat in component_stats.items()
        },
        "other_overhead_seconds": other_seconds,
        "other_overhead_fraction": 0.0 if total_runtime_seconds <= 0.0 else other_seconds / total_runtime_seconds,
        "cprofile_top": cprofile_text,
    }
    return payload


def main() -> None:
    """Run one component-level profile and save machine-readable outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_named_trajectory(args)
    component_stats = {
        "alignment_solver": ComponentStat(),
        "residual_computation": ComponentStat(),
        "weight_update": ComponentStat(),
        "regularization": ComponentStat(),
        "hard_core_extraction": ComponentStat(),
    }

    profile = cProfile.Profile()
    with instrument_components(component_stats):
        total_start = time.perf_counter()
        profile.enable()
        result = benchmark_soft_weight_robust_alignment(
            trajectory,
            core_fraction=args.fraction,
            rule=args.rule,
            delta=args.delta,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            sequence_continuity_strength=args.sequence_continuity_strength,
            sequence_continuity_passes=args.sequence_continuity_passes,
            contact_consistency_strength=args.contact_consistency_strength,
            contact_consistency_passes=args.contact_consistency_passes,
            contact_distance_cutoff=args.contact_distance_cutoff,
            contact_min_sequence_separation=args.contact_min_sequence_separation,
            contact_max_neighbors=args.contact_max_neighbors,
        )
        profile.disable()
        total_runtime_seconds = time.perf_counter() - total_start

    stats_stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stats_stream).sort_stats("cumtime")
    stats.print_stats("mdsuperpose", 25)
    stats.print_stats("numpy", 15)
    cprofile_text = stats_stream.getvalue()

    payload = build_profile_payload(
        args=args,
        trajectory=trajectory,
        component_stats=component_stats,
        total_runtime_seconds=total_runtime_seconds,
        benchmark_result=result,
        cprofile_text=cprofile_text,
    )

    prefix = args.output_dir / (
        f"profile_{args.named_dataset}_seq{args.sequence_continuity_strength:.2f}"
        f"_contact{args.contact_consistency_strength:.2f}"
    )
    json_path = Path(f"{prefix}.json")
    txt_path = Path(f"{prefix}_cprofile.txt")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    txt_path.write_text(cprofile_text)

    print("Profiling complete")
    print(f"dataset={args.named_dataset}")
    print(f"frames={trajectory.n_frames}")
    print(f"atoms={trajectory.n_atoms}")
    print(f"total_runtime_seconds={total_runtime_seconds:.6f}")
    for name, stat in component_stats.items():
        fraction = 0.0 if total_runtime_seconds <= 0.0 else 100.0 * stat.total_seconds / total_runtime_seconds
        print(f"{name}: seconds={stat.total_seconds:.6f}, calls={stat.calls}, percent={fraction:.2f}")
    print(f"other_overhead_seconds={payload['other_overhead_seconds']:.6f}")
    print(f"json_path={json_path}")
    print(f"cprofile_path={txt_path}")


if __name__ == "__main__":
    main()
