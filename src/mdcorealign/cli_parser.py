"""Argument-parser construction for the MDCoreAlign command-line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__

DEFAULT_CORE_FRACTION = 0.6
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_DELTA = 0.2
DEFAULT_RULE = "huber"
DEFAULT_FIT_SELECTION = "protein and name CA"
TESTED_UPDATE_ROUNDS = frozenset({1, 2, 4, 8})


class _StoreExplicit(argparse.Action):
    """Keep parser defaults distinct from options supplied by the user."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace._explicit_options = (
            *getattr(namespace, "_explicit_options", ()), self.dest
        )


class _BooleanExplicit(argparse.BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        namespace._explicit_options = (
            *getattr(namespace, "_explicit_options", ()), self.dest
        )


def _positive_update_rounds(value: str) -> int:
    """Parse a positive update budget and warn for unvalidated values."""

    try:
        rounds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--fast-update-rounds must be a positive integer"
        ) from exc
    if rounds < 1:
        raise argparse.ArgumentTypeError(
            "--fast-update-rounds must be at least 1"
        )
    if rounds not in TESTED_UPDATE_ROUNDS:
        tested = ", ".join(str(item) for item in sorted(TESTED_UPDATE_ROUNDS))
        print(
            "Warning: --fast-update-rounds="
            f"{rounds} is outside the tested set {{{tested}}}.",
            file=sys.stderr,
        )
    return rounds


class _AdvancedHelpAction(argparse.Action):
    """Display the isolated compatibility and research-option help page."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print(
            "Advanced and legacy align options\n"
            "\n"
            "Legacy modes (accepted for compatibility): default, long, long-fast,\n"
            "alignment-only, fast-global, diagnostic-rich. These are not recommended\n"
            "as first-line product workflows.\n"
            "\n"
            "Advanced controls include --contract, --gpu-precision,\n"
            "--gpu-rotation-solver, --reference-mode, --warm-start, robust-weight\n"
            "parameters, representative-frame budgets, and chunking:\n"
            "  --fast-representative-count: representative frames K (default 256)\n"
            "  --fast-update-rounds: robust update budget M (default 4)\n"
            "Experimental balanced-lite and motion-graph controls are also available. See\n"
            "docs/LEGACY_AND_ADVANCED.md and\n"
            "docs/cli_reference.md before changing them.\n"
        )
        parser.exit()


class _ApplyAdvancedHelpAction(argparse.Action):
    """Display source overrides and legacy export controls separately."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print(
            "Advanced apply options\n"
            "\n"
            "Source relocation: --input, --topology, --trajectory,\n"
            "--topology-format, --trajectory-format. Source hashes and atom order\n"
            "remain mandatory.\n"
            "\n"
            "Selections: --fit-selection, --apply-selection/--export-selection.\n"
            "Frames: --frames, --frame-range, --stride, --all-frames.\n"
            "Streaming: --chunk-size, --drop-unit-cell,\n"
            "--allow-large-text-output.\n"
            "\n"
            "The hidden --result-dir, --transforms, --purpose, --format and --out\n"
            "options are retained for legacy pre-VT02 results and compatibility.\n"
            "Normal VT02.2 use is: mdcorealign apply RESULT_DIR -o OUTPUT.\n"
        )
        parser.exit()


def _add_input_options(
    parser: argparse.ArgumentParser,
    *,
    concise: bool = False,
    advanced_only: bool = False,
) -> None:
    """Add common input options to a subcommand parser."""

    advanced = argparse.SUPPRESS if concise else None
    primary = argparse.SUPPRESS if advanced_only else None

    parser.add_argument("--input", default=None, help=primary or "Legacy single-file input: .npy/.npz array or multi-model PDB")
    parser.add_argument("--topology", default=None, help=primary or "Topology file for topology+trajectory workflows")
    parser.add_argument("--trajectory", default=None, help=primary or "Trajectory file for topology+trajectory workflows")
    parser.add_argument(
        "--input-format",
        action=_StoreExplicit,
        default="auto",
        choices=["auto", "array", "pdb", "ca-pdb"],
        help=primary or "Input parser to use. 'ca-pdb' preserves CA residue labels.",
    )
    parser.add_argument("--topology-format", action=_StoreExplicit, default=None, help=advanced or "Explicit topology format override, for example PDB, GRO, PSF, PRMTOP, TPR")
    parser.add_argument("--trajectory-format", action=_StoreExplicit, default=None, help=advanced or "Explicit trajectory format override, for example XTC, DCD, TRR, NC, PDB")
    parser.add_argument("--array-key", action=_StoreExplicit, default=None, help=advanced or "Key to read from .npz files; defaults to 'coords' or first key")
    parser.add_argument("--chain", action=_StoreExplicit, default=None, help=advanced or "Restrict fit atoms to chain ID; export scope is controlled separately")
    parser.add_argument(
        "--source-unit",
        action=_StoreExplicit,
        default="auto",
        choices=["auto", "angstrom", "nanometer"],
        help=advanced or (
            "Coordinate-unit provenance. Array input is converted only when "
            "explicitly set to 'nanometer'; auto never guesses from geometry."
        ),
    )
    parser.add_argument("--output-dir", default="mdcorealign_output", help=primary or "Directory for generated outputs")
    parser.add_argument(
        "--fit-selection",
        action=_StoreExplicit,
        default=DEFAULT_FIT_SELECTION,
        help=primary or "Atoms used to estimate stability weights and rigid transforms. Supported PDB selections: 'protein and name CA', 'protein and backbone', 'protein', 'all'.",
    )
    parser.add_argument(
        "--apply-selection",
        "--export-selection",
        default=None,
        help=primary or advanced or "Atoms written after applying fitted transforms. Alignment input loading defaults to --fit-selection; saved-result apply defaults to complete protein. Use 'protein' or 'all' to transform additional atoms without using them for weights/RMSF.",
    )


def _add_solver_options(
    parser: argparse.ArgumentParser,
    *,
    concise: bool = False,
    default_mode: str = "default",
) -> None:
    """Add common robust solver options."""

    from . import cli as commands

    advanced = argparse.SUPPRESS if concise else None

    parser.add_argument(
        "--mode",
        action=_StoreExplicit,
        default=default_mode,
        type=commands._parse_align_mode,
        metavar="{throughput,exact}",
        help=(
            "Execution mode: 'throughput' is the production default and uses "
            "fixed-budget shared-weight estimation followed by explicit all-frame fitting; "
            "'exact' is the slower chronological numerical reference/audit. "
            "Run --help-advanced for compatibility modes."
        ),
    )
    parser.add_argument(
        "--contract",
        default=None,
        choices=["exact", "throughput", "balanced-lite"],
        help=advanced or (
            "Advanced execution-contract alias. Prefer --mode throughput for production "
            "or --mode exact for numerical audit; balanced-lite remains experimental."
        ),
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        type=commands._parse_backend,
        metavar="{cpu,gpu}",
        help=(
            "Compute backend. GPU requires the optional CuPy installation and "
            "never silently falls back to CPU."
        ),
    )
    parser.add_argument(
        "--gpu-precision",
        action=_StoreExplicit,
        default="mixed",
        choices=["float64", "float32", "mixed"],
        help=advanced or (
            "GPU-only arithmetic policy; mixed estimates weights in float32 and "
            "performs final all-frame fitting in float64"
        ),
    )
    parser.add_argument(
        "--gpu-rotation-solver",
        action=_StoreExplicit,
        default="horn",
        choices=["svd", "cusolver_svd", "horn"],
        help=advanced or "GPU-only batched rotation solver",
    )
    parser.add_argument("--audit-subset-count", type=int, default=32, help=advanced)
    parser.add_argument("--audit-heldout-count", type=int, default=64, help=advanced)
    parser.add_argument(
        "--allow-limited-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=advanced or "Permit at most one bounded balanced-lite refinement",
    )
    parser.add_argument(
        "--refinement-representative-count",
        type=int,
        default=512,
        help=advanced,
    )
    parser.add_argument("--reference-mode", action=_StoreExplicit, default="first_frame", choices=["first_frame", "previous_frame"], help=advanced)
    parser.add_argument("--warm-start", action=_BooleanExplicit, default=True, help=advanced or "Legacy warm start; exact requires it, throughput uses independent updates")
    parser.add_argument("--rule", action=_StoreExplicit, default=DEFAULT_RULE, choices=["gaussian", "huber", "tukey"], help=advanced or "Robust rule; throughput supports only huber")
    parser.add_argument("--delta", action=_StoreExplicit, type=float, default=DEFAULT_DELTA, help=advanced or "Positive finite robust scale: throughput/exact default 1 A; legacy default 0.2 A")
    parser.add_argument("--max-iterations", action=_StoreExplicit, type=int, default=DEFAULT_MAX_ITERATIONS, help=advanced or "Exact/legacy IRLS limit; throughput uses --fast-update-rounds instead")
    parser.add_argument("--tolerance", action=_StoreExplicit, type=float, default=1e-6, help=advanced)
    parser.add_argument("--sequence-lambda", action=_StoreExplicit, type=float, default=0.0, help=advanced or "Sequence continuity strength [0,1]: throughput/exact default 0.05; legacy default 0")
    parser.add_argument("--contact-lambda", action=_StoreExplicit, type=float, default=0.0, help=advanced or "Optional legacy contact consistency smoothing strength")
    parser.add_argument("--core-fraction", type=float, default=DEFAULT_CORE_FRACTION, help=advanced)
    parser.add_argument("--max-weight-fit-frames", type=int, default=2000, help=advanced or "Maximum representative frames used to estimate long-mode weights")
    parser.add_argument("--frame-selection", default="uniform", choices=["uniform", "random", "stratified_blocks"], help=advanced or "Representative frame selection strategy for long mode")
    parser.add_argument("--random-seed", type=int, default=42, help=advanced or "Seed for random or stratified long-mode frame selection")
    parser.add_argument("--chunk-size", type=int, default=2000, help=advanced or "Frames per chunk for fitting and streaming RMSF")
    parser.add_argument(
        "--ordinary-rmsf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute the optional ordinary/full-set-aligned RMSF comparison. "
            "The primary continuous-weight analysis does not require it."
        ),
    )
    parser.add_argument(
        "--exact-representative-count",
        action=_StoreExplicit,
        type=int,
        default=2000,
        help=advanced or "Maximum representative frames for sequential_lean exact weight estimation",
    )
    parser.add_argument(
        "--fast-representative-count",
        action=_StoreExplicit,
        type=int,
        default=256,
        help=advanced
        or "Representative frames K for fixed-budget weight estimation (default 256)",
    )
    parser.add_argument(
        "--fast-update-rounds",
        action=_StoreExplicit,
        type=_positive_update_rounds,
        default=4,
        help=advanced
        or "Fixed robust update budget M for throughput weight estimation (default 4)",
    )
    parser.add_argument(
        "--descriptor-chunk-size",
        type=int,
        default=2000,
        help=advanced or "Frame chunk size for W5 descriptors",
    )
    parser.add_argument(
        "--local-block-count",
        type=int,
        default=4,
        help=advanced or "Temporal blocks for W5 reproducibility",
    )
    parser.add_argument(
        "--experimental-motion-graph",
        action="store_true",
        help=advanced or "Write soft exploratory co-motion/hinge diagnostics; no mature hard-domain claim",
    )
    if concise:
        parser.add_argument(
            "--help-advanced",
            nargs=0,
            action=_AdvancedHelpAction,
            help="Show legacy modes and research controls",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""

    from . import cli as commands

    parser = argparse.ArgumentParser(
        prog="mdcorealign",
        description="Continuous-weight alignment and flexibility profiling for protein conformational ensembles.",
    )
    parser.add_argument("--version", action="version", version=f"MDCoreAlign {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    align = subparsers.add_parser(
        "align",
        help="Run compact throughput analysis (or exact numerical audit)",
    )
    _add_input_options(align, concise=True)
    _add_solver_options(align, concise=True, default_mode="throughput")
    align.add_argument("--dry-run", action="store_true", help="Validate input selections and write audit files without running alignment")
    align.add_argument(
        "--local-deformation",
        action="store_true",
        help="Add the complementary alignment-free W5 diagnostic",
    )
    align.add_argument(
        "--legacy-artifacts",
        action="store_true",
        help="Also write deprecated flat compatibility artifacts",
    )
    align.add_argument("--out", default=None, help=argparse.SUPPRESS)
    align.add_argument(
        "--materialize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    align.set_defaults(func=commands._run_alignment)

    flexibility = subparsers.add_parser(
        "flexibility",
        help="Compatibility command for the earlier combined profile workflow",
    )
    _add_input_options(flexibility)
    _add_solver_options(flexibility)
    flexibility.add_argument("--system-name", default="toy", help="System label written to segment/summary outputs")
    flexibility.add_argument("--flexible-percentile", type=float, default=0.8)
    flexibility.add_argument("--min-segment-length", type=int, default=3)
    flexibility.add_argument("--merge-gap", type=int, default=1)
    flexibility.set_defaults(func=commands._run_flexibility)

    diagnostics = subparsers.add_parser(
        "diagnostics",
        help="Compatibility command for detailed convergence traces",
    )
    _add_input_options(diagnostics)
    _add_solver_options(diagnostics)
    diagnostics.add_argument("--system-name", default="toy")
    diagnostics.add_argument("--run-id", default="run0")
    diagnostics.add_argument("--trace", action="store_true", help="Write per-iteration trace.tsv")
    diagnostics.set_defaults(func=commands._run_diagnostics)

    audit = subparsers.add_parser("audit-preprocessing", help="Run lightweight trajectory preprocessing sanity checks")
    _add_input_options(audit)
    audit.add_argument("--system-name", default="input")
    audit.add_argument("--jump-threshold", type=float, default=15.0, help="Frame-to-frame displacement threshold in A")
    audit.set_defaults(func=commands._run_audit_preprocessing)

    scale_audit = subparsers.add_parser(
        "audit-coordinate-scale",
        help="Run a read-only chain-aware coordinate-unit integrity audit",
    )
    _add_input_options(scale_audit)
    scale_audit.add_argument("--system-name", default="input")
    scale_audit.set_defaults(func=commands._run_audit_coordinate_scale)

    local = subparsers.add_parser(
        "local-deformation",
        help="Run optional alignment-free chain-aware W5 diagnostics",
    )
    _add_input_options(local)
    local.add_argument("--window-size", type=int, default=5, choices=[5])
    local.add_argument("--output", default=None, help="Output path for local_deformation.tsv")
    local.add_argument("--descriptor-chunk-size", type=int, default=2000)
    local.add_argument("--block-count", type=int, default=4)
    local.add_argument("--deforming-percentile", type=float, default=0.8)
    local.add_argument("--min-segment-length", type=int, default=3)
    local.add_argument("--merge-gap", type=int, default=1)
    local.add_argument("--nonlocal-sample-count", type=int, default=512)
    local.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu", "gpu", "auto"],
    )
    local.add_argument(
        "--gpu-precision",
        default="mixed",
        choices=["float64", "float32", "mixed"],
    )
    local.add_argument(
        "--experimental-motion-graph",
        action="store_true",
        help="Add soft exploratory co-motion/hinge diagnostics",
    )
    local.set_defaults(func=commands._run_local_deformation)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Write annotated coordinates from a saved alignment result",
        description=(
            "Write visualization-ready coordinates without rerunning analysis. "
            "Use `mdcorealign apply RESULT_DIR -o aligned.pdb|aligned.cif|aligned.xtc`."
        ),
        epilog=(
            "PDB/mmCIF occupancy stores the reporting core label and B-factor stores "
            "robust RMSF in Angstrom. XTC also creates an annotated topology companion. "
            "PDB, mmCIF, and XTC are available in the standard installation."
        ),
    )
    apply_parser.add_argument(
        "result",
        nargs="?",
        help="Alignment result directory produced by `mdcorealign align`",
    )
    _add_input_options(apply_parser, concise=True, advanced_only=True)
    apply_parser.add_argument(
        "--transforms",
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--result-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--purpose",
        choices=["structure", "trajectory", "compatibility"],
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["mmcif", "xtc", "pdb"],
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--out",
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path; .pdb/.cif/.mmcif/.xtc selects the export contract",
    )
    apply_parser.add_argument(
        "--frames",
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--frame-range",
        default=None,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument("--stride", type=int, default=1, help=argparse.SUPPRESS)
    apply_parser.add_argument(
        "--all-selected-frames",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--all-frames",
        dest="all_selected_frames",
        action="store_true",
        help="Write all source frames for PDB/mmCIF instead of frame 0",
    )
    apply_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--drop-unit-cell",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--allow-large-text-output",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    apply_parser.add_argument(
        "--help-advanced",
        nargs=0,
        action=_ApplyAdvancedHelpAction,
        help="Show source overrides, frame selection, and legacy controls",
    )
    apply_parser.set_defaults(fit_selection=None)
    apply_parser.set_defaults(func=commands._run_apply)

    formats = subparsers.add_parser("formats", help="Print recommended, experimental, and unsupported input/output formats")
    formats.add_argument("--out", default=None, help="Optional TSV output path for the supported formats table")
    formats.set_defaults(func=commands._run_formats)

    doctor = subparsers.add_parser(
        "doctor",
        help="Report single-product dependencies and CPU/GPU readiness",
        description=(
            "Report single-product dependency versions and CPU/GPU readiness "
            "without changing the environment."
        ),
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit the diagnostic report as JSON",
    )
    doctor.set_defaults(func=commands._run_doctor)
    return parser
