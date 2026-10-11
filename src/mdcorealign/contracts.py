"""Explicit execution contracts for exact, throughput and diagnostic use."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mdsuperpose.fast_kabsch import (
    weighted_kabsch_batch,
)
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_mean,
)
from mdsuperpose.window_kinematics import (
    LocalDeformationResult,
    MotionDomainResult,
    scan_local_kinematics,
    scan_motion_domains,
)
from mdsuperpose.window_refine import (
    bounded_global_weight_refinement,
    exact_all_frame_fit,
    temporally_stratified_indices,
    validate_weighting_parameters,
)

from .experimental import (
    FastGlobalAlignmentResult,
    _core_mask,
    _jaccard,
    _spearman,
    solve_fast_global_alignment,
)

ExecutionContract = Literal[
    "exact",
    "throughput",
    "balanced-lite",
    "local-deformation",
]
BackendName = Literal["cpu", "gpu", "auto"]


@dataclass(slots=True)
class ExactAlignmentResult:
    """Exact sequential-weight estimate followed by fixed-weight all-frame fit."""

    mode: str
    frozen_weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray | None
    robust_rmsf: np.ndarray
    ordinary_rmsf: np.ndarray | None
    mean_residual: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    total_runtime_seconds: float
    timing: dict[str, float]
    metadata: dict[str, object]
    average_iterations_per_frame: float
    converged_fraction: float

    @property
    def mean_rmsd(self) -> float:
        return float(np.mean(self.rmsd))

    @property
    def mean_weighted_rmsd(self) -> float:
        return float(np.mean(self.weighted_rmsd))

    @property
    def n_frames(self) -> int:
        """Return the number of fitted frames."""

        return int(self.rmsd.size)

    @property
    def n_atoms(self) -> int:
        """Return the number of fitted atoms or residues."""

        return int(self.frozen_weights.size)

    @property
    def reference_mode(self) -> str:
        """Return the locked reference convention."""

        return "first_frame"

    @property
    def warm_start(self) -> bool:
        """Report the exact estimator's sequential warm-start semantics."""

        return True

    @property
    def average_runtime_per_frame(self) -> float:
        """Return total runtime divided by frame count."""

        return self.total_runtime_seconds / max(1, self.n_frames)

@dataclass(frozen=True, slots=True)
class SampledQualityAudit:
    """Fixed-cost quality audit computed from bounded frame samples."""

    quality_status: str
    warning: str | None
    core_jaccard_subset_a_b: float
    weight_spearman_subset_a_b: float
    candidate_core_jaccard_min: float
    candidate_weight_spearman_min: float
    effective_core_relative_difference: float
    heldout_residual_quantile_relative_difference: float
    subset_a_indices: np.ndarray
    subset_b_indices: np.ndarray
    heldout_indices: np.ndarray
    runtime_seconds: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "quality_status": self.quality_status,
            "warning": self.warning,
            "core_jaccard_subset_a_b": self.core_jaccard_subset_a_b,
            "weight_spearman_subset_a_b": self.weight_spearman_subset_a_b,
            "candidate_core_jaccard_min": self.candidate_core_jaccard_min,
            "candidate_weight_spearman_min": (
                self.candidate_weight_spearman_min
            ),
            "effective_core_relative_difference": (
                self.effective_core_relative_difference
            ),
            "heldout_residual_quantile_relative_difference": (
                self.heldout_residual_quantile_relative_difference
            ),
            "subset_a_count": int(self.subset_a_indices.size),
            "subset_b_count": int(self.subset_b_indices.size),
            "heldout_count": int(self.heldout_indices.size),
            "runtime_seconds": self.runtime_seconds,
            **self.metadata,
        }


@dataclass(slots=True)
class ContractExecutionResult:
    """One result with an explicit scientific and computational contract."""

    contract: str
    backend_requested: str
    backend_used: str
    alignment: ExactAlignmentResult | FastGlobalAlignmentResult | object | None
    local_deformation: LocalDeformationResult | object | None
    motion_graph: MotionDomainResult | None
    quality_status: str
    quality_warning: str | None
    sampled_audit: SampledQualityAudit | None
    refinement_performed: bool
    total_runtime_seconds: float
    timing: dict[str, float]
    metadata: dict[str, object]


def _validate_coords_and_chains(
    coords: np.ndarray,
    chain_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("coords requires at least one frame and 3 residues")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must have shape (n_residues,)")
    return values, chains


def solve_exact_contract(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 2000,
    chunk_size: int = 2000,
    store_aligned: bool = False,
    compute_ordinary_rmsf: bool = True,
    delta: float = 1.0,
    rule: str = "huber",
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    sequence_continuity_strength: float = 0.05,
    contact_consistency_strength: float = 0.0,
) -> ExactAlignmentResult:
    """Run the exact CPU audit/reference path."""

    total_start = time.perf_counter()
    requested = {
        "delta": delta, "rule": rule, "max_iterations": max_iterations,
        "tolerance": tolerance, "sequence_continuity_strength": sequence_continuity_strength,
        "contact_consistency_strength": contact_consistency_strength,
    }
    resolve_contract_parameters("exact", requested)
    values, chains = _validate_coords_and_chains(coords, chain_ids)
    selected = temporally_stratified_indices(
        values.shape[0],
        min(int(representative_count), values.shape[0]),
    )
    weight_start = time.perf_counter()
    estimate = estimate_representative_weights_sequential_mean(
        values[selected],
        values[0],
        rule=rule,
        delta=delta,
        sigma=delta,
        cutoff=delta,
        max_iterations=max_iterations,
        tolerance=tolerance,
        objective_tolerance=1e-5,
        objective_patience=2,
        sequence_continuity_strength=sequence_continuity_strength,
        contact_consistency_strength=contact_consistency_strength,
        sequence_chain_ids=chains,
        record_history=False,
    )
    weight_seconds = time.perf_counter() - weight_start
    weights = np.asarray(estimate.mean_weights, dtype=np.float64)
    fit_start = time.perf_counter()
    fit = exact_all_frame_fit(
        values,
        values[0],
        weights,
        chunk_size=chunk_size,
        return_aligned=store_aligned,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
    )
    fit_seconds = time.perf_counter() - fit_start
    assert fit.robust_rmsf is not None
    assert fit.mean_residual is not None
    assert fit.rmsd is not None
    assert fit.weighted_rmsd is not None
    total_seconds = time.perf_counter() - total_start
    return ExactAlignmentResult(
        mode="exact",
        frozen_weights=weights,
        rotations=fit.rotations,
        translations=fit.translations,
        aligned_coords=fit.aligned_coords,
        robust_rmsf=fit.robust_rmsf,
        ordinary_rmsf=fit.ordinary_rmsf,
        mean_residual=fit.mean_residual,
        rmsd=fit.rmsd,
        weighted_rmsd=fit.weighted_rmsd,
        total_runtime_seconds=total_seconds,
        timing={
            "weight_estimation": weight_seconds,
            "all_frame_fit_metrics_RMSF": fit_seconds,
            "aligned_coordinate_materialization": fit.timing[
                "aligned_output_materialization"
            ],
            "total_runtime_seconds": total_seconds,
        },
        metadata={
            "contract": "exact",
            "requested_parameters": requested,
            "backend": "cpu",
            "weight_estimator": "sequential_lean",
            "retention_mode": "mean_only",
            "warm_start_semantics": "chronological_sequential_lean",
            "public_default_changed": False,
            "representative_count": int(selected.size),
            "max_iterations_reached_count": (
                estimate.max_iterations_reached_count
            ),
            "mean_final_weight_change": estimate.mean_final_weight_change,
            "max_final_weight_change": estimate.max_final_weight_change,
            "mean_final_transform_change": estimate.mean_final_transform_change,
            "max_final_transform_change": estimate.max_final_transform_change,
            "ordinary_rmsf_computed": compute_ordinary_rmsf,
            "effective_parameters": {
                "delta": float(delta), "rule": rule,
                "max_iterations": int(max_iterations), "tolerance": float(tolerance),
                "sequence_continuity_strength": float(sequence_continuity_strength),
                "contact_consistency_strength": float(contact_consistency_strength),
                "warm_start": True,
                "weight_scale_parameter": {"huber": "delta", "gaussian": "sigma", "tukey": "cutoff"}[rule],
                "weight_scale": float(delta),
            },
        },
        average_iterations_per_frame=estimate.average_iterations_per_frame,
        converged_fraction=estimate.converged_fraction,
    )


def _offset_stratified_indices(
    n_frames: int,
    count: int,
    offset: float,
) -> np.ndarray:
    count = min(max(1, int(count)), int(n_frames))
    positions = (np.arange(count, dtype=np.float64) + offset) / count
    return np.unique(
        np.minimum((positions * n_frames).astype(np.int64), n_frames - 1)
    )


def _estimate_independent_weights(
    coords: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    chain_ids: np.ndarray | None,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
) -> np.ndarray:
    weights, _, _, _ = bounded_global_weight_refinement(
        coords[indices],
        reference,
        np.ones(coords.shape[1], dtype=np.float64),
        update_rounds=4,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        sequence_chain_ids=chain_ids,
    )
    return np.asarray(weights, dtype=np.float64)


def _heldout_residual_quantiles(
    coords: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fit = weighted_kabsch_batch(
        coords[indices],
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=False,
    )
    residuals = np.linalg.norm(
        fit.aligned_coords - reference[None, :, :],
        axis=2,
    )
    return np.asarray(
        np.quantile(residuals, [0.5, 0.9, 0.95]),
        dtype=np.float64,
    )


def run_fixed_cost_sampled_audit(
    coords: np.ndarray,
    candidate_weights: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    subset_count: int = 32,
    heldout_count: int = 64,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
) -> SampledQualityAudit:
    """Audit a candidate with frame counts bounded independently of trajectory length."""

    start = time.perf_counter()
    values, chains = _validate_coords_and_chains(coords, chain_ids)
    candidate = np.asarray(candidate_weights, dtype=np.float64)
    if candidate.shape != (values.shape[1],):
        raise ValueError("candidate_weights must match n_residues")
    subset_a = _offset_stratified_indices(
        values.shape[0],
        subset_count,
        0.2,
    )
    subset_b = _offset_stratified_indices(
        values.shape[0],
        subset_count,
        0.8,
    )
    heldout = _offset_stratified_indices(
        values.shape[0],
        heldout_count,
        0.5,
    )
    weights_a = _estimate_independent_weights(
        values,
        values[0],
        subset_a,
        chains,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
    )
    weights_b = _estimate_independent_weights(
        values,
        values[0],
        subset_b,
        chains,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
    )
    core_a = _core_mask(weights_a)
    core_b = _core_mask(weights_b)
    candidate_core = _core_mask(candidate)
    effective_a = float(np.sum(weights_a) / max(np.max(weights_a), 1e-12))
    effective_b = float(np.sum(weights_b) / max(np.max(weights_b), 1e-12))
    effective_difference = abs(effective_a - effective_b) / max(
        0.5 * (effective_a + effective_b),
        1e-12,
    )
    quantiles_a = _heldout_residual_quantiles(
        values,
        values[0],
        heldout,
        weights_a,
    )
    quantiles_b = _heldout_residual_quantiles(
        values,
        values[0],
        heldout,
        weights_b,
    )
    residual_difference = float(
        np.max(
            np.abs(quantiles_a - quantiles_b)
            / np.maximum(0.5 * (quantiles_a + quantiles_b), 1e-12)
        )
    )
    subset_core = _jaccard(core_a, core_b)
    subset_weight = _spearman(weights_a, weights_b)
    candidate_core_min = min(
        _jaccard(candidate_core, core_a),
        _jaccard(candidate_core, core_b),
    )
    candidate_weight_min = min(
        _spearman(candidate, weights_a),
        _spearman(candidate, weights_b),
    )
    if (
        subset_core >= 0.90
        and subset_weight >= 0.97
        and candidate_core_min >= 0.90
        and candidate_weight_min >= 0.97
        and effective_difference <= 0.10
        and residual_difference <= 0.10
    ):
        status = "high_confidence"
        warning = None
    elif (
        subset_core >= 0.80
        and subset_weight >= 0.90
        and candidate_core_min >= 0.80
        and candidate_weight_min >= 0.90
        and effective_difference <= 0.20
        and residual_difference <= 0.25
    ):
        status = "moderate_confidence"
        warning = (
            "Sampled audit is moderate; consider exact verification for "
            "high-stakes interpretation."
        )
    else:
        status = "low_confidence"
        warning = (
            "Sampled audit found unstable weights/core or held-out residuals. "
            "Run the exact contract explicitly before scientific interpretation."
        )
    return SampledQualityAudit(
        quality_status=status,
        warning=warning,
        core_jaccard_subset_a_b=subset_core,
        weight_spearman_subset_a_b=subset_weight,
        candidate_core_jaccard_min=candidate_core_min,
        candidate_weight_spearman_min=candidate_weight_min,
        effective_core_relative_difference=effective_difference,
        heldout_residual_quantile_relative_difference=residual_difference,
        subset_a_indices=subset_a,
        subset_b_indices=subset_b,
        heldout_indices=heldout,
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "audit_cost_scales_with_all_frames": False,
            "full_exact_fallback_performed": False,
            "thresholds": "balanced-lite-audit-v1",
        },
    )


def solve_balanced_lite_cpu(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    update_rounds: int = 4,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    audit_subset_count: int = 32,
    audit_heldout_count: int = 64,
    allow_limited_refinement: bool = True,
    refinement_representative_count: int = 512,
    include_w5_on_low_confidence: bool = False,
    chunk_size: int = 2000,
    store_aligned: bool = False,
) -> ContractExecutionResult:
    """Run throughput estimation, a fixed-cost audit and one bounded refinement."""

    total_start = time.perf_counter()
    values, chains = _validate_coords_and_chains(coords, chain_ids)
    first = solve_fast_global_alignment(
        values,
        chain_ids=chains,
        representative_count=representative_count,
        update_rounds=update_rounds,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        chunk_size=chunk_size,
        store_aligned=store_aligned,
        compute_confidence=False,
        collect_risk_signals=True,
    )
    audit = run_fixed_cost_sampled_audit(
        values,
        first.frozen_weights,
        chain_ids=chains,
        subset_count=audit_subset_count,
        heldout_count=audit_heldout_count,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
    )
    audit_seconds = audit.runtime_seconds
    final = first
    refinement_performed = False
    refinement_seconds = 0.0
    if allow_limited_refinement and audit.quality_status != "high_confidence":
        refinement_start = time.perf_counter()
        final = solve_fast_global_alignment(
            values,
            chain_ids=chains,
            representative_count=min(
                refinement_representative_count,
                values.shape[0],
            ),
            update_rounds=update_rounds,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
            chunk_size=chunk_size,
            store_aligned=store_aligned,
            compute_confidence=False,
            collect_risk_signals=True,
        )
        refinement_seconds = time.perf_counter() - refinement_start
        refinement_performed = True
        audit = run_fixed_cost_sampled_audit(
            values,
            final.frozen_weights,
            chain_ids=chains,
            subset_count=audit_subset_count,
            heldout_count=audit_heldout_count,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
        audit_seconds += audit.runtime_seconds
    local = None
    w5_seconds = 0.0
    if (
        include_w5_on_low_confidence
        and audit.quality_status != "high_confidence"
    ):
        w5_start = time.perf_counter()
        local = scan_local_kinematics(
            values,
            chain_ids=chains,
            evaluate_blocks=False,
            descriptor_chunk_size=chunk_size,
        )
        w5_seconds = time.perf_counter() - w5_start
    total_seconds = time.perf_counter() - total_start
    return ContractExecutionResult(
        contract="balanced-lite",
        backend_requested="cpu",
        backend_used="cpu",
        alignment=final,
        local_deformation=local,
        motion_graph=None,
        quality_status=audit.quality_status,
        quality_warning=audit.warning,
        sampled_audit=audit,
        refinement_performed=refinement_performed,
        total_runtime_seconds=total_seconds,
        timing={
            "level1_throughput": first.total_runtime_seconds,
            "sampled_audit": audit_seconds,
            "limited_refinement": refinement_seconds,
            "conditional_W5": w5_seconds,
            "full_exact_fallback": 0.0,
            "total_runtime_seconds": total_seconds,
        },
        metadata={
            "status": "experimental",
            "automatic_exact_fallback": False,
            "maximum_refinement_stages": 1,
            "exact_verification_recommended": (
                audit.quality_status != "high_confidence"
            ),
        },
    )


def resolve_contract_parameters(contract: str, options: dict[str, object]) -> dict[str, object]:
    """Validate explicit public options without mistaking parser defaults for requests."""

    resolved = dict(options)
    weighting_keys = {
        "delta", "rule", "sequence_continuity_strength", "contact_consistency_strength",
        "max_iterations", "tolerance", "warm_start", "reference_mode", "update_rounds",
    }
    if contract == "local-deformation":
        invalid = weighting_keys.intersection(resolved)
        if invalid:
            raise ValueError(f"local-deformation does not support weighting options: {sorted(invalid)}")
        return resolved
    if contract not in {"exact", "throughput", "balanced-lite"}:
        raise ValueError(f"unsupported execution contract: {contract}")
    if resolved.pop("reference_mode", "first_frame") != "first_frame":
        raise ValueError(f"{contract} requires reference_mode='first_frame'")
    warm_start = resolved.pop("warm_start", contract == "exact")
    if warm_start != (contract == "exact"):
        raise ValueError(
            f"{contract} does not support warm_start={warm_start}; "
            + ("exact uses chronological warm starts" if contract == "exact"
               else "fixed-budget throughput uses independent shared-weight updates")
        )
    if contract != "exact":
        if resolved.pop("rule", "huber") != "huber":
            raise ValueError(f"{contract} supports only rule='huber'")
        for name in ("max_iterations", "tolerance"):
            if name in resolved:
                raise ValueError(f"{contract} does not support {name}; use update_rounds for its fixed budget")
        if resolved.pop("contact_consistency_strength", 0.0) != 0.0:
            raise ValueError(f"{contract} does not support contact_consistency_strength")
        rounds = resolved.get("update_rounds", 4)
        if int(rounds) < 1:
            raise ValueError("update_rounds must be at least 1")
    elif "update_rounds" in resolved:
        raise ValueError("exact uses max_iterations, not update_rounds")
    resolved.setdefault("delta", 1.0)
    resolved.setdefault("sequence_continuity_strength", 0.05)
    validate_weighting_parameters(resolved["delta"], resolved["sequence_continuity_strength"])
    if contract == "exact":
        if resolved.get("rule", "huber") not in {"huber", "gaussian", "tukey"}:
            raise ValueError("rule must be huber, gaussian, or tukey")
        if int(resolved.get("max_iterations", 20)) < 1:
            raise ValueError("max_iterations must be at least 1")
        tolerance = resolved.get("tolerance", 1e-6)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        contact = resolved.get("contact_consistency_strength", 0.0)
        if not np.isfinite(contact) or not 0.0 <= contact <= 1.0:
            raise ValueError("contact_consistency_strength must be finite and in [0, 1]")
    return resolved


def solve_contract(
    coords: np.ndarray,
    *,
    contract: ExecutionContract,
    backend: BackendName = "cpu",
    chain_ids: np.ndarray | None = None,
    store_aligned: bool = False,
    optional_soft_graph: bool = False,
    **kwargs: object,
) -> ContractExecutionResult:
    """Execute a public contract and retain requested and kernel-effective settings."""

    options = resolve_contract_parameters(contract, kwargs)
    result = _execute_contract(
        coords, contract=contract, backend=backend, chain_ids=chain_ids,
        store_aligned=store_aligned, optional_soft_graph=optional_soft_graph, **options,
    )
    requested = {name: kwargs.get(name, "DEFAULT") for name in (
        "delta", "rule", "sequence_continuity_strength", "contact_consistency_strength",
        "max_iterations", "tolerance", "warm_start", "update_rounds",
    )}
    effective = (
        {} if result.alignment is None
        else dict(result.alignment.metadata.get("effective_parameters", {}))
    )
    result.metadata.update(requested_parameters=requested, effective_parameters=effective)
    if result.alignment is not None:
        result.alignment.metadata.update(requested_parameters=requested, effective_parameters=effective)
    return result


def _execute_contract(
    coords: np.ndarray,
    *,
    contract: ExecutionContract,
    backend: BackendName = "cpu",
    chain_ids: np.ndarray | None = None,
    store_aligned: bool = False,
    optional_soft_graph: bool = False,
    **kwargs: object,
) -> ContractExecutionResult:
    """Dispatch one explicit contract without changing the public default."""

    total_start = time.perf_counter()
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'gpu', or 'auto'")
    if contract == "exact" and backend == "gpu":
        raise ValueError(
            "The exact sequential_lean contract is CPU-only. Use "
            "--backend cpu or --backend auto."
        )
    if contract != "exact" and backend in {"gpu", "auto"}:
        from .gpu_backend import gpu_available, solve_contract_gpu

        if backend == "gpu" or gpu_available():
            return solve_contract_gpu(
                coords,
                contract=contract,
                chain_ids=chain_ids,
                store_aligned=store_aligned,
                optional_soft_graph=optional_soft_graph,
                **kwargs,
            )
    cpu_kwargs = dict(kwargs)
    for name, supported in (("precision", "float64"), ("rotation_solver", "svd")):
        if name in cpu_kwargs and cpu_kwargs[name] != supported:
            raise ValueError(f"CPU execution supports only {name}={supported!r}; request backend='gpu' for GPU policies")
    cpu_kwargs.pop("precision", None)
    cpu_kwargs.pop("rotation_solver", None)
    values, chains = _validate_coords_and_chains(coords, chain_ids)
    if contract == "exact":
        exact = solve_exact_contract(
            values,
            chain_ids=chains,
            store_aligned=store_aligned,
            **cpu_kwargs,
        )
        return ContractExecutionResult(
            contract=contract,
            backend_requested=backend,
            backend_used="cpu",
            alignment=exact,
            local_deformation=None,
            motion_graph=None,
            quality_status="exact_reference",
            quality_warning=None,
            sampled_audit=None,
            refinement_performed=False,
            total_runtime_seconds=exact.total_runtime_seconds,
            timing=exact.timing,
            metadata={
                "public_default_changed": False,
                "automatic_exact_fallback": False,
            },
        )
    if contract == "throughput":
        result = solve_fast_global_alignment(
            values,
            chain_ids=chains,
            store_aligned=store_aligned,
            compute_confidence=False,
            collect_risk_signals=True,
            **cpu_kwargs,
        )
        result.mode = "throughput"
        result.metadata.update(
            {
                "execution_contract": "throughput",
                "fixed_budget_estimator": True,
                "status": "production",
            }
        )
        return ContractExecutionResult(
            contract=contract,
            backend_requested=backend,
            backend_used="cpu",
            alignment=result,
            local_deformation=None,
            motion_graph=None,
            quality_status="completed",
            quality_warning=None,
            sampled_audit=None,
            refinement_performed=False,
            total_runtime_seconds=result.total_runtime_seconds,
            timing=result.timing,
            metadata={
                "automatic_exact_fallback": False,
                "fixed_budget": True,
                "quality_audit_performed": False,
                "public_default_changed": False,
            },
        )
    if contract == "balanced-lite":
        result = solve_balanced_lite_cpu(
            values,
            chain_ids=chains,
            store_aligned=store_aligned,
            **cpu_kwargs,
        )
        result.backend_requested = backend
        return result
    if contract == "local-deformation":
        local = scan_local_kinematics(
            values,
            chain_ids=chains,
            **cpu_kwargs,
        )
        graph = (
            scan_motion_domains(values, chain_ids=chains)
            if optional_soft_graph
            else None
        )
        total_seconds = time.perf_counter() - total_start
        return ContractExecutionResult(
            contract=contract,
            backend_requested=backend,
            backend_used="cpu",
            alignment=None,
            local_deformation=local,
            motion_graph=graph,
            quality_status="local_diagnostic",
            quality_warning=(
                "Local-deformation scores are not RMSD or mature hard "
                "biological-domain labels."
            ),
            sampled_audit=None,
            refinement_performed=False,
            total_runtime_seconds=total_seconds,
            timing={
                "W5": local.timing["total_seconds"],
                "soft_graph": (
                    graph.timing["total_seconds"] if graph is not None else 0.0
                ),
                "total_runtime_seconds": total_seconds,
            },
            metadata={
                "alignment_free": True,
                "hard_motion_domains_mature": False,
            },
        )
    raise ValueError(f"unsupported contract: {contract}")
