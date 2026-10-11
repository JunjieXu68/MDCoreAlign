"""Pilot tests for experimental local kinematics and motion domains."""

from __future__ import annotations

import numpy as np

from mdsuperpose.window_kinematics import (
    align_to_motion_domain,
    local_profile_rows,
    scan_local_kinematics,
    scan_motion_domains,
)


def compact_reference(n_residues: int, seed: int = 7) -> np.ndarray:
    """Build a non-degenerate chain with CA-like adjacent spacing."""

    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_residues, 3))
    coords = np.cumsum(steps, axis=0)
    adjacent = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return coords * (3.8 / np.median(adjacent))


def rotation(angle: float) -> np.ndarray:
    """Return one row-compatible z rotation."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def rigid_trajectory(n_residues: int = 40, n_frames: int = 80) -> np.ndarray:
    """Generate pure global rigid motion."""

    reference = compact_reference(n_residues)
    return np.asarray(
        [
            reference @ rotation(angle).T
            + np.asarray((angle, 2.0 * angle, -angle))
            for angle in np.linspace(-0.8, 0.8, n_frames)
        ]
    )


def domain_trajectory(sizes: tuple[int, ...], n_frames: int = 100) -> np.ndarray:
    """Generate coherent rigid domains linked by changing hinges."""

    reference = compact_reference(sum(sizes))
    frames: list[np.ndarray] = []
    for angle in np.linspace(-0.7, 0.7, n_frames):
        frame = reference.copy()
        start = sizes[0]
        for domain_index, size in enumerate(sizes[1:], start=1):
            pivot = reference[start - 1]
            stop = start + size
            frame[start:stop] = (
                (frame[start:stop] - pivot)
                @ rotation(angle * domain_index).T
                + pivot
            )
            start = stop
        frames.append(frame)
    return np.asarray(frames)


def boundary_positions(labels: np.ndarray) -> np.ndarray:
    """Return zero-based domain starts after the first domain."""

    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def test_pure_rigid_motion_has_no_local_deformation_or_false_domain() -> None:
    result = scan_motion_domains(rigid_trajectory(), evaluate_blocks=False)
    assert float(np.max(result.local.local_deformation_index)) < 1e-12
    np.testing.assert_array_equal(np.unique(result.motion_domain_id), [1])
    assert result.metadata["single_global_core_required"] is False


def test_local_loop_deformation_is_localised() -> None:
    rng = np.random.default_rng(12)
    reference = compact_reference(44)
    coords = np.repeat(reference[None, :, :], 120, axis=0)
    known_loop = np.arange(17, 24)
    coords[:, known_loop] += rng.normal(
        scale=1.2, size=(coords.shape[0], known_loop.size, 3)
    )
    result = scan_local_kinematics(coords)
    top = np.argsort(result.local_deformation_index)[-known_loop.size :]
    assert len(set(top).intersection(known_loop.tolist())) >= 6
    selected = result.locally_deforming_segments > 0
    assert np.count_nonzero(selected[known_loop]) >= 6
    assert result.metadata["score_called_rmsd"] is False
    assert result.metadata["score_called_global_mobility"] is False


def test_two_equal_rigid_domains_and_hinge_are_recovered() -> None:
    result = scan_motion_domains(
        domain_trajectory((20, 20)),
        minimum_domain_size=6,
        evaluate_blocks=False,
    )
    boundaries = boundary_positions(result.motion_domain_id)
    assert np.any(np.abs(boundaries - 20) <= 1)
    assert np.unique(result.motion_domain_id).size == 2


def test_unequal_rigid_domains_are_not_forced_to_one_core() -> None:
    result = scan_motion_domains(
        domain_trajectory((28, 12)),
        minimum_domain_size=6,
        evaluate_blocks=False,
    )
    boundaries = boundary_positions(result.motion_domain_id)
    assert np.any(np.abs(boundaries - 28) <= 1)
    assert np.unique(result.motion_domain_id).size == 2


def test_three_domain_motion_recovers_two_boundaries() -> None:
    result = scan_motion_domains(
        domain_trajectory((15, 15, 15)),
        minimum_domain_size=6,
        evaluate_blocks=False,
    )
    boundaries = boundary_positions(result.motion_domain_id)
    assert np.any(np.abs(boundaries - 15) <= 1)
    assert np.any(np.abs(boundaries - 30) <= 1)
    assert np.unique(result.motion_domain_id).size == 3


def test_scale_and_rigid_transform_invariance() -> None:
    coords = domain_trajectory((20, 20))
    baseline = scan_motion_domains(coords, evaluate_blocks=False)
    transformed = coords @ rotation(0.63).T + np.asarray((5.0, -3.0, 8.0))
    for scale in (0.1, 1.0, 10.0):
        observed = scan_motion_domains(
            transformed * scale, evaluate_blocks=False
        )
        np.testing.assert_allclose(
            observed.local.local_deformation_index,
            baseline.local.local_deformation_index,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_array_equal(
            observed.motion_domain_id, baseline.motion_domain_id
        )


def test_chain_break_is_forced_and_segments_do_not_cross_it() -> None:
    rng = np.random.default_rng(5)
    reference = compact_reference(40)
    coords = np.repeat(reference[None, :, :], 100, axis=0)
    coords[:, 17:23] += rng.normal(scale=0.8, size=(100, 6, 3))
    chains = np.asarray(["A"] * 20 + ["B"] * 20)
    positions = np.r_[np.arange(1, 21), np.arange(1, 21)]
    result = scan_motion_domains(
        coords,
        chain_ids=chains,
        residue_positions=positions,
        minimum_domain_size=5,
        evaluate_blocks=False,
    )
    assert 20 in boundary_positions(result.motion_domain_id)
    segment = result.local.locally_deforming_segments
    assert not (
        segment[19] > 0
        and segment[19] == segment[20]
    )


def test_outlier_frame_does_not_create_persistent_local_deformation() -> None:
    coords = rigid_trajectory()
    coords[10, 18:22] += 50.0
    result = scan_local_kinematics(coords)
    assert float(np.max(result.local_deformation_index)) < 1e-10


def test_optional_domain_alignment_uses_explicit_anchor() -> None:
    coords = domain_trajectory((20, 20))
    domains = scan_motion_domains(
        coords, minimum_domain_size=6, evaluate_blocks=False
    )
    aligned = align_to_motion_domain(
        coords,
        domains,
        selection="user_specified",
        domain_id=1,
    )
    anchor_delta = (
        aligned.aligned_coords[:, aligned.anchor_mask]
        - coords[0][None, aligned.anchor_mask]
    )
    anchor_rmsd = np.sqrt(
        np.mean(np.sum(anchor_delta * anchor_delta, axis=2), axis=1)
    )
    assert float(np.max(anchor_rmsd)) < 1e-10
    assert float(np.max(aligned.rmsd)) > 0.1
    assert aligned.metadata["rmsd_role"] == "downstream_geometric_audit"


def test_profile_schema_uses_local_kinematic_semantics() -> None:
    coords = domain_trajectory((12, 12), n_frames=40)
    result = scan_motion_domains(
        coords, minimum_domain_size=5, evaluate_blocks=False
    )
    rows = local_profile_rows(result)
    assert len(rows) == 24
    assert {
        "local_deformation_index",
        "local_rigidity_score",
        "locally_deforming_segment_id",
        "motion_domain_id",
        "hinge_boundary_score_after_residue",
        "inter_domain_edge_instability",
    }.issubset(rows[0])


def test_graph_reuses_full_frame_w5_and_reports_nonlocal_sampling() -> None:
    coords = domain_trajectory((20, 20), n_frames=600)
    local = scan_local_kinematics(coords, evaluate_blocks=False)
    graph = scan_motion_domains(
        coords,
        minimum_domain_size=6,
        nonlocal_sample_count=128,
        evaluate_blocks=False,
    )
    np.testing.assert_array_equal(
        graph.local.local_deformation_index,
        local.local_deformation_index,
    )
    assert graph.metadata["local_edge_sampling"] == "all_frames_exact_W5_reuse"
    assert graph.metadata["nonlocal_edge_sampling"] == "temporally_stratified"
    assert graph.metadata["nonlocal_edge_sample_count"] == 128
