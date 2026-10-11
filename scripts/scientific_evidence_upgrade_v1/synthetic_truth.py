"""Deterministic controlled trajectories and independent truth metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mdsuperpose.fast_kabsch import (
    apply_transforms_batch,
    weighted_kabsch_transform_batch,
)


@dataclass(frozen=True, slots=True)
class SyntheticTrajectory:
    """One trajectory with known global transform and internal deformation."""

    observed: np.ndarray
    internal: np.ndarray
    reference: np.ndarray
    true_stable_mask: np.ndarray | None
    true_stability: np.ndarray
    true_rotations: np.ndarray
    true_translations: np.ndarray
    mobile_mask: np.ndarray
    loop_masks: tuple[np.ndarray, ...]
    fixed_domain_mask: np.ndarray | None
    moving_domain_mask: np.ndarray | None
    hinge_angles_deg: np.ndarray | None
    metadata: dict[str, object]


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ],
        dtype=np.float64,
    )


def _uniform_rotations(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    quaternions = rng.normal(size=(n_frames, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    rotations[0] = np.eye(3)
    return rotations


def _translations(rng: np.random.Generator, n_frames: int, maximum: float = 30.0) -> np.ndarray:
    directions = rng.normal(size=(n_frames, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(0.0, maximum, size=n_frames)
    output = directions * magnitudes[:, None]
    output[0] = 0.0
    return output


def _segment_mask(n_residues: int, center_fraction: float, length_fraction: float) -> np.ndarray:
    length = min(20, max(8, int(round(n_residues * length_fraction))))
    center = int(round(center_fraction * (n_residues - 1)))
    start = min(max(1, center - length // 2), n_residues - length - 1)
    mask = np.zeros(n_residues, dtype=bool)
    mask[start : start + length] = True
    return mask


def _smooth_loop_motion(
    reference: np.ndarray,
    mask: np.ndarray,
    n_frames: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    directions = rng.normal(size=(indices.size, 3))
    for _ in range(4):
        padded = np.pad(directions, ((1, 1), (0, 0)), mode="edge")
        directions = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    time = np.linspace(0.0, 6.0 * np.pi, n_frames)
    temporal = np.sin(time + phase) + 0.35 * np.sin(0.43 * time + 0.7 * phase)
    temporal -= temporal[0]
    temporal /= max(float(np.max(np.abs(temporal))), 1e-12)
    position = np.linspace(-1.0, 1.0, indices.size)
    envelope = np.cos(0.5 * np.pi * position) ** 2
    displacement = np.zeros((n_frames, reference.shape[0], 3), dtype=np.float64)
    displacement[:, indices, :] = (
        amplitude
        * temporal[:, None, None]
        * envelope[None, :, None]
        * directions[None, :, :]
    )
    displacement[0] = 0.0
    return displacement


def _hinge_motion(
    reference: np.ndarray,
    moving_mask: np.ndarray,
    n_frames: int,
    amplitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    moving_indices = np.flatnonzero(moving_mask)
    boundary = moving_indices[0]
    origin = reference[max(0, boundary - 1)]
    local = reference[min(reference.shape[0] - 1, boundary + 2)] - reference[max(0, boundary - 2)]
    if np.linalg.norm(local) < 1e-8:
        local = np.asarray([1.0, 0.0, 0.0])
    axis = local / np.linalg.norm(local)
    phase = np.linspace(0.0, 4.0 * np.pi, n_frames)
    angles = np.deg2rad(amplitude_deg) * np.sin(phase)
    angles[0] = 0.0
    output = np.repeat(reference[None, :, :], n_frames, axis=0)
    centered = reference[moving_mask] - origin
    for frame, angle in enumerate(angles):
        row_operator = _rotation_matrix(axis, float(angle)).T
        output[frame, moving_mask] = centered @ row_operator + origin
    return output, np.degrees(angles)


def _heteroscedastic_motion(
    reference: np.ndarray,
    n_frames: int,
    amplitude: float,
    rng: np.random.Generator,
    *,
    uniform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_residues = reference.shape[0]
    if uniform:
        sigma = np.full(n_residues, amplitude, dtype=np.float64)
    else:
        phase = np.linspace(0.0, 3.0 * np.pi, n_residues)
        profile = 0.5 + 0.5 * np.sin(phase + 0.4) ** 2
        profile += 0.75 * np.exp(-0.5 * ((np.arange(n_residues) - 0.72 * n_residues) / max(4.0, 0.08 * n_residues)) ** 2)
        profile = (profile - np.min(profile)) / max(float(np.ptp(profile)), 1e-12)
        sigma = 0.08 + amplitude * profile
    noise = rng.normal(size=(n_frames, n_residues, 3))
    for frame in range(1, n_frames):
        noise[frame] = 0.85 * noise[frame - 1] + math.sqrt(1.0 - 0.85**2) * noise[frame]
    noise -= noise[0:1]
    motion = reference[None, :, :] + noise * sigma[None, :, None]
    motion[0] = reference
    return motion, sigma


def generate_synthetic_trajectory(
    reference: np.ndarray,
    *,
    scenario: str,
    n_frames: int,
    seed: int,
    local_amplitude_A: float,
    hinge_amplitude_deg: float,
) -> SyntheticTrajectory:
    """Generate one registered controlled trajectory from a real CA template."""

    target = np.asarray(reference, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 20:
        raise ValueError("reference must have shape (N,3) with N>=20")
    if scenario not in {f"S{index}" for index in range(7)}:
        raise ValueError("scenario must be S0-S6")
    rng = np.random.default_rng(int(seed))
    n_residues = target.shape[0]
    loop_one = _segment_mask(n_residues, 0.35, 0.10)
    loop_two = _segment_mask(n_residues, 0.72, 0.09)
    fixed_domain = np.arange(n_residues) < n_residues // 2
    moving_domain = ~fixed_domain
    internal = np.repeat(target[None, :, :], n_frames, axis=0)
    stable_mask: np.ndarray | None
    mobile_mask = np.zeros(n_residues, dtype=bool)
    loop_masks: tuple[np.ndarray, ...] = ()
    hinge_angles: np.ndarray | None = None
    stability = np.ones(n_residues, dtype=np.float64)
    fixed_output: np.ndarray | None = None
    moving_output: np.ndarray | None = None

    if scenario == "S0":
        stable_mask = np.ones(n_residues, dtype=bool)
    elif scenario == "S1":
        internal += _smooth_loop_motion(target, loop_one, n_frames, local_amplitude_A, rng)
        mobile_mask = loop_one.copy()
        stable_mask = ~mobile_mask
        loop_masks = (loop_one,)
        stability = stable_mask.astype(np.float64)
    elif scenario == "S2":
        internal += _smooth_loop_motion(target, loop_one, n_frames, local_amplitude_A, rng)
        internal += _smooth_loop_motion(target, loop_two, n_frames, local_amplitude_A, rng)
        mobile_mask = loop_one | loop_two
        stable_mask = ~mobile_mask
        loop_masks = (loop_one, loop_two)
        stability = stable_mask.astype(np.float64)
    elif scenario in {"S3", "S4"}:
        internal, hinge_angles = _hinge_motion(
            target,
            moving_domain,
            n_frames,
            hinge_amplitude_deg,
        )
        mobile_mask = moving_domain.copy()
        stable_mask = fixed_domain.copy()
        fixed_output = fixed_domain.copy()
        moving_output = moving_domain.copy()
        stability = stable_mask.astype(np.float64)
        if scenario == "S4":
            local_loop = loop_one & fixed_domain
            if np.count_nonzero(local_loop) < 5:
                local_loop = _segment_mask(n_residues, 0.25, 0.08) & fixed_domain
            internal += _smooth_loop_motion(
                target,
                local_loop,
                n_frames,
                local_amplitude_A,
                rng,
            )
            mobile_mask |= local_loop
            stable_mask &= ~local_loop
            stability = stable_mask.astype(np.float64)
            loop_masks = (local_loop,)
    elif scenario == "S5":
        internal, sigma = _heteroscedastic_motion(
            target,
            n_frames,
            local_amplitude_A,
            rng,
            uniform=False,
        )
        inverse = 1.0 / np.maximum(sigma * sigma, 1e-12)
        stability = (inverse - np.min(inverse)) / max(float(np.ptp(inverse)), 1e-12)
        count = max(1, int(np.floor(0.60 * n_residues)))
        order = np.lexsort((np.arange(n_residues), -stability))
        stable_mask = np.zeros(n_residues, dtype=bool)
        stable_mask[order[:count]] = True
        mobile_mask = ~stable_mask
    else:
        internal, _ = _heteroscedastic_motion(
            target,
            n_frames,
            local_amplitude_A,
            rng,
            uniform=True,
        )
        stability = np.full(n_residues, 0.5, dtype=np.float64)
        stable_mask = None
        mobile_mask = np.ones(n_residues, dtype=bool)

    rotations = _uniform_rotations(rng, n_frames)
    translations_forward = _translations(rng, n_frames)
    observed = internal @ rotations + translations_forward[:, None, :]
    true_translations = -np.einsum(
        "fi,fij->fj",
        translations_forward,
        np.swapaxes(rotations, 1, 2),
    )
    return SyntheticTrajectory(
        observed=observed,
        internal=internal,
        reference=target,
        true_stable_mask=stable_mask,
        true_stability=stability,
        true_rotations=rotations,
        true_translations=true_translations,
        mobile_mask=mobile_mask,
        loop_masks=loop_masks,
        fixed_domain_mask=fixed_output,
        moving_domain_mask=moving_output,
        hinge_angles_deg=hinge_angles,
        metadata={
            "scenario": scenario,
            "seed": int(seed),
            "n_frames": int(n_frames),
            "local_amplitude_A": float(local_amplitude_A),
            "hinge_amplitude_deg": float(hinge_amplitude_deg),
            "transform_convention": "aligned = observed @ rotation.T + translation",
            "truth_scope": "generative_not_biological",
        },
    )


def aligned_from_transform(
    coords: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Apply stored MDCoreAlign transforms through an independent expression."""

    return np.einsum("fni,fji->fnj", coords, rotations) + translations[:, None, :]


def rotation_geodesic_degrees(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return SO(3) geodesic errors in degrees."""

    relative = np.einsum("fij,fkj->fik", estimated, truth)
    cosines = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosines))


def rmsf(coords: np.ndarray) -> np.ndarray:
    """Compute per-residue Cartesian RMSF in the supplied coordinate frame."""

    values = np.asarray(coords, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))


def infer_transforms(source: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer rigid transforms between paired coordinate frames."""

    rotations = np.empty((source.shape[0], 3, 3), dtype=np.float64)
    translations = np.empty((source.shape[0], 3), dtype=np.float64)
    for start in range(0, source.shape[0], 500):
        stop = min(start + 500, source.shape[0])
        mobile = np.asarray(source[start:stop], dtype=np.float64)
        target = np.asarray(aligned[start:stop], dtype=np.float64)
        mobile_center = np.mean(mobile, axis=1)
        target_center = np.mean(target, axis=1)
        mobile_centered = mobile - mobile_center[:, None, :]
        target_centered = target - target_center[:, None, :]
        covariance = mobile_centered.swapaxes(1, 2) @ target_centered
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
        signs = np.where(np.linalg.det(u @ vt) < 0.0, -1.0, 1.0)
        u[:, :, 2] *= signs[:, None]
        row_operator = u @ vt
        rotations[start:stop] = np.swapaxes(row_operator, 1, 2)
        translations[start:stop] = target_center - np.einsum(
            "fi,fij->fj",
            mobile_center,
            row_operator,
        )
    return rotations, translations


def gauge_to_reference(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map an arbitrary output gauge to the registered frame-0 reference."""

    result = weighted_kabsch_transform_batch(
        aligned[0:1],
        reference,
        np.ones(reference.shape[0], dtype=np.float64),
        validate_inputs=False,
    )
    rotations = np.repeat(result.rotations, aligned.shape[0], axis=0)
    translations = np.repeat(result.translations, aligned.shape[0], axis=0)
    return apply_transforms_batch(aligned, rotations, translations)


def binary_ranking_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute AUROC and average precision from deterministic residue scores."""

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(truth, dtype=bool)
    positives = int(np.count_nonzero(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return {"AUROC": float("nan"), "AUPRC": float("nan")}
    order = np.lexsort((np.arange(values.size), -values))
    sorted_labels = labels[order].astype(np.float64)
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1.0 - sorted_labels)
    tpr = np.concatenate(([0.0], true_positive / positives, [1.0]))
    fpr = np.concatenate(([0.0], false_positive / negatives, [1.0]))
    auroc = float(np.trapz(tpr, fpr))
    precision = true_positive / np.arange(1, labels.size + 1)
    auprc = float(np.sum(precision * sorted_labels) / positives)
    return {"AUROC": auroc, "AUPRC": auprc}


def binary_mask_metrics(estimated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute exact binary core-recovery metrics."""

    left = np.asarray(estimated, dtype=bool)
    right = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(left & right))
    fp = int(np.count_nonzero(left & ~right))
    fn = int(np.count_nonzero(~left & right))
    tn = int(np.count_nonzero(~left & ~right))
    union = tp + fp + fn
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    return {
        "Jaccard": tp / union if union else 1.0,
        "Dice": 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "MCC": (tp * tn - fp * fn) / denominator if denominator else float("nan"),
    }


def top_cardinality_mask(scores: np.ndarray, cardinality: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(values.size), -values))
    output = np.zeros(values.size, dtype=bool)
    output[order[: int(cardinality)]] = True
    return output
