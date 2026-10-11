"""Alignment and superposition utilities."""

from __future__ import annotations

import numpy as np


def kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute the optimal rotation matrix from mobile to reference."""

    covariance = mobile.T @ reference
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    determinant = np.linalg.det(u @ vt)
    correction = np.eye(3)
    correction[2, 2] = -1.0 if determinant < 0.0 else 1.0
    return u @ correction @ vt


def _batched_kabsch_rotations(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute optimal rotation matrices for a batch of centered frames."""

    covariance = np.einsum("fmi,mj->fij", mobile, reference)
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    uv_t = np.matmul(u, vt)
    determinant = np.linalg.det(uv_t)
    correction = np.broadcast_to(np.eye(3), (mobile.shape[0], 3, 3)).copy()
    correction[:, 2, 2] = np.where(determinant < 0.0, -1.0, 1.0)
    return np.matmul(np.matmul(u, correction), vt)


def align_frame(frame: np.ndarray, reference: np.ndarray, atom_mask: np.ndarray | None = None) -> np.ndarray:
    """Align a single frame onto a reference structure."""

    if atom_mask is None:
        mobile_fit = frame
        ref_fit = reference
    else:
        mobile_fit = frame[atom_mask]
        ref_fit = reference[atom_mask]

    mobile_center = mobile_fit.mean(axis=0)
    ref_center = ref_fit.mean(axis=0)
    rotation = kabsch_rotation(mobile_fit - mobile_center, ref_fit - ref_center)
    return (frame - mobile_center) @ rotation + ref_center


def align_trajectory(
    coords: np.ndarray,
    reference: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Align all frames onto a fixed reference."""

    if atom_mask is None:
        mobile_fit = coords
        ref_fit = reference
    else:
        mobile_fit = coords[:, atom_mask, :]
        ref_fit = reference[atom_mask]

    mobile_center = mobile_fit.mean(axis=1)
    ref_center = ref_fit.mean(axis=0)
    rotations = _batched_kabsch_rotations(
        mobile_fit - mobile_center[:, None, :],
        ref_fit - ref_center[None, :],
    )
    return np.einsum("fai,fij->faj", coords - mobile_center[:, None, :], rotations) + ref_center[None, None, :]


def medoid_frame_index(coords: np.ndarray) -> int:
    """Return the medoid-like frame index using centroid distance."""

    flattened = coords.reshape(coords.shape[0], -1)
    center = flattened.mean(axis=0, keepdims=True)
    distances = np.sum((flattened - center) ** 2, axis=1)
    return int(np.argmin(distances))
