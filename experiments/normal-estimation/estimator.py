"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
DISTANCE_DECAY = 2.0
ROBUST_CUTOFF = 2.5
ROBUST_REWEIGHTING_STEPS = 2
MAD_TO_SIGMA = 1.4826
BATCH_SIZE = 2_048


def estimate_normals(
    points: np.ndarray,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Estimate unoriented normals with two-step robust weighted PCA.

    An initial Gaussian distance-weighted fit supplies a provisional tangent
    plane and a robust MAD residual scale. Two Cauchy IRLS steps then update the
    plane while holding that scale fixed, avoiding progressively stronger
    rejection caused solely by scale contraction.
    """
    del query_indices
    if neighbor_indices.shape[1] < NEIGHBOR_COUNT:
        msg = f"At least {NEIGHBOR_COUNT} cached neighbors are required"
        raise ValueError(msg)

    estimated = np.empty((neighbor_indices.shape[0], 3), dtype=np.float64)
    for start in range(0, neighbor_indices.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, neighbor_indices.shape[0])
        neighborhoods = points[neighbor_indices[start:stop, :NEIGHBOR_COUNT]]
        distances = neighbor_distances[start:stop, :NEIGHBOR_COUNT]
        radius = distances[:, -1:]
        scaled_squared = np.divide(
            distances * distances,
            radius * radius,
            out=np.zeros_like(distances, dtype=np.float64),
            where=radius > 0.0,
        )
        distance_weights = np.exp(-DISTANCE_DECAY * scaled_squared)
        distance_weights /= distance_weights.sum(axis=1, keepdims=True)

        centroid = np.einsum(
            "nk,nki->ni", distance_weights, neighborhoods, optimize=True
        )
        centered = neighborhoods - centroid[:, None, :]
        covariance = np.einsum(
            "nk,nki,nkj->nij",
            distance_weights,
            centered,
            centered,
            optimize=True,
        )
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]

        initial_residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
        residual_median = np.median(initial_residuals, axis=1, keepdims=True)
        robust_scale = MAD_TO_SIGMA * np.median(
            np.abs(initial_residuals - residual_median), axis=1, keepdims=True
        )
        scale_floor = np.finfo(np.float64).eps * np.maximum(radius, 1.0)
        robust_scale = np.maximum(robust_scale, scale_floor)

        for _ in range(ROBUST_REWEIGHTING_STEPS):
            residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
            normalized = residuals / (ROBUST_CUTOFF * robust_scale)
            robust_weights = 1.0 / (1.0 + normalized * normalized)

            weights = distance_weights * robust_weights
            weights /= weights.sum(axis=1, keepdims=True)
            centroid = np.einsum("nk,nki->ni", weights, neighborhoods, optimize=True)
            centered = neighborhoods - centroid[:, None, :]
            covariance = np.einsum(
                "nk,nki,nkj->nij", weights, centered, centered, optimize=True
            )
            _, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]

        estimated[start:stop] = normals

    return estimated
