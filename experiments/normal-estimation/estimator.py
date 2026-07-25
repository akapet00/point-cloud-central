"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

LOCAL_NEIGHBOR_COUNT = 112
INITIAL_SUPPORT_COUNT = 160
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
    """Estimate unoriented normals with broad initialization and local IRLS.

    A 160-neighbor Gaussian tail stabilizes only the provisional tangent plane,
    while its bandwidth remains the 112th-neighbor distance. Two Cauchy IRLS
    steps then fit only the local 112-neighbor patch, limiting the final
    estimator's exposure to curvature and neighborhood bias.
    """
    del query_indices
    if neighbor_indices.shape[1] < INITIAL_SUPPORT_COUNT:
        msg = f"At least {INITIAL_SUPPORT_COUNT} cached neighbors are required"
        raise ValueError(msg)

    estimated = np.empty((neighbor_indices.shape[0], 3), dtype=np.float64)
    for start in range(0, neighbor_indices.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, neighbor_indices.shape[0])
        initial_neighborhoods = points[
            neighbor_indices[start:stop, :INITIAL_SUPPORT_COUNT]
        ]
        initial_distances = neighbor_distances[start:stop, :INITIAL_SUPPORT_COUNT]
        bandwidth = initial_distances[
            :, LOCAL_NEIGHBOR_COUNT - 1 : LOCAL_NEIGHBOR_COUNT
        ]
        scaled_squared = np.divide(
            initial_distances * initial_distances,
            bandwidth * bandwidth,
            out=np.zeros_like(initial_distances, dtype=np.float64),
            where=bandwidth > 0.0,
        )
        spatial_kernel = np.exp(-DISTANCE_DECAY * scaled_squared)
        initial_weights = spatial_kernel / spatial_kernel.sum(axis=1, keepdims=True)

        centroid = np.einsum(
            "nk,nki->ni", initial_weights, initial_neighborhoods, optimize=True
        )
        centered = initial_neighborhoods - centroid[:, None, :]
        covariance = np.einsum(
            "nk,nki,nkj->nij",
            initial_weights,
            centered,
            centered,
            optimize=True,
        )
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]

        neighborhoods = initial_neighborhoods[:, :LOCAL_NEIGHBOR_COUNT]
        distance_weights = spatial_kernel[:, :LOCAL_NEIGHBOR_COUNT].copy()
        distance_weights /= distance_weights.sum(axis=1, keepdims=True)
        centered = neighborhoods - centroid[:, None, :]
        scale_floor = np.finfo(np.float64).eps * np.maximum(bandwidth, 1.0)

        for _ in range(ROBUST_REWEIGHTING_STEPS):
            residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
            residual_median = np.median(residuals, axis=1, keepdims=True)
            robust_scale = MAD_TO_SIGMA * np.median(
                np.abs(residuals - residual_median), axis=1, keepdims=True
            )
            robust_scale = np.maximum(robust_scale, scale_floor)
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
