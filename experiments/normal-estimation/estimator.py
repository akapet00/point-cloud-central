"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
DISTANCE_DECAY = 1.0
BATCH_SIZE = 2_048


def estimate_normals(
    points: np.ndarray,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Estimate unoriented normals with distance-weighted local PCA.

    Gaussian weights decrease with squared distance from the query, using the
    selected neighborhood radius as a density-adaptive bandwidth. This keeps
    all 112 samples for noise averaging while reducing the leverage of distant
    points, which are more likely to cross curved features.
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
        radius = distances[:, -1:, ...]
        scaled_squared = np.divide(
            distances * distances,
            radius * radius,
            out=np.zeros_like(distances, dtype=np.float64),
            where=radius > 0.0,
        )
        weights = np.exp(-DISTANCE_DECAY * scaled_squared)
        weights /= weights.sum(axis=1, keepdims=True)

        centroid = np.einsum("nk,nki->ni", weights, neighborhoods, optimize=True)
        centered = neighborhoods - centroid[:, None, :]
        covariance = np.einsum(
            "nk,nki,nkj->nij", weights, centered, centered, optimize=True
        )
        _, eigenvectors = np.linalg.eigh(covariance)
        estimated[start:stop] = eigenvectors[:, :, 0]

    return estimated
