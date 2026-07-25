"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
BATCH_SIZE = 2_048


def estimate_normals(
    points: np.ndarray,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Estimate unoriented normals with fixed-neighborhood PCA.

    The normal is the eigenvector associated with the smallest eigenvalue of
    the local covariance matrix. ``neighbor_distances`` is unused by this
    baseline but remains available for weighted or adaptive estimators.
    """
    del query_indices, neighbor_distances
    if neighbor_indices.shape[1] < NEIGHBOR_COUNT:
        msg = f"At least {NEIGHBOR_COUNT} cached neighbors are required"
        raise ValueError(msg)

    estimated = np.empty((neighbor_indices.shape[0], 3), dtype=np.float64)
    for start in range(0, neighbor_indices.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, neighbor_indices.shape[0])
        neighborhoods = points[neighbor_indices[start:stop, :NEIGHBOR_COUNT]]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True)
        _, eigenvectors = np.linalg.eigh(covariance)
        estimated[start:stop] = eigenvectors[:, :, 0]

    return estimated
