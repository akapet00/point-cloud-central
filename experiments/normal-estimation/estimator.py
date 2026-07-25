"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

SMALL_NEIGHBOR_COUNT = 80
LARGE_NEIGHBOR_COUNT = 224
DISTANCE_DECAY = 2.0
SURFACE_VARIATION_THRESHOLD = 0.01
BATCH_SIZE = 2_048


def _weighted_pca(
    neighborhoods: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian-weighted PCA normals and dimensionless surface variation."""
    radius = distances[:, -1:]
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
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total_variance = eigenvalues.sum(axis=1)
    surface_variation = np.divide(
        eigenvalues[:, 0],
        total_variance,
        out=np.zeros_like(total_variance),
        where=total_variance > 0.0,
    )
    return eigenvectors[:, :, 0], surface_variation


def estimate_normals(
    points: np.ndarray,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Estimate unoriented normals by confidence-selecting one of two PCA scales.

    A low 80-neighbor surface variation indicates a reliable local tangent, so
    the more curvature-local normal is used. Otherwise the 224-neighbor normal
    supplies stronger noise averaging. Both fits retain density-scaled Gaussian
    weighting.
    """
    del query_indices
    if neighbor_indices.shape[1] < LARGE_NEIGHBOR_COUNT:
        msg = f"At least {LARGE_NEIGHBOR_COUNT} cached neighbors are required"
        raise ValueError(msg)

    estimated = np.empty((neighbor_indices.shape[0], 3), dtype=np.float64)
    for start in range(0, neighbor_indices.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, neighbor_indices.shape[0])
        neighborhoods = points[neighbor_indices[start:stop, :LARGE_NEIGHBOR_COUNT]]
        distances = neighbor_distances[start:stop, :LARGE_NEIGHBOR_COUNT]

        small_normals, small_variation = _weighted_pca(
            neighborhoods[:, :SMALL_NEIGHBOR_COUNT],
            distances[:, :SMALL_NEIGHBOR_COUNT],
        )
        large_normals, _ = _weighted_pca(neighborhoods, distances)
        use_small = small_variation <= SURFACE_VARIATION_THRESHOLD
        estimated[start:stop] = np.where(
            use_small[:, None], small_normals, large_normals
        )

    return estimated
