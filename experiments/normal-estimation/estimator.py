"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
INITIAL_NEIGHBOR_COUNT = 224
DISTANCE_DECAY = 2.0
TUKEY_CUTOFFS = (4.62, 2.77)
MAX_BROAD_BLEND = 0.10
MAD_TO_SIGMA = 1.4826
BATCH_SIZE = 2_048


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return one weighted median per row, retaining a singleton axis."""
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative_weights = np.cumsum(sorted_weights, axis=1)
    threshold = 0.5 * sorted_weights.sum(axis=1, keepdims=True)
    median_indices = np.argmax(cumulative_weights >= threshold, axis=1)
    return np.take_along_axis(sorted_values, median_indices[:, None], axis=1)


def estimate_normals(
    points: np.ndarray,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Estimate normals from broad initialization and local robust PCA.

    A 224-neighbor Gaussian tail stabilizes the provisional tangent under
    positional noise. Two Tukey-biweight IRLS steps then refine that normal
    using only the query-local 112-neighbor patch. A final confidence-weighted
    blend borrows broad-scale stability only when the local normal is ambiguous
    and the two scales agree.
    """
    del query_indices
    if neighbor_indices.shape[1] < INITIAL_NEIGHBOR_COUNT:
        msg = f"At least {INITIAL_NEIGHBOR_COUNT} cached neighbors are required"
        raise ValueError(msg)

    estimated = np.empty((neighbor_indices.shape[0], 3), dtype=np.float64)
    for start in range(0, neighbor_indices.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, neighbor_indices.shape[0])
        initial_neighborhoods = points[
            neighbor_indices[start:stop, :INITIAL_NEIGHBOR_COUNT]
        ]
        initial_distances = neighbor_distances[start:stop, :INITIAL_NEIGHBOR_COUNT]
        bandwidth = initial_distances[:, NEIGHBOR_COUNT - 1 : NEIGHBOR_COUNT]
        scaled_squared = np.divide(
            initial_distances * initial_distances,
            bandwidth * bandwidth,
            out=np.zeros_like(initial_distances, dtype=np.float64),
            where=bandwidth > 0.0,
        )
        initial_weights = np.exp(-DISTANCE_DECAY * scaled_squared)
        initial_weights /= initial_weights.sum(axis=1, keepdims=True)

        centroid = np.einsum(
            "nk,nki->ni", initial_weights, initial_neighborhoods, optimize=True
        )
        initial_centered = initial_neighborhoods - centroid[:, None, :]
        covariance = np.einsum(
            "nk,nki,nkj->nij",
            initial_weights,
            initial_centered,
            initial_centered,
            optimize=True,
        )
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        broad_normals = normals.copy()

        neighborhoods = initial_neighborhoods[:, :NEIGHBOR_COUNT]
        distance_weights = initial_weights[:, :NEIGHBOR_COUNT]
        distance_weights /= distance_weights.sum(axis=1, keepdims=True)
        centered = neighborhoods - centroid[:, None, :]
        scale_floor = np.finfo(np.float64).eps * np.maximum(bandwidth, 1.0)
        for tukey_cutoff in TUKEY_CUTOFFS:
            residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
            residual_median = _weighted_median(residuals, distance_weights)
            robust_scale = MAD_TO_SIGMA * _weighted_median(
                np.abs(residuals - residual_median), distance_weights
            )
            robust_scale = np.maximum(robust_scale, scale_floor)
            normalized = (residuals - residual_median) / (tukey_cutoff * robust_scale)
            inside = normalized * normalized < 1.0
            robust_weights = np.square(1.0 - normalized * normalized) * inside

            weights = distance_weights * robust_weights
            weights /= weights.sum(axis=1, keepdims=True)
            centroid = np.einsum("nk,nki->ni", weights, neighborhoods, optimize=True)
            centered = neighborhoods - centroid[:, None, :]
            covariance = np.einsum(
                "nk,nki,nkj->nij", weights, centered, centered, optimize=True
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]

        agreement = np.einsum("ni,ni->n", normals, broad_normals, optimize=True)
        aligned_broad = np.where(
            agreement[:, None] < 0.0, -broad_normals, broad_normals
        )
        agreement = np.abs(agreement)
        ambiguity = np.divide(
            eigenvalues[:, 0],
            eigenvalues[:, 1],
            out=np.zeros_like(eigenvalues[:, 0]),
            where=eigenvalues[:, 1] > 0.0,
        )
        blend = MAX_BROAD_BLEND * np.clip(ambiguity, 0.0, 1.0) * agreement**2
        normals = (1.0 - blend[:, None]) * normals + blend[:, None] * aligned_broad
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, np.finfo(np.float64).tiny)

        estimated[start:stop] = normals

    return estimated
