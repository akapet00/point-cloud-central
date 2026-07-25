"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
INITIAL_NEIGHBOR_COUNT = 224
DISTANCE_DECAY = 2.0
ROBUST_CUTOFFS = (2.5, 1.5)
MAD_TO_SIGMA = 1.4826
JET_RIDGE = 1e-6
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
    """Estimate normals with robust PCA followed by a quadratic local jet.

    A broad Gaussian PCA stabilizes the provisional tangent under noise, and
    two local Cauchy refinements limit outlier leverage. A weighted quadratic
    height fit in the final PCA frame then separates curvature from the linear
    slope at the query.
    """
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

        neighborhoods = initial_neighborhoods[:, :NEIGHBOR_COUNT]
        distance_weights = initial_weights[:, :NEIGHBOR_COUNT]
        distance_weights /= distance_weights.sum(axis=1, keepdims=True)
        centered = neighborhoods - centroid[:, None, :]
        scale_floor = np.finfo(np.float64).eps * np.maximum(bandwidth, 1.0)
        for robust_cutoff in ROBUST_CUTOFFS:
            residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
            residual_median = _weighted_median(residuals, distance_weights)
            robust_scale = MAD_TO_SIGMA * _weighted_median(
                np.abs(residuals - residual_median), distance_weights
            )
            robust_scale = np.maximum(robust_scale, scale_floor)
            normalized = (residuals - residual_median) / (robust_cutoff * robust_scale)
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

        query_points = points[query_indices[start:stop]]
        offsets = neighborhoods - query_points[:, None, :]
        safe_bandwidth = np.maximum(bandwidth, np.finfo(np.float64).eps)
        tangent_x = eigenvectors[:, :, 1]
        tangent_y = eigenvectors[:, :, 2]
        local_x = (
            np.einsum("nki,ni->nk", offsets, tangent_x, optimize=True) / safe_bandwidth
        )
        local_y = (
            np.einsum("nki,ni->nk", offsets, tangent_y, optimize=True) / safe_bandwidth
        )
        local_z = (
            np.einsum("nki,ni->nk", offsets, normals, optimize=True) / safe_bandwidth
        )
        design = np.stack(
            (
                np.ones_like(local_x),
                local_x,
                local_y,
                local_x * local_x,
                local_x * local_y,
                local_y * local_y,
            ),
            axis=2,
        )
        normal_matrix = np.einsum(
            "nk,nki,nkj->nij", weights, design, design, optimize=True
        )
        normal_matrix += JET_RIDGE * np.eye(6)[None, :, :]
        right_hand_side = np.einsum(
            "nk,nki,nk->ni", weights, design, local_z, optimize=True
        )
        coefficients = np.linalg.solve(normal_matrix, right_hand_side)

        normals = (
            normals
            - coefficients[:, 1, None] * tangent_x
            - coefficients[:, 2, None] * tangent_y
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        estimated[start:stop] = normals

    return estimated
