"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
FINAL_NEIGHBOR_COUNT = 128
FINAL_STATISTIC_COUNT = 64
INITIAL_NEIGHBOR_COUNT = 224
DISTANCE_DECAY = 2.0
TUKEY_CUTOFFS = (4.62, 2.77)
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
    positional noise. Two Tukey-biweight IRLS steps refine that normal; the
    final 128-point covariance uses tangent-plane distance after the first
    correction and robust statistics from its nearest 64 samples.
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

        scale_floor = np.finfo(np.float64).eps * np.maximum(bandwidth, 1.0)
        query_points = points[query_indices[start:stop]]
        refinement_counts = (NEIGHBOR_COUNT, FINAL_NEIGHBOR_COUNT)
        statistic_counts = (NEIGHBOR_COUNT, FINAL_STATISTIC_COUNT)
        for pass_index, (tukey_cutoff, refinement_count, statistic_count) in enumerate(
            zip(TUKEY_CUTOFFS, refinement_counts, statistic_counts, strict=True)
        ):
            neighborhoods = initial_neighborhoods[:, :refinement_count]
            if pass_index == 0:
                distance_weights = initial_weights[:, :refinement_count].copy()
            else:
                query_offsets = neighborhoods - query_points[:, None, :]
                normal_offsets = np.einsum(
                    "nki,ni->nk", query_offsets, normals, optimize=True
                )
                tangent_squared = np.maximum(
                    np.einsum(
                        "nki,nki->nk", query_offsets, query_offsets, optimize=True
                    )
                    - normal_offsets * normal_offsets,
                    0.0,
                )
                scaled_tangent_squared = np.divide(
                    tangent_squared,
                    bandwidth * bandwidth,
                    out=np.zeros_like(tangent_squared),
                    where=bandwidth > 0.0,
                )
                distance_weights = np.exp(-DISTANCE_DECAY * scaled_tangent_squared)
            distance_weights /= distance_weights.sum(axis=1, keepdims=True)
            centered = neighborhoods - centroid[:, None, :]
            residuals = np.einsum("nki,ni->nk", centered, normals, optimize=True)
            statistic_residuals = residuals[:, :statistic_count]
            statistic_weights = distance_weights[:, :statistic_count]
            residual_median = _weighted_median(statistic_residuals, statistic_weights)
            robust_scale = MAD_TO_SIGMA * _weighted_median(
                np.abs(statistic_residuals - residual_median), statistic_weights
            )
            robust_scale = np.maximum(robust_scale, scale_floor)
            normalized = (residuals - residual_median) / (tukey_cutoff * robust_scale)
            inside = normalized * normalized < 1.0
            robust_weights = np.square(1.0 - normalized * normalized) * inside

            weights = distance_weights * robust_weights
            weights /= weights.sum(axis=1, keepdims=True)
            centroid = np.einsum("nk,nki->ni", weights, neighborhoods, optimize=True)
            covariance = np.einsum(
                "nk,nki,nkj->nij",
                weights,
                neighborhoods - centroid[:, None, :],
                neighborhoods - centroid[:, None, :],
                optimize=True,
            )
            _, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]

        estimated[start:stop] = normals

    return estimated
