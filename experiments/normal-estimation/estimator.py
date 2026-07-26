"""Agent-editable normal estimator.

This is the only source file changed by the research loop. It receives point
coordinates and precomputed neighbors, never references or condition labels.
"""

from __future__ import annotations

import numpy as np

NEIGHBOR_COUNT = 112
FIRST_STATISTIC_COUNT = 64
FINAL_NEIGHBOR_COUNT = 128
FINAL_STATISTIC_COUNT = 32
INITIAL_NEIGHBOR_COUNT = 224
DISTANCE_DECAY = 2.0
TUKEY_CUTOFFS = (4.15, 2.77, 2.77)
THIRD_REFINEMENT_MAX_THICKNESS = 0.1
THIRD_NORMAL_EXTRAPOLATION = 0.8
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
    positional noise. Two Tukey-biweight IRLS steps refine that normal using
    query-local residual statistics. A third step is accepted only for a thin
    local sheet, where another redescending fit is unlikely to reject noise.
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

        scale_floor = np.finfo(np.float64).eps * np.maximum(bandwidth, 1.0)
        refinement_counts = (
            NEIGHBOR_COUNT,
            FINAL_NEIGHBOR_COUNT,
            FINAL_NEIGHBOR_COUNT,
        )
        statistic_counts = (
            FIRST_STATISTIC_COUNT,
            FINAL_STATISTIC_COUNT,
            FINAL_STATISTIC_COUNT,
        )
        preceding_normals = normals
        for step, (tukey_cutoff, refinement_count, statistic_count) in enumerate(
            zip(TUKEY_CUTOFFS, refinement_counts, statistic_counts, strict=True)
        ):
            neighborhoods = initial_neighborhoods[:, :refinement_count]
            distance_weights = initial_weights[:, :refinement_count].copy()
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
            refined_normals = eigenvectors[:, :, 0]
            if step == 2:
                thin_sheet = robust_scale <= THIRD_REFINEMENT_MAX_THICKNESS * bandwidth
                dot = np.einsum("ni,ni->n", refined_normals, normals, optimize=True)
                aligned_normals = refined_normals * np.where(
                    dot[:, None] < 0.0, -1.0, 1.0
                )
                preceding_dot = np.einsum(
                    "ni,ni->n", preceding_normals, normals, optimize=True
                )
                aligned_preceding = preceding_normals * np.where(
                    preceding_dot[:, None] < 0.0, -1.0, 1.0
                )
                preceding_update = normals - aligned_preceding
                latest_update = aligned_normals - normals
                preceding_squared = np.einsum(
                    "ni,ni->n", preceding_update, preceding_update, optimize=True
                )
                projection_scale = np.divide(
                    np.einsum(
                        "ni,ni->n", latest_update, preceding_update, optimize=True
                    ),
                    preceding_squared,
                    out=np.zeros_like(preceding_squared),
                    where=preceding_squared > np.finfo(np.float64).eps,
                )
                orthogonal_update = (
                    latest_update - projection_scale[:, None] * preceding_update
                )
                latest_length = np.linalg.norm(latest_update, axis=1, keepdims=True)
                orthogonal_length = np.linalg.norm(
                    orthogonal_update, axis=1, keepdims=True
                )
                orthogonal_update = np.divide(
                    orthogonal_update * latest_length,
                    orthogonal_length,
                    out=np.zeros_like(orthogonal_update),
                    where=orthogonal_length > np.finfo(np.float64).eps,
                )
                extrapolated = (
                    aligned_normals + THIRD_NORMAL_EXTRAPOLATION * orthogonal_update
                )
                extrapolated /= np.linalg.norm(extrapolated, axis=1, keepdims=True)
                normals = np.where(thin_sheet, extrapolated, normals)
            else:
                if step == 1:
                    preceding_normals = normals
                normals = refined_normals

        estimated[start:stop] = normals

    return estimated
