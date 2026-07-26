"""Evaluate predictions without importing agent-editable estimator code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data" / "PCPNet"
ARRAY_DIR = DATA_DIR / "arrays"
CACHE_DIR = DATA_DIR / "cache"
JOB_ROOT = DATA_DIR / "jobs"
CONDITIONS = ("clean", "low", "medium", "high", "stripe", "gradient")
PGP_THRESHOLDS = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])


@dataclass(frozen=True)
class ShapeResult:
    """Metrics for one PCPNet shape and condition."""

    name: str
    condition: str
    queries: int
    squared_error_sum: float
    pgp_counts: np.ndarray

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.squared_error_sum / self.queries))


class EvaluationError(RuntimeError):
    """Raised when predictions violate the evaluator contract."""


def anonymous_order(shape_id: str) -> str:
    """Match the stable anonymous order used by the prediction runner."""
    return hashlib.sha256(f"normal-search-v1:{shape_id}".encode()).hexdigest()


def angular_errors(
    estimated_normals: np.ndarray,
    reference_normals: np.ndarray,
) -> np.ndarray:
    """Return sign-invariant angular errors in degrees."""
    estimated = np.asarray(estimated_normals, dtype=np.float64)
    reference = np.asarray(reference_normals, dtype=np.float64)
    if (
        estimated.shape != reference.shape
        or estimated.ndim != 2
        or estimated.shape[1] != 3
    ):
        msg = f"Expected estimated normals with shape {reference.shape}; got {estimated.shape}"
        raise EvaluationError(msg)
    if not np.all(np.isfinite(estimated)):
        msg = "Predictions contain non-finite normals"
        raise EvaluationError(msg)

    lengths = np.linalg.norm(estimated, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        msg = "Predictions contain zero-length normals"
        raise EvaluationError(msg)
    estimated = estimated / lengths[:, np.newaxis]
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    dots = np.abs(np.einsum("ij,ij->i", estimated, reference))
    return np.rad2deg(np.arccos(np.clip(dots, 0.0, 1.0)))


def evaluate_shape(
    shape: dict[str, object],
    tier: str,
    item_id: str,
) -> ShapeResult:
    """Evaluate saved predictions for one shape."""
    shape_id = str(shape["id"])
    name = str(shape["name"])
    condition = str(shape["condition"])
    reference_normals = np.load(ARRAY_DIR / f"{name}.normals.npy", mmap_mode="r")
    with np.load(CACHE_DIR / tier / f"{shape_id}.npz") as cache:
        query_indices = cache["query_indices"]

    prediction_path = JOB_ROOT / tier / "output" / f"{item_id}.normals.npy"
    if not prediction_path.is_file():
        msg = f"Missing predictions: {prediction_path}"
        raise EvaluationError(msg)
    estimated = np.load(prediction_path, allow_pickle=False)
    errors = angular_errors(estimated, reference_normals[query_indices])
    return ShapeResult(
        name=name,
        condition=condition,
        queries=errors.size,
        squared_error_sum=float(np.dot(errors, errors)),
        pgp_counts=np.count_nonzero(
            errors[:, np.newaxis] < PGP_THRESHOLDS,
            axis=0,
        ),
    )


def aggregate(
    results: list[ShapeResult],
    tier: str,
    run_summary: dict[str, object],
) -> dict[str, object]:
    """Aggregate using PCPNet's mean of per-shape RMSE values."""
    found_conditions = {result.condition for result in results}
    if found_conditions != set(CONDITIONS):
        msg = f"Expected all six conditions; got {sorted(found_conditions)}"
        raise EvaluationError(msg)

    by_condition: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        selected = [result for result in results if result.condition == condition]
        query_count = sum(result.queries for result in selected)
        pgp_counts = np.sum([result.pgp_counts for result in selected], axis=0)
        by_condition[condition] = {
            "rmse": float(np.mean([result.rmse for result in selected])),
            "pgp": {
                str(int(threshold)): float(count / query_count)
                for threshold, count in zip(PGP_THRESHOLDS, pgp_counts, strict=True)
            },
            "queries": query_count,
            "shapes": len(selected),
        }

    return {
        "tier": tier,
        "rmse": float(
            np.mean([float(metrics["rmse"]) for metrics in by_condition.values()])
        ),
        "conditions": by_condition,
        "runtime_s": float(run_summary["runtime_s"]),
        "peak_memory_mb": float(run_summary["peak_memory_mb"]),
        "queries": sum(result.queries for result in results),
        "shapes": len(results),
        "shape_results": [
            {
                "name": result.name,
                "condition": result.condition,
                "rmse": result.rmse,
                "queries": result.queries,
            }
            for result in results
        ],
    }


def print_summary(summary: dict[str, object]) -> None:
    """Print stable, grep-friendly experiment output."""
    print("---")
    print(f"tier:            {summary['tier']}")
    print(f"rmse:            {summary['rmse']:.6f}")
    conditions = summary["conditions"]
    assert isinstance(conditions, dict)
    for condition in CONDITIONS:
        print(f"rmse_{condition}: {conditions[condition]['rmse']:.6f}")
    print(f"runtime_s:       {summary['runtime_s']:.3f}")
    print(f"peak_memory_mb:  {summary['peak_memory_mb']:.1f}")
    print(f"queries:         {summary['queries']}")
    print(f"shapes:          {summary['shapes']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=("development", "validation"), default="development"
    )
    parser.add_argument("--json", type=Path, help="write full metrics as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest_path = DATA_DIR / f"{args.tier}-manifest.json"
    job_dir = JOB_ROOT / args.tier
    summary_path = job_dir / "output" / "run.json"
    if not manifest_path.is_file():
        msg = f"Missing prepared tier: {manifest_path}"
        raise SystemExit(msg)
    if not summary_path.is_file():
        msg = f"Missing estimator run: {summary_path}. Run `uv run run.py --tier {args.tier}`."
        raise SystemExit(msg)

    manifest = json.loads(manifest_path.read_text())
    source_ids = json.loads((job_dir / "input" / "job.json").read_text())["items"]
    if len(source_ids) != len(manifest["shapes"]):
        raise EvaluationError("Prediction job does not match the prepared tier")
    order = sorted(
        range(len(manifest["shapes"])),
        key=lambda index: anonymous_order(str(manifest["shapes"][index]["id"])),
    )
    item_by_shape = {
        shape_index: source_ids[position] for position, shape_index in enumerate(order)
    }
    run_summary = json.loads(summary_path.read_text())
    estimator_hash = hashlib.sha256(
        (EXPERIMENT_DIR / "estimator.py").read_bytes()
    ).hexdigest()
    if run_summary.get("estimator_sha256") != estimator_hash:
        raise EvaluationError("Predictions do not match the current estimator")
    if run_summary.get("tier") != args.tier:
        msg = "Estimator run summary does not match the requested tier"
        raise EvaluationError(msg)

    results: list[ShapeResult] = []
    for index, shape in enumerate(manifest["shapes"], start=1):
        result = evaluate_shape(shape, args.tier, item_by_shape[index - 1])
        results.append(result)
        print(
            f"[{index:02d}/{len(manifest['shapes']):02d}] scored",
            file=sys.stderr,
        )

    summary = aggregate(results, args.tier, run_summary)
    print_summary(summary)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
