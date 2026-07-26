"""Rebuild the tracked publication snapshot from canonical local records."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
RECORD_DIR = EXPERIMENT_DIR / "records"
PUBLICATION_DIR = EXPERIMENT_DIR / "publication" / "run-01"
CONDITIONS = ("clean", "low", "medium", "high", "stripe", "gradient")
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20_260_726
RESULT_FILES = (
    "baseline-test-results.json",
    "final-results.json",
)
PROVENANCE_CONSTANTS = {
    "version": 1,
    "study": "Explainable normal-estimation search on PCPNet",
    "search_started_utc": "2026-07-25T14:46:32Z",
    "search_ended_utc": "2026-07-26T08:34:42Z",
    "harness_commit": "f33a54480e0eeb90671227a3a41de5d7ee21dfdd",
    "finalist_commit": "c529cefe0007af493961dd386de9f5323568936b",
    "finalist_iteration": 279,
    "finalist_estimator_sha256": (
        "39065aea80932c3bb07cc0b4bfeaa7d0b9fafa01f580a980ef984fc4146e86cb"
    ),
    "baseline_commit": "b787b409d4250130d9519c528daded678b76910e",
    "baseline_estimator_sha256": (
        "95d82614e94ad4391cb0475805fa458403a49475256379f8f4ee34d0acb04f1f"
    ),
}


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON object from disk."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_records() -> list[dict[str, object]]:
    """Load and validate local canonical records."""
    records = [
        load_json(path) for path in sorted(RECORD_DIR.glob("[0-9][0-9][0-9][0-9].json"))
    ]
    if [record["iteration"] for record in records] != list(range(len(records))):
        raise RuntimeError("Canonical records are missing or non-contiguous")
    return records


def write_records(records: list[dict[str, object]]) -> None:
    """Write compact JSONL and tabular record views."""
    with (PUBLICATION_DIR / "records.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")

    result_source = EXPERIMENT_DIR / "results.tsv"
    if not result_source.is_file():
        raise FileNotFoundError(f"Missing regenerated result view: {result_source}")
    shutil.copy2(result_source, PUBLICATION_DIR / "results.tsv")

    fields = [
        "iteration",
        "status",
        "dev_rmse",
        "val_rmse",
        *[f"dev_{condition}" for condition in CONDITIONS],
        *[f"val_{condition}" for condition in CONDITIONS],
        "runtime_s",
        "memory_mb",
        "description",
    ]
    with (PUBLICATION_DIR / "condition-results.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            scores = record.get("condition_rmse", {})
            development = scores.get("development", {})
            validation = scores.get("validation", {})
            row: dict[str, object] = {
                "iteration": record["iteration"],
                "status": record["status"],
                "dev_rmse": record.get("dev_rmse"),
                "val_rmse": record.get("val_rmse"),
                "runtime_s": record.get("runtime_s"),
                "memory_mb": record.get("memory_mb"),
                "description": record["description"],
            }
            row.update(
                {
                    f"dev_{condition}": development.get(condition)
                    for condition in CONDITIONS
                }
            )
            row.update(
                {
                    f"val_{condition}": validation.get(condition)
                    for condition in CONDITIONS
                }
            )
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )


def paired_by_condition(
    baseline: dict[str, object], final: dict[str, object]
) -> dict[str, np.ndarray]:
    """Align baseline and finalist per-shape RMSE by name and condition."""
    baseline_rows = {
        (str(row["name"]), str(row["condition"])): float(row["rmse"])
        for row in baseline["shape_results"]
    }
    final_rows = {
        (str(row["name"]), str(row["condition"])): float(row["rmse"])
        for row in final["shape_results"]
    }
    if baseline_rows.keys() != final_rows.keys():
        raise RuntimeError("Baseline and finalist test shape sets differ")
    return {
        condition: np.asarray(
            [
                (baseline_rows[key], final_rows[key])
                for key in sorted(final_rows)
                if key[1] == condition
            ],
            dtype=np.float64,
        )
        for condition in CONDITIONS
    }


def bootstrap_results(
    baseline: dict[str, object], final: dict[str, object]
) -> dict[str, object]:
    """Compute paired shape-level bootstrap intervals stratified by condition."""
    paired = paired_by_condition(baseline, final)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    condition_deltas: list[np.ndarray] = []
    condition_stats: dict[str, object] = {}

    for condition, values in paired.items():
        count = values.shape[0]
        indices = rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
        baseline_samples = values[indices, 0].mean(axis=1)
        final_samples = values[indices, 1].mean(axis=1)
        delta_samples = final_samples - baseline_samples
        condition_deltas.append(delta_samples)
        final_ci = np.quantile(final_samples, [0.025, 0.975])
        delta_ci = np.quantile(delta_samples, [0.025, 0.975])
        condition_stats[condition] = {
            "shapes": count,
            "baseline_rmse": float(values[:, 0].mean()),
            "final_rmse": float(values[:, 1].mean()),
            "delta_rmse": float((values[:, 1] - values[:, 0]).mean()),
            "relative_improvement_percent": float(
                100.0
                * (values[:, 0].mean() - values[:, 1].mean())
                / values[:, 0].mean()
            ),
            "final_rmse_ci95": [float(final_ci[0]), float(final_ci[1])],
            "delta_rmse_ci95": [float(delta_ci[0]), float(delta_ci[1])],
        }

    delta_average = np.mean(np.column_stack(condition_deltas), axis=1)
    # Match the original publication analysis: estimate the finalist average
    # interval with an independent stratified bootstrap draw.
    final_average_samples: list[np.ndarray] = []
    for values in paired.values():
        count = values.shape[0]
        indices = rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
        final_average_samples.append(values[indices, 1].mean(axis=1))
    final_average = np.mean(np.column_stack(final_average_samples), axis=1)
    return {
        "version": 1,
        "method": (
            "paired nonparametric bootstrap over PCPNet shape entries, "
            "stratified by condition"
        ),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": BOOTSTRAP_SEED,
        "conditions": condition_stats,
        "average": {
            "baseline_rmse": float(baseline["rmse"]),
            "final_rmse": float(final["rmse"]),
            "delta_rmse": float(final["rmse"] - baseline["rmse"]),
            "relative_improvement_percent": float(
                100.0 * (baseline["rmse"] - final["rmse"]) / baseline["rmse"]
            ),
            "final_rmse_ci95": [
                float(np.quantile(final_average, 0.025)),
                float(np.quantile(final_average, 0.975)),
            ],
            "delta_rmse_ci95": [
                float(np.quantile(delta_average, 0.025)),
                float(np.quantile(delta_average, 0.975)),
            ],
        },
        "interpretation": (
            "Negative RMSE deltas favor the finalist. Intervals describe variation "
            "across the finite PCPNet test shape entries; they do not correct for "
            "repeated model selection on development and validation tiers."
        ),
    }


def write_provenance(records: list[dict[str, object]]) -> None:
    """Write stable study metadata and checksums for publication evidence."""
    evidence_names = (
        "baseline-test-results.json",
        "bootstrap-deltas.png",
        "bootstrap-results.json",
        "condition-results.tsv",
        "final-results.json",
        "qualitative-errors.png",
        "records.jsonl",
        "results.tsv",
    )
    checksums = {
        name: hashlib.sha256((PUBLICATION_DIR / name).read_bytes()).hexdigest()
        for name in evidence_names
    }
    provenance = {
        **PROVENANCE_CONSTANTS,
        "records": len(records),
        "agent_iterations": len(records) - 1,
        "successful_agent_iterations": sum(
            record.get("dev_rmse") is not None for record in records[1:]
        ),
        "crashed_agent_iterations": sum(
            record["status"] == "crash" for record in records[1:]
        ),
        "measured_records_including_baseline": sum(
            record.get("dev_rmse") is not None for record in records
        ),
        "validation_evaluations": sum(
            record.get("val_rmse") is not None for record in records
        ),
        "statuses": {
            status: sum(record["status"] == status for record in records)
            for status in ("keep", "provisional", "discard", "crash")
        },
        "agent": {
            "provider": "cody",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "pi_version": "0.82.0",
            "fresh_session_per_iteration": True,
            "offline_environment": True,
        },
        "environment": {
            "operating_system": "macOS 26.5.2 (25F84)",
            "cpu": "Apple M4 Max",
            "memory_bytes": 137438953472,
            "python": "3.12.11",
            "numpy": "2.5.1",
            "scipy": "1.18.0",
            "uv": "0.11.29",
        },
        "protocol": {
            "metric": "sign-invariant angular RMSE in degrees",
            "aggregation": (
                "mean per-shape RMSE within condition, then equal mean over six "
                "conditions"
            ),
            "development": "1,000 deterministic queries per training point cloud",
            "validation": "all official queries from three validation geometries",
            "test": "complete official testset_all.txt; 108 entries and 540,000 queries",
            "test_overlap_note": (
                "The official test list includes validation geometries and is not a "
                "strictly disjoint third split."
            ),
            "test_evaluation_policy": (
                "The finalist was frozen before one test evaluation. The baseline was "
                "evaluated afterward only for controlled reporting."
            ),
        },
        "checksums_sha256": checksums,
        "limitations": [
            (
                "The study has no randomized unphased control run, so it does not "
                "identify the causal effect of the workflow."
            ),
            "Development and validation were used repeatedly during model selection.",
            "API token counts and monetary cost were not persisted by the loop.",
        ],
    }
    (PUBLICATION_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main() -> None:
    """Regenerate publication records, summaries, bootstrap, and provenance."""
    records = load_records()
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    write_records(records)

    baseline = load_json(PUBLICATION_DIR / "baseline-test-results.json")
    final = load_json(PUBLICATION_DIR / "final-results.json")
    bootstrap = bootstrap_results(baseline, final)
    (PUBLICATION_DIR / "bootstrap-results.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n"
    )
    write_provenance(records)
    print(f"Rebuilt publication snapshot at {PUBLICATION_DIR}")


if __name__ == "__main__":
    main()
