"""Initialize and inspect persistent state for the stateless research loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RECORD_DIR = EXPERIMENT_DIR / "records"
STATE_PATH = EXPERIMENT_DIR / "state.json"
RESULTS_PATH = EXPERIMENT_DIR / "results.tsv"
NOTES_PATH = EXPERIMENT_DIR / "notes.md"
STOP_PATH = EXPERIMENT_DIR / "STOP"


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=EXPERIMENT_DIR, text=True
    ).strip()


def evaluate(tier: str) -> dict[str, object]:
    subprocess.run(
        ["uv", "run", "run.py", "--tier", tier],
        cwd=EXPERIMENT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["uv", "run", "evaluate.py", "--tier", tier],
        cwd=EXPERIMENT_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    metrics: dict[str, object] = {"condition_rmse": {}}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"rmse", "runtime_s", "peak_memory_mb"}:
            metrics[key] = float(value.strip())
        elif separator and key.startswith("rmse_"):
            metrics["condition_rmse"][key.removeprefix("rmse_")] = float(value.strip())
    if not {"rmse", "runtime_s", "peak_memory_mb"}.issubset(metrics):
        raise RuntimeError(f"Could not parse {tier} baseline metrics")
    return metrics


def initialize() -> None:
    """Measure and record the clean baseline."""
    if git("status", "--porcelain"):
        raise SystemExit("Working tree must be clean before initialization")
    if RECORD_DIR.exists() or any(
        path.exists() for path in (STATE_PATH, RESULTS_PATH, NOTES_PATH)
    ):
        raise SystemExit("Research state already exists")
    if not git("branch", "--show-current").startswith("research/"):
        raise SystemExit("Initialize on a branch named research/<tag>")

    print("Measuring baseline development and validation scores...")
    development = evaluate("development")
    validation = evaluate("validation")
    commit = git("rev-parse", "HEAD")
    frontier = {
        "commit": commit,
        "dev_rmse": development["rmse"],
        "val_rmse": validation["rmse"],
        "runtime_s": development["runtime_s"],
        "status": "keep",
    }
    record = {
        "version": 1,
        "harness_commit": commit,
        "estimator_sha256": hashlib.sha256(
            (EXPERIMENT_DIR / "estimator.py").read_bytes()
        ).hexdigest(),
        "iteration": 0,
        "candidate_commit": commit,
        "dev_rmse": development["rmse"],
        "val_rmse": validation["rmse"],
        "runtime_s": development["runtime_s"],
        "memory_mb": development["peak_memory_mb"],
        "status": "keep",
        "complexity": "baseline",
        "description": "fixed k=112 PCA",
        "rationale": "local covariance tangent plane",
        "source": "Hoppe1992",
        "kind": "parameter",
        "condition_rmse": {
            "development": development["condition_rmse"],
            "validation": validation["condition_rmse"],
        },
        "validation_interval": 10,
        "attempts_since_validation": 0,
        "frontier": frontier,
        "validated": {
            key: frontier[key]
            for key in ("commit", "dev_rmse", "val_rmse", "runtime_s")
        },
    }

    temporary_dir = EXPERIMENT_DIR / ".records.tmp"
    temporary_dir.mkdir()
    (temporary_dir / "0000.json").write_text(json.dumps(record, indent=2) + "\n")
    temporary_dir.replace(RECORD_DIR)
    NOTES_PATH.write_text(
        "# Research notes\n\n"
        "## Frontier\n\n"
        f"- Validated baseline: development {development['rmse']:.6f}, "
        f"validation {validation['rmse']:.6f}.\n"
        "- Method: fixed-neighborhood PCA with k=112.\n\n"
        "## Confirmed findings\n\n- None beyond the baseline.\n\n"
        "## Rejected ideas\n\n- None.\n\n"
        "## Promising next ideas\n\n"
        "- Establish a coarse fixed-k curve.\n"
        "- Try simple distance weighting and geometry-driven scale selection.\n"
    )
    STOP_PATH.unlink(missing_ok=True)
    rebuild_views()
    print(
        f"Initialized at {commit[:7]}: dev={development['rmse']:.6f}, "
        f"val={validation['rmse']:.6f}"
    )


def rebuild_views() -> None:
    """Ask the controller module to regenerate state and results views."""
    from loop import load_records, sync_views

    sync_views(load_records())


def show_status() -> None:
    if not RECORD_DIR.is_dir():
        print("Research state is not initialized")
        return
    rebuild_views()
    print(STATE_PATH.read_text().rstrip())
    print(f"head: {git('rev-parse', 'HEAD')}")
    status = git("status", "--porcelain", "--untracked-files=no")
    print(f"working_tree: {'clean' if not status else 'dirty'}")
    print(f"stop_requested: {STOP_PATH.exists()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "stop", "resume"))
    return parser.parse_args()


def main() -> None:
    command = parse_arguments().command
    if command == "init":
        initialize()
    elif command == "status":
        show_status()
    elif command == "stop":
        STOP_PATH.touch()
        print("Stop requested")
    elif command == "resume":
        STOP_PATH.unlink(missing_ok=True)
        print("Stop request cleared")


if __name__ == "__main__":
    main()
