"""Execute editable estimator code on one anonymous prepared job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ESTIMATOR_PATH = Path(__file__).resolve().with_name("estimator.py")


def load_estimator():
    """Load the agent-edited estimator."""
    spec = importlib.util.spec_from_file_location("normal_estimator", ESTIMATOR_PATH)
    if spec is None or spec.loader is None:
        msg = f"Cannot load {ESTIMATOR_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = getattr(module, "estimate_normals", None)
    if not callable(estimator):
        msg = "estimator.py must define estimate_normals"
        raise RuntimeError(msg)
    return estimator


def peak_memory_mb() -> float:
    """Return whole-process peak resident memory in MiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return float(usage / divisor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    item_ids = json.loads((input_dir / "job.json").read_text())["items"]

    start = time.perf_counter()
    estimator = load_estimator()
    initialization_s = time.perf_counter() - start

    for index, item_id in enumerate(item_ids, start=1):
        points = np.load(input_dir / f"{item_id}.points.npy", mmap_mode="r")
        with np.load(input_dir / f"{item_id}.neighbors.npz") as cache:
            query_indices = cache["query_indices"].astype(np.int64)
            neighbor_indices = cache["neighbor_indices"].astype(np.int64)
            neighbor_distances = cache["neighbor_distances"].copy()

        item_start = time.perf_counter()
        estimated = np.asarray(
            estimator(
                np.asarray(points),
                query_indices,
                neighbor_indices,
                neighbor_distances,
            )
        )
        item_seconds = time.perf_counter() - item_start
        expected_shape = (query_indices.size, 3)
        if estimated.shape != expected_shape:
            msg = f"Expected predictions {expected_shape}, got {estimated.shape}"
            raise ValueError(msg)
        np.save(output_dir / f"{item_id}.normals.npy", estimated)
        print(
            f"[{index:02d}/{len(item_ids):02d}] item complete: {item_seconds:.3f} s",
            file=sys.stderr,
        )

    summary = {
        "runtime_s": time.perf_counter() - start,
        "initialization_s": initialization_s,
        "peak_memory_mb": peak_memory_mb(),
        "items": len(item_ids),
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
