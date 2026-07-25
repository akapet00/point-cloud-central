"""Run the editable estimator in an isolated, label-free job directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data" / "PCPNet"
INPUT_DIR = DATA_DIR / "inputs"
CACHE_DIR = DATA_DIR / "cache"
JOB_ROOT = DATA_DIR / "jobs"
ESTIMATOR_PATH = EXPERIMENT_DIR / "estimator.py"
WORKER_PATH = EXPERIMENT_DIR / "worker.py"
WORKER_TIMEOUT_SECONDS = 300
WORKER_MEMORY_BYTES = 8 * 1024**3
WORKER_OUTPUT_BYTES = 512 * 1024**2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=("development", "validation"), default="development"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_manifest = INPUT_DIR / args.tier / "inputs.json"
    cache_dir = CACHE_DIR / args.tier
    if not input_manifest.is_file() or not cache_dir.is_dir():
        msg = (
            f"Missing prepared tier. Run `uv run prepare.py cache --tier {args.tier}`."
        )
        raise SystemExit(msg)
    if sys.platform != "darwin":
        raise SystemExit("The estimator sandbox is currently configured for macOS")

    job_dir = JOB_ROOT / args.tier
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    shutil.rmtree(job_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)
    output_dir.mkdir()

    source_ids = json.loads(input_manifest.read_text())["shape_ids"]
    source_ids = sorted(source_ids, key=anonymous_order)
    item_ids = [f"item-{index:03d}" for index in range(1, len(source_ids) + 1)]
    for source_id, item_id in zip(source_ids, item_ids, strict=True):
        shutil.copyfile(
            INPUT_DIR / args.tier / f"{source_id}.points.npy",
            input_dir / f"{item_id}.points.npy",
        )
        shutil.copyfile(
            cache_dir / f"{source_id}.npz",
            input_dir / f"{item_id}.neighbors.npz",
        )
    (input_dir / "job.json").write_text(json.dumps({"items": item_ids}) + "\n")

    command = [
        "sandbox-exec",
        "-p",
        sandbox_profile(input_dir, output_dir),
        sys.executable,
        str(WORKER_PATH),
        str(input_dir),
        str(output_dir),
    ]
    environment = {
        "HOME": str(output_dir),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
    }
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        preexec_fn=limit_worker,
    )
    started = time.monotonic()
    while process.poll() is None:
        elapsed = time.monotonic() - started
        memory_bytes = process_group_memory_bytes(process.pid)
        if elapsed > WORKER_TIMEOUT_SECONDS or memory_bytes > WORKER_MEMORY_BYTES:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            reason = (
                "timed out"
                if elapsed > WORKER_TIMEOUT_SECONDS
                else "exceeded memory limit"
            )
            raise SystemExit(f"Estimator worker {reason}")
        time.sleep(0.2)
    return_code = process.returncode
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    summary_path = output_dir / "run.json"
    summary = json.loads(summary_path.read_text())
    summary["tier"] = args.tier
    summary["estimator_sha256"] = hashlib.sha256(
        ESTIMATOR_PATH.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("---")
    print(f"tier:            {args.tier}")
    print(f"runtime_s:       {summary['runtime_s']:.3f}")
    print(f"peak_memory_mb:  {summary['peak_memory_mb']:.1f}")
    print(f"shapes:          {summary['items']}")


def process_group_memory_bytes(process_group: int) -> int:
    """Return total resident memory for the isolated process group."""
    result = subprocess.run(
        ["ps", "-o", "rss=", "-g", str(process_group)],
        text=True,
        capture_output=True,
        check=False,
    )
    return sum(int(value) for value in result.stdout.split()) * 1024


def limit_worker() -> None:
    """Apply CPU and output-size limits before starting the worker."""
    for limit, requested in (
        (resource.RLIMIT_CPU, WORKER_TIMEOUT_SECONDS),
        (resource.RLIMIT_FSIZE, WORKER_OUTPUT_BYTES),
    ):
        _, hard = resource.getrlimit(limit)
        value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(limit, (value, value))


def anonymous_order(shape_id: str) -> str:
    """Return a stable permutation key that reveals no condition label."""
    return hashlib.sha256(f"normal-search-v1:{shape_id}".encode()).hexdigest()


def sandbox_profile(input_dir: Path, output_dir: Path) -> str:
    """Deny project and user data, allowing read-only input and writable output."""
    protected = (REPO_ROOT, Path.home())
    denies = " ".join(f'(subpath "{path.resolve()}")' for path in protected)
    allowed = (
        Path(sys.base_prefix),
        Path(sys.prefix),
        WORKER_PATH,
        ESTIMATOR_PATH,
        input_dir,
        output_dir,
    )
    allows = " ".join(f'(subpath "{path.resolve()}")' for path in allowed)
    return " ".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny signal)",
            f'(allow process-exec (literal "{Path(sys.executable).resolve()}"))',
            f"(deny file-read* {denies})",
            "(deny file-write*)",
            f"(allow file-read* {allows})",
            f'(allow file-write* (subpath "{output_dir.resolve()}"))',
        ]
    )


if __name__ == "__main__":
    main()
