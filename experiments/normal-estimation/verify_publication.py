"""Verify the frozen normal-estimation publication snapshot and Git refs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PUBLICATION_DIR = EXPERIMENT_DIR / "publication" / "run-01"
RELEASE_TAG = "normal-estimation-study-v1"
EXPECTED_FIGURE_FILES = (
    "accuracy-runtime.png",
    "benchmark-comparison.png",
    "bootstrap-deltas-blog.png",
    "condition-progress.png",
    "qualitative-errors-blog.png",
    "search-progress.png",
    "validation-progress.png",
)


def git(*arguments: str) -> str:
    """Run a read-only Git command in the repository."""
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    """Validate checksums, record count, estimator hashes, and candidate refs."""
    provenance = json.loads((PUBLICATION_DIR / "provenance.json").read_text())
    for name, expected in provenance["checksums_sha256"].items():
        actual = hashlib.sha256((PUBLICATION_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {expected}")

    figure_dir = PUBLICATION_DIR / "figures"
    missing_figures = [
        name for name in EXPECTED_FIGURE_FILES if not (figure_dir / name).is_file()
    ]
    if missing_figures:
        raise RuntimeError(
            "Missing publication-ready figures: " + ", ".join(missing_figures)
        )

    record_count = sum(
        bool(line.strip())
        for line in (PUBLICATION_DIR / "records.jsonl").read_text().splitlines()
    )
    if record_count != provenance["records"]:
        raise RuntimeError(
            f"Expected {provenance['records']} records, found {record_count}"
        )

    estimator_hash = hashlib.sha256(
        (EXPERIMENT_DIR / "estimator.py").read_bytes()
    ).hexdigest()
    if estimator_hash != provenance["finalist_estimator_sha256"]:
        raise RuntimeError("Tracked estimator.py is not the frozen finalist")

    if not git("tag", "--list", RELEASE_TAG):
        raise RuntimeError(f"Missing release tag: {RELEASE_TAG}")

    missing_tags: list[str] = []
    records = [
        json.loads(line)
        for line in (PUBLICATION_DIR / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records[1:]:
        if record.get("candidate_commit") is None:
            continue
        tag = f"normal-search/{int(record['iteration']):04d}"
        if not git("tag", "--list", tag):
            missing_tags.append(tag)
    if missing_tags:
        examples = ", ".join(missing_tags[:5])
        raise RuntimeError(f"Missing {len(missing_tags)} candidate tags: {examples}")

    print(
        f"Verified {record_count} records, "
        f"{len(provenance['checksums_sha256'])} checksums, "
        f"{len(EXPECTED_FIGURE_FILES)} publication figures, and "
        f"{sum(record.get('candidate_commit') is not None for record in records[1:])} "
        "candidate refs."
    )


if __name__ == "__main__":
    main()
