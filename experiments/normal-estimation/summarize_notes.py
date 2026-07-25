"""Rebuild concise cross-iteration notes from canonical experiment records."""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RECORD_DIR = EXPERIMENT_DIR / "records"
NOTES_PATH = EXPERIMENT_DIR / "notes.md"


def main() -> None:
    records = [
        json.loads(path.read_text()) for path in sorted(RECORD_DIR.glob("*.json"))
    ]
    if not records:
        raise SystemExit("No experiment records")
    latest = records[-1]
    validated = latest["validated"]
    frontier = latest["frontier"]

    lines = [
        "# Research notes",
        "",
        "## Frontier",
        "",
        f"- Validated: dev {validated['dev_rmse']:.6f}, val {validated['val_rmse']:.6f}, commit {validated['commit'][:7]}.",
        f"- Current: dev {frontier['dev_rmse']:.6f}, status {frontier['status']}, commit {frontier['commit'][:7]}.",
        "",
        "## Retained findings",
        "",
    ]
    retained_commits = {frontier["commit"], validated["commit"]}
    retained = [
        record
        for record in records[1:]
        if record.get("candidate_commit") in retained_commits
        and record["status"] in {"keep", "provisional"}
    ]
    lines.extend(
        f"- #{record['iteration']}: {record['description']} — {record['rationale']}"
        for record in retained[-20:]
    )
    if not retained:
        lines.append("- None beyond the baseline.")

    lines.extend(["", "## Recent rejected ideas", ""])
    rejected = [
        record for record in records[1:] if record["status"] in {"discard", "crash"}
    ]
    lines.extend(
        f"- #{record['iteration']} ({record['status']}): {record['description']}"
        for record in rejected[-30:]
    )
    if not rejected:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Guidance",
            "",
            "- Prefer one geometrically motivated change at a time.",
            "- Avoid repeating descriptions already listed above.",
            "- Use results.tsv for complete quantitative history.",
            "",
        ]
    )
    temporary = NOTES_PATH.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(NOTES_PATH)


if __name__ == "__main__":
    main()
