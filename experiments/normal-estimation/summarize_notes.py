"""Rebuild concise cross-iteration memory from canonical experiment records."""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RECORD_DIR = EXPERIMENT_DIR / "records"
NOTES_PATH = EXPERIMENT_DIR / "notes.md"
CONDITION_MEMORY_PATH = EXPERIMENT_DIR / "condition-memory.md"
CONDITIONS = ("clean", "low", "medium", "high", "stripe", "gradient")


def atomic_text(path: Path, text: str) -> None:
    """Replace one generated text view atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def condition_scores(
    records: list[dict[str, object]], commit: object, tier: str
) -> dict[str, float] | None:
    """Find recorded condition scores for a candidate commit and tier."""
    for record in reversed(records):
        if record.get("candidate_commit") != commit:
            continue
        tiers = record.get("condition_rmse")
        if not isinstance(tiers, dict) or not isinstance(tiers.get(tier), dict):
            continue
        return {key: float(value) for key, value in tiers[tier].items()}
    return None


def failure_summary(record: dict[str, object]) -> str:
    """Return the most precise durable diagnostic available for a crash."""
    failure = record.get("failure")
    if not isinstance(failure, dict):
        return "legacy record has no structured failure diagnostic"
    stage = failure.get("stage", "unknown stage")
    message = failure.get("message", "unknown failure")
    return f"{stage}: {message}"


def build_notes(records: list[dict[str, object]]) -> str:
    """Build compact cross-iteration methodological memory."""
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

    lines.extend(["", "## Recent rejected ideas and failures", ""])
    rejected = [
        record for record in records[1:] if record["status"] in {"discard", "crash"}
    ]
    for record in rejected[-30:]:
        diagnostic = (
            f" — {failure_summary(record)}" if record["status"] == "crash" else ""
        )
        lines.append(
            f"- #{record['iteration']} ({record['status']}): "
            f"{record['description']}{diagnostic}"
        )
    if not rejected:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Guidance",
            "",
            "- Read condition-memory.md before choosing the next hypothesis.",
            "- Make one independently ablatable change per experiment.",
            "- Avoid repeating descriptions already listed above.",
            "- Use results.tsv for complete quantitative history.",
            "",
        ]
    )
    return "\n".join(lines)


def build_condition_memory(records: list[dict[str, object]]) -> str:
    """Build per-condition score and trade-off memory for the research agent."""
    latest = records[-1]
    baseline = records[0]
    frontier = latest["frontier"]
    validated = latest["validated"]
    baseline_dev = condition_scores(
        records, baseline["candidate_commit"], "development"
    )
    frontier_dev = condition_scores(records, frontier["commit"], "development")
    validated_val = condition_scores(records, validated["commit"], "validation")
    baseline_val = condition_scores(records, baseline["candidate_commit"], "validation")
    if baseline_dev is None or frontier_dev is None:
        raise RuntimeError(
            "Frontier condition metrics are missing from canonical records"
        )

    lines = [
        "# Per-condition research memory",
        "",
        "Lower RMSE is better. Deltas are candidate minus incumbent, so negative is an improvement.",
        "This file is regenerated from canonical records after every iteration.",
        "",
        "## Current frontier",
        "",
        "| Condition | Baseline dev | Frontier dev | Dev delta | Baseline val | Validated val | Val delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        baseline_validation = baseline_val.get(condition) if baseline_val else None
        current_validation = validated_val.get(condition) if validated_val else None
        validation_cells = (
            (
                f"{baseline_validation:.6f}",
                f"{current_validation:.6f}",
                f"{current_validation - baseline_validation:+.6f}",
            )
            if baseline_validation is not None and current_validation is not None
            else (
                "—",
                "—",
                "—",
            )
        )
        lines.append(
            f"| {condition} | {baseline_dev[condition]:.6f} | "
            f"{frontier_dev[condition]:.6f} | "
            f"{frontier_dev[condition] - baseline_dev[condition]:+.6f} | "
            f"{validation_cells[0]} | {validation_cells[1]} | "
            f"{validation_cells[2]} |"
        )

    measured = [
        record
        for record in records[1:]
        if isinstance(record.get("condition_rmse"), dict)
        and isinstance(record["condition_rmse"].get("development"), dict)
    ]
    lines.extend(
        [
            "",
            "## Development trade-offs by measured attempt",
            "",
            "Each delta compares the candidate with the frontier from which that attempt started.",
            "",
            "| Iteration | Status | Clean | Low | Medium | High | Stripe | Gradient |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    records_by_iteration = {int(record["iteration"]): record for record in records}
    for record in measured[-30:]:
        scores = record["condition_rmse"]["development"]
        previous = records_by_iteration[int(record["iteration"]) - 1]
        incumbent = condition_scores(
            records, previous["frontier"]["commit"], "development"
        )
        if incumbent is None:
            continue
        deltas = [float(scores[name]) - incumbent[name] for name in CONDITIONS]
        lines.append(
            f"| {record['iteration']} | {record['status']} | "
            + " | ".join(f"{delta:+.6f}" for delta in deltas)
            + " |"
        )

    lines.extend(["", "## Interpretation", ""])
    if measured:
        for record in measured[-12:]:
            scores = record["condition_rmse"]["development"]
            previous = records_by_iteration[int(record["iteration"]) - 1]
            incumbent = condition_scores(
                records, previous["frontier"]["commit"], "development"
            )
            if incumbent is None:
                continue
            improved = [
                name for name in CONDITIONS if float(scores[name]) < incumbent[name]
            ]
            worsened = [
                name for name in CONDITIONS if float(scores[name]) >= incumbent[name]
            ]
            lines.append(
                f"- #{record['iteration']} {record['description']}: improved "
                f"{', '.join(improved) or 'none'}; worsened "
                f"{', '.join(worsened) or 'none'}."
            )
    else:
        lines.append("- No post-baseline candidates have been measured.")
    lines.append("")
    return "\n".join(lines)


def rebuild_research_memory() -> None:
    """Regenerate all agent-facing research memory from immutable records."""
    records = [
        json.loads(path.read_text()) for path in sorted(RECORD_DIR.glob("*.json"))
    ]
    if not records:
        raise SystemExit("No experiment records")
    atomic_text(NOTES_PATH, build_notes(records))
    atomic_text(CONDITION_MEMORY_PATH, build_condition_memory(records))


def main() -> None:
    rebuild_research_memory()


if __name__ == "__main__":
    main()
