"""Run one fresh, bounded Pi process per normal-estimation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PROMPT_PATH = EXPERIMENT_DIR / "iteration.md"
PROPOSAL_PATH = EXPERIMENT_DIR / "proposal.json"
AGENT_WORKSPACE = EXPERIMENT_DIR / ".agent-workspace"
STATE_PATH = EXPERIMENT_DIR / "state.json"
RESULTS_PATH = EXPERIMENT_DIR / "results.tsv"
STOP_PATH = EXPERIMENT_DIR / "STOP"
LOCK_PATH = EXPERIMENT_DIR / ".loop.lock"
RECORD_DIR = EXPERIMENT_DIR / "records"
LOG_DIR = EXPERIMENT_DIR / "agent-logs"
ESTIMATOR_PATH = EXPERIMENT_DIR / "estimator.py"
RESULT_FIELDS = (
    "iteration",
    "candidate_commit",
    "dev_rmse",
    "val_rmse",
    "runtime_s",
    "memory_mb",
    "status",
    "failure_stage",
    "failure_message",
    "complexity",
    "description",
    "rationale",
    "source",
)
AGENT_READ_FILES = (
    "state.json",
    "results.tsv",
    "notes.md",
    "condition-memory.md",
    "README.md",
    "iteration.md",
)


class ExperimentFailure(RuntimeError):
    """A stage-specific failure suitable for a canonical experiment record."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        command: list[str] | None = None,
        return_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.command = command
        self.return_code = return_code
        self.detail = detail

    def as_record(self) -> dict[str, object]:
        diagnostic: dict[str, object] = {
            "stage": self.stage,
            "message": str(self),
        }
        if self.command is not None:
            diagnostic["command"] = self.command
        if self.return_code is not None:
            diagnostic["return_code"] = self.return_code
        if self.detail:
            diagnostic["detail"] = self.detail
        return diagnostic


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=EXPERIMENT_DIR, text=True
    ).strip()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_records() -> list[dict[str, object]]:
    paths = sorted(RECORD_DIR.glob("[0-9][0-9][0-9][0-9].json"))
    records = [json.loads(path.read_text()) for path in paths]
    if not records or [record["iteration"] for record in records] != list(
        range(len(records))
    ):
        raise RuntimeError("Experiment records are missing or non-contiguous")
    return records


def sync_views(records: list[dict[str, object]]) -> dict[str, object]:
    """Regenerate disposable state and TSV views from canonical records."""
    baseline = records[0]
    latest = records[-1]
    state = {
        "version": 1,
        "iteration": latest["iteration"],
        "validation_interval": latest["validation_interval"],
        "attempts_since_validation": latest["attempts_since_validation"],
        "baseline": baseline["validated"],
        "validated": latest["validated"],
        "frontier": latest["frontier"],
    }
    atomic_json(STATE_PATH, state)
    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        for record in records:
            failure = record.get("failure")
            failure = failure if isinstance(failure, dict) else {}
            row = {
                field: "" if record.get(field) is None else record.get(field, "")
                for field in RESULT_FIELDS
            }
            row["failure_stage"] = failure.get("stage", "")
            row["failure_message"] = failure.get("message", "")
            writer.writerow(row)
    return state


def validate_record_frontier(records: list[dict[str, object]]) -> None:
    """Verify that canonical records point to durable matching estimators."""
    for label in ("validated", "frontier"):
        commit = str(records[-1][label]["commit"])
        try:
            estimator = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{commit}:experiments/normal-estimation/estimator.py",
                ],
                cwd=EXPERIMENT_DIR,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Recorded {label} commit is unavailable: {commit}"
            ) from error
        matching_records = [
            record
            for record in records
            if record.get("candidate_commit") == commit
            and record.get("status") in {"keep", "provisional"}
        ]
        if not matching_records:
            raise RuntimeError(f"No retained canonical record owns {label} {commit}")
        expected_hash = str(matching_records[-1]["estimator_sha256"])
        actual_hash = __import__("hashlib").sha256(estimator).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Recorded {label} estimator hash does not match {commit}"
            )


def log_tail(path: Path, *, lines: int = 40) -> str | None:
    """Return a bounded diagnostic tail from a command log."""
    if not path.is_file():
        return None
    selected = path.read_text(errors="replace").splitlines()[-lines:]
    return "\n".join(selected) or None


def evaluate(tier: str, log_path: Path) -> dict[str, object]:
    commands = (
        (f"{tier}.prediction", ["uv", "run", "run.py", "--tier", tier]),
        (f"{tier}.scoring", ["uv", "run", "evaluate.py", "--tier", tier]),
    )
    with log_path.open("w") as log:
        for stage, command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=EXPERIMENT_DIR,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                log.flush()
                raise ExperimentFailure(
                    stage,
                    f"command exited with status {error.returncode}",
                    command=command,
                    return_code=error.returncode,
                    detail=log_tail(log_path),
                ) from error

    metrics: dict[str, object] = {"condition_rmse": {}}
    try:
        for line in log_path.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"rmse", "runtime_s", "peak_memory_mb"}:
                metrics[key] = float(value.strip())
            elif separator and key.startswith("rmse_"):
                condition_metrics = metrics["condition_rmse"]
                assert isinstance(condition_metrics, dict)
                condition_metrics[key.removeprefix("rmse_")] = float(value.strip())
    except ValueError as error:
        raise ExperimentFailure(
            f"{tier}.metrics",
            f"could not parse numeric metric: {error}",
            detail=log_tail(log_path),
        ) from error
    missing = {"rmse", "runtime_s", "peak_memory_mb"} - metrics.keys()
    if missing:
        raise ExperimentFailure(
            f"{tier}.metrics",
            f"missing metrics: {', '.join(sorted(missing))}",
            detail=log_tail(log_path),
        )
    return metrics


def format_and_lint_candidate() -> None:
    """Apply deterministic safe fixes and formatting, then verify Ruff."""
    commands = (
        ("quality.ruff-fix", ["uv", "run", "ruff", "check", "--fix", "estimator.py"]),
        ("quality.format", ["uv", "run", "ruff", "format", "estimator.py"]),
        ("quality.lint", ["uv", "run", "ruff", "check", "estimator.py"]),
    )
    for stage, command in commands:
        result = subprocess.run(
            command,
            cwd=EXPERIMENT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise ExperimentFailure(
                stage,
                f"Ruff exited with status {result.returncode}",
                command=command,
                return_code=result.returncode,
                detail=detail or None,
            )


def prepare_proposal() -> None:
    atomic_json(
        PROPOSAL_PATH,
        {
            "description": "REPLACE",
            "rationale": "REPLACE",
            "complexity": "low",
            "kind": "mechanism",
            "source": "original",
        },
    )


def validate_proposal() -> dict[str, str]:
    proposal = json.loads(PROPOSAL_PATH.read_text())
    expected = {"description", "rationale", "complexity", "kind", "source"}
    if set(proposal) != expected:
        raise ValueError("proposal.json has an invalid schema")
    if proposal["complexity"] not in {"baseline", "low", "medium"}:
        raise ValueError("Invalid complexity")
    if proposal["kind"] not in {"parameter", "mechanism", "simplification"}:
        raise ValueError("Invalid experiment kind")
    if any(
        not isinstance(proposal[key], str) or not proposal[key].strip()
        for key in expected
    ):
        raise ValueError("Proposal values must be non-empty strings")
    if "REPLACE" in proposal.values():
        raise ValueError("Proposal template was not completed")
    return proposal


def prepare_agent_workspace() -> None:
    """Expose only the files Pi needs, with editable copies isolated from the repo."""
    shutil.rmtree(AGENT_WORKSPACE, ignore_errors=True)
    AGENT_WORKSPACE.mkdir()
    for name in AGENT_READ_FILES:
        shutil.copy2(EXPERIMENT_DIR / name, AGENT_WORKSPACE / name)
    shutil.copy2(ESTIMATOR_PATH, AGENT_WORKSPACE / "estimator.py")
    shutil.copy2(PROPOSAL_PATH, AGENT_WORKSPACE / "proposal.json")


def launch_pi(args: argparse.Namespace, iteration: int, log_path: Path) -> int:
    command = [
        "pi",
        "--no-session",
        "--print",
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-builtin-tools",
        "--extension",
        str(EXPERIMENT_DIR / "agent-tools.ts"),
        "--tools",
        "read_research_file,replace_research_text",
        "--thinking",
        args.thinking,
    ]
    if args.model:
        command.extend(["--model", args.model])
    command.append((AGENT_WORKSPACE / "iteration.md").read_text())
    with log_path.open("w") as log:
        log.write(f"=== iteration {iteration}: {datetime.now(UTC).isoformat()} ===\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=AGENT_WORKSPACE,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PI_OFFLINE": "1"},
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write("\nPi invocation timed out.\n")
            return 124
    return result.returncode


def snapshot_protected_files() -> dict[Path, bytes]:
    """Snapshot ignored research memory so Pi cannot rewrite its own history."""
    paths = [
        STATE_PATH,
        RESULTS_PATH,
        EXPERIMENT_DIR / "notes.md",
        EXPERIMENT_DIR / "condition-memory.md",
        PROPOSAL_PATH,
    ]
    paths.extend(RECORD_DIR.glob("*.json"))
    return {path: path.read_bytes() for path in paths if path.is_file()}


def restore_protected_files(snapshot: dict[Path, bytes]) -> None:
    protected_paths = [
        STATE_PATH,
        RESULTS_PATH,
        EXPERIMENT_DIR / "notes.md",
        EXPERIMENT_DIR / "condition-memory.md",
        PROPOSAL_PATH,
    ]
    protected_paths.extend(RECORD_DIR.glob("*.json"))
    for path in protected_paths:
        if path not in snapshot and path.is_file():
            path.unlink()
    for path, content in snapshot.items():
        path.write_bytes(content)


def reconcile_pending_tag(iteration: int) -> None:
    """Roll back an unrecorded candidate left by interrupted finalization."""
    tag = f"normal-search/{iteration:04d}"
    if not git("tag", "--list", tag):
        return
    if (RECORD_DIR / f"{iteration:04d}.json").exists():
        raise RuntimeError(f"Tag and record already exist for iteration {iteration}")

    candidate_commit = git("rev-list", "-n", "1", tag)
    if git("rev-parse", "HEAD") == candidate_commit:
        parent = git("rev-parse", f"{candidate_commit}^")
        subprocess.run(
            ["git", "reset", "--hard", parent],
            cwd=EXPERIMENT_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(["git", "tag", "-d", tag], cwd=EXPERIMENT_DIR, check=True)


def commit_candidate(iteration: int, description: str) -> str:
    subprocess.run(["git", "add", "estimator.py"], cwd=EXPERIMENT_DIR, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Experiment {iteration}: {description}"],
        cwd=EXPERIMENT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    commit = git("rev-parse", "HEAD")
    if git("diff-tree", "--no-commit-id", "--name-only", "-r", commit) != (
        "experiments/normal-estimation/estimator.py"
    ):
        raise RuntimeError("Candidate commit changed files outside estimator.py")
    subprocess.run(
        ["git", "tag", f"normal-search/{iteration:04d}", commit],
        cwd=EXPERIMENT_DIR,
        check=True,
    )
    return commit


def base_record(
    iteration: int,
    state: dict[str, object],
    proposal: dict[str, str],
) -> dict[str, object]:
    return {
        "version": 1,
        "harness_commit": git("rev-parse", "HEAD"),
        "estimator_sha256": sha256(ESTIMATOR_PATH),
        "iteration": iteration,
        "candidate_commit": None,
        "dev_rmse": None,
        "val_rmse": None,
        "runtime_s": None,
        "memory_mb": None,
        "status": "crash",
        "complexity": proposal["complexity"],
        "description": proposal["description"],
        "rationale": proposal["rationale"],
        "source": proposal["source"],
        "kind": proposal["kind"],
        "condition_rmse": {},
        "failure": None,
        "validation_interval": state["validation_interval"],
        "attempts_since_validation": int(state["attempts_since_validation"]) + 1,
        "frontier": state["frontier"],
        "validated": state["validated"],
    }


def materially_simpler_or_faster(
    proposal: dict[str, str],
    development: dict[str, float],
    frontier: dict[str, object],
) -> bool:
    """Accept near-equal scores only for explicit, measurable simplifications."""
    if proposal["kind"] != "simplification":
        return False
    frontier_runtime = frontier.get("runtime_s")
    return frontier_runtime is not None and development["runtime_s"] <= 0.9 * float(
        frontier_runtime
    )


def sha256(path: Path) -> str:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def decide(
    record: dict[str, object],
    state: dict[str, object],
    proposal: dict[str, str],
    development: dict[str, float],
) -> None:
    frontier = dict(state["frontier"])
    validated = dict(state["validated"])
    record["dev_rmse"] = development["rmse"]
    record["runtime_s"] = development["runtime_s"]
    record["memory_mb"] = development["peak_memory_mb"]
    record["condition_rmse"] = {"development": development["condition_rmse"]}

    gain = float(frontier["dev_rmse"]) - development["rmse"]
    if not math.isfinite(development["rmse"]) or (
        gain < 0.05
        and not (
            gain >= -0.01
            and materially_simpler_or_faster(proposal, development, frontier)
        )
    ):
        record["status"] = "discard"
        return

    attempts = int(state["attempts_since_validation"]) + 1
    validate_now = (
        attempts >= int(state["validation_interval"])
        or float(validated["dev_rmse"]) - development["rmse"] >= 0.20
        or proposal["kind"] == "mechanism"
        or proposal["complexity"] == "medium"
    )
    if not validate_now:
        record["status"] = "provisional"
        record["frontier"] = {
            "commit": record["candidate_commit"],
            "dev_rmse": development["rmse"],
            "val_rmse": validated["val_rmse"],
            "runtime_s": development["runtime_s"],
            "status": "provisional",
        }
        return

    validation = evaluate("validation", EXPERIMENT_DIR / "validation.log")
    record["val_rmse"] = validation["rmse"]
    record["condition_rmse"]["validation"] = validation["condition_rmse"]
    record["attempts_since_validation"] = 0
    validation_gain = float(validated["val_rmse"]) - validation["rmse"]
    validated_simplification = (
        proposal["kind"] == "simplification"
        and validation_gain >= -0.01
        and development["runtime_s"] <= 0.9 * float(validated["runtime_s"])
    )
    if validation_gain < 0.05 and not validated_simplification:
        record["status"] = "discard"
        record["frontier"] = validated | {"status": "keep"}
        return

    accepted = {
        "commit": record["candidate_commit"],
        "dev_rmse": development["rmse"],
        "val_rmse": validation["rmse"],
        "runtime_s": development["runtime_s"],
    }
    record["status"] = "keep"
    record["validated"] = accepted
    record["frontier"] = accepted | {"status": "keep"}


def rebuild_research_memory() -> None:
    """Regenerate agent-facing notes from canonical records."""
    subprocess.run(
        ["uv", "run", "summarize_notes.py"],
        cwd=EXPERIMENT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def finalize_record(
    record: dict[str, object], restore_commit: str, operational_commit: str
) -> None:
    subprocess.run(
        ["git", "reset", "--hard", restore_commit],
        cwd=EXPERIMENT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    retained_candidate = (
        record["status"] in {"keep", "provisional"}
        and restore_commit == record["candidate_commit"]
    )
    if restore_commit != operational_commit and not retained_candidate:
        estimator = ESTIMATOR_PATH.read_bytes()
        subprocess.run(
            ["git", "reset", "--hard", operational_commit],
            cwd=EXPERIMENT_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if ESTIMATOR_PATH.read_bytes() != estimator:
            ESTIMATOR_PATH.write_bytes(estimator)
            subprocess.run(
                ["git", "add", "estimator.py"], cwd=EXPERIMENT_DIR, check=True
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Experiment {record['iteration']}: restore validated frontier",
                ],
                cwd=EXPERIMENT_DIR,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    atomic_json(RECORD_DIR / f"{int(record['iteration']):04d}.json", record)
    sync_views(load_records())
    rebuild_research_memory()
    PROPOSAL_PATH.unlink(missing_ok=True)


def run_iteration(args: argparse.Namespace) -> str:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked tree must be clean before an iteration")

    records = load_records()
    validate_record_frontier(records)
    state = sync_views(records)
    rebuild_research_memory()
    iteration = int(state["iteration"]) + 1
    reconcile_pending_tag(iteration)

    frontier_commit = str(state["frontier"]["commit"])
    frontier_estimator = subprocess.check_output(
        [
            "git",
            "show",
            f"{frontier_commit}:experiments/normal-estimation/estimator.py",
        ],
        cwd=EXPERIMENT_DIR,
    )
    if ESTIMATOR_PATH.read_bytes() != frontier_estimator:
        raise RuntimeError(
            "The current estimator does not match the recorded frontier; "
            "restore it before resuming"
        )
    operational_commit = git("rev-parse", "HEAD")

    snapshot = snapshot_protected_files()
    prepare_proposal()
    prepare_agent_workspace()
    estimator_before = ESTIMATOR_PATH.read_bytes()
    agent_log = LOG_DIR / f"{iteration:04d}.log"
    return_code = launch_pi(args, iteration, agent_log)
    workspace_estimator = AGENT_WORKSPACE / "estimator.py"
    workspace_proposal = AGENT_WORKSPACE / "proposal.json"
    proposal_content = (
        workspace_proposal.read_bytes() if workspace_proposal.is_file() else None
    )
    estimator_content = (
        workspace_estimator.read_bytes() if workspace_estimator.is_file() else None
    )
    restore_protected_files(snapshot)
    if proposal_content is not None:
        PROPOSAL_PATH.write_bytes(proposal_content)
    if estimator_content is not None:
        ESTIMATOR_PATH.write_bytes(estimator_content)
    shutil.rmtree(AGENT_WORKSPACE, ignore_errors=True)
    state = sync_views(load_records())

    fallback = {
        "description": "agent invocation failed",
        "rationale": "no valid bounded proposal was produced",
        "complexity": "baseline",
        "kind": "parameter",
        "source": "original",
    }
    proposal_failure: ExperimentFailure | None = None
    try:
        proposal = validate_proposal()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        proposal = fallback
        proposal_failure = ExperimentFailure("proposal.validation", str(error))

    changed = git("diff", "--name-only")
    record = base_record(iteration, state, proposal)
    edit_failure = proposal_failure
    if edit_failure is None and return_code != 0:
        edit_failure = ExperimentFailure(
            "agent.invocation",
            f"Pi exited with status {return_code}",
            return_code=return_code,
            detail=log_tail(agent_log),
        )
    if edit_failure is None and changed != "experiments/normal-estimation/estimator.py":
        paths = changed.splitlines() if changed else []
        edit_failure = ExperimentFailure(
            "agent.edit-scope",
            f"expected only estimator.py to change; changed paths: {paths or 'none'}",
        )
    if edit_failure is None and ESTIMATOR_PATH.read_bytes() == estimator_before:
        edit_failure = ExperimentFailure(
            "agent.no-change", "agent left estimator.py unchanged"
        )
    if edit_failure is not None:
        record["failure"] = edit_failure.as_record()
        finalize_record(record, operational_commit, operational_commit)
        return "crash"

    try:
        format_and_lint_candidate()
        if ESTIMATOR_PATH.read_bytes() == estimator_before:
            raise ExperimentFailure(
                "quality.no-change",
                "automatic formatting removed the agent's entire change",
            )
        record["estimator_sha256"] = sha256(ESTIMATOR_PATH)
        record["candidate_commit"] = commit_candidate(
            iteration, proposal["description"]
        )
        development = evaluate("development", EXPERIMENT_DIR / "run.log")
        decide(record, state, proposal, development)
    except ExperimentFailure as error:
        record["failure"] = error.as_record()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        command = None
        return_code = None
        detail = None
        if isinstance(error, subprocess.CalledProcessError):
            command = [str(part) for part in error.cmd]
            return_code = error.returncode
            detail = str(error.stderr or error.stdout or "").strip() or None
        failure = ExperimentFailure(
            "controller.finalization",
            f"{type(error).__name__}: {error}",
            command=command,
            return_code=return_code,
            detail=detail,
        )
        record["failure"] = failure.as_record()

    restore_commit = str(record["frontier"]["commit"])
    finalize_record(record, restore_commit, operational_commit)
    return str(record["status"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--model")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--max-failures", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not PROMPT_PATH.is_file() or not RECORD_DIR.is_dir():
        raise SystemExit("Run `uv run manage.py init` first")
    if args.max_iterations is not None and args.max_iterations < 0:
        raise SystemExit("--max-iterations must be non-negative")

    LOG_DIR.mkdir(exist_ok=True)
    acquire_lock()
    launches = 0
    failures = 0
    try:
        while not STOP_PATH.exists():
            if args.max_iterations is not None and launches >= args.max_iterations:
                break
            if STOP_PATH.exists():
                break
            status = run_iteration(args)
            launches += 1
            failures = failures + 1 if status == "crash" else 0
            iteration = load_records()[-1]["iteration"]
            print(f"iteration {iteration}: {status}", flush=True)
            if failures >= args.max_failures:
                raise SystemExit("Too many consecutive failed iterations")
            time.sleep(args.delay)
    finally:
        LOCK_PATH.unlink(missing_ok=True)

    reason = "STOP file" if STOP_PATH.exists() else "iteration limit"
    print(f"Loop stopped by {reason}.")


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().split("=", 1)[1])
            os.kill(pid, 0)
        except (OSError, ValueError, IndexError):
            LOCK_PATH.unlink()
        else:
            raise SystemExit(f"Loop already running with PID {pid}")
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as lock:
        lock.write(f"pid={os.getpid()}\n")


if __name__ == "__main__":
    main()
