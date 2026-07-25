# Explainable normal-estimation search

This experiment tests how closely a simple, explainable, PCA-derived local method can approach state-of-the-art normal estimation on PCPNet. Pi can change both the algorithm and its global hyperparameters, but every improvement must have a clear geometric rationale, remain compact, and be independently ablatable.

Research uses a stateless outer loop: one fresh Pi process performs exactly one experiment, writes its result to disk, and exits. Git and concise local records—not conversation history—carry knowledge between iterations.

## Scientific protocol

The estimator receives only point coordinates, official query indices, and precomputed nearest-neighbor indices and distances. It never receives reference normals, shape names, or perturbation labels.

Scores use sign-invariant angular error. PCPNet's mean per-shape RMSE is computed for each of six conditions, then the six condition scores are averaged equally.

Evaluation tiers:

1. **Development:** 1,000 deterministic queries from each training point cloud; used for routine experiments.
2. **Validation:** all official queries from PCPNet's three validation geometries; used periodically to promote changes.
3. **Test:** complete official test lists; used only after finalists are frozen. PCPNet's published test lists include the validation geometries, so final results must state that the benchmark is not a strictly disjoint third split.

Normal orientation is outside this experiment's scope.

## Files

| File | Purpose | Research agent access |
| --- | --- | --- |
| `estimator.py` | Complete candidate algorithm | **Editable** |
| `agent-tools.ts` | Path-scoped read/edit tools for isolated Pi invocations | Read-only |
| `iteration.md` | Instructions for one bounded Pi invocation | Read-only |
| `batch-strategy.md` | Ordered hypotheses and batch stopping rules | Read-only |
| `loop.py` | Fresh-process controller | Read-only |
| `manage.py`, `loop.py`, `summarize_notes.py` | Trusted initialization, evaluation, decisions, and records | Read-only |
| `prepare.py` | Download and cache PCPNet | Read-only |
| `run.py`, `worker.py` | Isolated prediction runner | Read-only |
| `evaluate.py` | Protected scoring | Read-only |
| `records/` | Canonical immutable record per completed experiment | Generated, untracked |
| `state.json`, `results.tsv` | Regenerated views of canonical records | Generated, untracked |
| `notes.md` | Concise cross-iteration research knowledge and failure stages | Generated, untracked |
| `condition-memory.md` | Per-condition frontier and candidate trade-offs | Generated, untracked |
| `agent-logs/` | Full output from each Pi process | Generated, untracked |

The baseline estimator is fixed-neighborhood PCA with 112 nearest neighbors.

## Dataset setup

Run from this directory:

```bash
uv run prepare.py download
uv run prepare.py verify
uv run prepare.py cache --tier development
uv run prepare.py cache --tier validation
```

PCPNet lives under ignored `data/PCPNet/`. The archive is about 964 MB and extracts to about 2.7 GB; arrays and neighbor caches need additional space.

## Manual evaluation

```bash
uv run run.py --tier development
uv run evaluate.py --tier development
uv run run.py --tier validation
uv run evaluate.py --tier validation --json validation.json
```

`run.py` gives editable code anonymous, label-free inputs. On macOS it applies an OS sandbox that blocks raw data, references, protected source, other tiers, prior results, network access, and writes outside predictions. `evaluate.py` never imports editable code.

The test tier is absent from prediction and evaluation CLIs. It requires a separate human-controlled procedure after finalists are frozen.

The primary score is `rmse` (lower is better). JSON output also contains per-condition RMSE and PGP. Reported runtime includes estimator import, predictions, and prediction writes, but excludes prepared neighbor search.

## Running the search

First commit this harness on a clean baseline branch. Create a research branch, then initialize its local records:

```bash
git switch -c research/normal-estimation-01
cd experiments/normal-estimation
uv run manage.py init
```

Run a small pilot with five fresh Pi contexts:

```bash
caffeinate -ims uv run loop.py --max-iterations 5
```

Inspect progress:

```bash
uv run manage.py status
column -ts $'\t' results.tsv
```

Continue without an iteration limit:

```bash
caffeinate -ims uv run loop.py
```

Request a graceful stop after the current Pi invocation:

```bash
uv run manage.py stop
```

Clear the stop request before resuming:

```bash
uv run manage.py resume
caffeinate -ims uv run loop.py
```

Each measured candidate receives a durable `normal-search/NNNN` Git tag, including discarded attempts. The controller writes one canonical record only after restoring the chosen frontier; state, TSV, notes, and per-condition memory can be rebuilt from those records. Agent edits receive automatic Ruff safe fixes and formatting before evaluation. Failures retain the exact stage, command status, and a bounded diagnostic in the canonical record; repeated crashes stop the loop for review.

Optional controller flags:

```text
--max-iterations N   maximum fresh Pi launches
--model PATTERN      Pi model pattern
--thinking LEVEL     defaults to high
--delay SECONDS      delay between launches
```

Each launch uses `pi --no-session --print` with `PI_OFFLINE=1`, no context files, auto-discovered extensions, skills, templates, or built-in tools. A small explicit extension exposes path-scoped reads and exact replacements only inside an isolated workspace. Pi has no prior conversation session and only proposes one edit plus metadata; trusted controller code owns Git, linting, evaluation, validation, decisions, rollback, tagging, and canonical records. Knowledge survives through records, regenerated views, per-condition memory, the tracked batch strategy, durable Git tags, and logs.

## Simplicity standard

A retained mechanism must:

1. address a recognizable geometric issue such as noise, curvature, density, or outliers;
2. admit a short mathematical explanation;
3. be removable in an independent ablation;
4. earn enough accuracy or efficiency to justify its complexity.

Prefer the simpler and faster candidate when validation RMSE differs by less than 0.05 degrees. Neural networks, learned predictors, condition-specific branches, per-shape tables, global reconstruction, orientation propagation, and opaque heuristic pipelines are prohibited.

## Published references

PFF-Net reports:

| Method | Clean | Low | Medium | High | Stripe | Gradient | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA | 12.29 | 12.87 | 18.38 | 27.52 | 13.66 | 12.81 | 16.25 |
| PFF-Net | 3.32 | 8.34 | 15.63 | 20.94 | 4.10 | 3.92 | 9.38 |

The published PCA row uses CGAL and is a protocol check rather than a bit-for-bit NumPy baseline.

A later notebook at `notebooks/03-normal-estimation-search/main.ipynb` will generate a progress plot from `results.tsv`: all attempts, retained improvements, running best, reference lines, per-condition progress, and the accuracy-runtime Pareto frontier.

## References

- Guerrero et al., [PCPNet: Learning Local Shape Properties from Raw Point Clouds](https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/)
- Hoppe et al., *Surface Reconstruction from Unorganized Points*, SIGGRAPH 1992
- Shi et al., [PFF-Net: Patch Feature Fitting for Point Cloud Normal Estimation](https://arxiv.org/abs/2511.21365)
