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
| `loop.py` | Fresh-process controller | Read-only |
| `manage.py`, `summarize_notes.py` | Trusted initialization and generated research memory | Read-only |
| `prepare.py` | Download and cache PCPNet | Read-only |
| `run.py`, `worker.py` | Isolated prediction runner | Read-only |
| `evaluate.py` | Protected scoring | Read-only |
| `records/` | Canonical immutable record per completed experiment | Generated, untracked |
| `state.json`, `results.tsv` | Regenerated views of canonical records | Generated, untracked |
| `notes.md` | Concise cross-iteration research knowledge and failure stages | Generated, untracked |
| `condition-memory.md` | Per-condition frontier and candidate trade-offs | Generated, untracked |
| `publication/run-01/` | Records, condition history, test metrics, uncertainty, and provenance | Frozen snapshot |
| `export_publication.py` | Rebuild the snapshot from canonical local records and test metrics | Read-only |
| `verify_publication.py` | Verify snapshot checksums, estimator hash, record count, and Git refs | Read-only |
| `agent-logs/` | Full output from each Pi process | Generated, untracked |

The completed search began from fixed-neighborhood PCA with 112 nearest neighbors. The tracked `estimator.py` is the frozen finalist from iteration 279.

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

The test tier is absent from prediction and evaluation CLIs. It requires a separate human-controlled procedure after finalists are frozen. The one-time finalist and controlled fixed-PCA results are preserved under `publication/run-01/`; do not overwrite them. The original fixed-PCA estimator was evaluated afterward, without further model selection, to provide the controlled comparison.

The primary score is `rmse` (lower is better). JSON output also contains per-condition RMSE and PGP. Reported runtime includes estimator import, predictions, and prediction writes, but excludes prepared neighbor search.

## Completed study and reproducibility

The search is complete. The frozen release is tagged `normal-estimation-study-v1`, and the retained estimator is iteration 279.

The tracked publication snapshot under `publication/run-01/` contains all 323 canonical records, aggregate and per-condition history, controlled baseline and finalist test metrics, paired bootstrap intervals, checksums, and provenance. The final analysis notebook consumes this snapshot and runs without the PCPNet dataset; the optional qualitative figure is regenerated only when local test data are available and `estimator.py` matches the frozen hash.

To verify the tracked analysis from the repository root:

```bash
uv sync --locked --dev
uv run ruff check experiments notebooks
uv run ruff format --check experiments notebooks
uv run experiments/normal-estimation/verify_publication.py
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/03-normal-estimation-search/main.ipynb \
  --output /tmp/normal-estimation-analysis.ipynb
```

The original autonomous loop remains in the release for auditability. Starting a new search from the frozen branch is intentionally unsupported because `estimator.py` is no longer the fixed-PCA baseline. Create a fresh branch from the original harness commit `b787b40` or adapt the initialization protocol explicitly.

To rebuild the publication snapshot from preserved local records and test metrics:

```bash
cd experiments/normal-estimation
uv run export_publication.py
```

Candidate commits referenced by records are preserved by the `normal-search/NNNN` tags. Publish those tags together with the release tag when mirroring the full study history.

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

The final analysis notebook at `notebooks/03-normal-estimation-search/main.ipynb` rebuilds the search summary from the tracked publication snapshot, visualizes all attempts without overcrowding, reports per-condition progress and the accuracy-runtime frontier, and compares the frozen official-test result with representative references from the PFF-Net PCPNet table. It writes publication-ready figures with article-scale typography to `publication/run-01/figures/`.

## References

- Guerrero et al., [PCPNet: Learning Local Shape Properties from Raw Point Clouds](https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/)
- Hoppe et al., *Surface Reconstruction from Unorganized Points*, SIGGRAPH 1992
- Shi et al., [PFF-Net: Patch Feature Fitting for Point Cloud Normal Estimation](https://arxiv.org/abs/2511.21365)
