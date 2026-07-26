# Point Cloud Central

A collection of point-cloud processing experiments. Educational explorations remain self-contained in notebooks, while reproducible command-line studies live under `experiments/`.

## Experiments

| Experiment | Description |
| --- | --- |
| [Explainable normal-estimation search](experiments/normal-estimation/README.md) | Stateless Pi search for simple, PCA-derived normal estimators on PCPNet. |

## Notebooks

| Notebook | Description |
| --- | --- |
| [Geometry feature extraction](notebooks/00-geometry-feature-extraction/main.ipynb) | Preprocessing, PCA-based geometric features, plane segmentation, and surface-area estimation for mobile laser scanning data. |
| [Optimal neighborhood](notebooks/01-optimal-neighborhood/main.ipynb) | Eigenentropy-based neighborhood selection and point-normal evaluation on the Stanford Bunny. |
| [Surface-point extraction](notebooks/02-surface-points-extraction/main.ipynb) | Query-point classification and boundary extraction from synthetic point clouds. |
| [Normal-estimation final analysis](notebooks/03-normal-estimation-search/main.ipynb) | Complete search history, controlled test comparison, uncertainty, per-condition trends, and runtime trade-offs. |

## Setup

The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --dev
uv run jupyter lab
```

Open any `main.ipynb` from JupyterLab and select the project environment when prompted. The notebooks resolve shared assets from the repository root, so they also work when launched from their own directories.

The surface-point notebook retains an optional [Polatory](https://github.com/polatory/polatory) example. It skips that cell when Polatory is unavailable and includes a SciPy-only implementation that works with the standard project environment.

Notebook outputs are intentionally cleared before commit.

## Code quality

Ruff checks and formats Python cells in the notebooks:

```bash
uv run ruff check notebooks
uv run ruff format --check notebooks
```

To apply formatting and safe fixes:

```bash
uv run ruff format notebooks
uv run ruff check --fix notebooks
```

## Data and references

The UTWENTE mobile laser scanning sample is associated with Florent Poux's [3D point-cloud feature extraction tutorial](https://www.youtube.com/watch?v=WKSJcG97gE4). The notebook links to the original download location.

The Bunny experiment uses the [Stanford Bunny](https://graphics.stanford.edu/data/3Dscanrep/), distributed by the Stanford Computer Graphics Laboratory. The repository contains a sampled point cloud and corresponding normals.

See the notebooks for algorithm-specific papers and references.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
