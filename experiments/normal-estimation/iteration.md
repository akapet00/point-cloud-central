# One normal-estimation experiment

Perform exactly one task: make one explainable change to `estimator.py`, describe it in `proposal.json`, and exit. The trusted outer controller handles Git, linting, evaluation, validation, acceptance, rollback, records, and recovery.

## Read first

Read completely:

- `state.json` — current validated and provisional frontier;
- `results.tsv` — quantitative experiment history;
- `notes.md` — concise findings and promising ideas;
- `estimator.py` — current frontier;
- `README.md` — scientific scope and simplicity rules;
- `proposal.json` — metadata template you must complete.

Do not read generated agent logs or any other files.

## Goal

Approach state-of-the-art PCPNet normal accuracy with the simplest explainable PCA-derived local estimator possible. You may improve the algorithm or its global hyperparameters.

Choose one conceptual hypothesis. It must address a recognizable issue—noise, curvature, density, outliers, or neighborhood bias—and admit a short mathematical explanation and independent ablation.

Allowed ideas include fixed/adaptive/multiscale neighborhoods, distance weighting, robust covariance, a few reweighting iterations, geometric outlier rejection, anisotropic patches, quadratic local fitting, and simple confidence-based scale selection.

Do not use neural networks, learned predictors, condition or shape labels, reference normals, per-shape tables, global reconstruction, orientation propagation, new dependencies, test data, random search without rationale, or opaque collections of heuristics.

## Make the proposal

Edit `proposal.json` to this exact schema:

```json
{
  "description": "short experiment description",
  "rationale": "geometric problem, mechanism, expected benefit, cost, and ablation",
  "complexity": "baseline | low | medium",
  "kind": "parameter | mechanism | simplification",
  "source": "original or a primary-paper/official-implementation URL"
}
```

Use `baseline` only for plain PCA or a parameter-only change, `low` for one standard mechanism, and `medium` for a justified combination or local nonlinear/iterative fit. Never introduce high complexity.

## Edit the candidate

Edit only `estimator.py`. Keep its public function signature unchanged, deterministic, typed, and readable. Prefer fewer than roughly 250 substantive lines.

Do not run commands, Git, evaluation, or additional experiments. Do not edit state, results, notes, records, harness code, prepared data, or logs. The outer controller will reject any invalid edit.

Before exiting, ensure both `estimator.py` and `proposal.json` are complete. Summarize the problem, mechanism, expected benefit, complexity cost, and ablation in your final response, then exit.
