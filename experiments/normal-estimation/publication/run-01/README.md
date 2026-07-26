# Normal-estimation search publication snapshot

This directory freezes the machine-readable evidence used by the accompanying article.

## Files

- `records.jsonl`: all 323 canonical records: one baseline and 322 agent iterations.
- `results.tsv`: aggregate search history.
- `condition-results.tsv`: per-condition development and validation history.
- `baseline-test-results.json`: controlled fixed-k PCA test result.
- `final-results.json`: frozen finalist test result.
- `bootstrap-results.json`: paired shape-level bootstrap comparison.
- `bootstrap-deltas.png`: plotted paired bootstrap differences.
- `qualitative-errors.png`: illustrative sharp-feature angular-error maps.
- `provenance.json`: commits, hashes, protocol, environment, and limitations.

The official PCPNet test list includes the validation geometries, so the test split is not strictly disjoint. The finalist was frozen before its single official-test evaluation. The baseline was evaluated afterward only to provide a controlled same-harness comparison.

Run `uv run experiments/normal-estimation/export_publication.py` from the repository root to rebuild the tabular evidence, bootstrap metrics, and provenance from preserved local records. The final analysis notebook reads this directory but does not modify it.
