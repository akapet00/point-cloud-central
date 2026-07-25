# Batch strategy

Use this order to keep the search interpretable and prevent repeated ad hoc proposals. Each iteration still changes exactly one mechanism or one tightly scoped parameter. Skip an item only when a canonical record already tests it or a prior result directly invalidates its premise.

## Current batch: characterize scale before adding complexity

1. **Fixed neighborhood size, k=160, with retained Gaussian decay 2.0.** The current weighting improves clean, low-noise, stripe, and gradient conditions but hurts medium and high noise. A larger patch is the simplest test of whether more samples recover noise robustness while Gaussian weighting limits curvature bias.
2. **Fixed neighborhood size, k=224, with retained Gaussian decay 2.0.** Run only if k=160 improves the equal-condition development score or clearly improves medium/high noise without catastrophic clean-geometry loss.
3. **Fixed neighborhood size, k=80, with retained Gaussian decay 2.0.** Run after the larger-scale direction is resolved; this anchors the curvature/locality side of the scale curve.
4. **Two-scale confidence selection.** Only after fixed scales establish the useful endpoints, select between the best small and large scales using one dimensionless geometric confidence. Do not repeat the unmeasured 32/64/112 implementation unchanged.
5. **One-step robust covariance reweighting.** Target outliers and high noise only if fixed-scale experiments show that support size alone cannot improve those conditions.
6. **Quadratic jet refinement.** Retry only after the earlier prediction failure has a precise diagnostic and the implementation directly addresses it; keep the fit normalized and independently ablatable.

## Batch rules

- Preserve the validated Gaussian-weighted k=112 estimator until a candidate passes the existing development and validation thresholds.
- Prefer parameter probes before new mechanisms; they create interpretable anchors for later adaptive methods.
- Do not retest Gaussian decay 1.0 or 3.0 at k=112: both were measured and rejected.
- Treat a crash as an implementation result, not evidence against the scientific hypothesis. Read its structured failure stage before deciding whether to retry.
- Stop this batch for human review after three consecutive crashes, after all fixed-scale probes fail to improve the aggregate score, or before introducing a second new mechanism.
- Validation remains controller-owned. Never optimize directly against condition labels or validation outputs inside the estimator.
