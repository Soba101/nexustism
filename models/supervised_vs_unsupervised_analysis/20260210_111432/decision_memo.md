# Decision Memo: Supervised vs Unsupervised Embedding Notebook

## Observed performance
- Runtime top model (cells 19/30): `all-mpnet-base-v2` with Spearman `0.4837` and ROC-AUC `0.7793`.
- Runner-up `unsupervised_simcse_best`; Spearman gap `0.0036`.

## Trustworthiness of comparison
- Significance (cell 36): 1/5 pairwise Spearman comparisons significant at p < 0.05.
- Winner consistency (cell 31): runtime=`all-mpnet-base-v2` vs artifact=`all-mpnet-base-v2`.
- Ranking metrics are now query-group-aware and report grouping diagnostics.
- Adapter-only supervised checkpoints now fail fast.

## Deployment recommendation
- Recommendation: **No-go** until corrected evaluation pipeline is rerun.
- P0: query-group-aware ranking metrics + corrected bootstrap significance.
- P0: `ARTIFACT_SOURCE_MODE="current"` for same-run diagnostics.
- P1: merged supervised weights required (or explicit adapter loading support).
- P1: enable HDBSCAN or exclude cluster evidence from winner rationale.