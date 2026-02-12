# Decision Memo: Supervised vs Unsupervised Embedding Notebook

## Observed performance
- Runtime top model (cells 19/30): `supervised_v4_cosine` with Spearman `0.5012` and ROC-AUC `0.7894`.
- Runner-up `all-mpnet-base-v2`; Spearman gap `0.0175`.

## Trustworthiness of comparison
- Significance (cell 36): 2/5 pairwise Spearman comparisons significant at p < 0.05.
- Winner consistency (cell 31): runtime=`supervised_v4_cosine` vs artifact=`supervised_v4_cosine`.
- Ranking metrics are now query-group-aware and report grouping diagnostics.
- Adapter-only supervised checkpoints now fail fast.

## Deployment recommendation
- Recommendation: **No-go** until corrected evaluation pipeline is rerun.
- P0: query-group-aware ranking metrics + corrected bootstrap significance.
- P0: `ARTIFACT_SOURCE_MODE="current"` for same-run diagnostics.
- P1: merged supervised weights required (or explicit adapter loading support).
- P1: enable HDBSCAN or exclude cluster evidence from winner rationale.