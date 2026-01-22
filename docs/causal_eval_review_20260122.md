# Causal Evaluation Review (2026-01-22)

## Run context
- Notebook: `causal_detection_pipeline.ipynb`
- Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Labels: heuristic causal pairs + generated negatives
- Split: stratified fallback (temporal split lacked positives)
- Hints: cause/effect tags + category + delta-hours

## Latest metrics (evaluation_metrics.json)
- Stratified holdout: ROC-AUC 0.983, F1 0.869, Precision 0.900, Recall 0.840
- Temporal holdout: ROC-AUC 0.957, F1 0.441, Precision 0.306, Recall 0.789, size 2880
- Directionality accuracy: 0.760

## Critique
- Directionality improved materially with tags and time/category hints (0.76), now usable for candidate ranking.
- Temporal holdout precision is still low (0.306), so false positives remain high on real-time data.
- Stratified metrics remain optimistic; treat them as sanity checks, not primary performance.

## Acceptability
- Acceptable for ranking/reranking candidate causes.
- Not acceptable for automated causal assertions without human review due to low temporal precision.

## Recommended next steps
- Build a small human-labeled directional set (200-500 pairs) to calibrate thresholds.
- Tighten negative construction around same-category, similar-text, short time-window pairs.
- Evaluate with strict temporal holdout only when reporting headline metrics.
