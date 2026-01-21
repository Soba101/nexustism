# Embedding Evaluation Review (2026-01-22)

## Scope and inputs
- Results from `nexustism/models/results/embedding_eval_snow_csv_20260122_031812.csv`
  and `nexustism/models/results/embedding_eval_servicenow_json_20260122_031812.csv`
- Models: MPNet baseline, V4 MPNet LoRA, EmbeddingGemma-300M
- Pairing: CSV uses fixed test pairs; JSON uses TF-IDF generated pairs
- Metrics: Spearman, Pearson, ROC-AUC, PR-AUC, F1, precision, recall, accuracy, plus
  score separation and confusion matrix (threshold selected on validation, reported on test)

## Methodology updates
- Pair metrics use a validation/test split with thresholds selected on validation.
- Retrieval metrics (Recall@k, MRR, nDCG) use category-based relevance.

## Dataset-level observations
- CSV dataset: 1000 pairs, balanced labels (500 positive, 500 negative); metrics reflect 200-pair test split.
- JSON dataset: 556 pairs with 196 positive and 360 negative labels; metrics reflect 88-pair test split.

## Key findings
- CSV results remain plausible:
  - MPNet baseline still has the highest Spearman (0.399) with weaker accuracy (0.59).
  - V4 MPNet LoRA slightly improves F1 (0.664) and accuracy (0.64) but lower Spearman (0.376).
  - EmbeddingGemma-300M trails on all metrics.
- JSON results are no longer perfect but still suspicious:
  - Spearman/Pearson are negative and ROC-AUC is ~0.34, suggesting label polarity or pair construction issues.
  - Precision/recall are stuck at 0.82/1.0 with all positives predicted, indicating thresholding is not meaningful for this set.
- Retrieval metrics now include Hit@k:
  - Recall@k remains tiny on CSV due to many relevant items; Hit@k is more interpretable here.

## Concerns and likely causes
- JSON dataset likely contains strong lexical cues tied to category or templated
  phrasing, so TF-IDF-based pair selection yields pairs that any model separates.
- Thresholds are now selected on validation and reported on test; this reduces inflation
  but still requires stronger test construction to be meaningful.
- Pair generation (TF-IDF neighbors) may not match the real retrieval task; it can
  create positives that are highly similar by surface form rather than semantics.
- No language split or per-language metrics, so multilingual robustness is untested.

## Recommendations
- Rebuild the JSON evaluation with stricter de-duplication and harder negatives
  (same product or short description overlap but different category/label).
- Split thresholds: choose best threshold on validation, report on a held-out test.
- Add retrieval metrics (Recall@k, MRR, nDCG) using a query-to-corpus setup that
  mirrors the intended usage (description-only search).
- Report per-language stats if multilingual content is expected.
