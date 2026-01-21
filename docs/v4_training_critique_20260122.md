# V4 Training Critique (2026-01-22)

## Summary
- V4 shows stronger ROC/PR AUC but only modest Spearman movement, suggesting the model is better at separating easy positives/negatives while ranking quality is roughly flat.
- Early stopping at <1 epoch implies the current validation signal saturates quickly; this can mean the eval set is too easy or overly lexical.

## Strengths
- Hard negatives and added augmentation improve robustness compared to V3.
- Learning-rate wiring and seed logging improve reproducibility.
- Classifier block is optional and more resilient to tiny classes.

## Risks and gaps
- Jaccard-thresholded positives encourage lexical overlap; the model can overfit to token cues instead of semantics.
- Hard negatives are still lexical; without semantic mining they can be “hard” by overlap yet still trivially separable.
- Threshold selection is on the evaluation set; this inflates F1/accuracy and masks calibration issues.
- Relationship classifier lacks the `causal` class in validation; reported metrics do not cover that class.
- Training uses `dummy_data_promax.csv` (10k incidents); domain shift risk remains if production data distribution differs.

## Recommendations
- Split train/val/test by incident ID and keep thresholding on validation only.
- Add semantic hard negatives using base-embedding nearest neighbors instead of only TF-IDF/Jaccard.
- Report retrieval metrics (Recall@k/MRR/nDCG) with a query-to-corpus setup.
- Track per-category performance to detect lexical leakage or overfitting to dominant classes.

## Is it worth finetuning other models?
Yes, but only after the evaluation is more reliable. V4 already delivers stable results, so new finetunes are justified when:
- You have enough labeled pairs (tens of thousands or more).
- Retrieval metrics plateau and you can afford a full re-index of embeddings.
- Multilingual coverage is needed beyond what V4 provides.

### Candidates (768-dim, local-friendly)
- `BAAI/bge-base-en-v1.5` for strong retrieval performance.
- `intfloat/e5-base-v2` for balanced semantic search.
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` if multilingual is a priority.

If multilingual is secondary and re-indexing is costly, keep V4 as the baseline and focus on better evaluation data and semantic hard negatives first.
