# Reranker and Causal Relations Notes

## Why a reranker
- Embeddings provide fast candidate recall but are weak at top-k precision for duplicates.
- A cross-encoder reranker scores the full text pair and reorders the top-k to reduce false positives.

## Recommended pipeline
1) Embedding retrieval (top-k candidates)
2) Reranker on top-k (reorder by relevance)
3) Optional causal classifier on reranked top-n

## Reranker implementation (local)
- Model type: cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Inputs: pair of ticket descriptions (query, candidate)
- Output: relevance score (float)
- Rerank: sort candidates by score, return top-n

## Causal relations implementation
- Model type: classifier on pair features or cross-encoder classifier
- Inputs: (earlier_ticket, later_ticket) with temporal ordering enforced
- Labels: `causal`, `related`, `duplicate`, `none` (or your current schema)
- Output: class probabilities + label

## Evaluation guidance
- Reranker: Precision@k / MRR / nDCG on a labeled duplicate set
- Causal classifier: per-class F1/ROC-AUC and a strict temporal split
- Keep thresholds tuned on validation only and reported on test

## Minimal labeling protocol
- Sample 200–500 candidate pairs from embedding top-k
- Label duplicate vs non-duplicate; if causal, also label direction
- Use this set for reranker and causal classifier evaluation
