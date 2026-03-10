# Model Registry — ITSM Nexus

> **Purpose**: Single source of truth for all embedding models used in this project.
> Keep this file in sync with 
exustism/notebook-fixes/deploy_nomic.py:_MODEL_REGISTRY.
>
> **Benchmark**: All Spearman/ROC-AUC figures use enchmark_v4_semantic_resnotes (grounded ground truth).
> Do NOT use enchmark_v4_group_category for model selection — it is systematically biased.
>
> **Promotion rule**: A model may be promoted to **Production** only if its Spearman
> on enchmark_v4_semantic_resnotes beats the current production baseline.

---

## Active Models

| ID | Model / Path | Mode | Spearman | ROC-AUC | Date | Status |
|----|--------------|------|----------|---------|------|--------|
| V6-Nomic-LoRA | 
exustism/models/real_servicenow_finetuned_nomic_lora/<br>eal_servicenow_v2_20260310_1045_merged | fine-tuned | **0.5472** | **0.8159** | 2026-03-10 | ✅ Production |
| V1.5-Nomic-ZeroShot | 
omic-ai/nomic-embed-text-v1.5 (HuggingFace) | zero-shot | 0.4476 | 0.7584 | pre-2026-03-10 | 🗄️ Archived (superseded by V6) |
| V4-MPNet-LoRA | 
exustism/models/real_servicenow_finetuned_mpnet_lora/<br>eal_servicenow_v2_20260104_2321 | fine-tuned | 0.4949 | 0.7857 | 2026-01-04 | 🗄️ Archived |

---

## Cross-Encoder (Reranker)

| Model | Source | Evaluated? | Status |
|-------|--------|-----------|--------|
| cross-encoder/ms-marco-MiniLM-L-12-v2 | HuggingFace | ⚠️ Not yet — see Issue #1 in ML_ENGINEERING_REMEDIATION_PLAN.md | Production (pending eval) |

---

## Causal Classifier

| Model | Path | F1 | Precision | Recall | Status |
|-------|------|----|-----------|--------|--------|
| Cross-encoder causal v1 | models/causal_classifier/causal_crossencoder_v1 | ⚠️ Not measured | — | — | Experimental (fallback to reranker if not found) |

> Cross-encoder eval and causal classifier eval are tracked in ML_ENGINEERING_REMEDIATION_PLAN.md.

---

## Promotion Checklist

Before promoting any model to **Production**:

- [ ] Evaluated on enchmark_v4_semantic_resnotes (Spearman > current production baseline of 0.5472)
- [ ] alidate_training_pairs.py passes with exit code 0 (no benchmark leakage)
- [ ] Cross-encoder ranking eval run (MRR@10 measured)
- [ ] MODEL_ID env var updated in .env
- [ ] API startup log shows new model_id and correct mode
- [ ] supabase/embed_incidents_nomic.py re-run to update all embeddings
- [ ] supabase/rebuild_v4_indexes.py re-run to rebuild HNSW index
- [ ] This file updated with new model row

---

## Known Non-Starters (Do Not Retry)

| Model | Reason | Evidence |
|-------|--------|----------|
| V5.3 Nomic LoRA (routing labels) | Catastrophic forgetting: Sp 0.4476 → 0.2040 on grounded benchmark | 
exustism/docs/v5_evaluation.md |

---

*Last updated: 2026-03-11*
