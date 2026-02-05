# ITSM Nexus - ML Pipeline

SentenceTransformer fine-tuning pipeline for IT Service Management ticket similarity and causal relationship detection.

- **Production Model:** V4 Cosine (Spearman 0.4949, ROC-AUC 0.7857)
- **Base Model:** all-mpnet-base-v2 (109M params, 768-dim)
- **Fine-tuning:** LoRA (r=16, α=32) with curriculum learning

---

## Two-Stage Pipeline

### Stage 1: Bi-Encoder (Fast Similarity Search)

Encodes tickets into 768-dim embeddings for cosine similarity ranking.

```python
query = "database connection timeout"
embedding = bi_encoder.encode(query)  # → shape (768,)
top_k_candidates = vector_db.search(embedding, k=20)
```

- **Model:** all-mpnet-base-v2 + LoRA adapters
- **Output:** 768-dimensional vector per ticket
- **Inference:** ~5ms per query

### Stage 2: Cross-Encoder (Causal Reranking)

Reranks top-K candidates for causal relationship detection.

```python
causal_scores = cross_encoder.predict([
    (query, candidate.description)
    for candidate in top_k_candidates
])
# → [0.89, 0.34, 0.12] (probability of causal relationship)
```

- **Model:** ms-marco-MiniLM-L-6-v2
- **Task:** Binary classification (causal vs non-causal)
- **Inference:** ~10-15ms per candidate

---

## Production Model (V4 Cosine)

| Property | Value |
|----------|-------|
| Path | `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20260104_2321` |
| Base | all-mpnet-base-v2 (109M params) |
| Embedding dim | 768 |
| Loss | Pure CosineSimilarityLoss |
| LoRA | r=16, α=32 |
| Training | 12 epochs (4 per curriculum phase), batch=32, lr=2e-5 |
| Spearman | 0.4949 |
| ROC-AUC | 0.7857 |
| F1 | 0.7134 |
| Adversarial ROC-AUC | 0.967 (verified semantic understanding) |

---

## Curriculum Learning

### Problem

Train/test distribution mismatch caused fine-tuned models to underperform baseline:

- Training data: separability=0.374, 0% overlap (too easy)
- Test data: separability=0.187, 54.4% overlap (realistic)

### Solution

3-phase progressive difficulty training (15,000 total pairs):

| Phase | Pairs | Positive Threshold | Negative Threshold | Epochs | Purpose |
|-------|-------|-------------------|-------------------|--------|---------|
| 1 (Easy) | 5K | pos ≥ 0.52 | neg ≤ 0.36 | 4 | Build foundation |
| 2 (Medium) | 5K | pos ≥ 0.40 | neg ≤ 0.45 | 4 | Bridge gap |
| 3 (Hard) | 5K | pos ≥ 0.30 | neg ≤ 0.50 | 4 | Match test difficulty |

Spearman progression: 0.413 → 0.487 → 0.498 across phases.

### Key Files

- `fix_train_test_mismatch.ipynb` - Generates curriculum datasets
- `data_new/curriculum_training_pairs_*.json` - Generated curriculum data
- [docs/curriculum_training_guide.md](docs/curriculum_training_guide.md) - Complete guide
- [docs/train_test_mismatch_analysis.md](docs/train_test_mismatch_analysis.md) - Root cause analysis

---

## Training

### Train V4 Cosine (Production)

```bash
jupyter notebook model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb
# Run all cells sequentially
```

### Generate Curriculum Dataset

```bash
jupyter notebook fix_train_test_mismatch.ipynb
# Generates 15K pairs across 3 difficulty phases into data_new/
```

### Training Config

```python
CONFIG = {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'use_curriculum': True,
    'epochs_per_phase': 4,
    'batch_size': 32,
    'lr': 2e-5,
    'max_seq_length': 256,
    'seed': 42
}
```

### Device Auto-Detection

Priority: CUDA → MPS (Apple Silicon) → CPU

- CUDA: batch_size=16-32
- MPS: batch_size=8, FP16 disabled
- CPU: batch_size=8, FP16 disabled

---

## Data Organization

### Training Data

| File | Purpose |
|------|---------|
| `data_new/curriculum_training_pairs_*.json` | Curriculum datasets (15K pairs) - **use these** |
| `data_new/fixed_training_pairs.json` | Validated training pairs |
| `data_new/fixed_test_pairs.json` | Validated test pairs |
| `data/servicenow_incidents_full.json` | Full incident dataset |
| `data/training_pairs.json` | Original pairs (legacy) |

Always use `data_new/curriculum_training_pairs_*.json`, not raw pairs from `data/`.

### Text Preprocessing

Context metadata placed at **end** of text to prevent shortcut learning:

```
{Short Description}. {Description} (Context: [{Service} | {Service offering}] [{Category} | {Subcategory}] Group: {Assignment group}.)
```

### Pair Generation Strategy

- **40% positives:** TF-IDF > 0.5 (any category)
- **30% hard negatives:** Same category, TF-IDF < 0.3 (prevents category shortcuts)
- **30% easy negatives:** Different category, TF-IDF < 0.3

---

## Evaluation

### Run Evaluation

```bash
jupyter notebook evaluate_model_v2.ipynb
```

### Adversarial Diagnostic

Models must pass adversarial testing before production:

- **Test:** Cross-category positives (high TF-IDF) + same-category negatives (low TF-IDF)
- **Pass criteria:** ROC-AUC ≥ 0.70 AND F1 ≥ 0.70
- **Purpose:** Verify model learns semantic content, not category shortcuts

### Production Checklist

- [ ] Spearman correlation ≥ 0.80
- [ ] ROC-AUC ≥ 0.95
- [ ] Adversarial diagnostic PASSED (ROC-AUC ≥ 0.70, F1 ≥ 0.70)
- [ ] Holdout metrics within 5% of eval metrics
- [ ] `training_metadata.json` saved with model

---

## Model Deployment

```python
from deploy_model_v4 import V4CosineModelDeployment

model = V4CosineModelDeployment()

# Compute similarity
similarity = model.compute_similarity(
    "User cannot login to SAP",
    "SAP authentication failed"
)

# Find similar tickets
results = model.find_similar(
    query="Email not working",
    candidates=["Outlook crashes", "VPN issue", "Cannot send emails"],
    top_k=3
)
```

Deployment script: `notebook-fixes/deploy_model_v4.py`

### Generate Embeddings

```bash
python supabase/embed_incidents_v4_cosine.py
python supabase/rebuild_v4_indexes.py
```

---

## Model Comparison

| Model | Parameters | Embedding Dim | Speed | Use Case |
|-------|-----------|---------------|-------|----------|
| MPNet-base + LoRA | 109M | 768 | Medium | **Production** |
| Nomic-embed-text-v1.5 | 137M | 768 | Fast | Speed-optimized alternative |
| Qwen3-Embedding-8B | 8B | 768 (truncated) | Slow | Experimental |

### Training Notebooks

| Notebook | Model | Notes |
|----------|-------|-------|
| `model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb` | MPNet + LoRA | **Production (V4 Cosine)** |
| `model_promax_mpnet_lorapeft_v3.ipynb` | MPNet + LoRA | V3 iteration |
| `model_promax_nomic_v3_clean.ipynb` | Nomic v1.5 | Standard accuracy |
| `model_promax_nomic_speed_v3_clean.ipynb` | Nomic v1.5 | Speed-optimized (FP16) |
| `qwen3_embedding_768.ipynb` | Qwen3-8B | Experimental |

---

## Known Issues

- **InputExample pickling:** Must use `num_workers=0` in DataLoader
- **MPS memory:** May require batch_size=8, restart kernel if OOM
- **NLTK data:** Download required: `wordnet`, `omw-1.4`, `stopwords`, `punkt`
- **Category shortcuts:** Models can exploit metadata prefixes; adversarial diagnostic verifies semantic learning
- **Train/test mismatch:** Use curriculum learning to solve (see curriculum_training_guide.md)
- **Non-curriculum training pairs:** Causes train/test mismatch. Always use `data_new/curriculum_training_pairs_*.json`

---

## Documentation

- [docs/model_pipeline.md](docs/model_pipeline.md) - Complete training pipeline reference
- [docs/curriculum_training_guide.md](docs/curriculum_training_guide.md) - Curriculum learning guide
- [docs/train_test_mismatch_analysis.md](docs/train_test_mismatch_analysis.md) - Root cause analysis
- [docs/causal_pipeline.md](docs/causal_pipeline.md) - Causal classifier (CrossEncoder)
- [docs/NOTEBOOK_CELL_MAP.md](docs/NOTEBOOK_CELL_MAP.md) - Cell-by-cell notebook guide

---

**Last Updated:** January 29, 2026
