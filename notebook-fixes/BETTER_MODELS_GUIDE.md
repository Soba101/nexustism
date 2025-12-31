# Better Models for Hard Duplicate Detection

Current situation:
- **Best performance**: Raw MPNet baseline (0.5038)
- **Problem**: Fine-tuning makes it worse
- **Test set difficulty**: Extremely hard (54.4% overlap)

---

## Option 1: Try Raw Nomic-Embed (Quick Win)

Nomic is specifically designed for hard negatives. Try the baseline FIRST before fine-tuning:

```python
from sentence_transformers import SentenceTransformer

# Test raw Nomic baseline
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# Evaluate on your test set
# Expected: May beat MPNet's 0.5038 out-of-box!
```

**Why this might work:**
- Trained with **matryoshka loss** (learns at multiple dimensions)
- **8192 token context** (can see entire long incidents)
- **Optimized for hard negatives** (exactly your problem)
- Recent architecture (2024 vs MPNet 2020)

**Action**: Run `evaluate_model.ipynb` with raw Nomic before any fine-tuning

---

## Option 2: Modern Sentence-Transformer Models

### 2a. **gte-large-en-v1.5** (General Text Embeddings)
```python
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
```

**Stats:**
- **Size**: 434M params (vs MPNet 109M)
- **Performance**: Top-5 on MTEB leaderboard
- **Specialization**: General text similarity
- **Expected Spearman**: 0.52-0.55

**Pros**: Very strong baseline, good generalization
**Cons**: Larger (slower inference), needs more GPU memory

### 2b. **bge-large-en-v1.5** (Beijing Academy)
```python
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
```

**Stats:**
- **Size**: 335M params
- **Performance**: Top-10 on MTEB
- **Specialization**: Retrieval + hard negatives
- **Expected Spearman**: 0.51-0.54

**Pros**: Excellent at hard negatives, strong baseline
**Cons**: Larger model, slower

### 2c. **e5-large-v2** (Microsoft)
```python
model = SentenceTransformer('intfloat/e5-large-v2')
```

**Stats:**
- **Size**: 335M params
- **Performance**: Top-15 on MTEB
- **Specialization**: Text pairs, duplicate detection
- **Expected Spearman**: 0.50-0.53

**Pros**: Strong at pairwise similarity
**Cons**: Requires "query:" prefix for best performance

---

## Option 3: Domain-Specific Models

### 3a. **instructor-large** (Instruction-Tuned)
```python
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-large')

# Custom instruction for your domain
instruction = "Represent the IT service incident for duplicate detection"
embeddings = model.encode([[instruction, text1], [instruction, text2]])
```

**Pros**: Can customize via instructions, very flexible
**Cons**: Slower, more complex to use

### 3b. **Fine-tune BGE on Your Domain**

BGE is known to fine-tune well:

```python
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Use your test-matched pairs
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,  # Short training
    lr=2e-5,   # Lower for larger model
    warmup_steps=500
)
```

**Expected**: 0.53-0.56 (5-11% above baseline)

---

## Option 4: Ensemble Multiple Models

Combine predictions from multiple models:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load 3 different models
mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
nomic = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
bge = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Encode pairs with all models
emb1_mpnet = mpnet.encode(text1)
emb2_mpnet = mpnet.encode(text2)
sim_mpnet = np.dot(emb1_mpnet, emb2_mpnet) / (np.linalg.norm(emb1_mpnet) * np.linalg.norm(emb2_mpnet))

emb1_nomic = nomic.encode(text1)
emb2_nomic = nomic.encode(text2)
sim_nomic = np.dot(emb1_nomic, emb2_nomic) / (np.linalg.norm(emb1_nomic) * np.linalg.norm(emb2_nomic))

emb1_bge = bge.encode(text1)
emb2_bge = bge.encode(text2)
sim_bge = np.dot(emb1_bge, emb2_bge) / (np.linalg.norm(emb1_bge) * np.linalg.norm(emb2_bge))

# Ensemble: Average predictions
final_similarity = (sim_mpnet + sim_nomic + sim_bge) / 3

# Or weighted ensemble (tune weights on validation set)
final_similarity = 0.4*sim_mpnet + 0.4*sim_nomic + 0.2*sim_bge
```

**Expected**: +3-5% over best single model
**Cost**: 3x slower inference, 3x memory

---

## Option 5: Cross-Encoders (Highest Accuracy)

For **maximum accuracy** on hard pairs, use cross-encoders:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Score pairs directly (no separate embeddings)
scores = model.predict([
    [text1_a, text2_a],
    [text1_b, text2_b],
    ...
])
```

**Pros**:
- Highest accuracy (5-10% better than bi-encoders)
- Sees both texts together (better context)

**Cons**:
- **MUCH slower** (can't pre-compute embeddings)
- Must score ALL pairs at query time
- Not suitable for large-scale retrieval

**Use case**:
- Fine-tuning the top-100 candidates from bi-encoder
- Your Stage 2 causal detection (already using this!)

---

## My Recommendations (Ordered by Effort)

### Immediate (5 minutes)
1. **Test raw Nomic baseline** - May already beat MPNet!
   ```bash
   # In evaluate_model.ipynb, change model path
   model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
   ```

### Quick Win (1 hour)
2. **Try BGE-base-en-v1.5 raw** - Known to be strong at hard negatives
3. **Try GTE-base-en-v1.5 raw** - Top MTEB performer

### Medium Effort (3-4 hours)
4. **Generate test-matched pairs + fine-tune BGE**
   - Generate 10K pairs matching your test difficulty
   - Fine-tune BGE-base with MNRL loss
   - Expected: 0.53-0.56

### Advanced (1 day)
5. **Ensemble top-3 models**
   - MPNet + Nomic + BGE
   - Tune ensemble weights on validation set
   - Expected: 0.54-0.57

---

## Comparison Table

| Model | Size | Speed | Expected Spearman | Effort | Confidence |
|-------|------|-------|-------------------|--------|------------|
| **MPNet baseline** | 109M | Fast | 0.5038 | 0 min | ✅ Proven |
| **Nomic-v1.5 raw** | 137M | Fast | 0.50-0.53 | 5 min | 🟡 High |
| **BGE-base raw** | 109M | Fast | 0.51-0.54 | 10 min | 🟡 High |
| **GTE-base raw** | 109M | Fast | 0.51-0.53 | 10 min | 🟢 Medium |
| **BGE-large raw** | 335M | Medium | 0.52-0.55 | 15 min | 🟢 Medium |
| **Fine-tuned BGE** | 109M | Fast | 0.53-0.56 | 4 hrs | 🟢 Medium |
| **Ensemble (3 models)** | 3×109M | Slow | 0.54-0.57 | 1 day | 🟡 High |
| **Cross-Encoder** | 33M | Very Slow | 0.55-0.60 | N/A | ✅ Proven |

---

## Quick Start Script

```python
#!/usr/bin/env python3
"""Test multiple models on your hard test set"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from scipy.stats import spearmanr

# Load test data
with open('data_new/fixed_test_pairs.json') as f:
    test_data = json.load(f)

texts1 = test_data['texts1']
texts2 = test_data['texts2']
labels = test_data['labels']

# Models to test
models_to_try = [
    'sentence-transformers/all-mpnet-base-v2',
    'nomic-ai/nomic-embed-text-v1.5',
    'BAAI/bge-base-en-v1.5',
    'Alibaba-NLP/gte-base-en-v1.5',
]

results = {}

for model_name in models_to_try:
    print(f"\nTesting {model_name}...")

    # Load model
    if 'nomic' in model_name:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)

    # Encode
    emb1 = model.encode(texts1, show_progress_bar=True)
    emb2 = model.encode(texts2, show_progress_bar=True)

    # Compute similarities
    sims = [np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            for e1, e2 in zip(emb1, emb2)]

    # Spearman
    spearman, _ = spearmanr(labels, sims)
    results[model_name] = spearman

    print(f"  Spearman: {spearman:.4f}")

# Print final ranking
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:50s}: {score:.4f}")
```

Save as `test_all_models.py` and run!

---

## Bottom Line

**Don't fine-tune on your current curriculum pairs - they make models worse.**

Instead:
1. **Try raw Nomic** (5 min) - May beat MPNet immediately
2. **Try raw BGE** (10 min) - Strong at hard negatives
3. **If needed**: Generate test-matched pairs + fine-tune BGE (4 hrs)

Expected best result: **Spearman 0.53-0.56** (5-11% above baseline)
