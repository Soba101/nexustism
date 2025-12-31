# Complete Two-Stage Pipeline: Similarity + Causal Detection

**Your Use Case:** Find duplicate tickets AND identify root cause relationships

---

## Overview

You need **TWO separate models** for two different tasks:

### Stage 1: Similarity (Bi-Encoder) - **What you're training now**
**Task:** Find duplicate/similar tickets
**Question:** "Is Ticket A similar to Ticket B?"
**Model:** MPNet with LoRA (`model_promax_mpnet_lorapeft-v2.ipynb`)
**Loss:** CosineSimilarityLoss ✅ **CORRECT CHOICE**
**Output:** Similarity score 0.0-1.0

### Stage 2: Causal Classification (Cross-Encoder) - **Already exists**
**Task:** Determine if one ticket caused another
**Question:** "Did Ticket A cause Ticket B?"
**Model:** MiniLM CrossEncoder (`NLI.ipynb`)
**Loss:** Binary Cross-Entropy
**Output:** Causal probability 0.0-1.0

---

## Why You Need Both

### Example Scenario

**New Ticket:**
```
Ticket #5234: "Users reporting VPN connection failures starting 2:30 PM"
```

**Your Goals:**
1. **Find duplicates** - Are other users reporting the same issue?
2. **Find root cause** - What incident caused this?

### Stage 1: Similarity Search (Fast - 0.1 seconds)

Use MPNet bi-encoder to find top-K similar tickets from database:

```python
query = "Users reporting VPN connection failures starting 2:30 PM"
candidates = similarity_search(query, top_k=100)

Results:
1. Ticket #5231: "Cannot connect to VPN, timeout errors" (similarity: 0.89) ← DUPLICATE
2. Ticket #5233: "VPN gateway unreachable" (similarity: 0.87) ← DUPLICATE
3. Ticket #5220: "VPN server high CPU usage at 2:15 PM" (similarity: 0.72)
4. Ticket #5100: "Network switch failure in datacenter B" (similarity: 0.68)
5. ... (96 more)
```

**What this tells you:**
- Tickets #5231, #5233 are likely **duplicates** (>0.85 similarity)
- Tickets #5220, #5100 might be **related** but not duplicates

### Stage 2: Causal Classification (Slower - 10 seconds)

Use CrossEncoder to determine causality on top candidates:

```python
# Check if earlier tickets caused #5234
for candidate in candidates:
    if candidate.created_before(ticket_5234):
        causal_score = causal_classifier(candidate, ticket_5234)

Results:
Ticket #5220 → #5234: causal_score = 0.91 ✅ CAUSAL
  "VPN server high CPU at 2:15 PM" likely caused "VPN failures at 2:30 PM"

Ticket #5100 → #5234: causal_score = 0.15 ❌ NOT CAUSAL
  "Network switch failure" is unrelated (different datacenter)
```

**What this tells you:**
- **Root cause:** Ticket #5220 (VPN server high CPU)
- **Effect:** Ticket #5234 (VPN connection failures)

---

## Why CosineSimilarity is Correct for Stage 1

You're absolutely right to use `CosineSimilarityLoss` for similarity detection!

### What You Need from Stage 1

1. **Duplicate Detection:** Score how similar two tickets are
2. **Ranking:** Given a ticket, rank all others by similarity
3. **Threshold:** Decide if similarity > 0.85 → duplicate

### Why CosineSimilarityLoss Works

```python
# Training pair 1:
Ticket A: "VPN connection failed"
Ticket B: "Cannot connect to VPN"
Label: 1.0 (similar/duplicate)

# Model learns:
similarity(A, B) should be ~1.0
```

```python
# Training pair 2:
Ticket C: "VPN connection failed"
Ticket D: "Printer jammed"
Label: 0.0 (not similar)

# Model learns:
similarity(C, D) should be ~0.0
```

**Outcome:** Model learns to predict **exact similarity scores**, which you can threshold:
- Score > 0.85 → Duplicate
- Score 0.60-0.85 → Related
- Score < 0.60 → Unrelated

---

## Complete Workflow

### 1. Training Phase (What you're doing now)

**Train Similarity Model (MPNet with LoRA):**
```bash
jupyter notebook model_promax_mpnet_lorapeft-v2.ipynb
# Trains on 15K pairs with curriculum learning
# Output: Bi-encoder for fast similarity search
```

**Train Causal Model (CrossEncoder):**
```bash
jupyter notebook NLI.ipynb
# Trains on relationship pairs (duplicate/causal/related/unrelated)
# Output: CrossEncoder for causal classification
```

### 2. Production Deployment

**Step 1: Embed all tickets (one-time)**
```python
# Use trained MPNet bi-encoder
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('models/real_servicenow_finetuned_mpnet_lora')

# Embed all 10,000 incidents
all_tickets = load_incidents()
embeddings = model.encode(all_tickets['text'].tolist())

# Store in Supabase/PostgreSQL with pgvector
upload_to_database(all_tickets, embeddings)
```

**Step 2: Similarity search (real-time, <100ms)**
```python
def find_similar_tickets(new_ticket_text, top_k=100):
    """Find top-K similar tickets using bi-encoder"""

    # Encode new ticket
    query_embedding = model.encode([new_ticket_text])[0]

    # Vector search in database
    results = db.vector_search(
        query_embedding=query_embedding,
        top_k=top_k,
        similarity_threshold=0.5
    )

    return results
```

**Step 3: Causal detection (on-demand, ~10s for 100 pairs)**
```python
def find_root_cause(ticket, similar_tickets):
    """Use CrossEncoder to find causal tickets"""

    from sentence_transformers import CrossEncoder
    causal_model = CrossEncoder('models/causal_classifier')

    # Only check tickets created BEFORE this one
    prior_tickets = [t for t in similar_tickets if t.created_at < ticket.created_at]

    # Classify causality for each pair
    causal_scores = []
    for prior_ticket in prior_tickets:
        score = causal_model.predict([[prior_ticket.text, ticket.text]])[0]
        causal_scores.append((prior_ticket, score))

    # Sort by causal probability
    causal_scores.sort(key=lambda x: x[1], reverse=True)

    # Return tickets with causal_score > 0.7
    root_causes = [(t, s) for t, s in causal_scores if s > 0.7]
    return root_causes
```

**Step 4: Complete pipeline**
```python
def analyze_new_ticket(ticket_text):
    """Complete two-stage analysis"""

    # Stage 1: Find similar tickets (fast)
    similar = find_similar_tickets(ticket_text, top_k=100)

    # Classify duplicates (threshold-based)
    duplicates = [t for t in similar if t.similarity > 0.85]
    related = [t for t in similar if 0.60 < t.similarity <= 0.85]

    # Stage 2: Find root causes (slower, but only on top-100)
    root_causes = find_root_cause(ticket_text, similar)

    return {
        'duplicates': duplicates,      # Same issue
        'related': related,            # Similar issues
        'root_causes': root_causes,    # What caused this
    }
```

---

## Example Output

```python
ticket = "Users reporting email delivery delays starting 3:45 PM"
results = analyze_new_ticket(ticket)

print(results)
```

**Output:**
```json
{
  "duplicates": [
    {
      "id": "INC-5441",
      "text": "Email messages stuck in queue, not delivering",
      "similarity": 0.91,
      "created_at": "2024-12-24 15:50"
    },
    {
      "id": "INC-5439",
      "text": "Outlook showing send/receive errors",
      "similarity": 0.87,
      "created_at": "2024-12-24 15:47"
    }
  ],
  "related": [
    {
      "id": "INC-5420",
      "text": "Exchange server high memory usage",
      "similarity": 0.74,
      "created_at": "2024-12-24 15:30"
    }
  ],
  "root_causes": [
    {
      "id": "INC-5420",
      "text": "Exchange server high memory usage at 3:30 PM",
      "causal_score": 0.89,
      "explanation": "Server resource issue likely caused email delays"
    }
  ]
}
```

**Insight:**
- **Duplicates:** INC-5441, INC-5439 (same issue reported by different users)
- **Root Cause:** INC-5420 (Exchange server memory issue caused the email delays)
- **Action:** Fix the root cause (INC-5420), close duplicates

---

## Current Training Setup - Recommendations

Since you want **duplicate detection** (not pure ranking), **CosineSimilarityLoss is correct!**

But to improve performance, focus on:

### 1. More Training Epochs ✅

```python
# Current: 2 epochs per phase (6 total)
# Recommended: 4-5 epochs per phase (12-15 total)

'phase1_epochs': 4,
'phase2_epochs': 5,
'phase3_epochs': 6,  # Most time on hardest pairs
```

### 2. Lower Learning Rate ✅

```python
# Current: 5e-5
# Recommended: 3e-5 (more stable convergence)

'lr': 3e-5,
```

### 3. More Warmup ✅

```python
# Current: 100 steps
# Recommended: 500 steps

'warmup_steps': 500,
```

### 4. Add Weight Decay ✅

```python
# In model.fit():
optimizer_params={
    'lr': CONFIG['lr'],
    'weight_decay': 0.01  # Prevent overfitting
}
```

### 5. Better Evaluation ✅

```python
# Use full eval set, not subset
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,  # All 1,000, not 100
    name='eval_full'
)
```

---

## Don't Change to MNRL!

You were right to question my MNRL recommendation. Here's why:

### MNRL is for:
- ❌ Pure ranking tasks (find top-K, don't care about scores)
- ❌ When you only need relative ordering
- ❌ Semantic search (Google-style)

### CosineSimilarity is for:
- ✅ **Duplicate detection** ← YOUR TASK
- ✅ Threshold-based decisions (>0.85 = duplicate)
- ✅ Absolute similarity scores matter
- ✅ Need to distinguish duplicates from related tickets

**Your workflow:**
1. If similarity > 0.85 → **Duplicate** (close/merge ticket)
2. If similarity 0.60-0.85 → **Related** (link tickets)
3. If similarity < 0.60 → **Unrelated** (new issue)

**This requires accurate absolute scores, which CosineSimilarity provides!**

---

## Final Recommendation

### For Similarity Model (Stage 1):

**Keep CosineSimilarityLoss**, but improve training:

```python
CONFIG = {
    'lr': 3e-5,              # Lower (was 5e-5)
    'warmup_steps': 500,     # Higher (was 100)
    'phase1_epochs': 4,      # Was 2
    'phase2_epochs': 5,      # Was 2
    'phase3_epochs': 6,      # Was 2
    'weight_decay': 0.01,    # NEW
}
```

**Expected:** Spearman 0.51-0.52 (beats baseline)

### For Causal Model (Stage 2):

Already trained in `NLI.ipynb` - you're good!

---

## Summary

| Task | Model Type | Loss Function | Notebook | Status |
|------|-----------|---------------|----------|---------|
| **Duplicate Detection** | Bi-Encoder (MPNet) | CosineSimilarityLoss | `model_promax_mpnet_lorapeft-v2.ipynb` | ✅ Training now |
| **Causal Detection** | Cross-Encoder (MiniLM) | Binary Cross-Entropy | `NLI.ipynb` | ✅ Already exists |

**You're on the right track!** Just need more epochs and better hyperparameters. 🎯

See also:
- [docs/causal_pipeline.md](causal_pipeline.md) - Causal model details
- [docs/TRAINING_OPTIMIZATION_GUIDE.md](TRAINING_OPTIMIZATION_GUIDE.md) - Hyperparameter tuning
