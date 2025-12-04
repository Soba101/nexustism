# Causal Relationship Classifier Pipeline

> Documentation for training and using the CrossEncoder-based causal relationship classifier for ITSM tickets.

**Last Updated:** December 2025  
**Notebook:** `NLI.ipynb`  
**Related:** `docs/model_pipeline.md` (similarity model)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [When to Use This](#3-when-to-use-this)
4. [Data Preparation](#4-data-preparation)
5. [Training Configuration](#5-training-configuration)
6. [Training Process](#6-training-process)
7. [Evaluation](#7-evaluation)
8. [Inference](#8-inference)
9. [Two-Stage Pipeline Integration](#9-two-stage-pipeline-integration)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### What Is This?

A **binary classifier** that determines if one ITSM ticket **caused** another.

**Input:** Two ticket texts (Ticket A, Ticket B)  
**Output:** Probability that Ticket A caused Ticket B

### Example

```
Ticket A: "Database server crashed due to disk full at 3:15 PM"
Ticket B: "Multiple users reporting application timeout errors starting 3:20 PM"

Causal Score: 0.87 → ✅ CAUSAL (A likely caused B)
```

### Why CrossEncoder (Not Bi-Encoder)?

| Aspect | Bi-Encoder (Similarity) | CrossEncoder (Causal) |
|--------|-------------------------|------------------------|
| How it works | Encode separately, compare embeddings | Encode pair together, joint classification |
| Speed | Fast (pre-compute embeddings) | Slower (run model per pair) |
| Accuracy | Good for similarity | Better for relationships |
| Use case | Find top-K similar tickets | Classify specific pairs |

**Causal detection requires understanding the relationship between two texts, not just their similarity.** CrossEncoders excel at this.

---

## 2. Architecture

### Model Structure

```
┌─────────────────────────────────────────────────────────────┐
│                 CROSSENCODER ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: [CLS] Ticket A [SEP] Ticket B [SEP]                 │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │   Transformer (MiniLM-L6-v2)        │                    │
│  │   6 layers, 384 hidden dim          │                    │
│  │   ~23M parameters                   │                    │
│  └─────────────────────────────────────┘                    │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │   Classification Head               │                    │
│  │   Linear(384 → 1)                   │                    │
│  └─────────────────────────────────────┘                    │
│           │                                                  │
│           ▼                                                  │
│       Sigmoid → P(causal)                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Base Model

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Property | Value |
|----------|-------|
| Architecture | MiniLM (distilled BERT) |
| Layers | 6 |
| Hidden Dimension | 384 |
| Parameters | ~23M |
| Max Sequence Length | 512 tokens |
| Pre-training | MS MARCO passage ranking |

**Why this model?**
- Fast inference (~3x faster than BERT-base)
- Good balance of speed and accuracy
- Pre-trained on semantic similarity tasks
- Small enough for CPU inference if needed

---

## 3. When to Use This

### ✅ Use the Causal Classifier When:

1. **Root Cause Analysis**: Identify which tickets triggered cascading failures
2. **Incident Clustering**: Group related incidents by causal chains
3. **Automated Routing**: Route tickets to teams that resolved the root cause
4. **SLA Attribution**: Determine which incident is the "parent" for SLA tracking
5. **Knowledge Base**: Link resolution articles to upstream causes

### ❌ Don't Use When:

1. **Simple similarity search**: Use the bi-encoder similarity model instead
2. **Real-time bulk processing**: Too slow for >100 pairs/second
3. **No labeled causal data**: Requires training data with causal labels

### Typical Pipeline

```
New Ticket Arrives
        │
        ▼
┌─────────────────────────────┐
│ Stage 1: Similarity Model   │  (Fast: ~1ms per comparison)
│ Find top-10 similar tickets │
└─────────────────────────────┘
        │
        ▼ (Only 10 tickets)
┌─────────────────────────────┐
│ Stage 2: Causal Classifier  │  (Slower: ~50ms per pair)
│ Check for causal relations  │
└─────────────────────────────┘
        │
        ▼
    Results with causal flags
```

---

## 4. Data Preparation

### 4.1 Input Data Format

**File:** `data/relationship_pairs.json`

```json
[
  {
    "text_a": "Server database crashed due to disk full",
    "text_b": "Application reporting connection timeout errors",
    "label": "causal"
  },
  {
    "text_a": "User password reset request",
    "text_b": "Email server maintenance completed",
    "label": "none"
  },
  {
    "text_a": "Printer offline in building A",
    "text_b": "Printer offline in building A - second report",
    "label": "duplicate"
  },
  {
    "text_a": "VPN slow for remote users",
    "text_b": "Network latency issues detected",
    "label": "related"
  }
]
```

### 4.2 Label Distribution (Typical)

| Label | Description | Typical % |
|-------|-------------|-----------|
| `causal` | A caused B | 5-15% |
| `related` | Connected but not causal | 20-30% |
| `duplicate` | Same issue, different reports | 5-10% |
| `none` | Unrelated tickets | 50-70% |

### 4.3 Binary Conversion

For the causal classifier, we convert to binary:

```python
# Causal = 1, Everything else = 0
causal_pairs = [p for p in data if p['label'] == 'causal']
non_causal_pairs = [p for p in data if p['label'] in ['none', 'related', 'duplicate']]
```

### 4.4 Class Balancing

Causal pairs are typically rare (5-15%). Balance the dataset:

```python
# Undersample majority class to 1:3 ratio
max_non_causal = len(causal_pairs) * 3
non_causal_sample = random.sample(non_causal_pairs, min(len(non_causal_pairs), max_non_causal))

balanced_data = causal_pairs + non_causal_sample
random.shuffle(balanced_data)
```

**Why 1:3 ratio?**
- 1:1 would lose too much data
- Higher ratios (1:5+) make learning difficult
- 1:3 is empirically good for this task

---

## 5. Training Configuration

### 5.1 Hyperparameters

```python
CONFIG = {
    'base_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'output_dir': 'models/causal_classifier',
    
    # Training
    'epochs': 5,
    'batch_size': 16,
    'warmup_ratio': 0.1,
    'max_length': 512,
    
    # Data
    'max_pairs': 10000,      # Memory limit
    'test_split': 0.2,       # 20% for evaluation
    
    # Reproducibility
    'seed': 42
}
```

### 5.2 Device Selection

```python
import torch

if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
```

### 5.3 Memory Considerations

| Device | Max Pairs | Batch Size | Training Time |
|--------|-----------|------------|---------------|
| CUDA (8GB) | 20,000 | 32 | ~10 min |
| MPS (16GB) | 10,000 | 16 | ~20 min |
| CPU | 5,000 | 8 | ~60 min |

---

## 6. Training Process

### 6.1 Load and Prepare Data

```python
import json
import random
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split

# Load relationship data
with open('data/relationship_pairs.json', 'r') as f:
    rel_data = json.load(f)

# Binary conversion: causal vs non-causal
causal_pairs = [p for p in rel_data if p['label'] == 'causal']
non_causal_pairs = [p for p in rel_data if p['label'] in ['none', 'related', 'duplicate']]

print(f"Raw: {len(causal_pairs)} causal, {len(non_causal_pairs)} non-causal")

# Balance dataset (1:3 ratio)
max_non_causal = min(len(non_causal_pairs), len(causal_pairs) * 3)
non_causal_sample = random.sample(non_causal_pairs, max_non_causal)

balanced_data = causal_pairs + non_causal_sample
random.shuffle(balanced_data)

print(f"Balanced: {len(causal_pairs)} causal, {len(non_causal_sample)} non-causal")
```

### 6.2 Create Training Examples

```python
# Create InputExamples
examples = []
for pair in balanced_data:
    label = 1.0 if pair['label'] == 'causal' else 0.0
    examples.append(InputExample(
        texts=[pair['text_a'], pair['text_b']],
        label=label
    ))

# Train/eval split
train_examples, eval_examples = train_test_split(
    examples,
    test_size=0.2,
    random_state=CONFIG['seed'],
    stratify=[ex.label for ex in examples]
)

print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
```

### 6.3 Initialize CrossEncoder

```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# Initialize model
model = CrossEncoder(
    CONFIG['base_model'],
    num_labels=1,           # Binary classification (sigmoid output)
    max_length=512,
    device=device
)

print(f"✅ Model loaded on {device}")
```

### 6.4 Setup Evaluator

```python
# Prepare evaluation data
eval_sentence_pairs = [(ex.texts[0], ex.texts[1]) for ex in eval_examples]
eval_labels = [int(ex.label) for ex in eval_examples]

evaluator = CEBinaryClassificationEvaluator(
    sentence_pairs=eval_sentence_pairs,
    labels=eval_labels,
    name='causal_eval'
)
```

### 6.5 Train Model

```python
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_path = Path(CONFIG['output_dir']) / f"causal_{timestamp}"
save_path.mkdir(parents=True, exist_ok=True)

# DataLoader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=CONFIG['batch_size']
)

# Calculate warmup steps
total_steps = len(train_dataloader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

print(f"🚀 Training: {CONFIG['epochs']} epochs, {total_steps} steps, {warmup_steps} warmup")

# Train
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=CONFIG['epochs'],
    warmup_steps=warmup_steps,
    output_path=str(save_path),
    save_best_model=True,
    show_progress_bar=True
)

print(f"✅ Training complete! Model saved to {save_path}")
```

---

## 7. Evaluation

### 7.1 Load Best Model

```python
# Load the saved best model
best_model = CrossEncoder(str(save_path), device=device)
```

### 7.2 Generate Predictions

```python
# Predict on evaluation set
predictions = best_model.predict(eval_sentence_pairs)

# Convert to binary (threshold = 0.5)
binary_preds = (predictions > 0.5).astype(int)
```

### 7.3 Metrics

```python
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve
)

# Classification report
print("📋 Classification Report:")
print(classification_report(eval_labels, binary_preds, 
                           target_names=['Non-Causal', 'Causal']))

# ROC-AUC
roc_auc = roc_auc_score(eval_labels, predictions)
print(f"🎯 ROC-AUC: {roc_auc:.4f}")

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(eval_labels, predictions)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
best_f1 = f1_scores[best_idx]

print(f"🎯 Best F1: {best_f1:.4f} @ threshold={best_threshold:.3f}")

# Confusion matrix
cm = confusion_matrix(eval_labels, binary_preds)
print(f"\n📊 Confusion Matrix:\n{cm}")
```

### 7.4 Expected Performance

| Metric | Expected Range | Good Value |
|--------|----------------|------------|
| ROC-AUC | 0.70-0.85 | ≥ 0.75 |
| F1 (Causal) | 0.50-0.70 | ≥ 0.60 |
| Precision (Causal) | 0.55-0.75 | ≥ 0.65 |
| Recall (Causal) | 0.45-0.70 | ≥ 0.55 |

### 7.5 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Causal', 'Causal'],
            yticklabels=['Non-Causal', 'Causal'])
axes[0].set_title('Confusion Matrix')

# Score Distribution
causal_scores = predictions[np.array(eval_labels) == 1]
non_causal_scores = predictions[np.array(eval_labels) == 0]
axes[1].hist(non_causal_scores, bins=30, alpha=0.7, label='Non-Causal')
axes[1].hist(causal_scores, bins=30, alpha=0.7, label='Causal')
axes[1].axvline(0.5, color='black', linestyle='--', label='Threshold')
axes[1].set_title('Score Distribution')
axes[1].legend()

# Precision-Recall Curve
axes[2].plot(recall, precision)
axes[2].scatter([recall[best_idx]], [precision[best_idx]], color='red', s=100)
axes[2].set_title(f'PR Curve (Best F1={best_f1:.3f})')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')

plt.tight_layout()
plt.savefig(save_path / 'evaluation_plots.png')
plt.show()
```

---

## 8. Inference

### 8.1 Load Model

```python
from sentence_transformers.cross_encoder import CrossEncoder

# Load trained model
model = CrossEncoder('models/causal_classifier/causal_YYYYMMDD_HHMM')
```

### 8.2 Single Pair Prediction

```python
def detect_causal(text_a: str, text_b: str, threshold: float = 0.5) -> dict:
    """
    Detect if text_a caused text_b.
    
    Args:
        text_a: Potential cause ticket text
        text_b: Potential effect ticket text
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict with is_causal, confidence, threshold
    """
    score = model.predict([(text_a, text_b)])[0]
    
    return {
        'is_causal': bool(score > threshold),
        'confidence': float(score),
        'threshold': threshold
    }

# Example
result = detect_causal(
    "Server database crashed due to disk full",
    "Application reporting connection timeout errors"
)
print(f"Causal: {result['is_causal']}, Confidence: {result['confidence']:.3f}")
```

### 8.3 Batch Prediction

```python
def detect_causal_batch(pairs: list, threshold: float = 0.5) -> list:
    """
    Detect causal relationships for multiple pairs.
    
    Args:
        pairs: List of (text_a, text_b) tuples
        threshold: Classification threshold
    
    Returns:
        List of result dicts
    """
    scores = model.predict(pairs)
    
    return [
        {
            'pair': pairs[i],
            'is_causal': bool(scores[i] > threshold),
            'confidence': float(scores[i])
        }
        for i in range(len(pairs))
    ]

# Example
pairs = [
    ("Server crash", "App timeout"),
    ("Password reset", "Printer offline"),
]
results = detect_causal_batch(pairs)
```

### 8.4 Bidirectional Check

Causal relationships are directional. Check both directions:

```python
def detect_causal_bidirectional(text_a: str, text_b: str, threshold: float = 0.5) -> dict:
    """
    Check causal relationship in both directions.
    
    Returns:
        dict with a_causes_b, b_causes_a, and confidence scores
    """
    # Check A → B
    score_ab = model.predict([(text_a, text_b)])[0]
    
    # Check B → A
    score_ba = model.predict([(text_b, text_a)])[0]
    
    return {
        'a_causes_b': bool(score_ab > threshold),
        'a_causes_b_confidence': float(score_ab),
        'b_causes_a': bool(score_ba > threshold),
        'b_causes_a_confidence': float(score_ba),
        'any_causal': bool(score_ab > threshold or score_ba > threshold),
        'max_confidence': float(max(score_ab, score_ba))
    }

# Example
result = detect_causal_bidirectional(
    "DNS configuration changed",
    "Website not loading for users"
)
print(f"A→B: {result['a_causes_b']}, B→A: {result['b_causes_a']}")
```

---

## 9. Two-Stage Pipeline Integration

### 9.1 Complete Pipeline Code

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ITSMTicketAnalyzer:
    """
    Two-stage pipeline for ticket similarity and causal detection.
    
    Stage 1: Fast similarity search (Bi-Encoder)
    Stage 2: Causal relationship classification (CrossEncoder)
    """
    
    def __init__(self, 
                 similarity_model_path: str,
                 causal_model_path: str = None,
                 device: str = 'cpu'):
        """
        Initialize the analyzer.
        
        Args:
            similarity_model_path: Path to trained bi-encoder
            causal_model_path: Path to trained CrossEncoder (optional)
            device: 'cuda', 'mps', or 'cpu'
        """
        # Stage 1: Similarity model (always loaded)
        self.similarity_model = SentenceTransformer(similarity_model_path, device=device)
        
        # Stage 2: Causal model (optional)
        self.causal_model = None
        if causal_model_path:
            self.causal_model = CrossEncoder(causal_model_path, device=device)
        
        # Cache for embeddings
        self.ticket_embeddings = None
        self.ticket_ids = None
        self.ticket_texts = None
    
    def load_tickets(self, ticket_ids: list, ticket_texts: list):
        """
        Pre-compute embeddings for all tickets.
        """
        self.ticket_ids = ticket_ids
        self.ticket_texts = ticket_texts
        self.ticket_embeddings = self.similarity_model.encode(
            ticket_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"✅ Loaded {len(ticket_ids)} tickets")
    
    def find_similar(self, query_text: str, top_k: int = 10) -> list:
        """
        Stage 1: Find top-K similar tickets.
        """
        if self.ticket_embeddings is None:
            raise ValueError("No tickets loaded. Call load_tickets() first.")
        
        query_emb = self.similarity_model.encode(query_text, convert_to_numpy=True)
        similarities = cosine_similarity([query_emb], self.ticket_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'ticket_id': self.ticket_ids[i],
                'text': self.ticket_texts[i],
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]
    
    def analyze_ticket(self, 
                      new_ticket_text: str, 
                      top_k: int = 10,
                      causal_threshold: float = 0.5) -> dict:
        """
        Full two-stage analysis of a new ticket.
        
        Args:
            new_ticket_text: Text of the new ticket
            top_k: Number of similar tickets to find
            causal_threshold: Threshold for causal classification
        
        Returns:
            dict with similar tickets and causal relationships
        """
        # Stage 1: Find similar tickets
        similar = self.find_similar(new_ticket_text, top_k=top_k)
        
        # Stage 2: Check causal relationships (if model available)
        if self.causal_model:
            for ticket in similar:
                # Check both directions
                score_new_causes_old = self.causal_model.predict([
                    (new_ticket_text, ticket['text'])
                ])[0]
                score_old_causes_new = self.causal_model.predict([
                    (ticket['text'], new_ticket_text)
                ])[0]
                
                ticket['new_causes_this'] = bool(score_new_causes_old > causal_threshold)
                ticket['this_causes_new'] = bool(score_old_causes_new > causal_threshold)
                ticket['causal_confidence'] = float(max(score_new_causes_old, score_old_causes_new))
        
        return {
            'query': new_ticket_text,
            'similar_tickets': similar
        }


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ITSMTicketAnalyzer(
        similarity_model_path='models/v6_refactored_finetuned/...',
        causal_model_path='models/causal_classifier/causal_...',
        device='mps'  # or 'cuda' or 'cpu'
    )
    
    # Load existing tickets (do this once, cache the embeddings)
    ticket_ids = ['INC001', 'INC002', 'INC003', ...]
    ticket_texts = [
        "Database server crashed due to disk full",
        "Application timeout errors for multiple users",
        "Password reset requested by user",
        ...
    ]
    analyzer.load_tickets(ticket_ids, ticket_texts)
    
    # Analyze a new ticket
    new_ticket = "Users reporting slow application response times"
    results = analyzer.analyze_ticket(new_ticket, top_k=5)
    
    print(f"Query: {results['query']}")
    print("\nSimilar tickets:")
    for t in results['similar_tickets']:
        causal_flag = ""
        if t.get('this_causes_new'):
            causal_flag = " 🔥 CAUSED BY THIS"
        elif t.get('new_causes_this'):
            causal_flag = " → CAUSES THIS"
        
        print(f"  {t['ticket_id']}: similarity={t['similarity']:.3f}{causal_flag}")
        print(f"    {t['text'][:60]}...")
```

### 9.2 Performance Characteristics

| Stage | Operation | Latency | Throughput |
|-------|-----------|---------|------------|
| 1 | Encode new ticket | ~10ms | - |
| 1 | Similarity search (10K tickets) | ~5ms | - |
| 2 | Causal check (1 pair) | ~50ms | ~20/sec |
| 2 | Causal check (10 pairs, batch) | ~200ms | ~50/sec |

**Total for 10 candidates:** ~250ms (acceptable for real-time use)

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Low ROC-AUC (<0.65)

**Possible causes:**
- Insufficient causal examples in training data
- Labels are noisy/incorrect
- Model too small

**Solutions:**
- Add more labeled causal pairs
- Review and clean labels manually
- Try larger model: `cross-encoder/ms-marco-MiniLM-L-12-v2`

#### Class Imbalance Hurting Performance

**Symptoms:** High accuracy but low recall on causal class

**Solutions:**
- Increase causal ratio to 1:2 instead of 1:3
- Use class weights in loss function
- Oversample causal pairs (with augmentation)

#### MPS Memory Issues

**Symptoms:** Kernel crash during training

**Solutions:**
```python
# Reduce batch size
CONFIG['batch_size'] = 8

# Clear cache before training
import gc
gc.collect()
torch.mps.empty_cache()

# Limit training pairs
CONFIG['max_pairs'] = 5000
```

#### Model Not Learning

**Symptoms:** Loss doesn't decrease, metrics stay at random baseline

**Solutions:**
- Check data loading (ensure labels are correct)
- Increase learning rate: try default warmup
- Verify text preprocessing matches inference

### 10.2 Performance Tuning

| Issue | Solution |
|-------|----------|
| Training too slow | Increase batch_size, use CUDA |
| Low recall | Lower threshold (0.4 instead of 0.5) |
| Low precision | Raise threshold (0.6 instead of 0.5) |
| Overfitting | Reduce epochs, add dropout |
| Underfitting | Increase epochs, use larger model |

### 10.3 Debugging

```python
# Check prediction distribution
predictions = model.predict(eval_pairs)
print(f"Predictions: min={predictions.min():.3f}, max={predictions.max():.3f}, "
      f"mean={predictions.mean():.3f}, std={predictions.std():.3f}")

# Check for degenerate model (all same prediction)
if predictions.std() < 0.1:
    print("⚠️ Model outputs are too uniform - may not have learned")

# Inspect misclassifications
for i, (pair, label, pred) in enumerate(zip(eval_pairs, eval_labels, predictions)):
    if (pred > 0.5) != label:
        print(f"Misclassified: label={label}, pred={pred:.3f}")
        print(f"  A: {pair[0][:60]}...")
        print(f"  B: {pair[1][:60]}...")
```

---

## Appendix A: Model Artifacts

```
models/causal_classifier/causal_YYYYMMDD_HHMM/
├── config.json                    # Model configuration
├── model.safetensors              # Model weights (~90MB)
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json                 # Fast tokenizer
├── vocab.txt
├── training_metadata.json         # Training config and metrics
└── evaluation_plots.png           # Visualization
```

---

## Appendix B: Training Metadata Format

```json
{
  "training_date": "2025-12-04T15:30:00",
  "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "task": "binary_causal_classification",
  "config": {
    "epochs": 5,
    "batch_size": 16,
    "warmup_ratio": 0.1,
    "max_length": 512,
    "seed": 42
  },
  "device": "mps",
  "metrics": {
    "roc_auc": 0.78,
    "best_f1": 0.65,
    "best_threshold": 0.45
  },
  "data_stats": {
    "total_pairs": 4000,
    "train_pairs": 3200,
    "eval_pairs": 800,
    "causal_pairs": 1000,
    "non_causal_pairs": 3000
  }
}
```

---

## Appendix C: Quick Start

```bash
# 1. Ensure relationship data exists
ls data/relationship_pairs.json

# 2. Train causal classifier
jupyter notebook NLI.ipynb
# Run all cells

# 3. Use in inference
python -c "
from sentence_transformers.cross_encoder import CrossEncoder
model = CrossEncoder('models/causal_classifier/causal_...')
score = model.predict([('Server crashed', 'App unavailable')])[0]
print(f'Causal score: {score:.3f}')
"
```

---

*Document maintained by the Nexustism team. Last updated: December 2025*
