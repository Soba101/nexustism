# ITSM Ticket Similarity Model Training Pipeline

> Complete documentation for training the SentenceTransformer fine-tuned model for ITSM incident ticket similarity matching.

**Last Updated:** December 2025  
**Version:** v6 Refactored (v2-3)  
**Primary Notebook:** `finetune_v6_refactored_kaggle_v2-3.ipynb`

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Loading & Preprocessing](#3-data-loading--preprocessing)
4. [Pair Generation Strategy](#4-pair-generation-strategy)
5. [Model Architecture](#5-model-architecture)
6. [Training Configuration](#6-training-configuration)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Adversarial Diagnostic](#8-adversarial-diagnostic)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Production Deployment](#10-production-deployment)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Pipeline Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ITSM TICKET SIMILARITY PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw CSV    │───▶│ Preprocess   │───▶│    Pair      │───▶│   Training   │
│   Incidents  │    │   & Clean    │    │  Generation  │    │    Loop      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────────────────────────────────────┘
                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Evaluation  │───▶│  Adversarial │───▶│    Save      │───▶│  Production  │
│   Metrics    │    │  Diagnostic  │    │    Model     │    │   Deploy     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Two-Stage Inference Architecture

```
Stage 1: Similarity Model (Bi-Encoder)
├── Fast embedding generation (768-dim)
├── Cosine similarity for ranking
└── Returns top-K similar tickets

Stage 2: Causal Classifier (Cross-Encoder) [Optional]
├── Slower but more accurate
├── Binary classification: causal vs non-causal
└── Only runs on top-K candidates from Stage 1
└── See: docs/causal_pipeline.md
```

### Key Metrics Targets

| Metric | Target | Production Threshold |
|--------|--------|---------------------|
| Spearman Correlation | 0.75-0.85 | ≥ 0.80 |
| ROC-AUC | > 0.90 | ≥ 0.95 |
| PR-AUC | > 0.90 | ≥ 0.95 |
| Adversarial ROC-AUC | ≥ 0.70 | Required for deploy |
| Adversarial F1 | ≥ 0.70 | Required for deploy |

---

## 2. Environment Setup

### 2.1 Required Packages

```python
# Core ML Stack
sentence-transformers>=2.2.2
transformers>=4.38.0
torch                        # with MPS/CUDA support
scikit-learn==1.3.2
imbalanced-learn==0.12.0     # for SMOTE (optional)

# Data Processing
numpy==1.26.4
pandas
scipy

# NLP
nltk                         # wordnet, stopwords, punkt

# Visualization
matplotlib
seaborn

# Utilities
tqdm
joblib
```

### 2.2 Environment Variables

```python
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### 2.3 Device Detection

Priority order: **CUDA → MPS → CPU**

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(CONFIG['seed'])
    print(f"🚀 CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    torch.mps.manual_seed(CONFIG['seed'])
    print("🍎 MPS (Apple Silicon)")
else:
    device = "cpu"
    print("⚠️ CPU only")
```

### 2.4 NLTK Data Setup

Required resources downloaded automatically:

```python
import nltk
nltk.download('wordnet')      # Lemmatization
nltk.download('omw-1.4')      # Open Multilingual Wordnet
nltk.download('stopwords')    # English stop words
nltk.download('punkt')        # Sentence tokenizer
```

---

## 3. Data Loading & Preprocessing

### 3.1 Input Data Format

**Required CSV columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Number` | string | Ticket ID | INC0012345 |
| `Short Description` | string | Brief summary | "Email not working" |
| `Description` | string | Detailed description | "User reports Outlook crashes..." |
| `Category` | string | ITSM category | "Software" |
| `Subcategory` | string | Sub-classification | "Email" |
| `Service` | string | Affected service | "Microsoft 365" |
| `Service offering` | string | Specific offering | "Exchange Online" |
| `Assignment group` | string | Support team | "L2 Email Support" |

**⚠️ CRITICAL: Never include these columns to prevent data leakage:**

- `Resolution notes`
- `Resolution code`  
- `Closed by`
- Any post-resolution fields

### 3.2 Text Preprocessing Pipeline

```python
def normalize_field(val: str) -> str:
    """Clean individual field values."""
    s = str(val).strip()
    s = re.sub(r"\s+", " ", s)  # Collapse whitespace
    placeholders = {"", "nan", "none", "null", "unknown", "n/a", "na"}
    if s.lower() in placeholders:
        return ""
    return s

# Apply to all text columns
for col in required_cols:
    df[col] = df[col].fillna("").apply(normalize_field)

# Normalize casing for structured fields
context_cols = ["Service", "Service offering", "Category", "Subcategory", "Assignment group"]
for col in context_cols:
    df[col] = df[col].str.lower()
```

### 3.3 Contextual Text Construction (v6 Format)

**Format:** Context placed at END to reduce shortcut exploitation

```python
# Build bracketed context parts
df['context_service'] = "[" + df['Service'] + " | " + df['Service offering'] + "]"
df['context_category'] = "[" + df['Category'] + " | " + df['Subcategory'] + "]"
df['context_group'] = "Group: " + df['Assignment group'] + "."

# Core text (description only)
df['text_core'] = df['Short Description'] + ". " + df['Description']

# Full text with context at end
df['text'] = df['text_core'] + " (Context: " + df['context_service'] + " " + 
             df['context_category'] + " " + df['context_group'] + ")"
```

**Example output:**

```
Email not syncing on mobile device. User reports that Outlook app on iPhone 
stopped syncing emails since yesterday morning. Other apps work fine. 
(Context: [microsoft 365 | exchange online] [software | email] Group: L2 Email Support.)
```

**Why context at end?**

- Reduces shortcut exploitation (model can't just match category prefixes)
- Forces model to read full content before seeing metadata
- Improves generalization to mislabeled tickets

### 3.4 Data Quality Filters

```python
# Minimum text length
min_length = 10  # characters

# Filter short/empty texts
initial_count = len(df)
df = df[df['text'].str.len() >= min_length].copy()
dropped = initial_count - len(df)
print(f"⚠️ Dropped {dropped} incidents due to short/empty text")

# Create stratification ID for balanced splits
df['category_id'] = df.groupby(['Category', 'Subcategory']).ngroup()
```

---

## 4. Pair Generation Strategy

### 4.1 Three-Way Data Split

```python
from sklearn.model_selection import train_test_split

# 1. Holdout (10%) - completely unseen, for final validation
train_eval_df, holdout_df = train_test_split(
    df, 
    test_size=0.10, 
    stratify=df['category_id'], 
    random_state=42
)

# 2. Train/Eval split (85%/15% of remaining)
train_df, eval_df = train_test_split(
    train_eval_df, 
    test_size=0.15, 
    stratify=train_eval_df['category_id'], 
    random_state=42
)
```

**Final split proportions:**

- Train: ~76.5% of original data
- Eval: ~13.5% of original data
- Holdout: ~10% of original data (never seen during training)

### 4.2 TF-IDF Similarity Calculator

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class TextSimilarityCalculator:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000
        )
        self.tfidf = self.vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF matrix shape: {self.tfidf.shape}")
    
    def get_similarity(self, idx1, idx2):
        """Compute cosine similarity between two documents."""
        return (self.tfidf[idx1] @ self.tfidf[idx2].T).toarray()[0][0]
```

### 4.3 Hybrid Robust Pair Generation (v2-3 Strategy)

**The core innovation: Force semantic learning, prevent category shortcuts**

```python
def generate_robust_pairs(df, target_count, config):
    """
    Generates pairs that force the model to learn semantic content,
    not just category matching shortcuts.
    
    Strategy:
    - 40% positives: high TF-IDF (>0.5) - can be same OR cross category
    - 30% hard negatives: same category but low TF-IDF (<0.3)
    - 30% easy negatives: cross category and low TF-IDF (<0.3)
    """
    sim_calculator = TextSimilarityCalculator(df['text'].tolist())
    
    pos_target = int(target_count * 0.4)
    hard_neg_target = int(target_count * 0.3)
    easy_neg_target = target_count - pos_target - hard_neg_target
    
    pairs = []
    
    # 1. CONTENT POSITIVES: High TF-IDF similarity (any category)
    while len(pairs) < pos_target:
        i1, i2 = random.sample(all_indices, 2)
        if sim_calculator.get_similarity(i1, i2) > 0.5:
            pairs.append(InputExample(
                texts=[df.at[i1, 'text'], df.at[i2, 'text']], 
                label=1.0
            ))
    
    # 2. HARD NEGATIVES: Same category, low TF-IDF
    while len(hard_negatives) < hard_neg_target:
        # Sample from same category
        i1, i2 = random.sample(same_category_indices, 2)
        if sim_calculator.get_similarity(i1, i2) < 0.3:
            pairs.append(InputExample(
                texts=[df.at[i1, 'text'], df.at[i2, 'text']], 
                label=0.0
            ))
    
    # 3. EASY NEGATIVES: Different category, low TF-IDF
    while len(easy_negatives) < easy_neg_target:
        i1, i2 = random.sample(all_indices, 2)
        if df.at[i1, 'category_id'] != df.at[i2, 'category_id']:
            if sim_calculator.get_similarity(i1, i2) < 0.3:
                pairs.append(InputExample(
                    texts=[df.at[i1, 'text'], df.at[i2, 'text']], 
                    label=0.0
                ))
    
    return pairs
```

### 4.4 Pair Type Breakdown

| Type | Ratio | TF-IDF Threshold | Category Constraint | Label | Purpose |
|------|-------|------------------|---------------------|-------|---------|
| Content Positive | 40% | > 0.5 | Any | 1.0 | Learn semantic similarity |
| Hard Negative | 30% | < 0.3 | Same | 0.0 | Prevent category shortcuts |
| Easy Negative | 30% | < 0.3 | Different | 0.0 | Basic discrimination |

### 4.5 Why Hard Negatives Are Critical

**Without hard negatives:**

- Model learns: "same category = similar"
- Fails on mislabeled tickets
- High standard metrics, low adversarial metrics

**With hard negatives:**

- Model learns: "similar content = similar"
- Generalizes to edge cases
- Robust to category noise

---

## 5. Model Architecture

### 5.1 Base Model

**Model:** `sentence-transformers/all-mpnet-base-v2`

| Property | Value |
|----------|-------|
| Architecture | MPNet (Microsoft) |
| Embedding Dimension | 768 |
| Max Sequence Length | 384 tokens |
| Vocab Size | 30,527 |
| Parameters | ~110M |
| Pooling Strategy | Mean pooling |
| Pre-training | 1B+ sentence pairs |

### 5.2 Model Components

```
SentenceTransformer(
  (0): Transformer(
      MPNetModel with 12 layers, 12 attention heads, 768 hidden dim
  )
  (1): Pooling(
      Mean pooling over all token embeddings
  )
  (2): Normalize(
      L2 normalization (optional, applied at inference)
  )
)
```

### 5.3 Loss Function

**`CosineSimilarityLoss`** - optimizes for explicit similarity labels

```python
from sentence_transformers import losses

train_loss = losses.CosineSimilarityLoss(model)
```

**Why CosineSimilarityLoss (not MultipleNegativesRankingLoss)?**

| Aspect | CosineSimilarityLoss | MNRL |
|--------|---------------------|------|
| Pair labels | Uses explicit 0.0/1.0 labels | Ignores labels, uses in-batch negatives |
| Hard negatives | Respects our curated hard negatives | Mines its own (may miss edge cases) |
| Best for | Labeled pairs with specific negative mining | Positive-only pairs |

Our pipeline explicitly generates labeled hard negatives, so CosineSimilarityLoss is appropriate.

---

## 6. Training Configuration

### 6.1 Hyperparameters

```python
CONFIG = {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'output_dir': 'models/v6_refactored_finetuned',
    'source_data': 'data/dummy_data_promax.csv',
    
    # Training Hyperparameters
    'epochs': 3,              # Peak performance at epoch 2
    'batch_size': 8,          # Safe for 8GB GPU / 16GB MPS
    'lr': 2e-5,               # AdamW default for transformers
    'max_seq_length': 384,    # Covers 99%+ of ITSM tickets
    
    # Data Strategy
    'num_pairs': 2000,        # Total training pairs
    'pos_ratio': 0.4,         # 40% positive, 60% negative
    'eval_split': 0.15,       # Validation set proportion
    
    # Reproducibility
    'seed': 42
}
```

### 6.2 Training Schedule

```python
# Total training steps
total_steps = len(train_dataloader) * CONFIG['epochs']

# Warmup: 10% of total steps (linear warmup)
warmup_steps = int(total_steps * 0.1)

# Evaluation frequency: every 10% of training
evaluation_steps = int(total_steps * 0.1)

# Save best model based on Spearman correlation
save_best_model = True
```

### 6.3 DataLoader Configuration

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=CONFIG['batch_size'],
    num_workers=0,                    # Required for InputExample
    pin_memory=(device != 'cpu')      # Speed up GPU transfer
)
```

**Why `num_workers=0`?**

- `InputExample` objects can't be pickled for multiprocessing
- Setting workers > 0 causes serialization errors
- Single-threaded loading is sufficient for our data size

### 6.4 Memory Management

```python
import gc

def clear_memory():
    """Clear GPU/MPS cache before training."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Call before model loading and training
clear_memory()
```

### 6.5 Training Execution

```python
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Initialize model
model = SentenceTransformer(CONFIG['model_name'], device=device)
model.max_seq_length = CONFIG['max_seq_length']

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_path = Path(CONFIG['output_dir']) / f"v6_refactored_finetuned_{timestamp}"
save_path.mkdir(parents=True, exist_ok=True)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=CONFIG['epochs'],
    warmup_steps=warmup_steps,
    optimizer_params={'lr': CONFIG['lr']},
    output_path=str(save_path),
    evaluation_steps=evaluation_steps,
    save_best_model=True,
    show_progress_bar=True
)
```

---

## 7. Evaluation Framework

### 7.1 ITSMEvaluator Class

Custom evaluator tracking ITSM-specific metrics:

```python
from sentence_transformers.evaluation import SentenceEvaluator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

class ITSMEvaluator(SentenceEvaluator):
    def __init__(self, examples, batch_size=16, name=""):
        self.texts1 = [ex.texts[0] for ex in examples]
        self.texts2 = [ex.texts[1] for ex in examples]
        self.labels = np.array([ex.label for ex in examples])
        self.batch_size = batch_size
        self.name = name
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Encode pairs
        emb1 = model.encode(self.texts1, batch_size=self.batch_size)
        emb2 = model.encode(self.texts2, batch_size=self.batch_size)
        
        # Compute cosine similarity
        scores = np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        )
        
        # Calculate metrics
        spearman, _ = spearmanr(self.labels, scores)
        pearson, _ = pearsonr(self.labels, scores)
        roc_auc = roc_auc_score(self.labels, scores)
        pr_auc = average_precision_score(self.labels, scores)
        
        print(f"Epoch {epoch}: Spearman={spearman:.4f}, ROC-AUC={roc_auc:.4f}")
        
        # Return primary metric for model selection
        return spearman
```

### 7.2 Metrics Explained

| Metric | What It Measures | Good Value | Interpretation |
|--------|------------------|------------|----------------|
| **Spearman** | Ranking correlation | ≥ 0.80 | How well scores preserve similarity order |
| **Pearson** | Linear correlation | ≥ 0.90 | Raw score-label linear relationship |
| **ROC-AUC** | Discrimination ability | ≥ 0.95 | Can model distinguish similar/dissimilar? |
| **PR-AUC** | Precision-recall balance | ≥ 0.95 | Performance on imbalanced data |
| **F1 @ Best** | Optimal classification | ≥ 0.90 | Best threshold for binary decisions |

### 7.3 Evaluation Sets

| Set | Size | Purpose | When Used |
|-----|------|---------|-----------|
| **Eval** | 15% | During-training validation | Every evaluation_steps |
| **Holdout** | 10% | Final generalization check | After training complete |
| **Adversarial** | ~400 pairs | Category shortcut detection | After training complete |

### 7.4 Comprehensive Evaluation Function

```python
def run_eval(examples, model, name="eval"):
    """Run full evaluation with all metrics and visualizations."""
    texts1 = [ex.texts[0] for ex in examples]
    texts2 = [ex.texts[1] for ex in examples]
    labels = np.array([ex.label for ex in examples])
    
    # Encode
    emb1 = model.encode(texts1, batch_size=8, show_progress_bar=True)
    emb2 = model.encode(texts2, batch_size=8, show_progress_bar=True)
    
    # Cosine similarity
    scores = np.sum(emb1 * emb2, axis=1) / (
        np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
    )
    
    # Metrics
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Binary predictions at best threshold
    preds = (scores >= best_threshold).astype(int)
    f1 = f1_score(labels, preds)
    
    print(f"[{name}] ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
          f"F1={f1:.4f} @ threshold={best_threshold:.3f}")
    
    return {
        'scores': scores,
        'labels': labels,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'best_threshold': best_threshold
    }
```

---

## 8. Adversarial Diagnostic

### 8.1 Purpose

**Detect if the model is exploiting category shortcuts rather than learning true semantic similarity.**

High standard metrics (Spearman 0.85, ROC-AUC 0.99) can be misleading if the model simply learned:

- "Same category → similar"
- "Different category → dissimilar"

The adversarial diagnostic tests if the model **actually understands content**.

### 8.2 Adversarial Pair Types

```
┌─────────────────────────────────────────────────────────────┐
│                  ADVERSARIAL PAIR TYPES                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HARD POSITIVES (label=1.0)                                 │
│  ├── DIFFERENT categories                                   │
│  ├── HIGH content similarity (TF-IDF > 0.6)                │
│  └── Model SHOULD score HIGH if learning semantics         │
│                                                              │
│  HARD NEGATIVES (label=0.0)                                 │
│  ├── SAME category                                          │
│  ├── LOW content similarity (TF-IDF < 0.3)                 │
│  └── Model SHOULD score LOW despite matching category      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Implementation

```python
# Use holdout data with STRIPPED context (content only)
diag_df = holdout_incidents_df.copy()
diag_df['content_only_text'] = (
    diag_df['Short Description'] + ". " + diag_df['Description']
)

# Recompute TF-IDF on content-only text
diag_sim_calc = TextSimilarityCalculator(diag_df['content_only_text'].tolist())

hard_positives = []
hard_negatives = []

for attempt in range(50000):
    i1, i2 = random.sample(range(len(diag_df)), 2)
    cat1, cat2 = diag_df.at[i1, 'category_id'], diag_df.at[i2, 'category_id']
    tfidf_sim = diag_sim_calc.get_similarity(i1, i2)
    
    # Hard positive: DIFFERENT category, HIGH content similarity
    if cat1 != cat2 and tfidf_sim > 0.6 and len(hard_positives) < 200:
        hard_positives.append(InputExample(
            texts=[diag_df.at[i1, 'content_only_text'], 
                   diag_df.at[i2, 'content_only_text']],
            label=1.0
        ))
    
    # Hard negative: SAME category, LOW content similarity
    if cat1 == cat2 and tfidf_sim < 0.3 and len(hard_negatives) < 200:
        hard_negatives.append(InputExample(
            texts=[diag_df.at[i1, 'content_only_text'], 
                   diag_df.at[i2, 'content_only_text']],
            label=0.0
        ))
```

### 8.4 Pass/Fail Criteria

```
┌─────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC VERDICT                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ PASS: ROC-AUC ≥ 0.70 AND F1 ≥ 0.70                      │
│     → Model learns semantic content beyond category shortcuts│
│     → Safe to deploy to production                          │
│                                                              │
│  ❌ FAIL: Either metric < 0.70                              │
│     → Model exploiting category prefix matching             │
│     → Requires architecture/data changes before deployment  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 If Diagnostic Fails

**Required fixes:**

1. **Remove category from text entirely:**

   ```python
   df['text'] = df['Short Description'] + ". " + df['Description']
   ```

2. **Or move context to very end:**

   ```python
   df['text'] = df['text_core'] + " (Category: " + df['Category'] + ")"
   ```

3. **Generate 100% content-based pairs (ignore categories completely)**

4. **Increase hard negative ratio to 50%**

5. **Retrain and re-run diagnostic**

---

## 9. Inference Pipeline

### 9.1 Load Trained Model

```python
from sentence_transformers import SentenceTransformer

# Load the best model from training
model = SentenceTransformer('models/v6_refactored_finetuned/v6_refactored_finetuned_YYYYMMDD_HHMM')

# Verify model properties
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
print(f"Max sequence length: {model.max_seq_length}")
```

### 9.2 Preprocess New Tickets

```python
def preprocess_ticket(short_desc: str, description: str, 
                      category: str, subcategory: str,
                      service: str, offering: str, 
                      group: str) -> str:
    """
    Preprocess a new ticket for embedding.
    Must match training preprocessing exactly.
    """
    # Clean fields
    def clean(s):
        s = str(s).strip() if s else ""
        return "" if s.lower() in {"nan", "none", "null", "unknown"} else s
    
    short_desc = clean(short_desc)
    description = clean(description)
    category = clean(category).lower()
    subcategory = clean(subcategory).lower()
    service = clean(service).lower()
    offering = clean(offering).lower()
    group = clean(group).lower()
    
    # Build text (context at end)
    text_core = f"{short_desc}. {description}"
    context_parts = []
    if service or offering:
        context_parts.append(f"[{service} | {offering}]")
    if category or subcategory:
        context_parts.append(f"[{category} | {subcategory}]")
    if group:
        context_parts.append(f"Group: {group}.")
    
    if context_parts:
        return f"{text_core} (Context: {' '.join(context_parts)})"
    return text_core
```

### 9.3 Single Ticket Similarity Search

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_tickets(query_text: str, 
                        ticket_embeddings: np.ndarray,
                        ticket_ids: list, 
                        top_k: int = 10) -> list:
    """
    Find top-K most similar tickets to a query.
    
    Args:
        query_text: Preprocessed new ticket text
        ticket_embeddings: Pre-computed embeddings (N x 768)
        ticket_ids: List of ticket IDs corresponding to embeddings
        top_k: Number of results to return
    
    Returns:
        List of (ticket_id, similarity_score) tuples, sorted by score
    """
    # Encode query
    query_emb = model.encode(query_text, convert_to_numpy=True)
    
    # Compute cosine similarities
    similarities = cosine_similarity([query_emb], ticket_embeddings)[0]
    
    # Get top-K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(ticket_ids[i], float(similarities[i])) for i in top_indices]
```

### 9.4 Batch Embedding Computation

```python
def compute_all_embeddings(tickets: list, batch_size: int = 32) -> np.ndarray:
    """
    Pre-compute embeddings for all tickets in database.
    Run this once and cache the results.
    """
    embeddings = model.encode(
        tickets,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for faster cosine
    )
    return embeddings

# Save for later use
np.save('embeddings/all_ticket_embeddings.npy', embeddings)

# Load cached embeddings
embeddings = np.load('embeddings/all_ticket_embeddings.npy')
```

### 9.5 Two-Stage Pipeline (With Causal Detection)

For causal relationship detection, see **`docs/causal_pipeline.md`**

```python
# Stage 1: Find similar tickets (fast)
similar_tickets = find_similar_tickets(new_ticket, embeddings, ticket_ids, top_k=10)

# Stage 2: Check causal relationships (slower, more accurate)
# See docs/causal_pipeline.md for CrossEncoder implementation
for ticket_id, similarity in similar_tickets:
    causal_score = causal_model.predict([(new_ticket, ticket_text)])
    if causal_score > 0.5:
        print(f"Ticket {ticket_id} may be causally related")
```

---

## 10. Production Deployment

### 10.1 Model Artifacts

```
models/v6_refactored_finetuned/v6_refactored_finetuned_YYYYMMDD_HHMM/
├── config.json                      # Transformer configuration
├── config_sentence_transformers.json
├── model.safetensors                # Model weights (~440MB)
├── modules.json                     # Pipeline structure
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json                   # Fast tokenizer
├── tokenizer_config.json
├── vocab.txt
├── training_metadata.json           # Reproducibility manifest ⬅️ REQUIRED
├── 1_Pooling/
│   └── config.json
├── 2_Normalize/
│   └── config.json
└── eval/
    └── validation_eval_results.csv
```

### 10.2 Required Metadata (training_metadata.json)

```json
{
  "training_date": "2025-12-04T14:36:00",
  "model_name": "sentence-transformers/all-mpnet-base-v2",
  "config": {
    "epochs": 3,
    "batch_size": 8,
    "lr": 2e-5,
    "max_seq_length": 384,
    "num_pairs": 2000,
    "pos_ratio": 0.4,
    "seed": 42
  },
  "device": "mps",
  "final_spearman": 0.8460,
  "data_splits": {
    "train_incidents": 1200,
    "eval_incidents": 200,
    "holdout_incidents": 150,
    "train_pairs": 1700,
    "eval_pairs": 300,
    "holdout_pairs": 200
  },
  "pair_generation_strategy": {
    "type": "hybrid_robust",
    "positives_ratio": 0.4,
    "hard_negatives_ratio": 0.3,
    "easy_negatives_ratio": 0.3,
    "positive_threshold": "TF-IDF > 0.5",
    "negative_threshold": "TF-IDF < 0.3"
  }
}
```

### 10.3 Pre-Deployment Checklist

```
□ Spearman correlation ≥ 0.80
□ ROC-AUC ≥ 0.95
□ Adversarial diagnostic PASSED (ROC-AUC ≥ 0.70, F1 ≥ 0.70)
□ Holdout metrics within 5% of eval metrics
□ training_metadata.json saved
□ Model tested on real ServiceNow data (if available)
□ Inference latency benchmarked on target hardware
□ Memory usage profiled
□ Embeddings pre-computed for existing tickets
```

### 10.4 Example API Endpoint

```python
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load model and embeddings at startup
model = SentenceTransformer('models/v6_refactored_finetuned/...')
embeddings = np.load('embeddings/all_tickets.npy')
ticket_ids = load_ticket_ids()  # Your ticket ID list

@app.route('/api/similar', methods=['POST'])
def find_similar():
    data = request.json
    
    # Preprocess incoming ticket
    query = preprocess_ticket(
        data.get('short_desc', ''),
        data.get('description', ''),
        data.get('category', ''),
        data.get('subcategory', ''),
        data.get('service', ''),
        data.get('offering', ''),
        data.get('group', '')
    )
    
    # Find similar tickets
    results = find_similar_tickets(query, embeddings, ticket_ids, top_k=10)
    
    return jsonify({
        'query': data,
        'similar_tickets': [
            {'id': tid, 'score': round(score, 4)} 
            for tid, score in results
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 11. Troubleshooting

### 11.1 Common Issues

#### MPS/CUDA Out of Memory

**Symptoms:** Kernel crash during training or encoding

**Solutions:**

1. Reduce `batch_size` to 4
2. Clear cache before operations:

   ```python
   torch.mps.empty_cache()  # or torch.cuda.empty_cache()
   gc.collect()
   ```

3. Restart kernel to fully release memory
4. Use CPU for large encoding jobs:

   ```python
   model.encode(texts, device='cpu')
   ```

#### Low Adversarial Metrics

**Symptoms:** Standard metrics high (>0.90), adversarial metrics low (<0.70)

**Cause:** Model exploiting category shortcuts

**Fix:**

1. Remove category prefix from text completely
2. Move context to very end of text
3. Increase hard negative ratio to 50%
4. Retrain with epochs=5-8
5. Re-run adversarial diagnostic

#### NLTK Data Not Found

**Symptoms:** `LookupError: Resource 'wordnet' not found`

**Fix:**

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
```

#### InputExample Pickle Error

**Symptoms:** `Can't pickle InputExample` when using DataLoader

**Fix:** Set `num_workers=0`:

```python
DataLoader(examples, num_workers=0, ...)
```

#### Model Not Learning (Flat Metrics)

**Symptoms:** Spearman stays at ~0.5, no improvement across epochs

**Possible causes:**

1. Learning rate too low → try `lr=5e-5`
2. Batch size too small → try `batch_size=16`
3. Not enough training pairs → increase to 5000+
4. Data quality issues → check preprocessing

### 11.2 Performance Tuning

| Issue | Solution |
|-------|----------|
| Training too slow | Increase `batch_size` (if memory allows) |
| Low Spearman | Increase epochs to 8-12 |
| High variance | Increase `num_pairs` to 5000+ |
| Overfitting | Reduce epochs to 2-3 |
| Poor on short tickets | Reduce `max_seq_length` to 256 |
| Inference too slow | Pre-compute and cache embeddings |

### 11.3 Debugging Tips

```python
# Check GPU memory usage
def log_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"MPS: {allocated:.2f}GB allocated")

# Inspect training pairs
def inspect_pairs(pairs, n=3):
    print("Sample POSITIVE pairs:")
    for p in [p for p in pairs if p.label == 1.0][:n]:
        print(f"  Text 1: {p.texts[0][:80]}...")
        print(f"  Text 2: {p.texts[1][:80]}...")
        print()
    
    print("Sample NEGATIVE pairs:")
    for p in [p for p in pairs if p.label == 0.0][:n]:
        print(f"  Text 1: {p.texts[0][:80]}...")
        print(f"  Text 2: {p.texts[1][:80]}...")
        print()

# Check TF-IDF similarity distribution
def check_tfidf_distribution(calculator, indices, n_samples=100):
    sims = []
    for _ in range(n_samples):
        i, j = random.sample(indices, 2)
        sims.append(calculator.get_similarity(i, j))
    print(f"TF-IDF similarity: mean={np.mean(sims):.3f}, "
          f"std={np.std(sims):.3f}, range=[{min(sims):.3f}, {max(sims):.3f}]")
```

---

## Appendix A: File Structure

```
nexustism/
├── finetune_v6_refactored_kaggle_v2-3.ipynb    # Main training notebook
├── NLI.ipynb                                     # Causal classifier (see causal_pipeline.md)
├── requirements.txt
├── data/
│   ├── dummy_data_promax.csv                    # Training incidents
│   └── relationship_pairs.json                  # Causal/duplicate labels
├── models/
│   └── v6_refactored_finetuned/
│       └── v6_refactored_finetuned_YYYYMMDD_HHMM/
│           ├── model.safetensors
│           ├── training_metadata.json
│           └── eval/
├── embeddings/
│   └── all_ticket_embeddings.npy                # Cached embeddings
└── docs/
    ├── model_pipeline.md                        # This document
    ├── causal_pipeline.md                       # Causal classifier docs
    └── tuning_documentation.md                  # Historical experiments
```

---

## Appendix B: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1 | Oct 2024 | Initial category-based pair generation |
| v2-v3 | Nov 2024 | TF-IDF mining, hard negatives |
| v4 | Nov 2024 | Three-way split, holdout validation |
| v5 | Nov 2024 | MPS support, memory optimization |
| v6 | Nov 2024 | Contextual embeddings (prefix format) |
| v6-refactored | Dec 2024 | Context at end, robust pair generation |
| **v2-3** | **Dec 2024** | **Current: Hybrid 40/30/30 strategy, adversarial diagnostic** |

---

## Appendix C: Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Soba101/nexustism.git
cd nexustism

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
# Place incidents CSV at data/dummy_data_promax.csv

# 4. Train similarity model
jupyter notebook finetune_v6_refactored_kaggle_v2-3.ipynb
# Run all cells in order

# 5. Verify results
# - Check Spearman ≥ 0.80
# - Run adversarial diagnostic (must PASS)

# 6. (Optional) Train causal classifier
# See docs/causal_pipeline.md

# 7. Deploy
# Copy model from models/v6_refactored_finetuned/... to production
```

---

*Document maintained by the Nexustism team. Last updated: December 2025*
