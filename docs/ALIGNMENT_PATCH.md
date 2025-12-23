# Model Nomic Alignment Patch

This document outlines changes needed to align `model_promax_nomic.ipynb` and `model_promax_nomic_speed.ipynb` to use pre-generated pairs. **The only difference should be the `use_fp16` flag.**

## Changes Required for BOTH Notebooks

### 1. Replace Configuration Section (Cell 5 in both)

Replace the entire CONFIG dict with:

```python
# --- CONFIGURATION (V3 - Aligned with Pre-generated Pairs) ---
CONFIG = {
    # Model
    'model_name': 'nomic-ai/nomic-embed-text-v1.5',
    'output_dir': 'models/real_servicenow_finetuned_nomic',
    
    # Data
    'source_data': 'data_new/SNow_incident_ticket_data.csv',
    
    # Pre-generated pairs (use validated pairs)
    'use_pre_generated_pairs': True,
    'train_pairs_path': 'data_new/fixed_training_pairs.json',
    'eval_pairs_path': 'data_new/fixed_test_pairs.json',
    
    # Training hyperparameters
    'epochs': 4,
    'batch_size': 16,
    'lr': 2e-5,
    'max_seq_length': 512,
    'warmup_ratio': 0.1,
    
    # Speed optimizations
    'use_fp16': False,        # ← ONLY DIFFERENCE: True in speed version
    'dataloader_workers': 0,
    'eval_frequency': 1.0,
    
    # Data splits (legacy, only if not using pre-generated)
    'eval_split': 0.15,
    'holdout_split': 0.10,
    'min_text_length': 25,
    
    # Seed
    'seed': 42
}
```

### 2. Update Device Detection (after CONFIG dict)

Replace device detection with:

```python
# Device detection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.manual_seed_all(CONFIG['seed'])
    log(f"🚀 CUDA Detected: {torch.cuda.get_device_name(0)}")
    
    # Auto-increase batch size if GPU has enough memory
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = gpu_props.total_memory / 1024**3
    if gpu_mem_gb >= 16:
        CONFIG['batch_size'] = 32
        log(f"   GPU Memory: {gpu_mem_gb:.1f}GB - Using batch_size=32")
    else:
        log(f"   GPU Memory: {gpu_mem_gb:.1f}GB - Using batch_size=16")
    
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
    torch.mps.manual_seed(CONFIG['seed'])
    log("🍎 MPS (Apple Silicon) Detected")
    CONFIG['batch_size'] = 8
    CONFIG['use_fp16'] = False
else:
    DEVICE = 'cpu'
    log("⚠️ No GPU detected. Running on CPU.")
    CONFIG['batch_size'] = 8
    CONFIG['use_fp16'] = False

log(f"📊 Device: {DEVICE}, Batch Size: {CONFIG['batch_size']}")
log(f"\n🆕 V3 Improvements Active:")
log(f"   • Using pre-generated validated pairs: {CONFIG['use_pre_generated_pairs']}")
log(f"   • FP16 mixed precision: {CONFIG['use_fp16']}")
log(f"   • DataLoader workers: {CONFIG['dataloader_workers']}")
```

### 3. Add Pair Loading Function (after data loading imports)

Add this new function before the pair generation section:

```python
import json
from pathlib import Path

def load_pairs_from_json(pairs_path, config):
    """Load pre-generated training/eval pairs from JSON file."""
    pairs_file = Path(pairs_path)
    
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    log(f"📂 Loading pairs from: {pairs_file}")
    
    with open(pairs_file, 'r') as f:
        data = json.load(f)
    
    texts1 = data['texts1']
    texts2 = data['texts2']
    labels = data['labels']
    metadata = data.get('metadata', {})
    
    log(f"✅ Loaded {len(labels):,} pairs")
    log(f"   Positives: {sum(labels):,}")
    log(f"   Negatives: {len(labels) - sum(labels):,}")
    
    # Convert to InputExample format
    pairs = []
    for i in range(len(labels)):
        pairs.append(InputExample(
            texts=[texts1[i], texts2[i]],
            label=float(labels[i])
        ))
    
    return pairs, metadata
```

### 4. Simplify Training Loop

Replace the entire "Pair Generation" section and training code with:

```python
# ====================================================
# 6. Load Training & Evaluation Pairs
# ====================================================

if CONFIG['use_pre_generated_pairs']:
    log(f"\n{'='*80}")
    log("LOADING PRE-GENERATED PAIRS")
    log(f"{'='*80}")
    
    # Load training pairs
    train_pairs, train_metadata = load_pairs_from_json(CONFIG['train_pairs_path'], CONFIG)
    
    # Load eval pairs
    eval_pairs, eval_metadata = load_pairs_from_json(CONFIG['eval_pairs_path'], CONFIG)
    
    log(f"\n📊 Data Summary:")
    log(f"   Training pairs: {len(train_pairs):,}")
    log(f"   Eval pairs: {len(eval_pairs):,}")
    log(f"   Baseline separability: {train_metadata.get('baseline_separability', 'N/A')}")
else:
    # Legacy: Generate on the fly (optional)
    log("Generating pairs on-the-fly...")
    # [keep old pair generation code if needed]
    raise NotImplementedError("On-the-fly pair generation removed. Use pre-generated pairs.")

# ====================================================
# 7. Training
# ====================================================

from sentence_transformers import models, SentenceTransformer, losses
from torch.utils.data import DataLoader

log(f"\n{'='*80}")
log("TRAINING SENTENCE TRANSFORMER")
log(f"{'='*80}")

# Load model
model = SentenceTransformer(
    CONFIG['model_name'],
    device=DEVICE,
    trust_remote_code=True
)

# Create data loader
train_dataloader = DataLoader(
    train_pairs,
    shuffle=True,
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['dataloader_workers'],
    pin_memory=DEVICE == 'cuda'
)

# Define loss
train_loss = losses.ContrastiveLoss(model)

# Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=CONFIG['epochs'],
    warmup_steps=int(len(train_dataloader) * CONFIG['warmup_ratio']),
    optimizer_params={'lr': CONFIG['lr']},
    show_progress_bar=True,
    use_amp=CONFIG['use_fp16']  # ← FP16 flag used here
)

log(f"✅ Training complete!")

# ====================================================
# 8. Evaluation
# ====================================================

log(f"\n{'='*80}")
log("EVALUATION")
log(f"{'='*80}")

# Evaluate on eval set
eval_dataloader = DataLoader(
    eval_pairs,
    shuffle=False,
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['dataloader_workers']
)

embeddings1 = model.encode([ex.texts[0] for ex in eval_pairs], batch_size=CONFIG['batch_size'])
embeddings2 = model.encode([ex.texts[1] for ex in eval_pairs], batch_size=CONFIG['batch_size'])

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(embeddings1, embeddings2)
predictions = [scores[i,i] for i in range(len(eval_pairs))]

from sklearn.metrics import roc_auc_score, f1_score
labels = [ex.label for ex in eval_pairs]

auc = roc_auc_score(labels, predictions)
f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in predictions])

log(f"ROC-AUC: {auc:.4f}")
log(f"F1 Score: {f1:.4f}")

# Save model
model.save(CONFIG['output_dir'])
log(f"\n✅ Model saved to: {CONFIG['output_dir']}")
```

## Only Difference Between Files

### model_promax_nomic.ipynb (Accuracy/Standard)
```python
'use_fp16': False,
```

### model_promax_nomic_speed.ipynb (Speed-optimized)
```python
'use_fp16': True,
```

All other code should be identical.

---

## Summary of Benefits

✅ **Reproducible:** Uses validated training pairs  
✅ **Faster:** No on-the-fly TF-IDF computation  
✅ **Comparable:** Both notebooks use same pairs, only FP16 differs  
✅ **Simpler:** Removed curriculum learning complexity  
✅ **Quality:** +29.6% separability from validation notebook
