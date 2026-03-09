import json

# Create Phase 3 MNRL training notebook
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_cell(cell_type, source, metadata=None):
    cell = {
        'cell_type': cell_type,
        'source': source if isinstance(source, list) else [source],
        'metadata': metadata or {}
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    notebook['cells'].append(cell)

# Cell 1: Title
add_cell('markdown', '''# MPNet LoRA Fine-tuning - Phase 3 with MNRL Loss

**Goal**: Close 1.6% gap (0.4955 → 0.5038) by fixing task/loss mismatch

**Key Changes from Phase 3-only (CosineSimilarity)**:
- **Loss Function**: MultipleNegativesRankingLoss (MNRL) - optimized for ranking tasks
- **Training Data**: Positive pairs only (~2,500 pairs) - MNRL uses in-batch negatives
- **Batch Size**: 32 (up from 16) - enables 1,024 comparisons per batch vs 32
- **Epochs**: 16 (up from 12) - compensate for 50% data reduction
- **Expected**: Spearman 0.515-0.525 (beat baseline 0.5038 by 2-4%)

**Hypothesis**: CosineSimilarityLoss is a regression loss, but this is a RANKING task (find most similar incidents). MNRL directly optimizes for ranking.''')

# Cell 2: Configuration
add_cell('code', '''CONFIG = {
    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',

    # LoRA settings
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_target_modules': ['q', 'k', 'v'],

    # Training data - Phase 3 positives ONLY for MNRL
    'use_curriculum': False,
    'train_pairs_path': 'data_new/phase3_only_pairs.json',
    'filter_positives_only': True,  # NEW: MNRL needs positive pairs only
    'test_pairs_path': 'data_new/fixed_test_pairs.json',

    # MNRL-optimized hyperparameters
    'epochs': 16,  # Increase from 12 (compensate for 50% data reduction)
    'batch_size': 32,  # Increase from 16 (MNRL needs larger batches)
    'lr': 2e-5,  # Keep same as Phase 3-only
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_seq_length': 256,

    # Loss function
    'loss_type': 'mnrl',  # CHANGE from 'cosine_similarity'

    # Output
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',
    'save_name_prefix': 'phase3_mnrl',

    # System
    'seed': 42,
    'fp16': True,
}

print("Configuration loaded:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")''')

# Cell 3: Imports
add_cell('code', '''import os
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import gc

# Set seed
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Device setup
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Using Apple MPS")
else:
    device = 'cpu'
    print("Using CPU")

CONFIG['device'] = device''')

# Cell 4: Load data
add_cell('code', '''print("Loading Phase 3 training data...")
with open(CONFIG['train_pairs_path'], 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"\\nPhase 3 Data Statistics:")
print(f"  Total pairs in file: {len(train_data['texts1'])}")
print(f"  Metadata: {train_data['metadata']}")

print("\\nLoading test data...")
with open(CONFIG['test_pairs_path'], 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"  Test pairs: {len(test_data['texts1'])}")

# Create InputExamples - MNRL requires positive pairs only (no label parameter)
print("\\nFiltering for positive pairs (MNRL requirement)...")
if CONFIG['filter_positives_only']:
    train_examples = [
        InputExample(texts=[train_data['texts1'][i], train_data['texts2'][i]])
        for i in range(len(train_data['texts1']))
        if train_data['labels'][i] == 1.0  # Only positives
    ]
    print(f"  Filtered to {len(train_examples)} positive pairs (from {len(train_data['texts1'])} total)")
else:
    train_examples = [
        InputExample(texts=[train_data['texts1'][i], train_data['texts2'][i]],
                     label=float(train_data['labels'][i]))
        for i in range(len(train_data['texts1']))
    ]
    print(f"  Using all {len(train_examples)} pairs with labels")

print(f"\\nCreated {len(train_examples)} training examples for MNRL")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  In-batch negatives per anchor: {CONFIG['batch_size'] - 1}")
print(f"  Comparisons per batch: {CONFIG['batch_size']} * {CONFIG['batch_size']} = {CONFIG['batch_size']**2}")''')

# Cell 5: Initialize model with LoRA
add_cell('code', '''print("Initializing MPNet base model...")
model = SentenceTransformer(CONFIG['model_name'])
model.max_seq_length = CONFIG['max_seq_length']

if CONFIG['use_lora']:
    print("\\nApplying LoRA configuration...")

    # LoRA config
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['lora_target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Apply LoRA to transformer
    base_model = model[0].auto_model
    model[0].auto_model = get_peft_model(base_model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model[0].auto_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model[0].auto_model.parameters())

    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  LoRA config: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")

model = model.to(device)
print(f"\\nModel loaded on {device}")''')

# Cell 6: Setup training
add_cell('code', '''# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONFIG['batch_size'], num_workers=0)

# Loss function - MultipleNegativesRankingLoss for ranking task
print(f"\\nUsing loss function: {CONFIG['loss_type']}")
if CONFIG['loss_type'] == 'mnrl':
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print("  MNRL: Each anchor compares against all other samples in batch as negatives")
    print(f"  Effective learning signal: {CONFIG['batch_size']**2} comparisons per batch")
else:
    # Fallback to CosineSimilarity
    train_loss = losses.CosineSimilarityLoss(model)
    print("  CosineSimilarity: Direct regression on similarity scores")

# Training steps
num_train_steps = len(train_dataloader) * CONFIG['epochs']
warmup_steps = int(num_train_steps * CONFIG['warmup_ratio'])

print(f"\\nTraining configuration:")
print(f"  Total steps: {num_train_steps:,}")
print(f"  Warmup steps: {warmup_steps:,}")
print(f"  Batches per epoch: {len(train_dataloader)}")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['lr']}")''')

# Cell 7: Train
add_cell('code', '''# Clear GPU cache
if device == 'cuda':
    torch.cuda.empty_cache()
    gc.collect()

# Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_path = Path(CONFIG['output_dir']) / f"{CONFIG['save_name_prefix']}_{timestamp}"
save_path.parent.mkdir(parents=True, exist_ok=True)
save_path.mkdir(exist_ok=True)

print(f"\\nStarting training...")
print(f"Output path: {save_path}")
print("="*60)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=CONFIG['epochs'],
    warmup_steps=warmup_steps,
    optimizer_params={'lr': CONFIG['lr']},
    weight_decay=CONFIG['weight_decay'],
    output_path=str(save_path),
    save_best_model=True,
    show_progress_bar=True,
    use_amp=CONFIG['fp16'] and device == 'cuda'
)

print("\\n" + "="*60)
print(f"Training complete! Model saved to: {save_path}")''')

# Cell 8: Evaluate
add_cell('code', '''from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import spearmanr

print("\\nEvaluating trained model on test set...")

# Encode test pairs
test_emb1 = model.encode(test_data['texts1'], batch_size=32, device=device, show_progress_bar=True)
test_emb2 = model.encode(test_data['texts2'], batch_size=32, device=device, show_progress_bar=True)

# Compute similarities
test_similarities = np.array([
    np.dot(test_emb1[i], test_emb2[i]) / (np.linalg.norm(test_emb1[i]) * np.linalg.norm(test_emb2[i]))
    for i in range(len(test_emb1))
])

# Metrics
test_labels = np.array(test_data['labels'])
spearman_corr, _ = spearmanr(test_similarities, test_labels)
roc_auc = roc_auc_score(test_labels, test_similarities)

# Find best threshold
thresholds = np.linspace(0, 1, 101)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    preds = (test_similarities >= thresh).astype(int)
    f1 = f1_score(test_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Final metrics with best threshold
final_preds = (test_similarities >= best_thresh).astype(int)
accuracy = accuracy_score(test_labels, final_preds)

print("\\n" + "="*60)
print("TEST SET RESULTS (Phase 3 MNRL Model):")
print("="*60)
print(f"  Spearman Correlation: {spearman_corr:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Best Threshold: {best_thresh:.4f}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("="*60)

# Save metrics
metrics = {
    'spearman': float(spearman_corr),
    'roc_auc': float(roc_auc),
    'best_threshold': float(best_thresh),
    'f1': float(best_f1),
    'accuracy': float(accuracy),
    'config': CONFIG,
    'timestamp': timestamp
}

with open(save_path / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\\nMetrics saved to: {save_path / 'metrics.json'}")

# Comparison
print("\\n" + "="*60)
print("COMPARISON:")
print("="*60)
print("Baseline MPNet (no fine-tuning):    Spearman 0.5038 <-- TARGET")
print("MPNet LoRA (curriculum 3-phase):    Spearman 0.2906-0.3818")
print("MPNet LoRA (Phase 3 CosineSim):     Spearman 0.4955")
print("MPNet LoRA (Phase 2+3 CosineSim):   Spearman 0.4853")
print(f"MPNet LoRA (Phase 3 MNRL):          Spearman {spearman_corr:.4f}")
print("="*60)

if spearman_corr > 0.5038:
    improvement = ((spearman_corr - 0.5038) / 0.5038) * 100
    print(f"\\n🎉 SUCCESS! Beat baseline by {improvement:.1f}%")
    print("Next step: Try Experiment 2 (larger LoRA r=32) to push even higher")
elif spearman_corr > 0.500:
    gap = 0.5038 - spearman_corr
    print(f"\\n✓ Close! Only {gap:.4f} away from baseline")
    print("Next step: Try Experiment 2 (larger LoRA r=32) to close the gap")
elif spearman_corr > 0.4955:
    improvement = ((spearman_corr - 0.4955) / 0.4955) * 100
    print(f"\\n+ Improved over Phase 3 CosineSim by {improvement:.1f}%")
    print("Next step: Try Experiment 2 (larger LoRA r=32)")
else:
    print("\\n⚠ MNRL did not improve over CosineSimilarity")
    print("Next step: Try Experiment 3 (Cosine Annealing with warm restarts)")''')

# Save notebook
output_path = 'model_phase3_mnrl_mpnet_lorapeft.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Created {output_path}')
