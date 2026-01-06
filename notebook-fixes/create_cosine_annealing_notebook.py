import json

# Create Phase 3 Cosine Annealing training notebook
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
add_cell('markdown', '''# MPNet LoRA Fine-tuning - Phase 3 with Cosine Annealing

**Goal**: Close 1.6% gap (0.4955 → 0.5038) using advanced learning rate scheduling

**Experiment Results So Far**:
- ❌ MNRL failed: 0.3472 (threshold collapsed to 0.97)
- ✅ Phase 3 CosineSim: 0.4955 (current best fine-tuned)
- 🎯 Baseline MPNet: 0.5038 (target)

**Key Changes from Phase 3-only**:
- **Learning Rate Schedule**: Cosine annealing with warm restarts (2 cycles)
- **Higher Peak LR**: 3e-5 (up from 2e-5) - more aggressive exploration
- **Minimum LR**: 5e-6 - still learns during cycle valleys
- **Warm Restarts**: Escape local minima, explore different solutions
- **Expected**: Spearman 0.503-0.515 (approach or beat baseline)

**Hypothesis**: Phase 3's 54.4% overlap creates rugged loss landscape with local minima. Cosine annealing helps escape these traps.''')

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

    # Training data - Phase 3 with all pairs
    'use_curriculum': False,
    'train_pairs_path': 'data_new/phase3_only_pairs.json',
    'test_pairs_path': 'data_new/fixed_test_pairs.json',

    # Cosine Annealing hyperparameters
    'epochs': 14,  # Slight increase from 12
    'batch_size': 16,
    'lr': 3e-5,  # HIGHER than Phase 3-only (was 2e-5)
    'min_lr': 5e-6,  # Minimum LR for annealing
    'warmup_ratio': 0.15,  # Increased from 0.1
    'weight_decay': 0.01,
    'max_seq_length': 256,

    # Cosine Annealing with Warm Restarts
    'use_cosine_annealing': True,
    'num_cycles': 2,  # 2 complete cycles over 14 epochs
    'T_0': 7,  # First cycle: 7 epochs
    'T_mult': 1,  # Second cycle: 7 epochs

    # Loss function
    'loss_type': 'cosine_similarity',  # Keep what works

    # Output
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',
    'save_name_prefix': 'phase3_cosine_annealing',

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
print(f"  Total pairs: {len(train_data['texts1'])}")
print(f"  Metadata: {train_data['metadata']}")

print("\\nLoading test data...")
with open(CONFIG['test_pairs_path'], 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"  Test pairs: {len(test_data['texts1'])}")

# Create InputExamples with labels for CosineSimilarity
print("\\nCreating training examples...")
train_examples = [
    InputExample(texts=[train_data['texts1'][i], train_data['texts2'][i]],
                 label=float(train_data['labels'][i]))
    for i in range(len(train_data['texts1']))
]

print(f"Created {len(train_examples)} training examples")
print(f"Label distribution: min={min(train_data['labels']):.3f}, max={max(train_data['labels']):.3f}")''')

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

# Cell 6: Setup training with Cosine Annealing
add_cell('code', '''# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONFIG['batch_size'], num_workers=0)

# Loss function - CosineSimilarityLoss (proven to work)
print(f"\\nUsing loss function: {CONFIG['loss_type']}")
train_loss = losses.CosineSimilarityLoss(model)

# Training steps
num_train_steps = len(train_dataloader) * CONFIG['epochs']
warmup_steps = int(num_train_steps * CONFIG['warmup_ratio'])

print(f"\\nTraining configuration:")
print(f"  Total steps: {num_train_steps:,}")
print(f"  Warmup steps: {warmup_steps:,}")
print(f"  Batches per epoch: {len(train_dataloader)}")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Peak learning rate: {CONFIG['lr']}")
print(f"  Min learning rate: {CONFIG['min_lr']}")
if CONFIG['use_cosine_annealing']:
    print(f"\\nCosine Annealing with Warm Restarts:")
    print(f"  Number of cycles: {CONFIG['num_cycles']}")
    print(f"  T_0 (first cycle length): {CONFIG['T_0']} epochs")
    print(f"  T_mult (cycle multiplier): {CONFIG['T_mult']}")
    steps_per_epoch = len(train_dataloader)
    print(f"  Steps per cycle: {CONFIG['T_0'] * steps_per_epoch}")''')

# Cell 7: Custom training loop with Cosine Annealing
add_cell('code', '''from torch.optim import AdamW
import math

# Clear GPU cache
if device == 'cuda':
    torch.cuda.empty_cache()
    gc.collect()

# Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_path = Path(CONFIG['output_dir']) / f"{CONFIG['save_name_prefix']}_{timestamp}"
save_path.parent.mkdir(parents=True, exist_ok=True)
save_path.mkdir(exist_ok=True)

print(f"\\nStarting training with Cosine Annealing...")
print(f"Output path: {save_path}")
print("="*60)

if CONFIG['use_cosine_annealing']:
    # Custom training loop for cosine annealing
    # Get trainable parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': CONFIG['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG['lr'])

    # Cosine annealing scheduler
    steps_per_epoch = len(train_dataloader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG['T_0'] * steps_per_epoch,
        T_mult=CONFIG['T_mult'],
        eta_min=CONFIG['min_lr']
    )

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(CONFIG['epochs']):
        print(f"\\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Forward pass
            loss_value = train_loss(batch, None)

            # Backward pass
            loss_value.backward()

            # Optimizer step
            optimizer.step()
            scheduler.step()  # Update LR every step
            optimizer.zero_grad()

            epoch_loss += loss_value.item()
            global_step += 1

            # Log every 50 steps
            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}, final lr: {current_lr:.2e}")

    # Save model
    model.save(str(save_path))
    print(f"\\nModel saved to: {save_path}")

else:
    # Fallback to standard training
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

# Reload model for evaluation
model = SentenceTransformer(str(save_path))
model = model.to(device)

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
print("TEST SET RESULTS (Phase 3 Cosine Annealing):")
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

# Comprehensive comparison
print("\\n" + "="*60)
print("COMPREHENSIVE COMPARISON:")
print("="*60)
print("Baseline MPNet (no fine-tuning):        Spearman 0.5038 <-- TARGET")
print("MPNet LoRA (curriculum 3-phase):        Spearman 0.2906-0.3818")
print("MPNet LoRA (Phase 3 MNRL):              Spearman 0.3472")
print("MPNet LoRA (Phase 2+3 CosineSim):       Spearman 0.4853")
print("MPNet LoRA (Phase 3 CosineSim):         Spearman 0.4955")
print(f"MPNet LoRA (Phase 3 Cosine Annealing): Spearman {spearman_corr:.4f}")
print("="*60)

if spearman_corr > 0.5038:
    improvement = ((spearman_corr - 0.5038) / 0.5038) * 100
    print(f"\\n🎉 SUCCESS! Beat baseline by {improvement:.1f}%")
    print("Achievement unlocked: Fine-tuned model beats baseline!")
elif spearman_corr > 0.4955:
    improvement = ((spearman_corr - 0.4955) / 0.4955) * 100
    print(f"\\n✓ Improved over Phase 3 CosineSim by {improvement:.1f}%")
    gap = 0.5038 - spearman_corr
    print(f"  Gap to baseline: {gap:.4f} ({gap/0.5038*100:.1f}%)")
elif spearman_corr > 0.4853:
    print(f"\\n+ Better than Phase 2+3, but not Phase 3 CosineSim")
    print("  Phase 3 CosineSim (0.4955) remains best fine-tuned model")
else:
    print("\\n⚠ Did not improve - Phase 3 CosineSim (0.4955) remains best")
    print("\\nConclusion: Baseline MPNet (0.5038) may be optimal for this dataset")
    print("Fine-tuning introduces overfitting or conflicting objectives")''')

# Save notebook
output_path = 'model_phase3_cosine_annealing_mpnet_lorapeft.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Created {output_path}')
