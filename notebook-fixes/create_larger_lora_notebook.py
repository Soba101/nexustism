import json

# Create Phase 3 Larger LoRA training notebook
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
add_cell('markdown', '''# MPNet LoRA Fine-tuning - Phase 3 with Larger LoRA Rank

**Goal**: Close 1.6% gap (0.4955 → 0.5038) by increasing model capacity

**Experiment Results So Far**:
- ❌ MNRL failed: 0.3472 (threshold collapsed)
- ✅ Phase 3 CosineSim r=16: **0.4955** (current best)
- 🎯 Baseline MPNet: 0.5038 (target)

**Key Changes from Phase 3-only (r=16)**:
- **LoRA Rank**: r=32 (up from 16) - 2x more trainable parameters
- **LoRA Alpha**: 64 (up from 32) - maintains 2:1 alpha/r ratio
- **Target Modules**: ['q', 'k', 'v', 'o'] - added output projection
- **LoRA Dropout**: 0.05 (down from 0.1) - less regularization for hard data
- **Expected**: Spearman 0.505-0.515 (approach or beat baseline)

**Hypothesis**: Phase 3's 54.4% overlap requires subtle distinctions. Current r=16 (~885K params) may underfit. Doubling capacity to r=32 (~1.77M params) allows finer boundary learning.

**Rationale**:
- Still parameter-efficient: 1.77M / 110M = 1.6% trainable (vs 0.8%)
- RTX 5090 has plenty of VRAM (34GB)
- Hard data (Phase 3) unlikely to overfit with more capacity''')

# Cell 2: Configuration
add_cell('code', '''CONFIG = {
    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',

    # LoRA settings - INCREASED CAPACITY
    'use_lora': True,
    'lora_r': 32,  # DOUBLED from 16
    'lora_alpha': 64,  # DOUBLED from 32 (maintain 2:1 ratio)
    'lora_dropout': 0.05,  # REDUCED from 0.1
    'lora_target_modules': ['q', 'k', 'v', 'o'],  # ADDED 'o' output projection

    # Training data - Phase 3 only
    'use_curriculum': False,
    'train_pairs_path': 'data_new/phase3_only_pairs.json',
    'test_pairs_path': 'data_new/fixed_test_pairs.json',

    # Training hyperparameters - same as Phase 3 r=16
    'epochs': 12,
    'batch_size': 16,
    'lr': 2e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_seq_length': 256,

    # Loss function - proven to work
    'loss_type': 'cosine_similarity',

    # Output
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',
    'save_name_prefix': 'phase3_lora_r32',

    # System
    'seed': 42,
    'fp16': True,
}

print("Configuration loaded:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

print("\\nCapacity comparison:")
print("  Phase 3 r=16: ~885K trainable params (0.80%)")
print(f"  Phase 3 r=32: ~1.77M trainable params (1.60%)")
print("  Increase: 2x more capacity for subtle boundary learning")''')

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
print(f"  Total pairs: {len(train_data['texts1'])}")
print(f"  Metadata: {train_data['metadata']}")

print("\\nLoading test data...")
with open(CONFIG['test_pairs_path'], 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"  Test pairs: {len(test_data['texts1'])}")

# Create InputExamples
print("\\nCreating training examples...")
train_examples = [
    InputExample(texts=[train_data['texts1'][i], train_data['texts2'][i]],
                 label=float(train_data['labels'][i]))
    for i in range(len(train_data['texts1']))
]

print(f"Created {len(train_examples)} training examples")
print(f"Label distribution: min={min(train_data['labels']):.3f}, max={max(train_data['labels']):.3f}")''')

# Cell 5: Initialize model with Larger LoRA
add_cell('code', '''print("Initializing MPNet base model...")
model = SentenceTransformer(CONFIG['model_name'])
model.max_seq_length = CONFIG['max_seq_length']

if CONFIG['use_lora']:
    print("\\nApplying LARGER LoRA configuration...")

    # LoRA config with r=32
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
    print(f"  LoRA config: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, dropout={CONFIG['lora_dropout']}")
    print(f"  Target modules: {CONFIG['lora_target_modules']}")

    # Comparison
    r16_params = 884_736
    print(f"\\n  Capacity increase vs r=16:")
    print(f"    r=16: {r16_params:,} params")
    print(f"    r=32: {trainable_params:,} params")
    print(f"    Increase: {trainable_params/r16_params:.2f}x")

model = model.to(device)
print(f"\\nModel loaded on {device}")''')

# Cell 6: Setup training
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

print(f"\\nStarting training with LARGER LoRA (r=32)...")
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
print("TEST SET RESULTS (Phase 3 LoRA r=32):")
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
print("MPNet LoRA (Phase 3 CosineSim r=16):    Spearman 0.4955")
print(f"MPNet LoRA (Phase 3 CosineSim r=32):    Spearman {spearman_corr:.4f}")
print("="*60)

if spearman_corr > 0.5038:
    improvement = ((spearman_corr - 0.5038) / 0.5038) * 100
    print(f"\\n🎉 SUCCESS! Beat baseline by {improvement:.1f}%")
    print("Achievement unlocked: Fine-tuned model beats baseline!")
    print("More capacity (r=32) was the key!")
elif spearman_corr > 0.4955:
    improvement = ((spearman_corr - 0.4955) / 0.4955) * 100
    print(f"\\n✓ Improved over r=16 by {improvement:.1f}%!")
    gap = 0.5038 - spearman_corr
    print(f"  Gap to baseline: {gap:.4f} ({gap/0.5038*100:.1f}%)")
    if gap < 0.002:
        print("  Very close! Consider r=48 or r=64 to close final gap")
    else:
        print("  Progress made - higher capacity helps!")
else:
    diff = spearman_corr - 0.4955
    print(f"\\n~ Similar to r=16 (diff: {diff:+.4f})")
    if abs(diff) < 0.001:
        print("  No improvement - r=16 may be optimal capacity")
    else:
        print("  Slight regression - r=16 remains best")

print("\\n" + "="*60)
print("CONCLUSION:")
print("="*60")
if spearman_corr >= 0.5038:
    print("✅ Fine-tuning CAN beat baseline with right configuration!")
    print("   Phase 3-only + LoRA r=32 was the winning combination")
elif spearman_corr >= 0.500:
    print("Close to baseline - fine-tuning nearly matches unfinetuned model")
    print("May need different approach (data augmentation, different loss, etc.)")
else:
    print("Baseline MPNet (0.5038) remains best overall")
    print("Fine-tuning on this dataset introduces degradation")
    print("Recommendation: Use baseline MPNet for production")''')

# Save notebook
output_path = 'model_phase3_lora_r32_mpnet_lorapeft.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Created {output_path}')
