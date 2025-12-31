#!/usr/bin/env python3
"""
Update model_promax_mpnet_lorapeft.ipynb to use:
1. Pre-generated curriculum pairs
2. Higher learning rate (5e-5)
3. Simplified training (no on-the-fly pair generation)
"""

import json
import sys
from pathlib import Path

def update_config_cell(cell_source):
    """Update the CONFIG dictionary in the cell source"""

    # Find the latest curriculum pairs file
    data_dir = Path('data_new')
    curriculum_files = list(data_dir.glob('curriculum_training_pairs_*.json'))

    if not curriculum_files:
        print("ERROR: No curriculum_training_pairs_*.json found in data_new/")
        print("Please run fix_train_test_mismatch.ipynb first!")
        sys.exit(1)

    # Get the most recent file
    latest_curriculum = sorted(curriculum_files)[-1]
    print(f"Using curriculum dataset: {latest_curriculum.name}")

    # New CONFIG with pre-generated pairs
    new_config = f'''import random
import numpy as np
import torch
import logging

# Simple logging
def log(msg, level=logging.INFO):
    print(msg)

# --- CONFIGURATION (V3 - Curriculum Learning with Pre-generated Pairs) ---
CONFIG = {{
    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',

    # Data - USING PRE-GENERATED CURRICULUM PAIRS
    'use_pre_generated_pairs': True,  # NEW: Use pre-generated pairs
    'train_pairs_path': 'data_new/{latest_curriculum.name}',  # NEW: Curriculum dataset
    'source_data': 'data_new\\\\SNow_incident_ticket_data.csv',  # Fallback (not used)

    # LoRA/PEFT Configuration
    'use_lora': True,
    'lora_r': 16,              # Rank (8-32, higher = more capacity)
    'lora_alpha': 32,          # Scaling factor (typically 2*r)
    'lora_dropout': 0.1,       # Dropout for LoRA layers
    'lora_target_modules': ['query', 'key', 'value'],

    # Loss Function
    'use_multi_loss': False,   # Use single loss for simplicity
    'loss_type': 'cosine',     # 'cosine' or 'mnrl'

    # Training hyperparameters (UPDATED FOR CURRICULUM)
    'epochs': 6,               # 6 total epochs (2 per curriculum phase)
    'batch_size': 16,          # Will auto-reduce for MPS/CPU if needed
    'lr': 5e-5,                # INCREASED from 2e-5 (LoRA needs higher LR)
    'max_seq_length': 256,     # REDUCED from 384 (match baseline)
    'warmup_ratio': 0.1,

    # Curriculum Learning (NEW - Using phase_indicators from data)
    'use_curriculum': True,    # Train in 3 phases
    'epochs_per_phase': 2,     # 2 epochs per phase

    # Data splits (only used if NOT using pre-generated)
    'eval_split': 0.15,
    'holdout_split': 0.10,
    'min_text_length': 25,

    # Pair generation (LEGACY - not used with pre-generated pairs)
    'num_pairs': 50000,        # Not used
    'pos_ratio': 0.30,         # Not used

    # Reproducibility
    'seed': 42
}}

# Set seeds
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

log("="*70)
log("CONFIGURATION (V3 - Curriculum Learning)")
log("="*70)
log(f"Model: {{CONFIG['model_name']}}")
log(f"Output: {{CONFIG['output_dir']}}")
log(f"\\nData:")
log(f"  Using pre-generated pairs: {{CONFIG['use_pre_generated_pairs']}}")
log(f"  Pairs file: {{CONFIG['train_pairs_path']}}")
log(f"\\nLoRA Config:")
log(f"  Rank: {{CONFIG['lora_r']}}")
log(f"  Alpha: {{CONFIG['lora_alpha']}}")
log(f"  Dropout: {{CONFIG['lora_dropout']}}")
log(f"\\nTraining:")
log(f"  Total epochs: {{CONFIG['epochs']}}")
log(f"  Learning rate: {{CONFIG['lr']}} (INCREASED for LoRA)")
log(f"  Batch size: {{CONFIG['batch_size']}}")
log(f"  Max seq length: {{CONFIG['max_seq_length']}} (REDUCED to match baseline)")
log(f"\\nCurriculum:")
log(f"  Use curriculum: {{CONFIG['use_curriculum']}}")
log(f"  Epochs per phase: {{CONFIG['epochs_per_phase']}}")
log("="*70)
'''

    return new_config

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        sys.exit(1)

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find and update CONFIG cell (usually cell 5)
    config_cell_found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'CONFIG = {' in source and 'model_name' in source:
                print(f"Found CONFIG in cell {i}, updating...")
                new_config = update_config_cell(source)
                cell['source'] = new_config.split('\n')
                config_cell_found = True
                print("OK - Updated CONFIG cell")
                break

    if not config_cell_found:
        print("ERROR: Could not find CONFIG cell!")
        sys.exit(1)

    # Save updated notebook
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"\\nBacking up original to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    print(f"Writing updated notebook to: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\\n" + "="*70)
    print("SUCCESS: NOTEBOOK UPDATED")
    print("="*70)
    print("\\nChanges made:")
    print("  1. Learning rate: 2e-5 -> 5e-5 (LoRA needs higher LR)")
    print("  2. Max seq length: 384 -> 256 (match baseline)")
    print("  3. Using pre-generated curriculum pairs")
    print("  4. Curriculum learning enabled (3 phases, 2 epochs each)")
    print("\\nNext steps:")
    print("  1. Open model_promax_mpnet_lorapeft.ipynb in Jupyter")
    print("  2. Run all cells to train with new config")
    print("  3. Evaluate with evaluate_model.ipynb")
    print("\\nExpected results:")
    print("  - Better test performance (trained on hard pairs)")
    print("  - Likely to beat baseline (0.5038 Spearman)")
    print("="*70)

if __name__ == '__main__':
    main()
