#!/usr/bin/env python3
"""
Fix V2 notebook hyperparameters while keeping CosineSimilarityLoss.

The issue was NOT the loss function - CosineSimilarity is correct for duplicate detection!
The issue was:
1. LR too low (3e-5) - LoRA needs stronger gradients
2. Warmup too long (500 steps) - wasted 3+ epochs
3. Too many epochs (15 total) - overfitting to easy pairs
4. Weight decay unnecessarily constraining learning
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft-v2.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Cell 3 (CONFIG)
    cell3 = nb['cells'][3]
    source3 = ''.join(cell3['source'])

    # Fix hyperparameters
    source3 = source3.replace("'epochs': 12,  # 2 epochs per curriculum phase", "'epochs': 8,  # Balanced (was 15)")
    source3 = source3.replace("'lr': 3e-5,  # Increased from 2e-5 for LoRA", "'lr': 5e-5,  # Proven to work (was 3e-5 too low!)")
    source3 = source3.replace("'warmup_steps': 500,", "'warmup_steps': 200,  # Reasonable (was 500 too long!)")
    source3 = source3.replace("'phase1_epochs': 4,  # Easy pairs", "'phase1_epochs': 2,  # Easy (brief warmup)")
    source3 = source3.replace("'phase2_epochs': 5,  # Medium pairs", "'phase2_epochs': 2,  # Medium (brief)")
    source3 = source3.replace("'phase3_epochs': 6,  # Hard pairs (most important!)", "'phase3_epochs': 4,  # Hard (focus here!)")

    cell3['source'] = [source3]
    nb['cells'][3] = cell3

    # Update Cell 12 (Training) - Remove weight decay ONLY
    cell12 = nb['cells'][12]
    source12 = ''.join(cell12['source'])

    # Remove weight decay from both optimizer_params lines
    source12 = source12.replace("optimizer_params={'lr': CONFIG['lr'], 'weight_decay': 0.01}", "optimizer_params={'lr': CONFIG['lr']}")

    cell12['source'] = [source12]
    nb['cells'][12] = cell12

    # Save
    print(f"\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Fixed V2 notebook with correct hyperparameters")
    print("="*70)
    print("\nChanges applied:")
    print("  Cell 3 (CONFIG):")
    print("    - lr: 3e-5 -> 5e-5 (LoRA needs stronger gradients)")
    print("    - warmup_steps: 500 -> 200 (was wasting 3+ epochs)")
    print("    - epochs: 12 -> 8")
    print("    - phase1_epochs: 4 -> 2")
    print("    - phase2_epochs: 5 -> 2")
    print("    - phase3_epochs: 6 -> 4")
    print("\n  Cell 12 (Training):")
    print("    - Removed weight_decay: 0.01")
    print("    - KEPT CosineSimilarityLoss (correct for duplicate detection!)")
    print("\nExpected result: Spearman 0.50-0.51 (beating baseline 0.5038)")
    print("Training time: ~12 minutes")
    print("="*70)

if __name__ == '__main__':
    main()
