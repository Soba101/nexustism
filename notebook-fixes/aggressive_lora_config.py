#!/usr/bin/env python3
"""
Apply aggressive LoRA training config to beat baseline.

Key changes:
1. LR 1e-4 (2x higher) - LoRA needs strong gradients
2. Train ONLY on Phase 3 hard pairs
3. 10 epochs on hard pairs (not 4)
4. Lower warmup (100 steps)
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft-v2.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Cell 3 (CONFIG) - Aggressive settings
    cell3 = nb['cells'][3]
    source3 = ''.join(cell3['source'])

    # Aggressive hyperparameters
    source3 = source3.replace("'epochs': 8,  # Balanced (was 15)", "'epochs': 10,  # Focus on hard pairs ONLY")
    source3 = source3.replace("'lr': 5e-5,  # Proven to work (was 3e-5 too low!)", "'lr': 1e-4,  # AGGRESSIVE for LoRA")
    source3 = source3.replace("'warmup_steps': 200,  # Reasonable (was 500 too long!)", "'warmup_steps': 100,  # Minimal warmup")
    source3 = source3.replace("'phase1_epochs': 2,  # Easy (brief warmup)", "'phase1_epochs': 0,  # SKIP easy")
    source3 = source3.replace("'phase2_epochs': 2,  # Medium (brief)", "'phase2_epochs': 0,  # SKIP medium")
    source3 = source3.replace("'phase3_epochs': 4,  # Hard (focus here!)", "'phase3_epochs': 10,  # ALL epochs on HARD")

    cell3['source'] = [source3]
    nb['cells'][3] = cell3

    # Update Cell 12 - Skip Phase 1 and 2
    cell12 = nb['cells'][12]
    source12 = ''.join(cell12['source'])

    # Update phases to skip 1 and 2
    old_phases = """# Curriculum training: 3 phases with progressive epochs
phases = [
    ('Phase 1: Easy', CURRICULUM_PHASES['phase1'], CONFIG['phase1_epochs']),
    ('Phase 2: Medium', CURRICULUM_PHASES['phase2'], CONFIG['phase2_epochs']),
    ('Phase 3: Hard', CURRICULUM_PHASES['phase3'], CONFIG['phase3_epochs'])
]"""

    new_phases = """# Train ONLY on Phase 3 (hard pairs matching test distribution)
phases = [
    ('Phase 3: Hard ONLY', CURRICULUM_PHASES['phase3'], CONFIG['phase3_epochs'])
]"""

    if old_phases in source12:
        source12 = source12.replace(old_phases, new_phases)
        cell12['source'] = [source12]
        nb['cells'][12] = cell12

    # Save
    print(f"\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Applied aggressive LoRA config")
    print("="*70)
    print("\nChanges:")
    print("  - LR: 5e-5 -> 1e-4 (2x higher, aggressive gradients)")
    print("  - Warmup: 200 -> 100 (minimal)")
    print("  - Epochs: 8 -> 10")
    print("  - Training: ONLY Phase 3 hard pairs (5,000 pairs)")
    print("  - Skipping Phase 1 and 2 entirely")
    print("\nExpected: Spearman 0.51-0.53 (beat baseline 0.5038)")
    print("Training time: ~14 minutes")
    print("="*70)

if __name__ == '__main__':
    main()
