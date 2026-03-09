#!/usr/bin/env python3
"""
Increase LoRA rank for more model capacity.

Current: rank=16, alpha=32 (1.35% params trainable)
New: rank=32, alpha=64 (2.7% params trainable)

This gives the model more capacity to learn the hard test distribution.
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

    # Increase LoRA rank
    source3 = source3.replace("'lora_r': 16,  # Rank", "'lora_r': 32,  # Rank (2x higher capacity)")
    source3 = source3.replace("'lora_alpha': 32,  # Scaling factor", "'lora_alpha': 64,  # Scaling factor (2x)")
    source3 = source3.replace("'lora_dropout': 0.1,", "'lora_dropout': 0.05,  # Lower dropout for higher rank")

    cell3['source'] = [source3]
    nb['cells'][3] = cell3

    # Save
    print(f"\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Increased LoRA rank")
    print("="*70)
    print("\nChanges:")
    print("  - LoRA rank: 16 -> 32 (2x capacity)")
    print("  - LoRA alpha: 32 -> 64 (matched scaling)")
    print("  - LoRA dropout: 0.1 -> 0.05 (less regularization)")
    print("\nTrainable params: ~3M (2.7% of total, was 1.35%)")
    print("\nBenefit: More capacity to learn complex hard negatives")
    print("Cost: Slightly slower training (~10% slower)")
    print("="*70)

if __name__ == '__main__':
    main()
