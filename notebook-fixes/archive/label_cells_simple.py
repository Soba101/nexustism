#!/usr/bin/env python3
"""
Add simple text labels to each cell.
"""

import json
from pathlib import Path

# Cell descriptions in order
CELL_DESCRIPTIONS = [
    "Setup: Environment variables & package installation",
    "Imports: Core libraries (SentenceTransformers, PyTorch, sklearn)",
    "Markdown: Project Overview",
    "Config: Training parameters (LR, epochs, curriculum settings)",
    "Markdown: Data Loading Section",
    "Logging: Setup logging utilities",
    "Data Loading: Functions to load ServiceNow incidents",
    "Markdown: Data Preprocessing",
    "Data: Load incident data from JSON",
    "Markdown: Data Splitting",
    "Data Split: Split into train/eval/holdout sets",
    "Pair Loader: Load curriculum training pairs & test pairs",
    "Markdown: Pair Generation",
    "Pair Generation: TF-IDF classes (legacy, SKIPPED)",
    "Pair Generation: Generate pairs on-the-fly (legacy, SKIPPED)",
    "Markdown: Model Training",
    "Device: Auto-detect CUDA/MPS/CPU and set DEVICE variable",
    "Model Setup: LoRA initialization & loss functions",
    "Training: Main training loop with curriculum learning",
    "Markdown: Evaluation",
    "Evaluation: Score distribution diagnostic",
    "Evaluation: Cross-validation threshold function",
    "Evaluation: Comprehensive evaluation function",
    "Evaluation: Run final evaluation on all test sets",
    "Markdown: Visualization",
    "Visualization: Plot training metrics",
    "Markdown: Save Model",
    "Save: Save trained model & metadata",
    "Markdown: Borderline Test",
    "Evaluation: Borderline test (SKIPPED in curriculum mode)",
    "Markdown: Summary",
    "Summary: Print final results & next steps"
]

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"\nLabeling {len(nb['cells'])} cells...\n")

    # Ensure we have enough descriptions
    while len(CELL_DESCRIPTIONS) < len(nb['cells']):
        CELL_DESCRIPTIONS.append(f"Cell {len(CELL_DESCRIPTIONS)}")

    # Add labels
    for i, cell in enumerate(nb['cells']):
        desc = CELL_DESCRIPTIONS[i] if i < len(CELL_DESCRIPTIONS) else f"Cell {i}"

        # Add to metadata
        if 'metadata' not in cell:
            cell['metadata'] = {}
        cell['metadata']['description'] = desc

        # Print progress
        cell_type = "[MD]  " if cell['cell_type'] == 'markdown' else "[CODE]"
        print(f"  {cell_type} Cell {i:2d}: {desc}")

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup9')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing labeled notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: All cells labeled")
    print("="*70)
    print("\nCell execution order:")
    print("  1. Setup (Cell 0)")
    print("  2. Imports (Cell 1)")
    print("  3. Config (Cell 3)")
    print("  4. Load data (Cells 5-8)")
    print("  5. Split data (Cells 10-11)")
    print("  6. Load pairs (Cell 11) - CRITICAL")
    print("  7. Device detection (Cell 16) - CRITICAL")
    print("  8. Training (Cell 18) - Defines best_model")
    print("  9. Evaluation (Cells 20-23) - Uses best_model")
    print(" 10. Save (Cell 27)")
    print("\nIMPORTANT: Run cells in order! Cells 11, 16, 18 must run before evaluation.")
    print("="*70)

if __name__ == '__main__':
    main()
