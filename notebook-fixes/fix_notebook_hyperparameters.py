#!/usr/bin/env python3
"""
Fix hyperparameters in model_promax_mpnet_lorapeft.ipynb

Fixes:
1. Line 231: Update train_pairs_path to curriculum_training_pairs_complete.json
2. Line 246: Increase epochs from 6 to 12
3. Line 1597: Change optimizer_params to use CONFIG['lr']
4. Line 1619: Change optimizer_params to use CONFIG['lr']
"""

import json
import shutil
from datetime import datetime

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft.ipynb'

def main():
    # Backup original
    backup_path = f'{NOTEBOOK_PATH}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy(NOTEBOOK_PATH, backup_path)
    print(f"Created backup: {backup_path}")

    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes_made = []

    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue

        source = cell['source']
        modified = False

        # Fix 1: Update train_pairs_path
        for i, line in enumerate(source):
            if "'train_pairs_path': 'data_new/curriculum_training_pairs_20251224_065436.json'" in line:
                source[i] = line.replace(
                    "'train_pairs_path': 'data_new/curriculum_training_pairs_20251224_065436.json'",
                    "'train_pairs_path': 'data_new/curriculum_training_pairs_complete.json'"
                )
                changes_made.append(f"Line {i}: Updated train_pairs_path to curriculum_training_pairs_complete.json")
                modified = True

        # Fix 2: Increase epochs to 12
        for i, line in enumerate(source):
            if "'epochs': 6," in line and "# 6 total epochs" in line:
                source[i] = line.replace(
                    "'epochs': 6,               # 6 total epochs (2 per curriculum phase)",
                    "'epochs': 12,              # 12 total epochs (4 per curriculum phase)"
                )
                changes_made.append(f"Line {i}: Increased epochs from 6 to 12")
                modified = True

        # Fix 3 & 4: Change optimizer_params to use CONFIG['lr']
        for i, line in enumerate(source):
            if "optimizer_params={'lr': 2e-5}" in line:
                source[i] = line.replace(
                    "optimizer_params={'lr': 2e-5}",
                    "optimizer_params={'lr': CONFIG['lr']}"
                )
                source[i] = source[i].replace(
                    "# ✨ Reduced from 1e-4 to 2e-5 for LoRA",
                    "# ✨ Uses CONFIG['lr'] (5e-5)"
                )
                changes_made.append(f"Line {i}: Changed optimizer_params to use CONFIG['lr']")
                modified = True

   # Save modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("HYPERPARAMETER FIXES APPLIED")
    print(f"{'='*80}")
    print(f"\nChanges made:")
    for change in changes_made:
        print(f"  [OK] {change}")

    print(f"\nBackup saved to: {backup_path}")
    print(f"Notebook updated: {NOTEBOOK_PATH}")
    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
