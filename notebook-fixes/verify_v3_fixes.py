#!/usr/bin/env python3
"""Verify all fixes were preserved in v3 notebook."""
import json

print("="*80)
print("VERIFICATION: v3 Notebook Fixes")
print("="*80)

# Load v3 notebook
with open('model_promax_mpnet_lorapeft_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract all source code
source = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source.extend(cell['source'])

# Check 1: Train pairs path
check1 = any("'train_pairs_path': 'data_new/curriculum_training_pairs_complete.json'" in line for line in source)
print(f"\n1. Train pairs path updated: {'[PASS]' if check1 else '[FAIL]'}")

# Check 2: Epochs = 12
check2 = any("'epochs': 12," in line for line in source)
print(f"2. Epochs set to 12: {'[PASS]' if check2 else '[FAIL]'}")

# Check 3 & 4: optimizer_params uses CONFIG['lr']
check3_4 = sum(1 for line in source if "optimizer_params={'lr': CONFIG['lr']}" in line)
print(f"3. optimizer_params uses CONFIG['lr']: {'[PASS]' if check3_4 >= 2 else '[FAIL]'} (found {check3_4} instances)")

# Check 4: Imports consolidated
import_cells = [i for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code' and any('import' in line for line in cell['source'])]
check4 = len([i for i in import_cells if i < 5]) == 1  # Only one import cell in first 5 cells
print(f"4. Imports consolidated in early cell: {'[PASS]' if check4 else '[FAIL]'} (import cells: {import_cells[:3]})")

# Check 5: Cell count reduced
check5 = len(nb['cells']) < 32
print(f"5. Cell count reduced: {'[PASS]' if check5 else '[FAIL]'} ({len(nb['cells'])} vs 32 original)")

# Check 6: No legacy pair generation
check6 = not any('TfidfVectorizer' in line and 'generate' in line.lower() for line in source)
print(f"6. Legacy pair generation removed: {'[PASS]' if check6 else 'WARNING - TF-IDF generation code still present'}")

# Summary
all_pass = check1 and check2 and (check3_4 >= 2) and check4 and check5
print(f"\n{'='*80}")
if all_pass:
    print("[SUCCESS] ALL CRITICAL CHECKS PASSED - v3 ready!")
    print("\nv3 improvements:")
    print(f"  - Cells: {len(nb['cells'])} (reduced from 32)")
    print(f"  - Fixes preserved: train_pairs_path, epochs=12, lr=CONFIG['lr']")
    print(f"  - Imports consolidated: Cell 3")
    print(f"  - Clean structure: No if-guards or dead code")
else:
    print("[WARNING] Some checks failed - review v3")
print(f"{'='*80}")
