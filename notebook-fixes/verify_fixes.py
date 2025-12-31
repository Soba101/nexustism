#!/usr/bin/env python3
"""Verify all fixes were applied correctly."""
import json

print("="*80)
print("VERIFICATION: Fine-Tuning Fixes")
print("="*80)

# Load notebook
with open('model_promax_mpnet_lorapeft.ipynb', 'r', encoding='utf-8') as f:
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

# Check 5: Curriculum file exists
import os
check5 = os.path.exists('data_new/curriculum_training_pairs_complete.json')
file_size = os.path.getsize('data_new/curriculum_training_pairs_complete.json') / (1024*1024) if check5 else 0
print(f"4. Curriculum file exists: {'[PASS]' if check5 else '[FAIL]'} ({file_size:.1f} MB)")

# Summary
all_pass = check1 and check2 and (check3_4 >= 2) and check5
print(f"\n{'='*80}")
if all_pass:
    print("[SUCCESS] ALL CHECKS PASSED - Ready for training!")
    print("\nNext step: jupyter notebook model_promax_mpnet_lorapeft.ipynb")
else:
    print("[ERROR] SOME CHECKS FAILED - Review fixes")
print(f"{'='*80}")
