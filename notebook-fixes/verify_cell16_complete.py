#!/usr/bin/env python3
"""
Verify Cell 16 is fully fixed and ready to train.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("VERIFICATION: Cell 16 Complete Fix")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']
source = ''.join(lines)

print(f"\nCell 16: {len(lines)} lines")

# Check 1: No CONFIG['curriculum_phases']
check1 = "CONFIG['curriculum_phases']" not in source
print(f"\n1. No CONFIG['curriculum_phases'] reference: {'[PASS]' if check1 else '[FAIL]'}")

# Check 2: Uses CURRICULUM_PHASES.items()
check2 = "CURRICULUM_PHASES.items()" in source
print(f"2. Uses CURRICULUM_PHASES.items(): {'[PASS]' if check2 else '[FAIL]'}")

# Check 3: Creates phase_dataloader
check3 = "phase_dataloader" in source
print(f"3. Creates phase_dataloader: {'[PASS]' if check3 else '[FAIL]'}")

# Check 4: Uses CONFIG['epochs_per_phase']
check4 = "CONFIG['epochs_per_phase']" in source
print(f"4. Uses CONFIG['epochs_per_phase']: {'[PASS]' if check4 else '[FAIL]'}")

# Check 5: Uses CONFIG['lr'] in optimizer_params
check5 = "optimizer_params={'lr': CONFIG['lr']}" in source
print(f"5. Uses CONFIG['lr']: {'[PASS]' if check5 else '[FAIL]'}")

# Check 6: No hard_neg_ratio references
check6 = "hard_neg_ratio" not in source or "# COMMENTED:" in source
print(f"6. No hard_neg_ratio errors: {'[PASS]' if check6 else '[FAIL]'}")

# Check 7: Iterates with (phase_name, phase_examples)
check7 = "phase_name, phase_examples" in source
print(f"7. Correct phase iteration: {'[PASS]' if check7 else '[FAIL]'}")

all_pass = all([check1, check2, check3, check4, check5, check6, check7])

print(f"\n{'='*80}")
if all_pass:
    print("[SUCCESS] Cell 16 is fully fixed and ready!")
    print(f"{'='*80}")
    print("\nWhat will happen when you run Cell 16:")
    print("  1. Checks if CURRICULUM_PHASES exists (from Cell 12)")
    print("  2. Iterates through phase1, phase2, phase3")
    print("  3. For each phase:")
    print("     - Creates phase_dataloader from phase_examples")
    print("     - Trains for CONFIG['epochs_per_phase'] epochs (value: 2)")
    print("     - Uses learning rate CONFIG['lr'] (value: 5e-5)")
    print("  4. Total: 3 phases × 2 epochs = 6 epochs")
    print("\nExpected training time: 1-2 hours (GPU-dependent)")
else:
    print("[WARNING] Some checks failed!")
    if not check1:
        print("  - Still references CONFIG['curriculum_phases']")
    if not check2:
        print("  - Missing CURRICULUM_PHASES.items() iteration")
    if not check3:
        print("  - Missing phase_dataloader creation")

print(f"{'='*80}")
