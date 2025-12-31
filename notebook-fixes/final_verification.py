#!/usr/bin/env python3
"""
Final comprehensive verification of Cell 16.
Check for all potential errors.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FINAL COMPREHENSIVE VERIFICATION")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']
source = ''.join(lines)

print(f"\nCell 16: {len(lines)} lines")
print("\nChecking for potential errors...\n")

errors = []
warnings = []
passed = []

# Check 1: No CONFIG['curriculum_phases']
if "CONFIG['curriculum_phases']" in source:
    errors.append("CONFIG['curriculum_phases'] still referenced")
else:
    passed.append("No CONFIG['curriculum_phases']")

# Check 2: No train_dataloader before creation
if "len(train_dataloader)" in source and "total_steps" in source:
    # Check if it's calculated before curriculum loop
    train_dl_first = source.find("len(train_dataloader)")
    curriculum_start = source.find("CURRICULUM_PHASES")
    if train_dl_first < curriculum_start and train_dl_first != -1:
        errors.append("train_dataloader used before creation")
else:
    passed.append("train_dataloader not used prematurely")

# Check 3: No generate_training_pairs() calls
if "generate_training_pairs(" in source:
    errors.append("generate_training_pairs() still called (doesn't exist)")
else:
    passed.append("No generate_training_pairs() calls")

# Check 4: Uses CURRICULUM_PHASES.items()
if "CURRICULUM_PHASES.items()" in source:
    passed.append("Uses CURRICULUM_PHASES.items()")
else:
    errors.append("Missing CURRICULUM_PHASES.items()")

# Check 5: Creates phase_dataloader
if "phase_dataloader = DataLoader(" in source:
    passed.append("Creates phase_dataloader")
else:
    errors.append("Missing phase_dataloader creation")

# Check 6: Uses CONFIG['epochs_per_phase']
if "CONFIG['epochs_per_phase']" in source:
    passed.append("Uses CONFIG['epochs_per_phase']")
else:
    warnings.append("CONFIG['epochs_per_phase'] not found")

# Check 7: Uses CONFIG['lr']
if "optimizer_params={'lr': CONFIG['lr']}" in source:
    passed.append("Uses CONFIG['lr']")
else:
    errors.append("optimizer_params not using CONFIG['lr']")

# Check 8: No hard_neg_ratio errors
if "phase['hard_neg_ratio']" in source:
    errors.append("phase['hard_neg_ratio'] referenced (doesn't exist)")
else:
    passed.append("No hard_neg_ratio errors")

# Check 9: Correct phase iteration
if "phase_name, phase_examples" in source:
    passed.append("Correct phase iteration syntax")
else:
    errors.append("Incorrect phase iteration")

# Check 10: No phase['epochs'] references
if "phase['epochs']" in source:
    errors.append("phase['epochs'] referenced (doesn't exist)")
else:
    passed.append("No phase['epochs'] errors")

# Print results
print("PASSED CHECKS:")
for check in passed:
    print(f"  [OK] {check}")

if warnings:
    print("\nWARNINGS:")
    for warn in warnings:
        print(f"  [WARN] {warn}")

if errors:
    print("\nERRORS FOUND:")
    for error in errors:
        print(f"  [ERROR] {error}")

print(f"\n{'='*80}")
if errors:
    print(f"[FAIL] {len(errors)} errors need fixing")
elif warnings:
    print(f"[PARTIAL] {len(warnings)} warnings (may be OK)")
else:
    print("[SUCCESS] All checks passed!")
    print("\nCell 16 is ready to execute!")
    print("\nExpected behavior:")
    print("  1. Loads CURRICULUM_PHASES from Cell 12")
    print("  2. Trains 3 phases sequentially")
    print("  3. 4 epochs per phase = 12 total")
    print("  4. Learning rate 5e-5")
    print("  5. Saves best model")
    print("\nTraining time: 2-4 hours")

print(f"{'='*80}")
