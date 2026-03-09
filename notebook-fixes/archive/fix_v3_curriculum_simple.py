#!/usr/bin/env python3
"""
Simple fix for Cell 16 curriculum training.
Replace CONFIG['curriculum_phases'] with direct phase training.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING V3 CELL 16 - CURRICULUM TRAINING")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\nOriginal lines: {len(lines)}")

# Simple string replacements
new_lines = []
changes = 0

for i, line in enumerate(lines):
    original = line

    # Fix 1: Replace curriculum_phases iteration
    if "enumerate(CONFIG['curriculum_phases'])" in line:
        # Change to iterate over sorted CURRICULUM_PHASES items
        line = line.replace(
            "enumerate(CONFIG['curriculum_phases'])",
            "enumerate(sorted(CURRICULUM_PHASES.items()))"
        )
        line = line.replace(
            "for phase_idx, phase in",
            "for phase_idx, (phase_name, phase_examples) in"
        )
        changes += 1
        print(f"  Line {i+1}: Fixed iteration over CURRICULUM_PHASES")

    # Fix 2: Replace phase['epochs'] with CONFIG['epochs_per_phase']
    if "phase['epochs']" in line:
        line = line.replace("phase['epochs']", "CONFIG['epochs_per_phase']")
        changes += 1
        print(f"  Line {i+1}: Replaced phase['epochs'] with CONFIG['epochs_per_phase']")

    # Fix 3: Remove or comment out hard_neg_ratio lines
    if "hard_neg_ratio" in line:
        # Comment it out
        indent = len(line) - len(line.lstrip())
        line = ' ' * indent + '# ' + line.lstrip()
        changes += 1
        print(f"  Line {i+1}: Commented out hard_neg_ratio (not in pre-generated)")

    new_lines.append(line)

# Update cell
cell16['source'] = new_lines
nb['cells'][16] = cell16

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
print(f"[SUCCESS] Made {changes} changes to Cell 16")
print(f"{'='*80}")
print("\nNext: Cell 16 needs additional fix for phase training loop")
print("The iteration now gives (phase_name, phase_examples)")
print("But the training code needs to use phase_examples as the dataset")
print(f"{'='*80}")
