#!/usr/bin/env python3
"""
Fix the actual training loop inside curriculum phases.

After iterating over CURRICULUM_PHASES.items(), we get:
- phase_name: 'phase1', 'phase2', or 'phase3'
- phase_examples: list of InputExample objects

The training loop needs to:
1. Create DataLoader from phase_examples
2. Use that DataLoader in model.fit()
"""
import json
import re

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING V3 CELL 16 - PHASE TRAINING LOOP")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\nProcessing {len(lines)} lines...")

# Find the section where we need to add/fix DataLoader creation
new_lines = []
changes = 0
in_curriculum_loop = False
skip_until_fit = False

for i, line in enumerate(lines):
    # Track when we enter the curriculum loop
    if "for phase_idx, (phase_name, phase_examples) in enumerate" in line:
        in_curriculum_loop = True
        new_lines.append(line)
        continue

    # After the phase logging, add DataLoader creation
    if in_curriculum_loop and "# COMMENTED:" in line and "hard_neg_ratio" in line:
        # This is right after phase info logging
        # Add DataLoader creation code after this line
        new_lines.append(line)

        # Check if next lines already have DataLoader creation
        if i+1 < len(lines) and 'DataLoader' not in lines[i+1]:
            # Add DataLoader creation
            indent = '            '  # Match indentation inside loop
            new_lines.append(f'{indent}\n')
            new_lines.append(f'{indent}# Create DataLoader for this phase\n')
            new_lines.append(f'{indent}phase_dataloader = DataLoader(\n')
            new_lines.append(f'{indent}    phase_examples,\n')
            new_lines.append(f'{indent}    batch_size=CONFIG[\"batch_size\"],\n')
            new_lines.append(f'{indent}    shuffle=True,\n')
            new_lines.append(f'{indent}    num_workers=0\n')
            new_lines.append(f'{indent})\n')
            new_lines.append(f'{indent}\n')
            changes += 1
            print(f"  Line {i+1}: Added phase_dataloader creation")
        continue

    # Replace train_dataloader with phase_dataloader in model.fit() calls
    if in_curriculum_loop and 'train_dataloader' in line and 'model.fit' in ''.join(lines[max(0,i-2):i+1]):
        line = line.replace('train_dataloader', 'phase_dataloader')
        changes += 1
        print(f"  Line {i+1}: Changed train_dataloader to phase_dataloader")

    # Also replace in train_objectives parameter
    if in_curriculum_loop and 'train_dataloader' in line and 'train_objectives' in line:
        line = line.replace('train_dataloader', 'phase_dataloader')
        changes += 1
        print(f"  Line {i+1}: Changed train_dataloader to phase_dataloader in objectives")

    # Exit curriculum loop when we hit the fallback error handler
    if 'except RuntimeError' in line:
        in_curriculum_loop = False

    new_lines.append(line)

# Update cell
cell16['source'] = new_lines
nb['cells'][16] = cell16

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
if changes > 0:
    print(f"[SUCCESS] Made {changes} changes")
    print(f"{'='*80}")
    print("\nCell 16 will now:")
    print("  1. Iterate over CURRICULUM_PHASES.items()")
    print("  2. Create phase_dataloader from phase_examples")
    print("  3. Train each phase with phase_dataloader")
    print(f"{'='*80}")
else:
    print("[INFO] No additional changes needed")
    print(f"{'='*80}")
