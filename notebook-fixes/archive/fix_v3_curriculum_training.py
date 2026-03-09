#!/usr/bin/env python3
"""
Fix Cell 16 curriculum training logic.

The code tries to access CONFIG['curriculum_phases'] which doesn't exist.
It should use CURRICULUM_PHASES dictionary created by Cell 12 instead.

OLD (line 1170):
    for phase_idx, phase in enumerate(CONFIG['curriculum_phases']):

NEW:
    for phase_idx, (phase_name, phase_examples) in enumerate(CURRICULUM_PHASES.items()):
"""
import json
import re

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING V3 CELL 16 - CURRICULUM TRAINING LOGIC")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\nProcessing Cell 16: {len(lines)} lines")

# Track changes
changes_made = []
new_lines = []

i = 0
while i < len(lines):
    line = lines[i]

    # Fix 1: Replace CONFIG['curriculum_phases'] iteration
    if "for phase_idx, phase in enumerate(CONFIG['curriculum_phases']):" in line:
        indent = len(line) - len(line.lstrip())
        new_line = ' ' * indent + "for phase_idx, (phase_name, phase_examples) in enumerate(CURRICULUM_PHASES.items()):\n"
        new_lines.append(new_line)
        changes_made.append(f"Line {i+1}: Changed to iterate over CURRICULUM_PHASES.items()")
        i += 1
        continue

    # Fix 2: Replace phase['epochs'] with epochs_per_phase
    if "log(f\"\\n{'='*60}\")" in line or "log(f\"\\n📚 Phase {phase_idx + 1}: {phase['epochs']} epochs\")" in line:
        # These lines reference phase['epochs'] which won't exist
        # Replace with CONFIG['epochs_per_phase']
        if "phase['epochs']" in line:
            new_line = line.replace("phase['epochs']", "CONFIG['epochs_per_phase']")
            new_lines.append(new_line)
            changes_made.append(f"Line {i+1}: Changed phase['epochs'] to CONFIG['epochs_per_phase']")
            i += 1
            continue

    # Fix 3: Remove hard_neg_ratio logging (not applicable for pre-generated pairs)
    if "Hard neg ratio:" in line or "hard_neg_ratio" in line:
        # Skip this line
        changes_made.append(f"Line {i+1}: Removed hard_neg_ratio reference (not in pre-generated data)")
        i += 1
        continue

    # Fix 4: Replace generate_training_pairs call with direct use of phase_examples
    if "phase_train_examples = generate_training_pairs(" in line:
        # This is for the fallback/error handler - also needs fixing
        indent = len(line) - len(line.lstrip())
        # Skip to the closing parenthesis
        j = i
        while j < len(lines) and ')' not in lines[j]:
            j += 1
        # Replace entire generate_training_pairs call
        new_line = ' ' * indent + f"# Using pre-loaded phase examples\n"
        new_lines.append(new_line)
        new_lines.append(' ' * indent + f"phase_train_examples = phase_examples\n")
        changes_made.append(f"Lines {i+1}-{j+1}: Replaced generate_training_pairs with phase_examples")
        i = j + 1
        continue

    # Fix 5: Create DataLoader for phase_examples
    if "train_dataloader = DataLoader(" in line and i > 0:
        # Check if this is inside curriculum training block
        # Look back to see if we're in the curriculum section
        context_lines = ''.join(lines[max(0, i-10):i])
        if 'CURRICULUM' in context_lines or 'phase' in context_lines.lower():
            # This DataLoader needs to use phase_examples instead of train_examples
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + "phase_dataloader = DataLoader(\n"
            new_lines.append(new_line)
            changes_made.append(f"Line {i+1}: Changed to phase_dataloader for curriculum training")

            # Continue copying until we find the closing )
            i += 1
            while i < len(lines):
                line = lines[i]
                # Replace train_examples with phase_examples
                if 'train_examples' in line:
                    new_lines.append(line.replace('train_examples', 'phase_examples'))
                else:
                    new_lines.append(line)

                if ')' in line:
                    i += 1
                    break
                i += 1
            continue

    # Fix 6: Use phase_dataloader in model.fit()
    if "model.fit(" in line:
        # Check if we're in curriculum training section
        context_lines = ''.join(lines[max(0, i-20):i])
        if 'phase_idx' in context_lines or 'CURRICULUM' in context_lines:
            # Continue to train_objectives parameter
            new_lines.append(line)
            i += 1

            # Next line should be train_objectives
            line = lines[i]
            if 'train_objectives' in line and 'train_dataloader' in line:
                new_line = line.replace('train_dataloader', 'phase_dataloader')
                new_lines.append(new_line)
                changes_made.append(f"Line {i+1}: Changed to use phase_dataloader")
                i += 1
                continue

    # Default: keep line as-is
    new_lines.append(line)
    i += 1

# Update cell
cell16['source'] = new_lines
nb['cells'][16] = cell16

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
if changes_made:
    print("[SUCCESS] Cell 16 curriculum training logic fixed!")
    print(f"{'='*80}")
    print(f"\nChanges made ({len(changes_made)}):")
    for change in changes_made:
        print(f"  [OK] {change}")
else:
    print("[INFO] No changes needed - cell may already be fixed")

print(f"\n{'='*80}")
print("Cell 16 will now:")
print("  1. Iterate over CURRICULUM_PHASES (phase1, phase2, phase3)")
print("  2. Train each phase for CONFIG['epochs_per_phase'] epochs")
print("  3. Use pre-loaded phase_examples from Cell 12")
print(f"{'='*80}")
