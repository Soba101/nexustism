#!/usr/bin/env python3
"""
Fix Cell 16 in v3 notebook - remove the inverted guard condition.

The cell currently has:
    if not CONFIG.get('use_pre_generated_pairs', False):
        # ALL TRAINING CODE HERE

But CONFIG['use_pre_generated_pairs'] = True, so nothing runs!

Fix: Remove the guard and un-indent all the code.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING V3 CELL 16 - TRAINING EXECUTION")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\nLoaded notebook with {len(nb['cells'])} cells")

# Get Cell 16 (training execution)
cell16 = nb['cells'][16]
original_lines = cell16['source']

print(f"Cell 16 original line count: {len(original_lines)}")

# Find the problematic if statement
guard_line = "if not CONFIG.get('use_pre_generated_pairs', False):\n"
comment_line = "# Skip if using pre-generated pairs\n"

if guard_line in original_lines or comment_line in original_lines:
    print("\n[FOUND] Inverted guard condition at start of cell")

    # Remove the comment and if statement, and un-indent the rest
    new_lines = []
    skip_next = False

    for i, line in enumerate(original_lines):
        # Skip the comment line
        if line == comment_line:
            print(f"  Line {i+1}: Removing comment: {line.rstrip()}")
            skip_next = True
            continue

        # Skip the if statement line
        if skip_next and line == guard_line:
            print(f"  Line {i+1}: Removing guard: {line.rstrip()}")
            skip_next = False
            continue

        # Un-indent by 4 spaces (or 1 tab) for remaining lines
        if line.startswith('    '):
            new_lines.append(line[4:])  # Remove 4 spaces
        elif line.startswith('\t'):
            new_lines.append(line[1:])  # Remove 1 tab
        else:
            # Line not indented or blank - keep as is
            new_lines.append(line)

    # Update cell
    cell16['source'] = new_lines
    nb['cells'][16] = cell16

    print(f"\nCell 16 new line count: {len(new_lines)}")
    print(f"Lines removed: {len(original_lines) - len(new_lines)}")

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("[SUCCESS] Cell 16 fixed!")
    print(f"{'='*80}")
    print("\nChanges:")
    print("  [OK] Removed inverted guard condition")
    print("  [OK] Un-indented all training code")
    print("\nNow training will execute when you run Cell 16!")
    print(f"{'='*80}")

else:
    print("\n[INFO] Guard condition not found - cell may already be fixed")
    print("First 10 lines:")
    for i, line in enumerate(original_lines[:10]):
        print(f"  {i+1}: {line.rstrip()}")
