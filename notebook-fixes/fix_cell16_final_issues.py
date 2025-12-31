#!/usr/bin/env python3
"""
Fix final issues in Cell 16:
1. train_dataloader referenced before creation (line ~15)
2. Fallback handler calls generate_training_pairs() which doesn't exist
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING FINAL CELL 16 ISSUES")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\nProcessing {len(lines)} lines...")

new_lines = []
changes = 0

i = 0
while i < len(lines):
    line = lines[i]

    # Fix 1: Change train_dataloader length calculation to use first phase
    if "total_steps = len(train_dataloader) * CONFIG['epochs']" in line:
        indent = len(line) - len(line.lstrip())
        # Calculate based on first phase (they're all same size)
        new_lines.append(' ' * indent + "# Calculate warmup steps (use phase1 size as reference)\n")
        new_lines.append(' ' * indent + "sample_phase_size = len(list(CURRICULUM_PHASES.values())[0])\n")
        new_lines.append(' ' * indent + "batches_per_phase = (sample_phase_size + CONFIG['batch_size'] - 1) // CONFIG['batch_size']\n")
        new_lines.append(' ' * indent + "total_steps = batches_per_phase * CONFIG['epochs_per_phase'] * 3  # 3 phases\n")
        changes += 1
        print(f"  Line {i+1}: Fixed total_steps calculation")
        i += 1
        continue

    # Fix 2: Remove generate_training_pairs call in fallback
    if "phase_train_examples = generate_training_pairs(" in line:
        # This is in the fallback error handler
        # Skip lines until we find the closing parenthesis
        indent = len(line) - len(line.lstrip())

        # Just use phase_examples directly
        new_lines.append(' ' * indent + "# Use pre-loaded phase examples (already in memory)\n")
        new_lines.append(' ' * indent + "# phase_examples already available from loop\n")

        # Skip until closing parenthesis
        while i < len(lines) and ')' not in lines[i]:
            i += 1
        i += 1  # Skip the closing )

        changes += 1
        print(f"  Line {i}: Removed generate_training_pairs call in fallback")
        continue

    # Fix 3: In fallback, use phase_examples instead of phase_train_examples
    if i > 0 and "phase_train_examples" in line and "DataLoader" in ''.join(lines[i:i+5]):
        line = line.replace("phase_train_examples", "phase_examples")
        changes += 1
        print(f"  Line {i+1}: Changed phase_train_examples to phase_examples in fallback")

    new_lines.append(line)
    i += 1

# Update cell
cell16['source'] = new_lines
nb['cells'][16] = cell16

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
if changes > 0:
    print(f"[SUCCESS] Made {changes} fixes")
    print(f"{'='*80}")
    print("\nFixes applied:")
    print("  1. Fixed total_steps calculation (uses phase size)")
    print("  2. Removed generate_training_pairs() call in fallback")
    print("  3. Fallback uses phase_examples directly")
    print(f"\n{'='*80}")
    print("Cell 16 should now run without errors!")
    print(f"{'='*80}")
else:
    print("[INFO] No changes needed")
    print(f"{'='*80}")
