#!/usr/bin/env python3
"""
Fix epochs_per_phase to match total epochs.

CONFIG['epochs'] = 12 with comment "4 per curriculum phase"
But CONFIG['epochs_per_phase'] = 2

Should be: 12 total / 3 phases = 4 epochs per phase
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING epochs_per_phase CONFIG")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find CONFIG cell (should be Cell 6)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any("'epochs_per_phase':" in line for line in cell['source']):
        print(f"\nFound CONFIG in Cell {cell_idx}")

        # Replace epochs_per_phase value
        new_lines = []
        for line in cell['source']:
            if "'epochs_per_phase': 2" in line:
                # Change 2 to 4
                new_line = line.replace("'epochs_per_phase': 2", "'epochs_per_phase': 4")
                # Update comment
                new_line = new_line.replace("# 2 epochs per phase", "# 4 epochs per phase (12 total / 3 phases)")
                new_lines.append(new_line)
                print(f"  Changed: epochs_per_phase from 2 to 4")
            else:
                new_lines.append(line)

        cell['source'] = new_lines
        nb['cells'][cell_idx] = cell
        break

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
print("[SUCCESS] epochs_per_phase fixed!")
print(f"{'='*80}")
print("\nNew configuration:")
print("  Total epochs: 12")
print("  Curriculum phases: 3")
print("  Epochs per phase: 4")
print("  Training time: 2-4 hours (increased from 1-2 hours)")
print(f"{'='*80}")
