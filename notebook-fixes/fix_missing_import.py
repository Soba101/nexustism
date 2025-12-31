#!/usr/bin/env python3
"""
Add missing average_precision_score to imports in Cell 2.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING MISSING IMPORT: average_precision_score")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 2 (imports)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'from sklearn.metrics import' in source and 'roc_auc_score' in source:
            print(f"\nFound sklearn.metrics import in Cell {cell_idx}")

            # Find the line with f1_score and add average_precision_score after it
            new_lines = []
            for i, line in enumerate(cell['source']):
                new_lines.append(line)

                # After f1_score line, add average_precision_score
                if "f1_score," in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + "average_precision_score,\n")
                    print(f"  Added average_precision_score after f1_score")

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell
            break

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
print("[SUCCESS] Import added!")
print(f"{'='*80}")
print("\nNow sklearn.metrics imports:")
print("  - roc_auc_score")
print("  - precision_recall_curve")
print("  - f1_score")
print("  - average_precision_score  <-- NEW")
print("  - accuracy_score")
print("  - precision_score")
print("  - recall_score")
print("  - confusion_matrix")
print("  - roc_curve")
print(f"{'='*80}")
