#!/usr/bin/env python3
"""
Fix TFIDFSimilarityCalculator method name mismatch.
Code calls .similarity() but class defines .get_similarity()
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING TFIDF METHOD NAME")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 24 (adversarial diagnostic)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'class TFIDFSimilarityCalculator' in source and 'diag_tfidf.similarity(' in source:
            print(f"\nFound Cell {cell_idx} with method name mismatch")

            # Option 1: Change method name in class (get_similarity → similarity)
            # This is simpler than changing all call sites

            new_lines = []
            for line in cell['source']:
                # Change method definition
                if 'def get_similarity(self, idx1, idx2):' in line:
                    new_line = line.replace('get_similarity', 'similarity')
                    new_lines.append(new_line)
                    print(f"  Changed: def get_similarity() → def similarity()")
                else:
                    new_lines.append(line)

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell

            # Save
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)

            print(f"\n{'='*80}")
            print("[SUCCESS] Method name fixed!")
            print(f"{'='*80}")
            print("\nMethod is now: .similarity(idx1, idx2)")
            print("Code calls: diag_tfidf.similarity(i1, i2)")
            print("\nMatch! ✅")
            print(f"{'='*80}")
            break
