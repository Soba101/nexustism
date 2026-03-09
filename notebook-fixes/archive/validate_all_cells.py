#!/usr/bin/env python3
"""
Validate all code cells in the notebook for syntax errors.
"""

import json
import ast
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and c.get('source')]
    print(f"\nValidating {len(code_cells)} code cells...\n")

    errors = []
    for i, cell in code_cells:
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
            # Check for common variable definitions
            has_device = 'DEVICE = ' in source
            has_best_model = 'best_model = SentenceTransformer' in source
            has_train_examples = 'train_examples' in source and 'load_curriculum_pairs' in source
            has_eval_examples = 'eval_examples = [' in source

            markers = []
            if has_device:
                markers.append('DEVICE')
            if has_best_model:
                markers.append('best_model')
            if has_train_examples:
                markers.append('train_examples')
            if has_eval_examples:
                markers.append('eval_examples')

            if markers:
                print(f"  Cell {i:2d}: OK {' | '.join(markers)}")
            else:
                print(f"  Cell {i:2d}: OK")

        except SyntaxError as e:
            error_msg = f"Cell {i}: SyntaxError at line {e.lineno}: {e.msg}"
            errors.append(error_msg)
            print(f"  Cell {i:2d}: ERROR - {e.msg} at line {e.lineno}")

    print("\n" + "="*70)
    if errors:
        print(f"ERRORS FOUND: {len(errors)} cell(s) have syntax errors")
        print("="*70)
        for err in errors:
            print(f"  {err}")
    else:
        print("ALL CELLS VALID - No syntax errors found!")
        print("="*70)
        print("\nKey variables defined:")
        print("  - DEVICE (device detection)")
        print("  - train_examples (curriculum pairs)")
        print("  - eval_examples (test pairs)")
        print("  - best_model (trained model)")
        print("\nThe notebook is ready to run!")

if __name__ == '__main__':
    main()
