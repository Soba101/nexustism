#!/usr/bin/env python3
"""
Comprehensive verification of notebook fixes.
"""

import json
import ast
from pathlib import Path

def verify_notebook(notebook_path):
    """Verify all fixes are properly applied"""

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    code_cells = [c for c in cells if c['cell_type'] == 'code' and c.get('source')]

    print(f"\nNotebook structure:")
    print(f"  Total cells: {len(cells)}")
    print(f"  Code cells: {len(code_cells)}")
    print(f"  Markdown cells: {len(cells) - len(code_cells)}")

    # Check syntax
    print(f"\nSyntax validation:")
    syntax_errors = []
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code' or not cell.get('source'):
            continue
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
        except SyntaxError as e:
            syntax_errors.append((i, str(e)))

    if syntax_errors:
        print(f"  ERROR: {len(syntax_errors)} cells have syntax errors")
        for i, err in syntax_errors:
            print(f"    Cell {i}: {err[:80]}")
        return False
    else:
        print(f"  OK: All {len(code_cells)} code cells have valid syntax")

    # Find critical variables
    print(f"\nCritical variable definitions:")
    vars_found = {}

    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code' or not cell.get('source'):
            continue
        source = ''.join(cell['source'])

        if 'CONFIG = {' in source and 'model_name' in source:
            vars_found['CONFIG'] = i
        if "DEVICE = 'cuda'" in source or "DEVICE = 'mps'" in source or "DEVICE = 'cpu'" in source:
            if 'DEVICE' not in vars_found:  # Take first occurrence
                vars_found['DEVICE'] = i
        if 'train_examples' in source and 'load_curriculum' in source:
            vars_found['train_examples'] = i
        if 'eval_examples = [' in source and 'InputExample' in source:
            vars_found['eval_examples'] = i
        if 'best_model = SentenceTransformer' in source:
            vars_found['best_model'] = i

    for var in ['CONFIG', 'DEVICE', 'train_examples', 'eval_examples', 'best_model']:
        if var in vars_found:
            print(f"  OK: Cell {vars_found[var]:2d} defines {var}")
        else:
            print(f"  ERROR: {var} not found!")
            return False

    # Check fixes
    print(f"\nFix verification:")
    all_source = ''.join([''.join(c.get('source', [])) for c in code_cells])

    checks = [
        ('Device detection', 'DEVICE = ' in all_source and 'torch.cuda.is_available()' in all_source),
        ('Curriculum pair loading', 'load_curriculum_pairs' in all_source),
        ('Test pairs loading', 'fixed_test_pairs.json' in all_source),
        ('Skip guards', 'use_pre_generated_pairs' in all_source and 'SKIP_PAIR_GENERATION' in all_source),
        ('best_model definition', 'best_model = SentenceTransformer' in all_source),
        ('Unicode cleaned', '✅' not in all_source and '📦' not in all_source),
        ('Eval pairs loaded', 'eval_examples = [' in all_source and 'InputExample' in all_source),
        ('Borderline guard', 'if borderline_examples' in all_source or 'if len(borderline_examples)' in all_source),
    ]

    all_ok = True
    for check_name, result in checks:
        status = 'OK' if result else 'MISSING'
        print(f"  {status}: {check_name}")
        if not result:
            all_ok = False

    # Check execution order
    print(f"\nExecution order check:")
    order_ok = True

    if vars_found.get('train_examples', 999) >= vars_found.get('best_model', 0):
        print(f"  ERROR: train_examples (cell {vars_found.get('train_examples')}) must be before best_model (cell {vars_found.get('best_model')})")
        order_ok = False
    else:
        print(f"  OK: train_examples (cell {vars_found.get('train_examples')}) before best_model (cell {vars_found.get('best_model')})")

    if vars_found.get('DEVICE', 999) >= vars_found.get('best_model', 0):
        print(f"  ERROR: DEVICE (cell {vars_found.get('DEVICE')}) must be before best_model (cell {vars_found.get('best_model')})")
        order_ok = False
    else:
        print(f"  OK: DEVICE (cell {vars_found.get('DEVICE')}) before best_model (cell {vars_found.get('best_model')})")

    print(f"\n" + "="*70)
    if all_ok and order_ok and not syntax_errors:
        print("VERIFICATION PASSED: Notebook is ready to run!")
        print("="*70)
        print("\nTo run:")
        print("  1. Open model_promax_mpnet_lorapeft.ipynb")
        print("  2. Restart kernel (Kernel -> Restart & Clear Output)")
        print("  3. Run All (Cell -> Run All)")
        print("  4. Wait for training to complete")
        return True
    else:
        print("VERIFICATION FAILED: Issues found above")
        print("="*70)
        return False

if __name__ == '__main__':
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')
    success = verify_notebook(notebook_path)
    exit(0 if success else 1)
