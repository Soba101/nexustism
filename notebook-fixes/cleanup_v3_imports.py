#!/usr/bin/env python3
"""
Clean up scattered imports in v3 notebook.
Remove import statements from cells 4+ since Cell 2 has all imports.
"""

import json
import re

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

def is_import_line(line):
    """Check if a line is an import statement."""
    stripped = line.strip()
    return (
        stripped.startswith('import ') or
        stripped.startswith('from ') or
        (stripped.startswith('#') and 'import' in stripped.lower() and len(stripped) < 50)
    )

def clean_cell_imports(cell, cell_index):
    """Remove import statements from a code cell (except Cell 2)."""
    if cell['cell_type'] != 'code' or cell_index == 2:
        return cell, 0

    original_source = cell['source']
    cleaned_source = []
    removed_count = 0

    i = 0
    while i < len(original_source):
        line = original_source[i]

        # Skip import lines and their continuations
        if is_import_line(line):
            removed_count += 1
            # Skip continuation lines (lines ending with backslash or inside parentheses)
            while i + 1 < len(original_source):
                if line.rstrip().endswith('\\') or '(' in line and ')' not in line:
                    i += 1
                    line = original_source[i]
                    removed_count += 1
                else:
                    break
        else:
            cleaned_source.append(line)

        i += 1

    # Remove leading empty lines
    while cleaned_source and cleaned_source[0].strip() == '':
        cleaned_source.pop(0)

    cell['source'] = cleaned_source
    return cell, removed_count

def main():
    print("="*80)
    print("CLEANING UP IMPORTS IN v3 NOTEBOOK")
    print("="*80)

    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"\nProcessing {len(nb['cells'])} cells...")

    total_removed = 0
    cells_modified = 0

    # Clean each cell (skip Cell 2 which has consolidated imports)
    for i, cell in enumerate(nb['cells']):
        cleaned_cell, removed = clean_cell_imports(cell, i)
        nb['cells'][i] = cleaned_cell

        if removed > 0:
            cells_modified += 1
            total_removed += removed
            print(f"  Cell {i}: Removed {removed} import lines")

    # Save cleaned notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("[SUCCESS] Imports cleaned!")
    print(f"{'='*80}")
    print(f"\nCells modified: {cells_modified}")
    print(f"Import lines removed: {total_removed}")
    print(f"All imports now in Cell 2 only")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
