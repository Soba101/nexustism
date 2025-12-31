#!/usr/bin/env python3
"""
Fix JSON serialization error in V2 notebook Cell 16.
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft-v2.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Fix Cell 16 (Save Model & Metadata)
    cell = nb['cells'][16]
    source = ''.join(cell['source'])

    # Replace the problematic line
    old_line = "    'results': results,"
    new_line = "    'results': {k: float(v) if hasattr(v, 'item') else v for k, v in results.items()},"

    if old_line in source:
        source = source.replace(old_line, new_line)
        cell['source'] = [source]
        nb['cells'][16] = cell

        # Save first, then print
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print("="*70)
        print("SUCCESS: Fixed JSON serialization in Cell 16")
        print("="*70)
        print("\nFixed Cell 16: Added numpy -> Python type conversion")
        print("\nThe metadata will now save correctly by converting:")
        print("  numpy.float32 -> float")
        print("  numpy.float64 -> float")
        print("\nYou can now run Cell 16 without errors!")
        print("="*70)
    else:
        print("ERROR: Cell 16 pattern not found or already fixed")

if __name__ == '__main__':
    main()
