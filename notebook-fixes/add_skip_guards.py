#!/usr/bin/env python3
"""
Add SKIP_PAIR_GENERATION guards to legacy pair generation cells.
"""

import json
import sys
from pathlib import Path

def add_guard_to_cell(cell):
    """Add skip guard at the beginning of the cell"""
    source = cell.get('source', [])
    if not source:
        return False

    # Check if this is a pair generation cell
    source_text = ''.join(source)

    # Skip if guard already exists
    if 'SKIP_PAIR_GENERATION' in source_text and 'if not' in source_text:
        return False

    # Add guard to cells that generate pairs
    markers = [
        'Generate pairs for each split with reusable TF-IDF',
        'train_examples = generate_training_pairs',
        'eval_examples = generate_training_pairs'
    ]

    if any(marker in source_text for marker in markers):
        # Add guard at the beginning
        guard = [
            "# Skip if using pre-generated pairs\n",
            "if not CONFIG.get('use_pre_generated_pairs', False):\n",
            "    \n"
        ]

        # Indent all existing lines
        indented_source = []
        for line in source:
            if line.strip():  # Non-empty lines
                indented_source.append("    " + line)
            else:
                indented_source.append(line)

        # Add else clause
        else_clause = [
            "else:\n",
            "    log(\"⏭️  Skipping pair generation (using pre-generated curriculum pairs)\")\n",
            "    log(f\"   Loaded from: {CONFIG['train_pairs_path']}\")\n"
        ]

        cell['source'] = guard + indented_source + else_clause
        return True

    return False

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        sys.exit(1)

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Add guards to pair generation cells
    modified_count = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            if add_guard_to_cell(cell):
                modified_count += 1
                print(f"Added guard to cell {i}")

    if modified_count == 0:
        print("No cells needed modification (guards already exist or no matching cells found)")
        return

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup2')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Added skip guards to pair generation cells")
    print("="*70)
    print(f"\nModified {modified_count} cell(s)")
    print("\nThe cells will now:")
    print("  1. Check if use_pre_generated_pairs is True")
    print("  2. Skip pair generation if True")
    print("  3. Run legacy generation if False")
    print("\nRestart the kernel and run all cells from the beginning!")
    print("="*70)

if __name__ == '__main__':
    main()
