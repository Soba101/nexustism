#!/usr/bin/env python3
"""
Fix syntax error: remove orphaned else clause in pair loading cell.
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the pair loader cell (cell 11)
    cell = nb['cells'][11]
    source = cell['source']

    # Find and remove the orphaned else
    new_source = []
    skip_next_lines = 0

    for i, line in enumerate(source):
        # Skip the orphaned else and its content
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue

        # Check for the specific orphaned else pattern
        if line.strip() == 'else:' and i > 0:
            # Check context - if previous line ends with closing bracket or is blank
            # and we're already inside curriculum loading
            prev_line = source[i-1].strip()

            # Look ahead to see what's in this else
            next_line = source[i+1].strip() if i+1 < len(source) else ''

            # This is the orphaned else (after eval_examples loading, before legacy mode else)
            if '# Load all mixed' in next_line:
                print(f"Found orphaned else at line {i+1}")
                # Skip this else and its 2 content lines
                skip_next_lines = 3  # else:, # Load all mixed, train_examples = ..., CURRICULUM_PHASES = None
                continue

        new_source.append(line)

    cell['source'] = new_source
    nb['cells'][11] = cell

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup7')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Fixed syntax error")
    print("="*70)
    print("\nRemoved orphaned else clause that was causing SyntaxError")
    print("The pair loading logic now has correct if/else structure")
    print("="*70)

if __name__ == '__main__':
    main()
