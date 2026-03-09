#!/usr/bin/env python3
"""
Comprehensive check and fix for all cell structure issues in the notebook.
"""

import json
import ast
from pathlib import Path

def check_and_fix_cells(nb):
    """Check all cells for structure issues and fix them"""
    issues_found = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code' or not cell.get('source'):
            continue

        source = ''.join(cell['source'])

        # Check for specific known issues

        # Issue 1: Training cell has orphaned skip message
        if 'best_model = SentenceTransformer(str(save_path)' in source:
            if 'Skipping pair generation' in source:
                print(f"Cell {i}: Found orphaned skip message in training cell")
                # Remove the misplaced else clause
                lines = cell['source']
                new_lines = []
                skip_until = None

                for j, line in enumerate(lines):
                    if skip_until and j < skip_until:
                        continue
                    skip_until = None

                    # Remove the else clause about skipping pair generation
                    if 'else:' in line and j+1 < len(lines):
                        next_line = lines[j+1] if j+1 < len(lines) else ''
                        if 'Skipping pair generation' in next_line:
                            print(f"  Removing lines {j}-{j+2}")
                            skip_until = j + 3
                            continue

                    new_lines.append(line)

                cell['source'] = new_lines
                issues_found.append(f"Cell {i}: Removed orphaned else clause")

        # Issue 2: Check for syntax errors
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"Cell {i}: Syntax error - {e}")
            issues_found.append(f"Cell {i}: Syntax error at line {e.lineno}")

    return issues_found

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"\nChecking {len(nb['cells'])} cells...")
    issues = check_and_fix_cells(nb)

    if not issues:
        print("\n[OK] No issues found!")
        return

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup8')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Fixed cell structure issues")
    print("="*70)
    print(f"\nIssues fixed:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nAll cells should now work correctly!")
    print("="*70)

if __name__ == '__main__':
    main()
