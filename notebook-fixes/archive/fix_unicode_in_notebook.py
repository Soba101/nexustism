#!/usr/bin/env python3
"""
Remove problematic Unicode characters from notebook cells that might cause encoding errors.
"""

import json
import re
from pathlib import Path

def sanitize_source(source_lines):
    """Remove problematic Unicode characters"""
    sanitized = []
    modified = False

    replacements = {
        '✅': '[OK]',
        '📦': '[INSTALL]',
        '🧠': '[BUILD]',
        '📊': '[STATS]',
        '⏭️': '[SKIP]',
        '🎯': '[TARGET]',
        '💾': '[SAVE]',
        '✓': 'OK',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
    }

    for line in source_lines:
        new_line = line
        for emoji, replacement in replacements.items():
            if emoji in line:
                new_line = new_line.replace(emoji, replacement)
                modified = True
        sanitized.append(new_line)

    return sanitized, modified

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        return

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_cells = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('source'):
            sanitized, was_modified = sanitize_source(cell['source'])
            if was_modified:
                cell['source'] = sanitized
                modified_cells += 1
                print(f"Sanitized cell {i}")

    if modified_cells == 0:
        print("No Unicode characters found to replace")
        return

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup3')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing sanitized notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Removed problematic Unicode characters")
    print("="*70)
    print(f"\nModified {modified_cells} cell(s)")
    print("\nReplacements made:")
    print("  ✅ → [OK]")
    print("  📦 → [INSTALL]")
    print("  ⏭️ → [SKIP]")
    print("  (and other emojis)")
    print("="*70)

if __name__ == '__main__':
    main()
