#!/usr/bin/env python3
"""
Find and fix indentation error in notebook
"""

import json
from pathlib import Path

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("=" * 80)
print("FINDING INDENTATION ERROR")
print("=" * 80)
print()

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check each cell for the problematic pattern
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])

        # Look for cells with print statements and '=' characters (likely verification cells)
        for line_idx, line in enumerate(source):
            if 'log(f"\\n' in line or "log(f\"\\n" in line:
                print(f"Cell {cell_idx} (ID: {cell.get('id', 'unknown')}), Line {line_idx}:")
                print(f"  {line}")

                # Show surrounding lines for context
                start = max(0, line_idx - 3)
                end = min(len(source), line_idx + 4)
                print("\n  Context:")
                for i in range(start, end):
                    marker = ">>> " if i == line_idx else "    "
                    print(f"  {marker}{i}: {source[i]}", end='')
                print()

print("=" * 80)
print("Checking for indentation issues in verification code...")
print("=" * 80)

# Specifically check cells we modified (12, 16, 26)
target_ids = ['9df6368d', '95b6381e', 'f247e896']

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') in target_ids:
        source = cell.get('source', [])
        print(f"\n[Cell {cell_idx} - ID: {cell.get('id')}]")

        # Check each line for consistent indentation
        for line_idx, line in enumerate(source):
            if line.strip():  # Skip empty lines
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))

                if leading_tabs > 0 and leading_spaces > 0:
                    print(f"  Line {line_idx}: MIXED TABS/SPACES")
                    print(f"    {repr(line)}")

                # Look for problematic patterns
                if 'print(' in line and leading_spaces % 4 != 0:
                    print(f"  Line {line_idx}: NON-MULTIPLE-OF-4 INDENT ({leading_spaces} spaces)")
                    print(f"    {repr(line)}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
