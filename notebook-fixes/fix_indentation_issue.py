#!/usr/bin/env python3
"""
Fix indentation error in notebook
"""

import json
from pathlib import Path

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'
BACKUP_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb.backup_before_indent_fix'

print("=" * 80)
print("FIXING INDENTATION ERROR")
print("=" * 80)
print()

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create backup
print(f"Creating backup: {BACKUP_PATH}")
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# The error is at line 90 with log(f"\n{'='*70}")
# This pattern suggests it's in Cell 6 based on previous output

fixed = False
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') == '0bc626db':  # Cell 6
        print(f"\nFound Cell 6 (CONFIG cell)")

        source = cell['source']
        new_source = []

        for line_idx, line in enumerate(source):
            # Check for the problematic pattern
            if 'log(f"\\n' in line or "log(f\"\\n" in line:
                # Check indentation
                if line.startswith('log(') and not line.startswith(' '):
                    # Missing indentation
                    print(f"  Line {line_idx}: FIXING missing indentation")
                    print(f"    Before: {repr(line)}")
                    new_source.append('    ' + line)  # Add 4 spaces
                    print(f"    After: {repr('    ' + line)}")
                    fixed = True
                else:
                    new_source.append(line)
            else:
                new_source.append(line)

        if fixed:
            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            print(f"  [OK] Fixed indentation in Cell 6")

# Also check cells we modified
target_cells = [
    ('9df6368d', 'Cell 12 - Curriculum'),
    ('95b6381e', 'Cell 16 - Training'),
    ('f247e896', 'Cell 26 - Metadata')
]

for cell_id, cell_name in target_cells:
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('id') == cell_id:
            print(f"\nChecking {cell_name}")

            source = cell['source']
            has_issue = False

            for line_idx, line in enumerate(source):
                # Look for print statements with wrong indentation
                if 'print(' in line and line.strip().startswith('print('):
                    # Count leading whitespace
                    leading = len(line) - len(line.lstrip())

                    # If it's not a multiple of 4, there's likely an issue
                    if leading > 0 and leading % 4 != 0:
                        has_issue = True
                        print(f"  Line {line_idx}: SUSPICIOUS indent ({leading} spaces)")
                        print(f"    {repr(line[:50])}")

            if not has_issue:
                print(f"  [OK] No obvious issues")

if fixed:
    # Save notebook
    print(f"\nSaving fixed notebook...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print()
    print("=" * 80)
    print("[SUCCESS] Fixed indentation error!")
    print("=" * 80)
else:
    print()
    print("=" * 80)
    print("[INFO] No automatic fix applied - need manual inspection")
    print("=" * 80)
    print("\nLet me check line 90 specifically in each cell...")

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if len(source) >= 90:
                print(f"\nCell {cell_idx} (ID: {cell.get('id')}) has >=90 lines (total: {len(source)})")
                if len(source) > 90:
                    print(f"  Line 90: {repr(source[89][:80])}")  # Line 90 is index 89
