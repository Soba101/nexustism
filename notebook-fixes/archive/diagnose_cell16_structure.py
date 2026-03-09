#!/usr/bin/env python3
"""Diagnose Cell 16 structure to understand where to inject fixes."""
import json

nb = json.load(open('model_promax_mpnet_lorapeft_v3.ipynb', 'r', encoding='utf-8'))
lines = nb['cells'][16]['source']

print("CELL 16 STRUCTURE ANALYSIS")
print("="*80)

# Find key sections
in_curriculum = False
indent_level = 0

for i, line in enumerate(lines, 1):
    # Calculate indent
    indent = (len(line) - len(line.lstrip())) // 4

    # Key markers
    markers = []

    if 'for phase_idx' in line:
        markers.append('[PHASE LOOP START]')
        in_curriculum = True

    if 'DataLoader' in line:
        markers.append('[DATALOADER]')

    if 'model.fit' in line:
        markers.append('[MODEL.FIT]')

    if 'train_objectives' in line:
        markers.append('[TRAIN_OBJECTIVES]')

    if 'except RuntimeError' in line:
        in_curriculum = False
        markers.append('[ERROR HANDLER]')

    # Only print lines in curriculum section or key markers
    if markers or (in_curriculum and (
        'phase' in line.lower() or
        'dataloader' in line.lower() or
        'model.fit' in line or
        'log(' in line
    )):
        marker_str = ' '.join(markers) if markers else ''
        # Truncate line for display
        display_line = line.rstrip()[:60]
        if len(line.rstrip()) > 60:
            display_line += '...'
        print(f"{i:4} {'  '*indent}{display_line:65} {marker_str}")

print("\n" + "="*80)
print("Key findings:")
print("- Look for where DataLoader should be created inside phase loop")
print("- Check if phase_examples is used anywhere")
print("="*80)
