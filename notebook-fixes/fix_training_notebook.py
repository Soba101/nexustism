#!/usr/bin/env python3
"""
Fix training notebook bugs and add verification output

Fixes:
1. Cell 26 (metadata): Replace hardcoded LR with config.get('lr')
2. Cell 26 (metadata): Fix curriculum_learning logging
3. Cell 16 (training): Add training configuration verification
4. Cell 12 (curriculum): Add curriculum loading verification
"""

import json
from pathlib import Path

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'
BACKUP_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb.backup_metadata_fix'

print("="*80)
print("FIXING TRAINING NOTEBOOK")
print("="*80)
print()

# Load notebook
print(f"Loading: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create backup
print(f"Creating backup: {BACKUP_PATH}")
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

changes_made = []

# Fix 1 & 2: Cell 26 - Metadata function
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') == 'f247e896':
        print(f"\nFound Cell 26 (metadata function)")

        source = cell['source']
        new_source = []

        for line in source:
            # Fix 1: Replace hardcoded LR
            if '"learning_rate": 2e-5,' in line:
                new_source.append('            "learning_rate": config.get(\'lr\'),  # V2: Use actual config value\n')
                print("  [OK] Fixed hardcoded learning rate")
                changes_made.append("Fixed metadata learning_rate (Cell 26)")
                continue

            # Fix 2: Fix curriculum logging (replace entire curriculum_learning section)
            if '"curriculum_learning": {' in line:
                new_source.append('        "curriculum_learning": {\n')
                new_source.append('            "enabled": config.get(\'use_curriculum\', False),\n')
                new_source.append('            "num_phases": 3 if config.get(\'use_curriculum\') else 0,\n')
                new_source.append('            "epochs_per_phase": config.get(\'epochs_per_phase\', 4),\n')
                new_source.append('            "total_pairs": 15000 if config.get(\'use_curriculum\') else config.get(\'num_pairs\', 0)\n')
                new_source.append('        },\n')

                # Skip old lines until closing brace
                skip_until_brace = True
                print("  [OK] Fixed curriculum_learning metadata")
                changes_made.append("Fixed metadata curriculum_learning (Cell 26)")
                continue

            # Skip old curriculum_learning content
            if 'skip_until_brace' in locals() and skip_until_brace:
                if '},' in line and '"data_config"' not in line:
                    skip_until_brace = False
                continue

            new_source.append(line)

        cell['source'] = new_source
        nb['cells'][cell_idx] = cell
        break

# Fix 3: Cell 16 - Add training verification
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') == '95b6381e':
        print(f"\nFound Cell 16 (training loop)")

        source = cell['source']

        # Check if verification already exists
        if any('TRAINING CONFIGURATION VERIFICATION' in line for line in source):
            print("  [SKIP] Verification already exists")
        else:
            # Find where to insert (after CURRICULUM_PHASES check, before training)
            insert_idx = None
            for i, line in enumerate(source):
                if 'V2: Curriculum Learning' in line or 'if CONFIG.get(\'use_curriculum\')' in line:
                    # Find the line with first model.fit call
                    for j in range(i, len(source)):
                        if 'model.fit(' in source[j]:
                            insert_idx = j
                            break
                    break

            if insert_idx:
                verification_code = [
                    '\n',
                    '# Verify configuration before training\n',
                    'print("\\n" + "="*80)\n',
                    'print("TRAINING CONFIGURATION VERIFICATION")\n',
                    'print("="*80)\n',
                    'print(f"Learning Rate: {CONFIG[\'lr\']}")\n',
                    'print(f"Curriculum Learning: {CONFIG[\'use_curriculum\']}")\n',
                    'if CONFIG.get(\'use_curriculum\'):\n',
                    '    if CURRICULUM_PHASES:\n',
                    '        print(f"  Phases loaded: {len(CURRICULUM_PHASES)}")\n',
                    '        for phase_name, phase_data in sorted(CURRICULUM_PHASES.items()):\n',
                    '            print(f"    {phase_name}: {len(phase_data):,} pairs")\n',
                    '    else:\n',
                    '        print("  WARNING: CURRICULUM_PHASES is None/empty!")\n',
                    'print(f"Loss Function: MatryoshkaLoss + MultipleNegativesRankingLoss")\n',
                    'print(f"Warmup Ratio: {CONFIG[\'warmup_ratio\']}")\n',
                    'print(f"Batch Size: {CONFIG[\'batch_size\']}")\n',
                    'print("="*80 + "\\n")\n',
                    '\n'
                ]

                new_source = source[:insert_idx] + verification_code + source[insert_idx:]
                cell['source'] = new_source
                nb['cells'][cell_idx] = cell
                print("  [OK] Added training configuration verification")
                changes_made.append("Added training verification (Cell 16)")
            else:
                print("  [WARN] Could not find insertion point")

        break

# Fix 4: Cell 12 - Add curriculum verification
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') == '9df6368d':
        print(f"\nFound Cell 12 (curriculum loading)")

        source = cell['source']

        # Check if verification already exists
        if any('Curriculum Verification' in line for line in source):
            print("  [SKIP] Verification already exists")
        else:
            # Find where to insert (after CURRICULUM_PHASES assignment)
            insert_idx = None
            for i, line in enumerate(source):
                if 'CURRICULUM_PHASES = {' in line or 'CURRICULUM_PHASES = None' in line:
                    # Find the end of this block
                    for j in range(i+1, len(source)):
                        if source[j].strip() and not source[j].strip().startswith('#') and '{' not in source[j]:
                            insert_idx = j
                            break
                    break

            if insert_idx:
                verification_code = [
                    '\n',
                    '# Verify curriculum loaded correctly\n',
                    'if CONFIG.get(\'use_curriculum\') and CURRICULUM_PHASES:\n',
                    '    total_pairs = sum(len(v) for v in CURRICULUM_PHASES.values())\n',
                    '    print(f"\\n[OK] Curriculum Verification:")\n',
                    '    print(f"   Total pairs: {total_pairs:,}")\n',
                    '    print(f"   Phase 1 (easy): {len(CURRICULUM_PHASES.get(\'phase1\', [])):,}")\n',
                    '    print(f"   Phase 2 (medium): {len(CURRICULUM_PHASES.get(\'phase2\', [])):,}")\n',
                    '    print(f"   Phase 3 (hard): {len(CURRICULUM_PHASES.get(\'phase3\', [])):,}")\n',
                    '    \n',
                    '    assert len(CURRICULUM_PHASES) == 3, f"Expected 3 phases, got {len(CURRICULUM_PHASES)}"\n',
                    '    assert total_pairs == 15000, f"Expected 15K pairs, got {total_pairs:,}"\n',
                    '\n'
                ]

                new_source = source[:insert_idx] + verification_code + source[insert_idx:]
                cell['source'] = new_source
                nb['cells'][cell_idx] = cell
                print("  [OK] Added curriculum verification")
                changes_made.append("Added curriculum verification (Cell 12)")
            else:
                print("  [WARN] Could not find insertion point")

        break

# Save notebook
print(f"\nSaving updated notebook...")
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print("="*80)
print(f"[SUCCESS] Applied {len(changes_made)} changes!")
print("="*80)
for change in changes_made:
    print(f"  [OK] {change}")

print()
print("Next steps:")
print("  1. Open model_promax_mpnet_lorapeft_v3.ipynb")
print("  2. Restart kernel")
print("  3. Run all cells")
print("  4. Watch for verification output:")
print("     - 'TRAINING CONFIGURATION VERIFICATION' with LR=5e-6")
print("     - 'Curriculum Verification' with 3 phases x 5K pairs")
print("  5. After training, run evaluate_model_v2.ipynb")
print()
print(f"Backup saved to: {BACKUP_PATH}")
print("="*80)
