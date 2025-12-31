#!/usr/bin/env python3
"""
Update V2 notebook to use progressive epochs (more time on harder phases).
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft-v2.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Cell 3 (Config)
    cell3 = nb['cells'][3]
    source3 = ''.join(cell3['source'])

    # Replace epochs_per_phase with individual phase epochs
    old_config = "    'epochs_per_phase': 2,"
    new_config = """    # Progressive epochs - spend more time on harder phases
    'phase1_epochs': 4,  # Easy pairs
    'phase2_epochs': 5,  # Medium pairs
    'phase3_epochs': 6,  # Hard pairs (most important!)"""

    if old_config in source3:
        source3 = source3.replace(old_config, new_config)
        cell3['source'] = [source3]
        nb['cells'][3] = cell3
        print("Updated Cell 3: Added progressive epochs to CONFIG")
    else:
        print("Cell 3: Pattern not found, checking alternative...")
        # Try alternative pattern
        alt_old = "    'epochs_per_phase': 4,"
        if alt_old in source3:
            source3 = source3.replace(alt_old, new_config)
            cell3['source'] = [source3]
            nb['cells'][3] = cell3
            print("Updated Cell 3: Added progressive epochs to CONFIG (alternative)")

    # Update Cell 12 (Training loop)
    cell12 = nb['cells'][12]
    source12 = ''.join(cell12['source'])

    # Find and replace the phases definition
    old_phases = """# Curriculum training: 3 phases
phases = [
    ('Phase 1: Easy', CURRICULUM_PHASES['phase1']),
    ('Phase 2: Medium', CURRICULUM_PHASES['phase2']),
    ('Phase 3: Hard', CURRICULUM_PHASES['phase3'])
]"""

    new_phases = """# Curriculum training: 3 phases with progressive epochs
phases = [
    ('Phase 1: Easy', CURRICULUM_PHASES['phase1'], CONFIG['phase1_epochs']),
    ('Phase 2: Medium', CURRICULUM_PHASES['phase2'], CONFIG['phase2_epochs']),
    ('Phase 3: Hard', CURRICULUM_PHASES['phase3'], CONFIG['phase3_epochs'])
]"""

    if old_phases in source12:
        source12 = source12.replace(old_phases, new_phases)

        # Update the loop signature
        old_loop = "for phase_num, (phase_name, phase_data) in enumerate(phases, 1):"
        new_loop = "for phase_num, (phase_name, phase_data, phase_epochs) in enumerate(phases, 1):"
        source12 = source12.replace(old_loop, new_loop)

        # Update log message
        old_log = '    log(f"Epochs: {CONFIG[\'epochs_per_phase\']}")'
        new_log = '    log(f"Epochs: {phase_epochs}")'
        source12 = source12.replace(old_log, new_log)

        # Update model.fit epochs parameter
        old_fit_epochs = "        epochs=CONFIG['epochs_per_phase'],"
        new_fit_epochs = "        epochs=phase_epochs,"
        source12 = source12.replace(old_fit_epochs, new_fit_epochs)

        # Add weight decay to optimizer_params
        old_optimizer = "        optimizer_params={'lr': CONFIG['lr']}"
        new_optimizer = "        optimizer_params={'lr': CONFIG['lr'], 'weight_decay': 0.01}"
        source12 = source12.replace(old_optimizer, new_optimizer)

        cell12['source'] = [source12]
        nb['cells'][12] = cell12
        print("Updated Cell 12: Progressive epochs + weight decay")
    else:
        print("Cell 12: Pattern not found, may already be updated")

    # Save
    print(f"\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Updated V2 notebook with progressive epochs")
    print("="*70)
    print("\nChanges made:")
    print("  Cell 3 (Config):")
    print("    - phase1_epochs: 4 (was epochs_per_phase: 2)")
    print("    - phase2_epochs: 5")
    print("    - phase3_epochs: 6")
    print("\n  Cell 12 (Training):")
    print("    - Uses phase-specific epochs")
    print("    - Added weight_decay: 0.01")
    print("\nTotal epochs: 4 + 5 + 6 = 15 (was 6)")
    print("Training time: ~17-20 minutes (was ~4.5 minutes)")
    print("="*70)

if __name__ == '__main__':
    main()
