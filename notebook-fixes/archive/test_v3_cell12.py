#!/usr/bin/env python3
"""
Test if Cell 12 (curriculum loading) would execute properly.
"""
import json

print("="*80)
print("TESTING V3 CELL 12 EXECUTION")
print("="*80)

# Simulate CONFIG
CONFIG = {
    'use_pre_generated_pairs': True,
    'use_curriculum': True,
    'train_pairs_path': 'data_new/curriculum_training_pairs_complete.json'
}

print(f"\nCONFIG settings:")
print(f"  use_pre_generated_pairs: {CONFIG.get('use_pre_generated_pairs', False)}")
print(f"  use_curriculum: {CONFIG.get('use_curriculum', False)}")
print(f"  train_pairs_path: {CONFIG['train_pairs_path']}")

# Check if Cell 12 would execute its main branch
if CONFIG.get('use_pre_generated_pairs', False):
    print(f"\n[OK] Would enter PRE-GENERATED PAIRS branch")

    if CONFIG.get('use_curriculum', False):
        print(f"[OK] Would load curriculum phases")

        # Check file
        pairs_path = CONFIG['train_pairs_path']
        try:
            with open(pairs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"\n[OK] File loaded successfully")
            print(f"  Keys: {list(data.keys())}")
            print(f"  Size: {len(data.get('texts1', []))} pairs")
            print(f"  Has phase_indicators: {'phase_indicators' in data}")

            if 'phase_indicators' in data:
                from collections import Counter
                phase_counts = Counter(data['phase_indicators'])
                print(f"\n  Phase distribution:")
                for phase in sorted(phase_counts.keys()):
                    print(f"    Phase {phase}: {phase_counts[phase]:,} pairs")

        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
        except Exception as e:
            print(f"\n[ERROR] loading file: {e}")
    else:
        print(f"[SKIP] Would NOT load curriculum (use_curriculum=False)")
else:
    print(f"\n[SKIP] Would NOT enter pre-generated pairs branch")
    print(f"  Would use LEGACY MODE (generate on-the-fly)")

print("\n" + "="*80)
print("If this shows [OK] markers, Cell 12 should work correctly")
print("="*80)
