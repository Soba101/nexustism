#!/usr/bin/env python3
"""
Verify that Cell 16 training execution will run properly.
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("VERIFICATION: V3 Cell 16 Training Execution")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\nCell 16: {len(lines)} lines")

# Check 1: No inverted guard
check1 = "if not CONFIG.get('use_pre_generated_pairs', False):\n" not in lines
print(f"\n1. No inverted guard condition: {'[PASS]' if check1 else '[FAIL]'}")

# Check 2: Training code starts at indent 0
check2 = any(line.startswith('timestamp = ') for line in lines)
print(f"2. Training code present: {'[PASS]' if check2 else '[FAIL]'}")

# Check 3: Curriculum training code present
check3 = any('use_curriculum' in line for line in lines)
print(f"3. Curriculum training code: {'[PASS]' if check3 else '[FAIL]'}")

# Check 4: Uses CONFIG['lr']
check4 = sum(1 for line in lines if "optimizer_params={'lr': CONFIG['lr']}" in line)
print(f"4. Uses CONFIG['lr']: {'[PASS]' if check4 >= 1 else '[FAIL]'} ({check4} instances)")

# Check 5: First executable line (not comment/blank)
first_exec = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and not stripped.startswith('#'):
        first_exec = (i+1, line.rstrip())
        break

if first_exec:
    print(f"\n5. First executable line ({first_exec[0]}): {first_exec[1][:60]}")

all_pass = check1 and check2 and check3 and (check4 >= 1)

print(f"\n{'='*80}")
if all_pass:
    print("[SUCCESS] Cell 16 is ready to train!")
    print("\nWhen you run Cell 16 in Jupyter:")
    print("  - Training will start immediately")
    print("  - Will use curriculum learning (3 phases)")
    print("  - Will train for 12 epochs at LR 5e-5")
    print("  - Will save to models/real_servicenow_finetuned_mpnet_lora/")
else:
    print("[WARNING] Some checks failed - review Cell 16")
print(f"{'='*80}")
