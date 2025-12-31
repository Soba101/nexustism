#!/usr/bin/env python3
"""
Verify all GPU optimization fixes are applied to v3 notebook.
"""
import json
import re

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("VERIFICATION: GPU Optimization Fixes")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

all_checks_pass = True

# Check 1: Device detection uses strings (Cell 4)
print("\n[CHECK 1] Device Detection (Cell 4)")
cell4_source = ''.join(nb['cells'][4]['source'])

if "DEVICE = 'cuda'" in cell4_source:
    print("  [PASS] DEVICE = 'cuda' (string)")
elif "DEVICE = torch.device('cuda')" in cell4_source:
    print("  [FAIL] Still using torch.device('cuda') object")
    all_checks_pass = False
else:
    print("  [WARN] Device assignment not found")
    all_checks_pass = False

# Check 2: Batch size optimized (Cell 6)
print("\n[CHECK 2] Batch Size Optimization (Cell 6)")
cell6_source = ''.join(nb['cells'][6]['source'])
batch_match = re.search(r"'batch_size':\s*(\d+)", cell6_source)

if batch_match:
    batch_size = int(batch_match.group(1))
    if batch_size >= 64:
        print(f"  [PASS] batch_size = {batch_size} (optimized for RTX 5090)")
    elif batch_size >= 32:
        print(f"  [PASS] batch_size = {batch_size} (good for most GPUs)")
    else:
        print(f"  [WARN] batch_size = {batch_size} (conservative, could be higher)")
else:
    print("  [FAIL] batch_size not found in CONFIG")
    all_checks_pass = False

# Check 3: PEFT device placement (Cell 14)
print("\n[CHECK 3] PEFT Device Placement (Cell 14)")
cell14_source = ''.join(nb['cells'][14]['source'])

has_peft = 'get_peft_model(' in cell14_source
has_to_device = 'peft_model.to(device)' in cell14_source or 'peft_model = peft_model.to(device)' in cell14_source

if has_peft and has_to_device:
    print("  [PASS] PEFT model explicitly moved to device")
elif has_peft and not has_to_device:
    print("  [FAIL] PEFT model created but not moved to device")
    all_checks_pass = False
else:
    print("  [WARN] PEFT model creation not found")

# Check 4: Pin memory logic (any cell with DataLoader)
print("\n[CHECK 4] DataLoader pin_memory Setting")
found_pin_memory = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'DataLoader(' in source and 'pin_memory' in source:
            if "pin_memory=(DEVICE in ['cuda', 'mps'])" in source or \
               "pin_memory = (DEVICE in ['cuda', 'mps'])" in source:
                print(f"  [PASS] Cell {i}: pin_memory logic correct (works with string DEVICE)")
                found_pin_memory = True
                break

if not found_pin_memory:
    print("  [INFO] pin_memory setting not found (may be using defaults)")

# Check 5: AMP setting
print("\n[CHECK 5] Mixed Precision (use_amp) Setting")
found_amp = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "use_amp = DEVICE != 'cuda'" in source or \
           "use_amp = (DEVICE != 'cuda')" in source:
            print(f"  [PASS] Cell {i}: use_amp logic correct (False on CUDA)")
            found_amp = True
            break

if not found_amp:
    print("  [INFO] use_amp setting not found or using different logic")

# Summary
print("\n" + "="*80)
if all_checks_pass:
    print("[SUCCESS] All critical GPU fixes verified!")
    print("="*80)
    print("\nNotebook is ready for GPU-accelerated training.")
    print("\nNext steps:")
    print("  1. Restart Jupyter kernel")
    print("  2. Run cells 0-16")
    print("  3. Monitor Task Manager:")
    print("     - GPU Utilization: Should reach 90-98%")
    print("     - GPU Memory: Should use 12-16GB / 32GB")
    print("     - Training: 30-45 min per phase (vs 2-4 hours on CPU)")
    print("\nExpected speedup: 4-6x faster! 🚀")
else:
    print("[WARNING] Some checks did not pass")
    print("="*80)
    print("\nRun: python fix_gpu_utilization.py")

print("="*80)
