#!/usr/bin/env python3
"""
Fix GPU utilization issues in v3 notebook.

Fixes:
1. Change DEVICE from torch.device object to string
2. Add explicit PEFT model device placement
3. Optimize batch_size for RTX 5090 (32GB VRAM)
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING GPU UTILIZATION ISSUES")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_made = []

# Fix 1: Change device detection to use strings (Cell 4 or similar)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find device detection cell
        if 'torch.device(' in source and 'torch.cuda.is_available()' in source:
            print(f"\n[FIX 1] Found device detection in Cell {cell_idx}")

            new_lines = []
            for line in cell['source']:
                # Replace torch.device('cuda') with 'cuda'
                if "DEVICE = torch.device('cuda')" in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + "DEVICE = 'cuda'\n")
                    changes_made.append(f"Cell {cell_idx}: Changed torch.device('cuda') to 'cuda'")

                elif "DEVICE = torch.device('mps')" in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + "DEVICE = 'mps'\n")
                    changes_made.append(f"Cell {cell_idx}: Changed torch.device('mps') to 'mps'")

                elif "DEVICE = torch.device('cpu')" in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + "DEVICE = 'cpu'\n")
                    changes_made.append(f"Cell {cell_idx}: Changed torch.device('cpu') to 'cpu'")

                else:
                    new_lines.append(line)

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell

# Fix 2: Add PEFT device placement (Cell 14 or init_model_with_lora)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find PEFT model creation
        if 'get_peft_model(' in source and 'lora_config' in source:
            print(f"\n[FIX 2] Found PEFT initialization in Cell {cell_idx}")

            new_lines = []
            i = 0
            while i < len(cell['source']):
                line = cell['source'][i]
                new_lines.append(line)

                # After "peft_model = get_peft_model(...)", add device placement
                if 'peft_model = get_peft_model(' in line:
                    # Find the matching closing parenthesis
                    j = i
                    paren_count = line.count('(') - line.count(')')
                    while paren_count > 0 and j < len(cell['source']) - 1:
                        j += 1
                        paren_count += cell['source'][j].count('(') - cell['source'][j].count(')')
                        new_lines.append(cell['source'][j])

                    # Check if .to(device) already exists in next few lines
                    has_to_device = False
                    for k in range(j+1, min(j+5, len(cell['source']))):
                        if 'peft_model.to(' in cell['source'][k] or 'peft_model = peft_model.to(' in cell['source'][k]:
                            has_to_device = True
                            break

                    if not has_to_device:
                        # Add device placement
                        indent = len(line) - len(line.lstrip())
                        new_lines.append('\n')
                        new_lines.append(' ' * indent + '# Explicitly move PEFT model to device\n')
                        new_lines.append(' ' * indent + 'log(f"[GPU] Moving PEFT model to {device}")\n')
                        new_lines.append(' ' * indent + 'peft_model = peft_model.to(device)\n')
                        changes_made.append(f"Cell {cell_idx}: Added explicit PEFT device placement")

                    i = j + 1
                    continue

                i += 1

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell

# Fix 3: Increase batch_size for RTX 5090 (Cell 6 or CONFIG)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find CONFIG cell
        if "'batch_size':" in source and "'model_name':" in source:
            print(f"\n[FIX 3] Found CONFIG in Cell {cell_idx}")

            new_lines = []
            for line in cell['source']:
                # Change batch_size from 16 to 64
                if "'batch_size': 16" in line or "'batch_size':16" in line:
                    new_line = line.replace("'batch_size': 16", "'batch_size': 64")
                    new_line = new_line.replace("'batch_size':16", "'batch_size': 64")
                    new_lines.append(new_line)
                    changes_made.append(f"Cell {cell_idx}: Increased batch_size from 16 to 64 for RTX 5090")
                else:
                    new_lines.append(line)

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
if changes_made:
    print(f"[SUCCESS] Applied {len(changes_made)} fixes!")
    print(f"{'='*80}")
    for change in changes_made:
        print(f"  [OK] {change}")
    print(f"\n{'='*80}")
    print("GPU optimization complete!")
    print("\nNext steps:")
    print("  1. Wait for current training to finish")
    print("  2. Restart Jupyter kernel")
    print("  3. Re-run training with GPU properly utilized")
    print(f"{'='*80}")
else:
    print("[INFO] No changes made - fixes may already be applied")
    print(f"{'='*80}")
