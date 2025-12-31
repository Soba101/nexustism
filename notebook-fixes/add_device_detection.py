#!/usr/bin/env python3
"""
Add DEVICE detection cell to the notebook before model initialization.
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell that initializes the model
    model_init_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('source'):
            source_text = ''.join(cell['source'])
            if 'model = init_model_with_lora(CONFIG, DEVICE)' in source_text:
                model_init_idx = i
                break

    if model_init_idx is None:
        print("ERROR: Could not find model initialization cell")
        return

    # Check if device detection already exists before this cell
    for i in range(max(0, model_init_idx - 5), model_init_idx):
        cell = nb['cells'][i]
        if cell['cell_type'] == 'code' and cell.get('source'):
            source_text = ''.join(cell['source'])
            if 'DEVICE = ' in source_text and 'torch.cuda.is_available()' in source_text:
                print("Device detection already exists - no changes needed")
                return

    # Create device detection cell
    device_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "device_detection",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ========================================\n",
            "# Device Detection (CUDA/MPS/CPU)\n",
            "# ========================================\n",
            "\n",
            "import torch\n",
            "\n",
            "# Auto-detect device\n",
            "if torch.cuda.is_available():\n",
            "    DEVICE = 'cuda'\n",
            "    log(f\"[OK] Using CUDA: {torch.cuda.get_device_name(0)}\")\n",
            "    log(f\"   CUDA version: {torch.version.cuda}\")\n",
            "    log(f\"   Device capability: {torch.cuda.get_device_capability(0)}\")\n",
            "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
            "    DEVICE = 'mps'\n",
            "    log(\"[OK] Using MPS (Apple Silicon)\")\n",
            "else:\n",
            "    DEVICE = 'cpu'\n",
            "    log(\"[OK] Using CPU\")\n",
            "\n",
            "log(f\"\\nDevice set to: {DEVICE}\")\n"
        ]
    }

    # Insert before model initialization
    print(f"Inserting device detection cell before cell {model_init_idx}")
    nb['cells'].insert(model_init_idx, device_cell)

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup4')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Added device detection cell")
    print("="*70)
    print(f"\nInserted at position: {model_init_idx}")
    print("\nThe cell will:")
    print("  1. Auto-detect CUDA/MPS/CPU")
    print("  2. Set DEVICE variable")
    print("  3. Log device info")
    print("\nRestart kernel and run all cells from the beginning!")
    print("="*70)

if __name__ == '__main__':
    main()
