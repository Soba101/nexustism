#!/usr/bin/env python3
"""
Add guard to skip borderline evaluation when borderline_examples is empty.
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find borderline evaluation cell
    borderline_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('source'):
            source_text = ''.join(cell['source'])
            if 'borderline_results = comprehensive_eval(borderline_examples' in source_text:
                borderline_idx = i
                break

    if borderline_idx is None:
        print("Could not find borderline evaluation cell - may already be fixed")
        return

    cell = nb['cells'][borderline_idx]
    source = cell['source']

    # Check if guard already exists
    source_text = ''.join(source)
    if 'if borderline_examples' in source_text or 'if len(borderline_examples)' in source_text:
        print("Guard already exists - no changes needed")
        return

    # Wrap in conditional
    new_source = [
        "# Skip borderline evaluation if not using legacy mode\n",
        "if borderline_examples and len(borderline_examples) > 0:\n",
    ]

    # Indent all existing lines
    for line in source:
        if line.strip():
            new_source.append("    " + line)
        else:
            new_source.append(line)

    # Add else clause
    new_source.extend([
        "else:\n",
        "    log(\"\\n\" + \"=\"*60)\n",
        "    log(\"[SKIP] Borderline Test (not applicable for curriculum training)\")\n",
        "    log(\"=\"*60)\n",
        "    log(\"Using curriculum learning - borderline test not generated.\")\n",
        "    log(\"Evaluation uses fixed_test_pairs.json instead.\")\n"
    ])

    cell['source'] = new_source
    nb['cells'][borderline_idx] = cell

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup6')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Added borderline evaluation guard")
    print("="*70)
    print("\nBorderline test will be skipped when using curriculum training")
    print("(since borderline_examples is empty for curriculum mode)")
    print("="*70)

if __name__ == '__main__':
    main()
