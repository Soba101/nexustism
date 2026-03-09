#!/usr/bin/env python3
"""
Update pair loader to also load test/eval pairs from fixed_test_pairs.json.
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the pair loader cell
    pair_loader_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('source'):
            source_text = ''.join(cell['source'])
            if 'LOAD PRE-GENERATED CURRICULUM PAIRS' in source_text and 'load_curriculum_pairs' in source_text:
                pair_loader_idx = i
                break

    if pair_loader_idx is None:
        print("ERROR: Could not find pair loader cell")
        return

    cell = nb['cells'][pair_loader_idx]
    source = cell['source']

    # Check if eval pairs loading already exists
    source_text = ''.join(source)
    if 'fixed_test_pairs.json' in source_text and 'eval_examples' in source_text:
        print("Eval pairs loading already exists - no changes needed")
        return

    # Find where to insert eval loading code (after CURRICULUM_PHASES assignment)
    insert_idx = None
    for i, line in enumerate(source):
        if 'CURRICULUM_PHASES = None' in line or 'CURRICULUM_PHASES = {' in line:
            # Find the end of this block (after the closing brace or None assignment)
            if 'CURRICULUM_PHASES = None' in line:
                insert_idx = i + 1
                break
            else:
                # Find closing brace
                for j in range(i, len(source)):
                    if '}' in source[j] and 'phase3' in source[j-1]:
                        insert_idx = j + 1
                        break
                break

    if insert_idx is None:
        print("ERROR: Could not find insertion point")
        return

    # Add eval pairs loading code
    eval_loading_code = [
        "    \n",
        "    # Load test/eval pairs from fixed_test_pairs.json\n",
        "    test_pairs_path = 'data_new/fixed_test_pairs.json'\n",
        "    log(f\"\\nLoading test pairs from: {test_pairs_path}\")\n",
        "    \n",
        "    with open(test_pairs_path, 'r', encoding='utf-8') as f:\n",
        "        test_data = json.load(f)\n",
        "    \n",
        "    # Convert to InputExample\n",
        "    from sentence_transformers import InputExample\n",
        "    eval_examples = [\n",
        "        InputExample(texts=[t1, t2], label=float(label))\n",
        "        for t1, t2, label in zip(test_data['texts1'], test_data['texts2'], test_data['labels'])\n",
        "    ]\n",
        "    \n",
        "    # For curriculum training, we don't have separate holdout/borderline\n",
        "    # Use eval_examples for all evaluation metrics\n",
        "    holdout_examples = eval_examples  # Reuse for holdout metrics\n",
        "    borderline_examples = []  # Empty - not applicable for curriculum\n",
        "    \n",
        "    log(f\"[OK] Loaded {len(eval_examples):,} test pairs for evaluation\")\n",
        "    pos_count = sum(1 for ex in eval_examples if ex.label == 1.0)\n",
        "    log(f\"   Positives: {pos_count:,} ({100*pos_count/len(eval_examples):.1f}%)\")\n",
        "    log(f\"   Negatives: {len(eval_examples)-pos_count:,} ({100*(len(eval_examples)-pos_count)/len(eval_examples):.1f}%)\")\n",
        "    \n"
    ]

    # Insert the code
    source = source[:insert_idx] + eval_loading_code + source[insert_idx:]
    cell['source'] = source

    # Also need to handle the else clause (legacy mode)
    # Find the else clause and add variable initialization there too
    for i, line in enumerate(source):
        if line.strip() == 'else:' and i > 50:  # The else for legacy mode
            # Check if variables are already set
            next_lines = ''.join(source[i:i+10])
            if 'SKIP_PAIR_GENERATION = False' in next_lines and 'eval_examples' not in next_lines:
                # Add variable comment
                insert_at = i + 2  # After "SKIP_PAIR_GENERATION = False"
                source.insert(insert_at, "    # eval_examples, holdout_examples, borderline_examples will be generated below\n")
                break

    cell['source'] = source
    nb['cells'][pair_loader_idx] = cell

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup5')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUCCESS: Added test pairs loading")
    print("="*70)
    print("\nThe pair loader will now:")
    print("  1. Load curriculum training pairs (15K)")
    print("  2. Load test pairs from fixed_test_pairs.json (1K)")
    print("  3. Set eval_examples, holdout_examples, borderline_examples")
    print("\nThis fixes the 'eval_examples is not defined' error!")
    print("="*70)

if __name__ == '__main__':
    main()
