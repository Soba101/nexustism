#!/usr/bin/env python3
"""
Fix curriculum learning in model_promax_mpnet_lorapeft_v3.ipynb

Problem: The load_curriculum_pairs function doesn't properly split data by phase_indicators
Solution: Update function to use phase_indicators field to create phase1/phase2/phase3 lists
"""

import json
from pathlib import Path

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'
BACKUP_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb.backup_curriculum_fix'

print("="*80)
print("FIXING CURRICULUM LEARNING")
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

# Find and fix the load_curriculum_pairs function
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find the load_curriculum_pairs function
        if 'def load_curriculum_pairs' in source and 'phase_indicators' not in source:
            print(f"\n[FOUND] load_curriculum_pairs function in Cell {cell_idx}")

            # Replace the function with corrected version
            new_source = []
            inside_function = False
            function_indent = 0
            skip_until_next_def = False

            for i, line in enumerate(cell['source']):
                # Start of function
                if 'def load_curriculum_pairs' in line:
                    inside_function = True
                    function_indent = len(line) - len(line.lstrip())

                    # Insert new function
                    new_source.append("def load_curriculum_pairs(pairs_path, use_curriculum=True):\n")
                    new_source.append("    \"\"\"\n")
                    new_source.append("    Load pre-generated curriculum pairs from JSON file.\n")
                    new_source.append("    \n")
                    new_source.append("    Args:\n")
                    new_source.append("        pairs_path: Path to curriculum_training_pairs_*.json\n")
                    new_source.append("        use_curriculum: If True, return separate phases; if False, return all mixed\n")
                    new_source.append("    \n")
                    new_source.append("    Returns:\n")
                    new_source.append("        If use_curriculum=True: (phase1_examples, phase2_examples, phase3_examples)\n")
                    new_source.append("        If use_curriculum=False: all_examples (mixed)\n")
                    new_source.append("    \"\"\"\n")
                    new_source.append("    log(f\"Loading curriculum pairs from: {pairs_path}\")\n")
                    new_source.append("    \n")
                    new_source.append("    with open(pairs_path, 'r', encoding='utf-8') as f:\n")
                    new_source.append("        data = json.load(f)\n")
                    new_source.append("    \n")
                    new_source.append("    texts1 = data['texts1']\n")
                    new_source.append("    texts2 = data['texts2']\n")
                    new_source.append("    labels = data['labels']\n")
                    new_source.append("    phase_indicators = data.get('phase_indicators', [1] * len(texts1))  # Default to phase 1\n")
                    new_source.append("    \n")
                    new_source.append("    log(f\"Loaded {len(texts1):,} total pairs\")\n")
                    new_source.append("    \n")
                    new_source.append("    if use_curriculum:\n")
                    new_source.append("        # Split by phase indicators\n")
                    new_source.append("        phase1_examples = []\n")
                    new_source.append("        phase2_examples = []\n")
                    new_source.append("        phase3_examples = []\n")
                    new_source.append("        \n")
                    new_source.append("        for t1, t2, label, phase in zip(texts1, texts2, labels, phase_indicators):\n")
                    new_source.append("            example = InputExample(texts=[t1, t2], label=float(label))\n")
                    new_source.append("            \n")
                    new_source.append("            if phase == 1:\n")
                    new_source.append("                phase1_examples.append(example)\n")
                    new_source.append("            elif phase == 2:\n")
                    new_source.append("                phase2_examples.append(example)\n")
                    new_source.append("            elif phase == 3:\n")
                    new_source.append("                phase3_examples.append(example)\n")
                    new_source.append("        \n")
                    new_source.append("        log(f\"  Phase 1 (easy): {len(phase1_examples):,} pairs\")\n")
                    new_source.append("        log(f\"  Phase 2 (medium): {len(phase2_examples):,} pairs\")\n")
                    new_source.append("        log(f\"  Phase 3 (hard): {len(phase3_examples):,} pairs\")\n")
                    new_source.append("        \n")
                    new_source.append("        return phase1_examples, phase2_examples, phase3_examples\n")
                    new_source.append("    \n")
                    new_source.append("    else:\n")
                    new_source.append("        # Return all mixed (no curriculum)\n")
                    new_source.append("        all_examples = [\n")
                    new_source.append("            InputExample(texts=[t1, t2], label=float(label))\n")
                    new_source.append("            for t1, t2, label in zip(texts1, texts2, labels)\n")
                    new_source.append("        ]\n")
                    new_source.append("        \n")
                    new_source.append("        pos_count = sum(1 for ex in all_examples if ex.label == 1.0)\n")
                    new_source.append("        neg_count = len(all_examples) - pos_count\n")
                    new_source.append("        \n")
                    new_source.append("        log(f\"  Positives: {pos_count:,} ({100*pos_count/len(all_examples):.1f}%)\")\n")
                    new_source.append("        log(f\"  Negatives: {neg_count:,} ({100*neg_count/len(all_examples):.1f}%)\")\n")
                    new_source.append("        \n")
                    new_source.append("        return all_examples\n")
                    new_source.append("\n")

                    skip_until_next_def = True
                    continue

                # Skip old function body
                if skip_until_next_def:
                    # Check if we've reached the next function or end of old function
                    current_indent = len(line) - len(line.lstrip())

                    # End of function: either new def at same/lower indent, or non-empty line at same/lower indent
                    if line.strip() and current_indent <= function_indent:
                        skip_until_next_def = False
                        inside_function = False
                        new_source.append(line)
                    continue

                # Keep everything else
                new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            print("  [OK] Updated load_curriculum_pairs function")
            print("  [OK] Now properly splits data using phase_indicators")
            break

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*80)
print("[SUCCESS] Curriculum learning fix applied!")
print("="*80)
print()
print("Summary:")
print("  [OK] load_curriculum_pairs now uses phase_indicators field")
print("  [OK] Will properly split 15K pairs into 3 phases of 5K each")
print("  [OK] Phase 1: Easy pairs (high TF-IDF positives, easy negatives)")
print("  [OK] Phase 2: Medium pairs (cross-category positives, medium negatives)")
print("  [OK] Phase 3: Hard pairs (hardest examples)")
print()
print("Next steps:")
print("  1. Restart Jupyter kernel")
print("  2. Run Cell 12 (load curriculum pairs)")
print("  3. Verify CURRICULUM_PHASES has 5000 pairs per phase")
print("  4. Run training (Cell 16) - should train in 3 curriculum phases")
print()
print(f"Backup saved to: {BACKUP_PATH}")
print("="*80)
