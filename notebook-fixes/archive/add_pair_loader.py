#!/usr/bin/env python3
"""
Add pre-generated pair loading functionality to the LoRA notebook.
Inserts new cells right before the pair generation section.
"""

import json
import sys
from pathlib import Path

def create_pair_loader_cell():
    """Create the pair loader cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ========================================\n",
            "# LOAD PRE-GENERATED CURRICULUM PAIRS\n",
            "# ========================================\n",
            "\n",
            "import json\n",
            "from sentence_transformers import InputExample\n",
            "\n",
            "def load_curriculum_pairs(pairs_path, use_curriculum=True):\n",
            "    \"\"\"\n",
            "    Load pre-generated curriculum pairs from JSON file.\n",
            "    \n",
            "    Args:\n",
            "        pairs_path: Path to curriculum_training_pairs_*.json\n",
            "        use_curriculum: If True, return separate phases; if False, return all mixed\n",
            "    \n",
            "    Returns:\n",
            "        If use_curriculum=True: (phase1_pairs, phase2_pairs, phase3_pairs)\n",
            "        If use_curriculum=False: all_pairs (mixed)\n",
            "    \"\"\"\n",
            "    log(f\"\\nLoading curriculum pairs from: {pairs_path}\")\n",
            "    \n",
            "    with open(pairs_path, 'r', encoding='utf-8') as f:\n",
            "        data = json.load(f)\n",
            "    \n",
            "    texts1 = data['texts1']\n",
            "    texts2 = data['texts2']\n",
            "    labels = data['labels']\n",
            "    phase_indicators = data.get('phase_indicators', [1] * len(labels))\n",
            "    \n",
            "    log(f\"Loaded {len(labels):,} total pairs\")\n",
            "    log(f\"  Positives: {sum(labels):,} ({100*sum(labels)/len(labels):.1f}%)\")\n",
            "    log(f\"  Negatives: {len(labels) - sum(labels):,} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)\")\n",
            "    \n",
            "    # Show metadata\n",
            "    metadata = data.get('metadata', {})\n",
            "    if metadata:\n",
            "        log(f\"\\nCurriculum phases: {metadata.get('curriculum_phases', 'N/A')}\")\n",
            "        for phase_num in [1, 2, 3]:\n",
            "            phase_key = f'phase{phase_num}_config'\n",
            "            if phase_key in metadata:\n",
            "                phase_cfg = metadata[phase_key]\n",
            "                log(f\"  Phase {phase_num} ({phase_cfg.get('difficulty', 'N/A')}): \"\n",
            "                    f\"{phase_cfg.get('pairs', 0):,} pairs, \"\n",
            "                    f\"pos>={phase_cfg.get('pos_threshold', 'N/A')}, \"\n",
            "                    f\"neg<={phase_cfg.get('neg_threshold', 'N/A')}\")\n",
            "    \n",
            "    # Convert to InputExample format\n",
            "    if use_curriculum:\n",
            "        # Separate by phase\n",
            "        phase1_pairs = []\n",
            "        phase2_pairs = []\n",
            "        phase3_pairs = []\n",
            "        \n",
            "        for i in range(len(labels)):\n",
            "            example = InputExample(texts=[texts1[i], texts2[i]], label=float(labels[i]))\n",
            "            phase = phase_indicators[i]\n",
            "            if phase == 1:\n",
            "                phase1_pairs.append(example)\n",
            "            elif phase == 2:\n",
            "                phase2_pairs.append(example)\n",
            "            elif phase == 3:\n",
            "                phase3_pairs.append(example)\n",
            "        \n",
            "        log(f\"\\nSeparated into phases:\")\n",
            "        log(f\"  Phase 1: {len(phase1_pairs):,} pairs\")\n",
            "        log(f\"  Phase 2: {len(phase2_pairs):,} pairs\")\n",
            "        log(f\"  Phase 3: {len(phase3_pairs):,} pairs\")\n",
            "        \n",
            "        return phase1_pairs, phase2_pairs, phase3_pairs\n",
            "    else:\n",
            "        # Return all mixed\n",
            "        all_pairs = [\n",
            "            InputExample(texts=[texts1[i], texts2[i]], label=float(labels[i]))\n",
            "            for i in range(len(labels))\n",
            "        ]\n",
            "        log(f\"Returning {len(all_pairs):,} mixed pairs\")\n",
            "        return all_pairs\n",
            "\n",
            "# ========================================\n",
            "# LOAD OR GENERATE PAIRS\n",
            "# ========================================\n",
            "\n",
            "if CONFIG.get('use_pre_generated_pairs', False):\n",
            "    log(\"=\"*70)\n",
            "    log(\"USING PRE-GENERATED CURRICULUM PAIRS\")\n",
            "    log(\"=\"*70)\n",
            "    \n",
            "    pairs_path = CONFIG['train_pairs_path']\n",
            "    \n",
            "    if CONFIG.get('use_curriculum', False):\n",
            "        # Load phases separately for curriculum training\n",
            "        phase1_train, phase2_train, phase3_train = load_curriculum_pairs(\n",
            "            pairs_path, use_curriculum=True\n",
            "        )\n",
            "        \n",
            "        # For now, combine for evaluation split\n",
            "        # (In production, you'd want separate eval sets per phase)\n",
            "        train_examples = phase1_train + phase2_train + phase3_train\n",
            "        \n",
            "        log(f\"\\nTotal training examples: {len(train_examples):,}\")\n",
            "        log(\"\\nNote: Will train in 3 curriculum phases\")\n",
            "        \n",
            "        # Store phases for later use\n",
            "        CURRICULUM_PHASES = {\n",
            "            'phase1': phase1_train,\n",
            "            'phase2': phase2_train,\n",
            "            'phase3': phase3_train\n",
            "        }\n",
            "    else:\n",
            "        # Load all mixed\n",
            "        train_examples = load_curriculum_pairs(pairs_path, use_curriculum=False)\n",
            "        CURRICULUM_PHASES = None\n",
            "    \n",
            "    # Skip the pair generation cells below\n",
            "    SKIP_PAIR_GENERATION = True\n",
            "    \n",
            "else:\n",
            "    log(\"=\"*70)\n",
            "    log(\"GENERATING PAIRS ON-THE-FLY (LEGACY MODE)\")\n",
            "    log(\"=\"*70)\n",
            "    log(\"Note: Consider using pre-generated curriculum pairs instead!\")\n",
            "    log(\"      Run fix_train_test_mismatch.ipynb to generate them.\")\n",
            "    \n",
            "    SKIP_PAIR_GENERATION = False\n",
            "    CURRICULUM_PHASES = None\n"
        ]
    }

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        sys.exit(1)

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the pair generation section (usually around cell 11-12)
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if 'Pair Generation' in source or '5. Pair Generation' in source:
                insert_index = i
                print(f"Found pair generation section at cell {i}")
                break

    if insert_index is None:
        print("ERROR: Could not find pair generation section!")
        sys.exit(1)

    # Insert the pair loader cell right before pair generation
    pair_loader_cell = create_pair_loader_cell()
    nb['cells'].insert(insert_index, pair_loader_cell)

    print(f"Inserted pair loader cell at index {insert_index}")

    # Save updated notebook
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\\n" + "="*70)
    print("SUCCESS: Pair loader added to notebook")
    print("="*70)
    print("\\nAdded:")
    print("  - load_curriculum_pairs() function")
    print("  - Automatic loading of pre-generated pairs")
    print("  - Curriculum phase separation")
    print("  - Skip flag for legacy pair generation")
    print("\\nThe notebook will now:")
    print("  1. Load curriculum pairs if use_pre_generated_pairs=True")
    print("  2. Separate into 3 phases if use_curriculum=True")
    print("  3. Skip on-the-fly generation when using pre-generated pairs")
    print("="*70)

if __name__ == '__main__':
    main()
