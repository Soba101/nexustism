#!/usr/bin/env python3
"""Fix evaluate_model_v2.ipynb to match correct JSON format."""

import json

# Read notebook
with open('evaluate_model_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Section 4 data loading
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'test_pairs_path = DATA_DIR' in ''.join(cell['source']):
        nb['cells'][i]['source'] = [
            "# Load test pairs\n",
            "test_pairs_path = DATA_DIR / TEST_PAIRS_FILE\n",
            "print(f'Loading: {test_pairs_path}')\n",
            "\n",
            "with open(test_pairs_path, 'r', encoding='utf-8') as f:\n",
            "    test_pairs_data = json.load(f)\n",
            "\n",
            "# Extract data (format: {texts1: [...], texts2: [...], labels: [...]})\n",
            "test_texts1 = test_pairs_data['texts1']\n",
            "test_texts2 = test_pairs_data['texts2']\n",
            "test_labels = np.array(test_pairs_data['labels'])\n",
            "\n",
            "# Check if category data available\n",
            "has_categories = 'categories1' in test_pairs_data\n",
            "if has_categories:\n",
            "    print(f'  Category metadata found - adversarial diagnostic enabled')\n",
            "else:\n",
            "    print(f'  No category metadata - adversarial diagnostic will be skipped')\n",
            "\n",
            "# Validate\n",
            "validate_test_pairs(test_texts1, test_texts2, test_labels)\n"
        ]
        print(f"Fixed Section 4 at cell {i}")
        break

# Fix adversarial diagnostic
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'ADVERSARIAL DIAGNOSTIC TEST' in ''.join(cell['source']):
        nb['cells'][i]['source'] = open('adversarial_cell.txt', 'r', encoding='utf-8').read()
        print(f"Fixed adversarial diagnostic at cell {i}")
        break

# Save
with open('evaluate_model_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nUpdated notebook")
print(f"Total cells: {len(nb['cells'])}")
