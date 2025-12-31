#!/usr/bin/env python3
"""
Update evaluate_model_v2.ipynb to include the latest trained model
"""

import json

NOTEBOOK_PATH = 'evaluate_model_v2.ipynb'

print("Updating evaluation notebook with latest model...")

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the configuration cell with FINETUNED_MODELS
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'FINETUNED_MODELS = [' in source:
            print(f"\nFound FINETUNED_MODELS in cell {cell_idx}")

            # Replace the list
            new_source = []
            in_finetuned_models = False

            for line in cell['source']:
                if 'FINETUNED_MODELS = [' in line:
                    in_finetuned_models = True
                    new_source.append("FINETUNED_MODELS = [\n")
                    new_source.append("    'v6_refactored_finetuned/v6_refactored_finetuned_20251204_1424',\n")
                    new_source.append("    'real_servicenow_finetuned_mpnet/real_servicenow_v2_20251210_1939',\n")
                    new_source.append("    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1637',\n")
                    new_source.append("    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251227_0214',  # Latest with MatryoshkaLoss + MNRL\n")
                    continue

                if in_finetuned_models and ']' in line:
                    in_finetuned_models = False
                    new_source.append("]\n")
                    continue

                if not in_finetuned_models:
                    new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            print("  Updated FINETUNED_MODELS list")
            print("  Added: real_servicenow_v2_20251227_0214")
            break

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Updated {NOTEBOOK_PATH}")
print("\nNext step: Run the evaluation notebook to get results for your latest model")
print("  jupyter notebook evaluate_model_v2.ipynb")
