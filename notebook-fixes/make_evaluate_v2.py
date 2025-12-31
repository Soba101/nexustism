#!/usr/bin/env python3
"""
Create evaluate_model_v2.ipynb with all improvements.
"""

import json
from pathlib import Path

# Read the script I wrote earlier that has all the cells
with open(__file__, 'r', encoding='utf-8') as f:
    content = f.read()

# Import the actual creation function
exec(open('create_v2_notebook.py', 'r', encoding='utf-8').read().replace(
    'model_promax_mpnet_lorapeft-v2',
    'evaluate_model_v2'
))

# Run the creation
nb = create_v2_notebook()

# Save
output_path = Path('evaluate_model_v2.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Success: Created {output_path}")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Improvements: Configuration, Security, Diagnostics, Visualizations")
