#!/usr/bin/env python3
"""
Add descriptive labels to each cell in the notebook.
"""

import json
from pathlib import Path

def get_cell_description(cell, index):
    """Determine what a cell does based on its content"""
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if source.strip():
            first_line = source.split('\n')[0].strip('# ')
            return f"MARKDOWN: {first_line[:60]}"
        return "MARKDOWN: (empty)"

    if cell['cell_type'] != 'code':
        return f"{cell['cell_type'].upper()}"

    source = ''.join(cell.get('source', []))

    # Identify code cells by their content
    if 'ensure_packages' in source or 'pip install' in source:
        return "CODE: Package Installation & Environment Setup"
    elif 'from sentence_transformers import' in source and 'import torch' in source:
        return "CODE: Core Imports (SentenceTransformers, PyTorch, etc.)"
    elif 'CONFIG = {' in source and 'model_name' in source:
        return "CODE: Configuration Dictionary (Training Parameters)"
    elif 'def log(' in source or 'logging_buffer' in source:
        return "CODE: Logging Utilities"
    elif 'def load_incidents' in source:
        return "CODE: Data Loading Functions"
    elif 'df_incidents = load_incidents' in source:
        return "CODE: Load ServiceNow Incident Data"
    elif 'def split_data' in source:
        return "CODE: Data Splitting Function (Train/Eval/Holdout)"
    elif 'train_df, eval_df, holdout_df = split_data' in source:
        return "CODE: Split Dataset into Train/Eval/Holdout"
    elif 'def load_curriculum_pairs' in source:
        return "CODE: Curriculum Pair Loading Function"
    elif 'LOAD PRE-GENERATED CURRICULUM PAIRS' in source:
        return "CODE: Load Training & Test Pairs (Curriculum + Eval)"
    elif 'TFIDFSimilarityCalculator' in source and 'class' in source:
        return "CODE: TF-IDF Pair Generation Classes (Legacy)"
    elif 'Generate pairs for each split' in source and 'TFIDFSimilarityCalculator(' in source:
        return "CODE: Generate Pairs On-The-Fly (SKIPPED - Legacy Mode)"
    elif 'def init_model_with_lora' in source:
        return "CODE: LoRA Model Initialization Function"
    elif 'Device Detection' in source and 'DEVICE = ' in source:
        return "CODE: Device Detection (CUDA/MPS/CPU)"
    elif 'class HybridLoss' in source or 'class CombinedLoss' in source:
        return "CODE: Custom Loss Functions (MNRL + Cosine)"
    elif 'model = init_model_with_lora(CONFIG, DEVICE)' in source and 'model.fit' in source:
        return "CODE: Model Training with Curriculum Learning"
    elif 'SCORE DISTRIBUTION DIAGNOSTIC' in source:
        return "CODE: Score Distribution Analysis (Post-Training)"
    elif 'def get_cv_threshold' in source:
        return "CODE: Cross-Validation Threshold Calculation"
    elif 'def comprehensive_eval' in source:
        return "CODE: Comprehensive Evaluation Function"
    elif 'eval_results = comprehensive_eval' in source and 'FINAL RESULTS' in source:
        return "CODE: Final Evaluation on All Test Sets"
    elif 'def plot_' in source or 'plt.figure' in source:
        return "CODE: Visualization & Plotting"
    elif 'borderline_results = comprehensive_eval(borderline_examples' in source:
        return "CODE: Borderline Test Evaluation (Legacy Mode Only)"
    elif 'model.save(' in source or 'Save metadata' in source:
        return "CODE: Save Model & Training Metadata"
    else:
        # Generic description based on key functions
        if 'def ' in source:
            return "CODE: Helper Functions"
        elif 'import ' in source:
            return "CODE: Additional Imports"
        else:
            return "CODE: Execution Block"

def add_cell_labels(nb):
    """Add ID and description to each cell's metadata"""
    for i, cell in enumerate(nb['cells']):
        description = get_cell_description(cell, i)

        # Add to metadata
        if 'metadata' not in cell:
            cell['metadata'] = {}

        cell['metadata']['cell_label'] = description
        cell['metadata']['cell_index'] = i

        # For code cells, add a comment at the top if it doesn't exist
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = cell['source']
            first_line = source[0] if source else ''

            # Check if there's already a descriptive comment
            if not first_line.strip().startswith('#'):
                # Add label as comment
                label_comment = f"# CELL {i}: {description.replace('CODE: ', '')}\n"
                # Check if second line is a separator
                if len(source) > 0 and '===' in source[0]:
                    # Insert before separator
                    source.insert(0, label_comment)
                else:
                    source.insert(0, label_comment)
                cell['source'] = source

def print_cell_map(nb):
    """Print a map of all cells"""
    print("\n" + "="*80)
    print("NOTEBOOK CELL MAP")
    print("="*80)
    print(f"\nTotal cells: {len(nb['cells'])}\n")

    for i, cell in enumerate(nb['cells']):
        desc = cell.get('metadata', {}).get('cell_label', 'Unknown')
        cell_type = cell['cell_type']

        if cell_type == 'markdown':
            icon = "📝"
        elif cell_type == 'code':
            icon = "⚙️"
        else:
            icon = "❓"

        print(f"  {icon} Cell {i:2d}: {desc}")

    print("\n" + "="*80)

def main():
    notebook_path = Path('model_promax_mpnet_lorapeft.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Adding labels to {len(nb['cells'])} cells...")
    add_cell_labels(nb)

    # Print the map
    print_cell_map(nb)

    # Backup
    backup_path = notebook_path.with_suffix('.ipynb.backup9')
    print(f"\nBacking up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(notebook_path, 'r', encoding='utf-8')), f, indent=1)

    # Save
    print(f"Writing labeled notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*80)
    print("SUCCESS: Added cell labels")
    print("="*80)
    print("\nEach cell now has:")
    print("  1. A comment at the top describing its purpose")
    print("  2. Metadata with cell_label and cell_index")
    print("\nYou can now easily see what each cell does!")
    print("="*80)

if __name__ == '__main__':
    main()
