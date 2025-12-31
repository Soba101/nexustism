#!/usr/bin/env python3
"""
Add better baseline models to evaluate_model.ipynb.

This adds a new cell after the baseline evaluation that tests:
- Nomic-Embed-v1.5
- BGE-base-en-v1.5
- GTE-base-en-v1.5
- E5-base-v2
- MiniLM-L12-v2
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('evaluate_model.ipynb')

    # Define additional baseline models to test
    additional_baselines = [
        {
            'name': 'Nomic-Embed-v1.5',
            'model_id': 'nomic-ai/nomic-embed-text-v1.5',
            'trust_remote_code': True,
            'install_deps': ['einops'],  # Required dependency
        },
        {
            'name': 'BGE-base-en-v1.5',
            'model_id': 'BAAI/bge-base-en-v1.5',
            'trust_remote_code': False,
            'install_deps': [],
        },
        {
            'name': 'GTE-base-en-v1.5',
            'model_id': 'Alibaba-NLP/gte-base-en-v1.5',
            'trust_remote_code': True,
            'install_deps': [],
        },
        {
            'name': 'E5-base-v2',
            'model_id': 'intfloat/e5-base-v2',
            'trust_remote_code': False,
            'install_deps': [],
        },
        {
            'name': 'MiniLM-L12-v2',
            'model_id': 'sentence-transformers/all-MiniLM-L12-v2',
            'trust_remote_code': False,
            'install_deps': [],
        },
    ]

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell index after baseline evaluation (cell with "Baseline (Raw MPNet)")
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        if 'Baseline (Raw MPNet)' in source and 'Loading baseline model' in source:
            insert_index = i + 1
            break

    if insert_index is None:
        print("ERROR: Could not find baseline evaluation cell")
        return

    print(f"Found baseline cell, will insert new cell at index {insert_index}")

    # Create new cell for evaluating additional baseline models
    new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6b. Evaluate Additional Baseline Models\n",
            "\n",
            "Test modern high-performance models that may outperform MPNet baseline:\n",
            "- **Nomic-Embed-v1.5**: Designed for hard negatives\n",
            "- **BGE-base-en-v1.5**: Strong retrieval model\n",
            "- **GTE-base-en-v1.5**: Top MTEB performer\n",
            "- **E5-base-v2**: Microsoft's text embedding model\n",
            "- **MiniLM-L12-v2**: Fast and efficient baseline"
        ]
    }

    # Build the code cell source dynamically
    models_config_str = "additional_baselines = [\n"
    for config in additional_baselines:
        models_config_str += "    {\n"
        models_config_str += f"        'name': '{config['name']}',\n"
        models_config_str += f"        'model_id': '{config['model_id']}',\n"
        models_config_str += f"        'trust_remote_code': {config['trust_remote_code']},\n"
        deps_repr = repr(config['install_deps'])
        if config['install_deps']:
            models_config_str += f"        'install_deps': {deps_repr},  # Required dependency\n"
        else:
            models_config_str += f"        'install_deps': {deps_repr},\n"
        models_config_str += "    },\n"
    models_config_str += "]\n"

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# List of additional baseline models to evaluate\n",
            models_config_str,
            "\n",
            "print(\"=\"*80)\n",
            "print(\"EVALUATING ADDITIONAL BASELINE MODELS\")\n",
            "print(\"=\"*80)\n",
            "print(f\"\\nTesting {len(additional_baselines)} additional models...\\n\")\n",
            "\n",
            "# Track which models succeeded/failed\n",
            "successful_models = []\n",
            "failed_models = []\n",
            "\n",
            "for model_config in additional_baselines:\n",
            "    model_name = model_config['name']\n",
            "    model_id = model_config['model_id']\n",
            "    trust_remote = model_config['trust_remote_code']\n",
            "    deps = model_config['install_deps']\n",
            "    \n",
            "    print(f\"\\n{'-'*80}\")\n",
            "    print(f\"Testing: {model_name}\")\n",
            "    print(f\"Model ID: {model_id}\")\n",
            "    print(f\"-\"*80)\n",
            "    \n",
            "    try:\n",
            "        # Install dependencies if needed\n",
            "        if deps:\n",
            "            print(f\"Installing dependencies: {', '.join(deps)}\")\n",
            "            import subprocess\n",
            "            import sys\n",
            "            for dep in deps:\n",
            "                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', dep])\n",
            "            print(\"  ✓ Dependencies installed\")\n",
            "        \n",
            "        # Load model\n",
            "        print(f\"Loading model...\")\n",
            "        model = SentenceTransformer(\n",
            "            model_id,\n",
            "            device=device,\n",
            "            trust_remote_code=trust_remote\n",
            "        )\n",
            "        print(\"  ✓ Model loaded\")\n",
            "        \n",
            "        # Evaluate\n",
            "        results = evaluator.evaluate_model(\n",
            "            model=model,\n",
            "            model_name=f\"Baseline ({model_name})\",\n",
            "            texts1=test_texts1,\n",
            "            texts2=test_texts2,\n",
            "            labels=test_labels,\n",
            "            verbose=True\n",
            "        )\n",
            "        \n",
            "        successful_models.append(model_name)\n",
            "        \n",
            "        # Compare to MPNet baseline\n",
            "        mpnet_baseline = evaluator.results.get('Baseline (Raw MPNet)', {})\n",
            "        if mpnet_baseline:\n",
            "            mpnet_spearman = mpnet_baseline['spearman']\n",
            "            model_spearman = results['spearman']\n",
            "            delta = model_spearman - mpnet_spearman\n",
            "            pct_change = (delta / mpnet_spearman * 100) if mpnet_spearman else 0\n",
            "            \n",
            "            print(f\"\\n  Comparison to MPNet baseline:\")\n",
            "            print(f\"    MPNet Spearman:  {mpnet_spearman:.4f}\")\n",
            "            print(f\"    {model_name} Spearman:  {model_spearman:.4f}\")\n",
            "            print(f\"    Delta:           {delta:+.4f} ({pct_change:+.1f}%)\")\n",
            "            \n",
            "            if delta > 0.01:\n",
            "                print(f\"    ✓ {model_name} is BETTER than MPNet!\")\n",
            "            elif delta > -0.01:\n",
            "                print(f\"    ≈ {model_name} is similar to MPNet\")\n",
            "            else:\n",
            "                print(f\"    ✗ {model_name} is worse than MPNet\")\n",
            "        \n",
            "        # Clean up\n",
            "        del model\n",
            "        if device == 'cuda':\n",
            "            torch.cuda.empty_cache()\n",
            "        \n",
            "    except Exception as e:\n",
            "        print(f\"\\n  ✗ Error with {model_name}: {e}\")\n",
            "        print(f\"    Skipping this model...\")\n",
            "        failed_models.append((model_name, str(e)))\n",
            "        continue\n",
            "\n",
            "# Summary\n",
            "print(f\"\\n{'='*80}\")\n",
            "print(\"ADDITIONAL BASELINES EVALUATION COMPLETE\")\n",
            "print(f\"{'='*80}\")\n",
            "print(f\"\\nSuccessful: {len(successful_models)}/{len(additional_baselines)}\")\n",
            "if successful_models:\n",
            "    for model in successful_models:\n",
            "        print(f\"  ✓ {model}\")\n",
            "\n",
            "if failed_models:\n",
            "    print(f\"\\nFailed: {len(failed_models)}\")\n",
            "    for model, error in failed_models:\n",
            "        print(f\"  ✗ {model}: {error[:60]}...\")\n",
            "\n",
            "# Show top performers\n",
            "if len(evaluator.results) > 1:\n",
            "    print(f\"\\n{'='*80}\")\n",
            "    print(\"TOP 3 BASELINE MODELS (by Spearman)\")\n",
            "    print(f\"{'='*80}\")\n",
            "    \n",
            "    baseline_results = {name: res for name, res in evaluator.results.items() if 'Baseline' in name}\n",
            "    sorted_baselines = sorted(baseline_results.items(), key=lambda x: x[1]['spearman'], reverse=True)\n",
            "    \n",
            "    for i, (name, res) in enumerate(sorted_baselines[:3], 1):\n",
            "        print(f\"\\n{i}. {name}\")\n",
            "        print(f\"   Spearman: {res['spearman']:.4f}\")\n",
            "        print(f\"   ROC-AUC:  {res['roc_auc']:.4f}\")\n",
            "        print(f\"   F1:       {res['best_f1']:.4f}\")\n",
            "\n",
            "print(f\"\\n{'='*80}\")"
        ]
    }

    # Insert both cells
    nb['cells'].insert(insert_index, new_cell)
    nb['cells'].insert(insert_index + 1, code_cell)

    print(f"\\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\\n" + "="*70)
    print("SUCCESS: Added additional baseline models to evaluate_model.ipynb")
    print("="*70)
    print("\\nAdded models:")
    for config in additional_baselines:
        print(f"  • {config['name']} ({config['model_id']})")

    print("\\nNext steps:")
    print("  1. Open evaluate_model.ipynb")
    print("  2. Run all cells (or just the new cell 6b)")
    print("  3. Compare results to find the best baseline model")
    print("\\nExpected: Nomic or BGE may beat MPNet baseline (0.5038)")
    print("="*70)

if __name__ == '__main__':
    main()
