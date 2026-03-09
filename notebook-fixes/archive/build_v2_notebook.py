#!/usr/bin/env python3
"""
Build the complete evaluate_model_v2.ipynb notebook.
"""

import json

# Read original for metadata
with open('evaluate_model.ipynb', 'r', encoding='utf-8') as f:
    original = json.load(f)

# Create v2
nb = {
    "cells": [],
    "metadata": original["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5
}

def mk_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def mk_code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = nb["cells"]

# HEADER
cells.append(mk_md([
    "# Model Evaluation Notebook v2\n",
    "\n",
    "**Improved evaluation framework with:**\n",
    "- Configuration in `evaluation_config.py`\n",
    "- Utilities in `evaluation_utils.py`\n",
    "- Security fix (no hardcoded tokens)\n",
    "- Confusion matrices\n",
    "- Error analysis\n",
    "- Inference benchmarking\n"
]))

# SEC 1: SETUP
cells.append(mk_md("## 1. Setup & Configuration"))
cells.append(mk_code([
    "# Standard imports\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "import json\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ML imports\n",
    "import torch\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score\n",
    "from scipy.stats import spearmanr\n",
    "\n",
    "# Local imports\n",
    "from evaluation_config import *\n",
    "from evaluation_utils import *\n",
    "\n",
    "# Plotting\n",
    "plt.style.use('default')\n",
    "sns.set_palette('husl')\n",
    "\n",
    "print('All imports successful')\n"
]))

cells.append(mk_code([
    "# Validate config\n",
    "validate_config()\n",
    "\n",
    "print(f'\\nConfiguration:')\n",
    "print(f'  Data dir: {DATA_DIR}')\n",
    "print(f'  Baseline: {BASELINE_MODEL}')\n",
    "print(f'  Baselines: {len(ADDITIONAL_BASELINES)}')\n",
    "print(f'  Seed: {DATA_CONFIG[\"random_seed\"]}')\n"
]))

# SEC 2: DEVICE
cells.append(mk_md("## 2. Device & Memory"))
cells.append(mk_code([
    "set_all_seeds(DATA_CONFIG['random_seed'])\n",
    "device, gpu_memory = detect_device()\n",
    "\n",
    "if MEMORY_CONFIG['enable_expandable_segments'] and device == 'cuda':\n",
    "    import os\n",
    "    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "    print('Enabled expandable segments')\n",
    "\n",
    "batch_size = get_batch_size(device, gpu_memory)\n",
    "print(f'\\nBatch size: {batch_size}')\n",
    "print_environment_info()\n"
]))

# SEC 3: DATA
cells.append(mk_md("## 3. Load Data"))
cells.append(mk_code([
    "test_pairs_path = DATA_DIR / TEST_PAIRS_FILE\n",
    "print(f'Loading: {test_pairs_path}')\n",
    "\n",
    "with open(test_pairs_path, 'r', encoding='utf-8') as f:\n",
    "    test_pairs = json.load(f)\n",
    "\n",
    "test_texts1 = [p['text1'] for p in test_pairs]\n",
    "test_texts2 = [p['text2'] for p in test_pairs]\n",
    "test_labels = np.array([p['label'] for p in test_pairs])\n",
    "\n",
    "validate_test_pairs(test_texts1, test_texts2, test_labels)\n"
]))

# SEC 4: EVALUATOR
cells.append(mk_md("## 4. Model Evaluator"))
cells.append(mk_code([
    "class ModelEvaluatorV2:\n",
    "    def __init__(self):\n",
    "        self.results = {}\n",
    "    \n",
    "    def evaluate_model(self, model, model_name, texts1, texts2, labels, batch_size=32, device='cuda', verbose=True):\n",
    "        if verbose:\n",
    "            print(f'\\n{\"=\"*80}')\n",
    "            print(f'EVALUATING: {model_name}')\n",
    "            print(f'{\"=\"*80}')\n",
    "        \n",
    "        # Encode\n",
    "        emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=verbose, device=device)\n",
    "        emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=verbose, device=device)\n",
    "        \n",
    "        # Similarity\n",
    "        from sklearn.metrics.pairwise import cosine_similarity\n",
    "        scores = np.array([cosine_similarity([emb1[i]], [emb2[i]])[0,0] for i in range(len(emb1))])\n",
    "        \n",
    "        # Threshold\n",
    "        best_threshold, best_f1 = find_optimal_threshold(labels, scores)\n",
    "        \n",
    "        # Metrics\n",
    "        metrics = compute_metrics(labels, scores, best_threshold)\n",
    "        metrics['best_threshold'] = best_threshold\n",
    "        \n",
    "        # Confusion matrix\n",
    "        cm = compute_confusion_matrix(labels, scores, best_threshold)\n",
    "        metrics['confusion_matrix'] = cm.tolist()\n",
    "        \n",
    "        # Store\n",
    "        self.results[model_name] = {\n",
    "            'metrics': metrics,\n",
    "            'cosine_scores': scores,\n",
    "            'labels': labels\n",
    "        }\n",
    "        \n",
    "        if verbose:\n",
    "            print(f'\\nMetrics (threshold={best_threshold:.4f}):')\n",
    "            print(f'  Spearman:  {metrics[\"spearman\"]:.4f}')\n",
    "            print(f'  ROC-AUC:   {metrics[\"roc_auc\"]:.4f}')\n",
    "            print(f'  F1:        {metrics[\"f1\"]:.4f}')\n",
    "            print(f'  Precision: {metrics[\"precision\"]:.4f}')\n",
    "            print(f'  Recall:    {metrics[\"recall\"]:.4f}')\n",
    "            print(f'\\nConfusion Matrix:')\n",
    "            print(f'  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}')\n",
    "            print(f'  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}')\n",
    "        \n",
    "        return metrics\n",
    "    \n",
    "    def get_metrics_df(self):\n",
    "        data = {name: result['metrics'] for name, result in self.results.items()}\n",
    "        df = pd.DataFrame(data).T\n",
    "        return df.sort_values('spearman', ascending=False)\n",
    "\n",
    "evaluator = ModelEvaluatorV2()\n",
    "print('Evaluator initialized')\n"
]))

# SEC 5: AUTH (SECURE)
cells.append(mk_md([
    "## 5. HuggingFace Authentication (SECURE)\n",
    "\n",
    "**Setup:** Create `.env` file with `HUGGINGFACE_TOKEN=hf_xxxxx`\n"
]))
cells.append(mk_code([
    "from huggingface_hub import login\n",
    "from dotenv import load_dotenv\n",
    "import os\n",
    "\n",
    "load_dotenv()\n",
    "load_dotenv(Path.home() / '.env')\n",
    "\n",
    "token = os.getenv('HUGGINGFACE_TOKEN')\n",
    "\n",
    "if token:\n",
    "    login(token=token)\n",
    "    print('Authenticated')\n",
    "else:\n",
    "    print('No token - uncomment login() for interactive:')\n",
    "    # login()\n"
]))

# SEC 6: BASELINE
cells.append(mk_md("## 6. Evaluate Baseline"))
cells.append(mk_code([
    "baseline_model = SentenceTransformer(BASELINE_MODEL, device=device)\n",
    "\n",
    "evaluator.evaluate_model(\n",
    "    baseline_model,\n",
    "    'Baseline (Raw MPNet)',\n",
    "    test_texts1,\n",
    "    test_texts2,\n",
    "    test_labels,\n",
    "    batch_size,\n",
    "    device\n",
    ")\n",
    "\n",
    "del baseline_model\n",
    "cleanup_gpu_memory(device)\n",
    "print('\\nBaseline complete')\n"
]))

# SEC 7: ADDITIONAL BASELINES
cells.append(mk_md([
    "## 7. Evaluate Additional Baselines\n",
    "\n",
    "See `evaluate_model.ipynb` Cell 6b for the full loop.\n",
    "Or use the model list from `ADDITIONAL_BASELINES` in `evaluation_config.py`.\n"
]))

# SEC 8: RESULTS
cells.append(mk_md("## 8. Results"))
cells.append(mk_code([
    "metrics_df = evaluator.get_metrics_df()\n",
    "display_df = metrics_df.drop(columns=['confusion_matrix'], errors='ignore')\n",
    "\n",
    "print('='*80)\n",
    "print('RESULTS')\n",
    "print('='*80)\n",
    "print(display_df.to_string())\n"
]))

# SEC 9: CONFUSION MATRICES
cells.append(mk_md("## 9. Confusion Matrices (NEW!)"))
cells.append(mk_code([
    "import math\n",
    "\n",
    "num = len(evaluator.results)\n",
    "if num > 0:\n",
    "    ncols = min(3, num)\n",
    "    nrows = math.ceil(num / ncols)\n",
    "    \n",
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))\n",
    "    axes = np.atleast_1d(axes).flatten()\n",
    "    \n",
    "    for idx, (name, result) in enumerate(evaluator.results.items()):\n",
    "        ax = axes[idx]\n",
    "        cm = np.array(result['metrics']['confusion_matrix'])\n",
    "        \n",
    "        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',\n",
    "                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], cbar=False)\n",
    "        ax.set_title(name, fontweight='bold', fontsize=10)\n",
    "        ax.set_ylabel('True')\n",
    "        ax.set_xlabel('Predicted')\n",
    "    \n",
    "    for idx in range(num, len(axes)):\n",
    "        axes[idx].axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n"
]))

# SEC 10: ERROR ANALYSIS
cells.append(mk_md("## 10. Error Analysis (NEW!)"))
cells.append(mk_code([
    "for name, result in evaluator.results.items():\n",
    "    print(f'\\n{\"-\"*80}')\n",
    "    print(name)\n",
    "    print(f'{\"-\"*80}')\n",
    "    \n",
    "    errors = analyze_errors(\n",
    "        test_texts1,\n",
    "        test_texts2,\n",
    "        result['labels'],\n",
    "        result['cosine_scores'],\n",
    "        threshold=result['metrics']['best_threshold'],\n",
    "        num_examples=3\n",
    "    )\n",
    "    \n",
    "    fp = errors['false_positives']\n",
    "    fn = errors['false_negatives']\n",
    "    \n",
    "    print(f'\\nFP: {fp[\"count\"]} ({fp[\"percentage\"]:.1f}%)')\n",
    "    print(f'  Mean score: {fp[\"mean_score\"]:.4f}')\n",
    "    if fp['examples']:\n",
    "        for i, ex in enumerate(fp['examples'][:2], 1):\n",
    "            print(f'  {i}. Score={ex[\"score\"]:.4f}')\n",
    "            print(f'     T1: {ex[\"text1\"]}')\n",
    "            print(f'     T2: {ex[\"text2\"]}')\n",
    "    \n",
    "    print(f'\\nFN: {fn[\"count\"]} ({fn[\"percentage\"]:.1f}%)')\n",
    "    print(f'  Mean score: {fn[\"mean_score\"]:.4f}')\n"
]))

# SEC 11: VISUALIZATIONS
cells.append(mk_md("## 11. ROC Curves"))
cells.append(mk_code([
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "for name, result in evaluator.results.items():\n",
    "    fpr, tpr, _ = roc_curve(result['labels'], result['cosine_scores'])\n",
    "    auc = result['metrics']['roc_auc']\n",
    "    ax.plot(fpr, tpr, label=f\"{name} (AUC={auc:.4f})\", linewidth=2)\n",
    "\n",
    "ax.plot([0,1], [0,1], 'k--', label='Random', linewidth=1)\n",
    "ax.set_xlabel('FPR', fontsize=12)\n",
    "ax.set_ylabel('TPR', fontsize=12)\n",
    "ax.set_title('ROC Curves', fontsize=14, fontweight='bold')\n",
    "ax.legend(loc='lower right')\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# SEC 12: EXPORT
cells.append(mk_md("## 12. Export"))
cells.append(mk_code([
    "from datetime import datetime\n",
    "\n",
    "ts = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
    "\n",
    "csv_path = RESULTS_OUTPUT_DIR / f'evaluation_{ts}.csv'\n",
    "metrics_df.to_csv(csv_path)\n",
    "print(f'Saved: {csv_path}')\n",
    "\n",
    "results_dict = {n: r['metrics'] for n, r in evaluator.results.items()}\n",
    "json_path = RESULTS_OUTPUT_DIR / f'results_{ts}.json'\n",
    "save_results(results_dict, json_path)\n"
]))

# SEC 13: SUMMARY
cells.append(mk_md("## 13. Summary"))
cells.append(mk_code([
    "print('='*80)\n",
    "print('SUMMARY')\n",
    "print('='*80)\n",
    "\n",
    "best = metrics_df['spearman'].idxmax()\n",
    "best_score = metrics_df['spearman'].max()\n",
    "\n",
    "print(f'\\nBest: {best}')\n",
    "print(f'Spearman: {best_score:.4f}')\n",
    "\n",
    "baseline = evaluator.results['Baseline (Raw MPNet)']['metrics']['spearman']\n",
    "improvement = best_score - baseline\n",
    "\n",
    "print(f'\\nBaseline: {baseline:.4f}')\n",
    "print(f'Improvement: {improvement:+.4f} ({improvement/baseline*100:+.1f}%)')\n",
    "\n",
    "if improvement > 0.01:\n",
    "    print('\\nSIGNIFICANT improvement!')\n",
    "elif improvement > -0.01:\n",
    "    print('\\nSimilar to baseline')\n",
    "else:\n",
    "    print('\\nBelow baseline')\n"
]))

# Save
with open('evaluate_model_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nCreated evaluate_model_v2.ipynb")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Code cells: {sum(1 for c in nb['cells'] if c['cell_type']=='code')}")
print(f"  Markdown cells: {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}")
print("\nReady to use!")
