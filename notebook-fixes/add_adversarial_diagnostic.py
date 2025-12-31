#!/usr/bin/env python3
"""
Add adversarial diagnostic to evaluate_model_v2.ipynb.

This adds:
1. Category column loading from CSV
2. Category metadata in test pairs
3. Full adversarial diagnostic implementation
"""

import json

# Read current v2 notebook
with open('evaluate_model_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def mk_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def mk_code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

# Find where to insert adversarial diagnostic (after Section 9 - Fine-tuned models)
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 10. Results Comparison' in ''.join(cell['source']):
        insert_index = i
        break

if insert_index is None:
    print("Could not find insertion point")
    exit(1)

print(f"Found insertion point at cell {insert_index}")

# Create new adversarial diagnostic cells
adv_cells = []

# Section header
adv_cells.append(mk_md([
    "## 9a. Adversarial Diagnostic Test (NEW!)\n",
    "\n",
    "**Purpose:** Verify model learns semantic similarity, not category shortcuts.\n",
    "\n",
    "**Test Design:**\n",
    "- Cross-category positives (high TF-IDF, different categories) - Model must score HIGH\n",
    "- Same-category negatives (low TF-IDF, same category) - Model must score LOW\n",
    "\n",
    "**Pass Criteria:** ROC-AUC ≥ 0.70 AND F1 ≥ 0.70 (per CLAUDE.md)\n",
    "\n",
    "If model passes → Learned real semantics ✓  \n",
    "If model fails → Using category shortcuts ✗\n"
]))

# Implementation code
adv_cells.append(mk_code([
    "# Adversarial Diagnostic Test\n",
    "print('='*80)\n",
    "print('ADVERSARIAL DIAGNOSTIC TEST')\n",
    "print('='*80)\n",
    "\n",
    "# Check if we have category data in test pairs\n",
    "has_category_data = 'category1' in test_pairs[0] if test_pairs else False\n",
    "\n",
    "if not has_category_data:\n",
    "    print('\\nCategory data not found in test pairs.')\n",
    "    print('Skipping adversarial diagnostic.')\n",
    "    print('\\nTo enable: Re-generate test pairs with category metadata.')\n",
    "else:\n",
    "    # Extract category data\n",
    "    categories1 = [p['category1'] for p in test_pairs]\n",
    "    categories2 = [p['category2'] for p in test_pairs]\n",
    "    \n",
    "    # Compute TF-IDF similarities for all pairs\n",
    "    from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "    print('\\nComputing TF-IDF similarities for adversarial filtering...')\n",
    "    \n",
    "    tfidf = TfidfVectorizer()\n",
    "    all_texts = test_texts1 + test_texts2\n",
    "    tfidf.fit(all_texts)\n",
    "    \n",
    "    tfidf_matrix1 = tfidf.transform(test_texts1)\n",
    "    tfidf_matrix2 = tfidf.transform(test_texts2)\n",
    "    \n",
    "    tfidf_sims = np.array([\n",
    "        (tfidf_matrix1[i] * tfidf_matrix2[i].T).toarray()[0, 0]\n",
    "        for i in range(len(test_texts1))\n",
    "    ])\n",
    "    \n",
    "    # Create adversarial test set\n",
    "    # 1. Cross-category positives (high TF-IDF, different categories, label=1)\n",
    "    # 2. Same-category negatives (low TF-IDF, same category, label=0)\n",
    "    \n",
    "    cross_category = np.array([c1 != c2 for c1, c2 in zip(categories1, categories2)])\n",
    "    same_category = np.array([c1 == c2 for c1, c2 in zip(categories1, categories2)])\n",
    "    high_tfidf = tfidf_sims >= 0.5\n",
    "    low_tfidf = tfidf_sims < 0.3\n",
    "    \n",
    "    # Adversarial mask\n",
    "    adversarial_mask = (\n",
    "        (cross_category & high_tfidf & (test_labels == 1)) |  # Cross-cat positives\n",
    "        (same_category & low_tfidf & (test_labels == 0))      # Same-cat negatives\n",
    "    )\n",
    "    \n",
    "    adv_count = adversarial_mask.sum()\n",
    "    \n",
    "    if adv_count < 10:\n",
    "        print(f'\\nWarning: Only {adv_count} adversarial pairs found.')\n",
    "        print('Test set may not be suitable for adversarial diagnostic.')\n",
    "    else:\n",
    "        print(f'\\nAdversarial test set: {adv_count} pairs')\n",
    "        \n",
    "        cross_cat_pos = (cross_category & high_tfidf & (test_labels == 1)).sum()\n",
    "        same_cat_neg = (same_category & low_tfidf & (test_labels == 0)).sum()\n",
    "        \n",
    "        print(f'  Cross-category positives: {cross_cat_pos}')\n",
    "        print(f'  Same-category negatives:  {same_cat_neg}')\n",
    "        \n",
    "        # Evaluate each model on adversarial subset\n",
    "        print('\\n' + '='*80)\n",
    "        print('ADVERSARIAL DIAGNOSTIC RESULTS')\n",
    "        print('='*80)\n",
    "        \n",
    "        adv_results = {}\n",
    "        \n",
    "        for model_name, result in evaluator.results.items():\n",
    "            print(f'\\n{\"-\"*80}')\n",
    "            print(f'{model_name}')\n",
    "            print(f'{\"-\"*80}')\n",
    "            \n",
    "            # Get scores for adversarial subset\n",
    "            adv_labels = test_labels[adversarial_mask]\n",
    "            adv_scores = result['cosine_scores'][adversarial_mask]\n",
    "            \n",
    "            # Compute metrics\n",
    "            from sklearn.metrics import roc_auc_score, f1_score\n",
    "            \n",
    "            adv_roc_auc = roc_auc_score(adv_labels, adv_scores) if len(set(adv_labels)) > 1 else 0.0\n",
    "            adv_predictions = (adv_scores >= 0.5).astype(int)\n",
    "            adv_f1 = f1_score(adv_labels, adv_predictions)\n",
    "            \n",
    "            # Check pass criteria\n",
    "            passed = (adv_roc_auc >= 0.70) and (adv_f1 >= 0.70)\n",
    "            \n",
    "            adv_results[model_name] = {\n",
    "                'roc_auc': adv_roc_auc,\n",
    "                'f1': adv_f1,\n",
    "                'passed': passed\n",
    "            }\n",
    "            \n",
    "            print(f'  ROC-AUC: {adv_roc_auc:.4f} (>= 0.70 required)')\n",
    "            print(f'  F1:      {adv_f1:.4f} (>= 0.70 required)')\n",
    "            \n",
    "            if passed:\n",
    "                print(f'  Status:  PASSED - Model learned semantic similarity!')\n",
    "            else:\n",
    "                print(f'  Status:  FAILED - Model may be using category shortcuts!')\n",
    "                \n",
    "                # Provide diagnostic info\n",
    "                if adv_roc_auc < 0.70:\n",
    "                    print(f'    Issue: Poor ranking (ROC-AUC too low)')\n",
    "                if adv_f1 < 0.70:\n",
    "                    print(f'    Issue: Poor classification (F1 too low)')\n",
    "        \n",
    "        # Summary table\n",
    "        print(f'\\n{\"=\"*80}')\n",
    "        print('ADVERSARIAL DIAGNOSTIC SUMMARY')\n",
    "        print(f'{\"=\"*80}')\n",
    "        \n",
    "        passed_models = [name for name, res in adv_results.items() if res['passed']]\n",
    "        failed_models = [name for name, res in adv_results.items() if not res['passed']]\n",
    "        \n",
    "        print(f'\\nPassed: {len(passed_models)}/{len(adv_results)}')\n",
    "        for name in passed_models:\n",
    "            res = adv_results[name]\n",
    "            print(f'  {name:40s} (ROC-AUC={res[\"roc_auc\"]:.4f}, F1={res[\"f1\"]:.4f})')\n",
    "        \n",
    "        if failed_models:\n",
    "            print(f'\\nFailed: {len(failed_models)}/{len(adv_results)}')\n",
    "            for name in failed_models:\n",
    "                res = adv_results[name]\n",
    "                print(f'  {name:40s} (ROC-AUC={res[\"roc_auc\"]:.4f}, F1={res[\"f1\"]:.4f})')\n",
    "        \n",
    "        print(f'\\n{\"=\"*80}')\n",
    "        \n",
    "        # Store results in evaluator for later export\n",
    "        for model_name, res in adv_results.items():\n",
    "            evaluator.results[model_name]['metrics']['adversarial_roc_auc'] = res['roc_auc']\n",
    "            evaluator.results[model_name]['metrics']['adversarial_f1'] = res['f1']\n",
    "            evaluator.results[model_name]['metrics']['adversarial_passed'] = res['passed']\n"
]))

# Insert adversarial diagnostic cells
for i, cell in enumerate(adv_cells):
    nb['cells'].insert(insert_index + i, cell)

print(f"Inserted {len(adv_cells)} cells at index {insert_index}")

# Also need to update Section 4 (data loading) to include category
# Find Section 4
section4_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 4. Load Test Data' in ''.join(cell['source']):
        section4_index = i + 1  # Next cell is the code
        break

if section4_index:
    # Update data loading cell to extract categories
    nb['cells'][section4_index]['source'] = [
        "# Load test pairs\n",
        "test_pairs_path = DATA_DIR / TEST_PAIRS_FILE\n",
        "print(f'Loading: {test_pairs_path}')\n",
        "\n",
        "with open(test_pairs_path, 'r', encoding='utf-8') as f:\n",
        "    test_pairs = json.load(f)\n",
        "\n",
        "# Extract data\n",
        "test_texts1 = [p['text1'] for p in test_pairs]\n",
        "test_texts2 = [p['text2'] for p in test_pairs]\n",
        "test_labels = np.array([p['label'] for p in test_pairs])\n",
        "\n",
        "# Check if category data is available (for adversarial diagnostic)\n",
        "has_categories = 'category1' in test_pairs[0] if test_pairs else False\n",
        "if has_categories:\n",
        "    print('  Category metadata found - adversarial diagnostic enabled')\n",
        "else:\n",
        "    print('  No category metadata - adversarial diagnostic will be skipped')\n",
        "\n",
        "# Validate\n",
        "validate_test_pairs(test_texts1, test_texts2, test_labels)\n"
    ]
    print("Updated Section 4 data loading")

# Save updated notebook
with open('evaluate_model_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nUpdated evaluate_model_v2.ipynb")
print(f"  Total cells: {len(nb['cells'])}")
print(f"\nAdversarial diagnostic added!")
print(f"\nNext step: Update test pairs to include category metadata")
print(f"  Run: python add_categories_to_test_pairs.py")
