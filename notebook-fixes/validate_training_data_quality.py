#!/usr/bin/env python3
"""
Training Data Quality Validation

This script validates training pair quality by:
1. Testing baseline MPNet performance on training data
2. Analyzing label distribution and difficulty
3. Identifying potentially mislabeled pairs
4. Comparing training vs test data characteristics

Critical Question:
- If baseline performs WELL on training data (ROC-AUC >0.75) → fine-tuning is making it worse
- If baseline performs POORLY on training data (ROC-AUC <0.70) → data quality issue

Expected Runtime: 5-10 minutes on RTX 5090
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("TRAINING DATA QUALITY VALIDATION")
print("=" * 80)

# Configuration
TRAINING_PAIRS_FILE = Path('data_new/curriculum_training_pairs_complete.json')
TEST_PAIRS_FILE = Path('data_new/fixed_test_pairs.json')
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cuda'
BATCH_SIZE = 128

# Load training data
print(f"\n1. Loading training data...")
print(f"   File: {TRAINING_PAIRS_FILE}")

if not TRAINING_PAIRS_FILE.exists():
    print(f"   ❌ ERROR: Training file not found!")
    exit(1)

with open(TRAINING_PAIRS_FILE, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

train_texts1 = train_data['texts1']
train_texts2 = train_data['texts2']
train_labels = np.array(train_data['labels'])

print(f"   ✅ Loaded {len(train_labels):,} training pairs")
print(f"      Positives: {train_labels.sum():,} ({train_labels.mean()*100:.1f}%)")
print(f"      Negatives: {(1-train_labels).sum():,} ({(1-train_labels.mean())*100:.1f}%)")

# Load test data for comparison
print(f"\n2. Loading test data for comparison...")
print(f"   File: {TEST_PAIRS_FILE}")

with open(TEST_PAIRS_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_texts1 = test_data['texts1']
test_texts2 = test_data['texts2']
test_labels = np.array(test_data['labels'])

print(f"   ✅ Loaded {len(test_labels):,} test pairs")
print(f"      Positives: {test_labels.sum():,} ({test_labels.mean()*100:.1f}%)")
print(f"      Negatives: {(1-test_labels).sum():,} ({(1-test_labels.mean())*100:.1f}%)")

# Load baseline model
print(f"\n3. Loading baseline model...")
print(f"   Model: {BASELINE_MODEL}")
print(f"   Device: {DEVICE}")

model = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
print(f"   ✅ Model loaded")

# Evaluate baseline on TRAINING data
print(f"\n4. Evaluating baseline on TRAINING data...")
print(f"   Encoding {len(train_texts1):,} pairs...")

train_emb1 = model.encode(train_texts1, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE)
train_emb2 = model.encode(train_texts2, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE)

print(f"   Computing cosine similarities...")
train_scores = np.array([
    cosine_similarity([train_emb1[i]], [train_emb2[i]])[0,0]
    for i in range(len(train_emb1))
])

# Compute metrics on training data
train_roc_auc = roc_auc_score(train_labels, train_scores)
train_spearman = spearmanr(train_labels, train_scores)[0]

# Find optimal threshold
thresholds = np.linspace(train_scores.min(), train_scores.max(), 100)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    predictions = (train_scores >= threshold).astype(int)
    f1 = f1_score(train_labels, predictions, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Compute metrics at optimal threshold
train_predictions = (train_scores >= best_threshold).astype(int)
train_precision = precision_score(train_labels, train_predictions, zero_division=0)
train_recall = recall_score(train_labels, train_predictions, zero_division=0)
train_cm = confusion_matrix(train_labels, train_predictions)

print(f"\n   BASELINE PERFORMANCE ON TRAINING DATA:")
print(f"   " + "-" * 76)
print(f"   Spearman:  {train_spearman:.4f}")
print(f"   ROC-AUC:   {train_roc_auc:.4f}")
print(f"   F1:        {best_f1:.4f} (threshold={best_threshold:.4f})")
print(f"   Precision: {train_precision:.4f}")
print(f"   Recall:    {train_recall:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      TN={train_cm[0,0]:5d}  FP={train_cm[0,1]:5d}")
print(f"      FN={train_cm[1,0]:5d}  TP={train_cm[1,1]:5d}")

# Evaluate baseline on TEST data
print(f"\n5. Evaluating baseline on TEST data...")
print(f"   Encoding {len(test_texts1):,} pairs...")

test_emb1 = model.encode(test_texts1, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE)
test_emb2 = model.encode(test_texts2, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE)

print(f"   Computing cosine similarities...")
test_scores = np.array([
    cosine_similarity([test_emb1[i]], [test_emb2[i]])[0,0]
    for i in range(len(test_emb1))
])

# Compute metrics on test data
test_roc_auc = roc_auc_score(test_labels, test_scores)
test_spearman = spearmanr(test_labels, test_scores)[0]
test_predictions = (test_scores >= best_threshold).astype(int)
test_f1 = f1_score(test_labels, test_predictions, zero_division=0)
test_precision = precision_score(test_labels, test_predictions, zero_division=0)
test_recall = recall_score(test_labels, test_predictions, zero_division=0)
test_cm = confusion_matrix(test_labels, test_predictions)

print(f"\n   BASELINE PERFORMANCE ON TEST DATA:")
print(f"   " + "-" * 76)
print(f"   Spearman:  {test_spearman:.4f}")
print(f"   ROC-AUC:   {test_roc_auc:.4f}")
print(f"   F1:        {test_f1:.4f} (threshold={best_threshold:.4f})")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      TN={test_cm[0,0]:5d}  FP={test_cm[0,1]:5d}")
print(f"      FN={test_cm[1,0]:5d}  TP={test_cm[1,1]:5d}")

# Compare train vs test
print(f"\n6. Train vs Test Comparison...")
print(f"   " + "-" * 76)
print(f"   Metric         Train     Test      Delta    Interpretation")
print(f"   " + "-" * 76)

spearman_delta = train_spearman - test_spearman
roc_delta = train_roc_auc - test_roc_auc
f1_delta = best_f1 - test_f1

print(f"   Spearman:    {train_spearman:6.4f}  {test_spearman:6.4f}  {spearman_delta:+7.4f}  ", end="")
if abs(spearman_delta) < 0.02:
    print("Similar ✅")
elif spearman_delta > 0.02:
    print("Train easier ⚠️")
else:
    print("Test easier ⚠️")

print(f"   ROC-AUC:     {train_roc_auc:6.4f}  {test_roc_auc:6.4f}  {roc_delta:+7.4f}  ", end="")
if abs(roc_delta) < 0.02:
    print("Similar ✅")
elif roc_delta > 0.02:
    print("Train easier ⚠️")
else:
    print("Test easier ⚠️")

print(f"   F1:          {best_f1:6.4f}  {test_f1:6.4f}  {f1_delta:+7.4f}  ", end="")
if abs(f1_delta) < 0.02:
    print("Similar ✅")
elif f1_delta > 0.02:
    print("Train easier ⚠️")
else:
    print("Test easier ⚠️")

# Analyze score distributions
print(f"\n7. Score Distribution Analysis...")
print(f"   " + "-" * 76)

# Training data
train_pos_scores = train_scores[train_labels == 1]
train_neg_scores = train_scores[train_labels == 0]

print(f"   TRAINING DATA:")
print(f"   Positive pairs: mean={train_pos_scores.mean():.4f} std={train_pos_scores.std():.4f}")
print(f"   Negative pairs: mean={train_neg_scores.mean():.4f} std={train_neg_scores.std():.4f}")
print(f"   Separation:     {train_pos_scores.mean() - train_neg_scores.mean():.4f}")

# Test data
test_pos_scores = test_scores[test_labels == 1]
test_neg_scores = test_scores[test_labels == 0]

print(f"\n   TEST DATA:")
print(f"   Positive pairs: mean={test_pos_scores.mean():.4f} std={test_pos_scores.std():.4f}")
print(f"   Negative pairs: mean={test_neg_scores.mean():.4f} std={test_neg_scores.std():.4f}")
print(f"   Separation:     {test_pos_scores.mean() - test_neg_scores.mean():.4f}")

# Identify potentially mislabeled pairs
print(f"\n8. Identifying Potentially Mislabeled Pairs...")
print(f"   " + "-" * 76)

# False positives (labeled positive but scored low)
fp_threshold = 0.3
train_fp_candidates = np.where((train_labels == 1) & (train_scores < fp_threshold))[0]

print(f"\n   Positive pairs with LOW similarity scores (< {fp_threshold}):")
print(f"   Found {len(train_fp_candidates)} candidates ({len(train_fp_candidates)/train_labels.sum()*100:.1f}% of positives)")

if len(train_fp_candidates) > 0:
    print(f"\n   Top 5 examples (may be mislabeled):")
    for i, idx in enumerate(train_fp_candidates[:5]):
        print(f"\n   {i+1}. Score={train_scores[idx]:.4f}")
        print(f"      Text1: {train_texts1[idx][:100]}...")
        print(f"      Text2: {train_texts2[idx][:100]}...")

# False negatives (labeled negative but scored high)
fn_threshold = 0.7
train_fn_candidates = np.where((train_labels == 0) & (train_scores > fn_threshold))[0]

print(f"\n   Negative pairs with HIGH similarity scores (> {fn_threshold}):")
print(f"   Found {len(train_fn_candidates)} candidates ({len(train_fn_candidates)/(1-train_labels).sum()*100:.1f}% of negatives)")

if len(train_fn_candidates) > 0:
    print(f"\n   Top 5 examples (may be mislabeled):")
    for i, idx in enumerate(train_fn_candidates[:5]):
        print(f"\n   {i+1}. Score={train_scores[idx]:.4f}")
        print(f"      Text1: {train_texts1[idx][:100]}...")
        print(f"      Text2: {train_texts2[idx][:100]}...")

# Check curriculum phases (if available)
if 'phase_indicators' in train_data:
    phase_indicators = np.array(train_data['phase_indicators'])

    print(f"\n9. Curriculum Phase Analysis...")
    print(f"   " + "-" * 76)

    for phase in sorted(set(phase_indicators)):
        phase_mask = phase_indicators == phase
        phase_scores = train_scores[phase_mask]
        phase_labels = train_labels[phase_mask]

        phase_roc = roc_auc_score(phase_labels, phase_scores) if len(set(phase_labels)) > 1 else 0.0
        phase_spearman = spearmanr(phase_labels, phase_scores)[0] if len(set(phase_scores)) > 1 else 0.0

        print(f"\n   Phase {phase}: {phase_mask.sum():,} pairs")
        print(f"      Spearman:  {phase_spearman:.4f}")
        print(f"      ROC-AUC:   {phase_roc:.4f}")
        print(f"      Positives: {phase_labels.sum():,} ({phase_labels.mean()*100:.1f}%)")

# Final diagnosis
print(f"\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print(f"\nBaseline MPNet Performance:")
print(f"  Training ROC-AUC: {train_roc_auc:.4f}")
print(f"  Test ROC-AUC:     {test_roc_auc:.4f}")

if train_roc_auc > 0.80:
    print(f"\n⚠️  CRITICAL ISSUE: Baseline performs VERY WELL on training data!")
    print(f"    → Training data is TOO EASY for the model")
    print(f"    → Fine-tuning will likely overfit and degrade performance")
    print(f"    → Model needs HARDER examples to learn meaningful improvements")

    print(f"\n📊 RECOMMENDED ACTIONS:")
    print(f"    1. Generate harder negative pairs (TF-IDF 0.3-0.5, same category)")
    print(f"    2. Add cross-category hard positives (high TF-IDF, different categories)")
    print(f"    3. Remove easy pairs (baseline already handles them)")
    print(f"    4. Use adversarial mining (find pairs where baseline fails)")

elif train_roc_auc > 0.75:
    print(f"\n⚠️  WARNING: Baseline performs WELL on training data")
    print(f"    → Training data may be too easy")
    print(f"    → Fine-tuning gains will be modest")
    print(f"    → Consider adding harder examples")

    print(f"\n📊 RECOMMENDED ACTIONS:")
    print(f"    1. Add more hard negatives (same category, medium TF-IDF)")
    print(f"    2. Use lower learning rate (1e-6 or 2e-6)")
    print(f"    3. Reduce epochs (4-6 instead of 12)")
    print(f"    4. Focus on pairs where baseline scores are borderline")

elif train_roc_auc > 0.70:
    print(f"\n✅ MODERATE learning signal exists")
    print(f"    → Training data has some challenging examples")
    print(f"    → Fine-tuning should help but needs careful tuning")

    print(f"\n📊 RECOMMENDED ACTIONS:")
    print(f"    1. Use very low learning rate (1e-6 or 2e-6)")
    print(f"    2. Train for fewer epochs (4-6)")
    print(f"    3. Use simple loss function (CosineSimilarityLoss)")
    print(f"    4. Monitor validation metrics carefully")

else:
    print(f"\n❌ POOR baseline performance on training data")
    print(f"    → Data quality issue OR task is very difficult")
    print(f"    → Check for label noise and data quality")

    print(f"\n📊 RECOMMENDED ACTIONS:")
    print(f"    1. Manually review mislabeled candidates (see above)")
    print(f"    2. Check pair generation logic for bugs")
    print(f"    3. Validate that labels are correct")
    print(f"    4. Consider using different base model")

# Train-test gap analysis
if abs(train_roc_auc - test_roc_auc) > 0.05:
    print(f"\n⚠️  WARNING: Large train-test performance gap ({train_roc_auc - test_roc_auc:+.4f})")
    print(f"    → Training and test distributions may differ")
    print(f"    → Model may not generalize well")
else:
    print(f"\n✅ Train-test performance is consistent (gap: {train_roc_auc - test_roc_auc:+.4f})")

# Mislabeled pairs warning
mislabeled_pct = (len(train_fp_candidates) + len(train_fn_candidates)) / len(train_labels) * 100
if mislabeled_pct > 10:
    print(f"\n⚠️  WARNING: {mislabeled_pct:.1f}% of pairs may be mislabeled")
    print(f"    → High label noise will prevent learning")
    print(f"    → Review and clean training data before retraining")
elif mislabeled_pct > 5:
    print(f"\n⚠️  MODERATE label noise detected ({mislabeled_pct:.1f}% suspicious pairs)")
    print(f"    → Some pairs may need review")
else:
    print(f"\n✅ Label quality appears good ({mislabeled_pct:.1f}% suspicious pairs)")

print(f"\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

# Save detailed report
report_path = Path('models/results/training_data_validation_report.json')
report_path.parent.mkdir(parents=True, exist_ok=True)

report = {
    'training_data': {
        'file': str(TRAINING_PAIRS_FILE),
        'total_pairs': int(len(train_labels)),
        'positives': int(train_labels.sum()),
        'negatives': int((1-train_labels).sum()),
        'baseline_spearman': float(train_spearman),
        'baseline_roc_auc': float(train_roc_auc),
        'baseline_f1': float(best_f1),
        'optimal_threshold': float(best_threshold),
    },
    'test_data': {
        'file': str(TEST_PAIRS_FILE),
        'total_pairs': int(len(test_labels)),
        'positives': int(test_labels.sum()),
        'negatives': int((1-test_labels).sum()),
        'baseline_spearman': float(test_spearman),
        'baseline_roc_auc': float(test_roc_auc),
        'baseline_f1': float(test_f1),
    },
    'train_test_gap': {
        'spearman_delta': float(spearman_delta),
        'roc_auc_delta': float(roc_delta),
        'f1_delta': float(f1_delta),
    },
    'potentially_mislabeled': {
        'false_positive_candidates': int(len(train_fp_candidates)),
        'false_negative_candidates': int(len(train_fn_candidates)),
        'total_suspicious': int(len(train_fp_candidates) + len(train_fn_candidates)),
        'suspicious_percentage': float(mislabeled_pct),
    },
    'score_distributions': {
        'train_positive_mean': float(train_pos_scores.mean()),
        'train_positive_std': float(train_pos_scores.std()),
        'train_negative_mean': float(train_neg_scores.mean()),
        'train_negative_std': float(train_neg_scores.std()),
        'train_separation': float(train_pos_scores.mean() - train_neg_scores.mean()),
        'test_positive_mean': float(test_pos_scores.mean()),
        'test_positive_std': float(test_pos_scores.std()),
        'test_negative_mean': float(test_neg_scores.mean()),
        'test_negative_std': float(test_neg_scores.std()),
        'test_separation': float(test_pos_scores.mean() - test_neg_scores.mean()),
    },
}

with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Detailed report saved to: {report_path}")
