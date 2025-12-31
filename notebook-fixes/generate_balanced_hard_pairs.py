#!/usr/bin/env python3
"""
Balanced Hard Pairs Generation - Final Version

INSIGHT: Mine positives and negatives SEPARATELY with appropriate criteria

STRATEGY:
- Positives: Baseline score 0.50-0.70 + TF-IDF > 0.45 (borderline but truly similar)
- Negatives: Baseline score 0.40-0.60 + TF-IDF < 0.35 (borderline but truly different)

TARGET:
- 40% positives, 60% negatives (balanced)
- Baseline ROC-AUC 0.70-0.75 (challenging but learnable)

Expected Runtime: 15 minutes
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from tqdm import tqdm
import random

print("=" * 80)
print("BALANCED HARD PAIRS GENERATION")
print("=" * 80)

# Configuration
SOURCE_DATA = Path('data_new/SNow_incident_ticket_data.csv')
OUTPUT_FILE = Path('data_new/balanced_hard_training_pairs.json')
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cuda'
BATCH_SIZE = 128
RANDOM_SEED = 42

SAMPLE_SIZE = 5000
TARGET_NUM_PAIRS = 15000
TARGET_POSITIVE_RATIO = 0.40

# Separate criteria for positives and negatives
POSITIVE_CRITERIA = {
    'baseline_range': (0.50, 0.70),  # Baseline somewhat confident but not certain
    'tfidf_min': 0.45,                # Must have semantic overlap
}

NEGATIVE_CRITERIA = {
    'baseline_range': (0.40, 0.60),  # Baseline uncertain
    'tfidf_max': 0.35,                # Must be semantically different
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load and prepare data
print(f"\n1. Loading and preparing data...")
df = pd.read_csv(SOURCE_DATA)

df['combined_text'] = (
    df['Description'].fillna('') + ' ' +
    '(Context: [' + df['Service'].fillna('') + ' | ' + df['Service offering'].fillna('') + '] ' +
    '[' + df['Category'].fillna('') + ' | ' + df['Subcategory'].fillna('') + '] ' +
    'Group: ' + df['Assignment group'].fillna('') + '.)'
)

df = df[df['combined_text'].str.len() >= 25].reset_index(drop=True)
categories = df['Category'].fillna('Unknown').values

# Sample for efficiency
if len(df) > SAMPLE_SIZE:
    sample_indices = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
    df = df.iloc[sample_indices].reset_index(drop=True)
    categories = categories[sample_indices]

texts = df['combined_text'].values
print(f"   ✅ Using {len(texts):,} incidents")

# Load model and encode
print(f"\n2. Encoding incidents...")
model = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE, normalize_embeddings=True)
print(f"   ✅ Encoded")

# Compute TF-IDF
print(f"\n3. Computing TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
tfidf_matrix = tfidf.fit_transform(texts)
print(f"   ✅ TF-IDF ready")

# Mine POSITIVES
print(f"\n4. Mining POSITIVE pairs...")
num_positives_target = int(TARGET_NUM_PAIRS * TARGET_POSITIVE_RATIO)
print(f"   Target: {num_positives_target:,} pairs")

positive_pairs = []
NUM_BATCHES = 200
PAIRS_PER_BATCH = 25000

for batch_idx in tqdm(range(NUM_BATCHES), desc="Mining positives"):
    if len(positive_pairs) >= num_positives_target:
        break

    idx1 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)
    idx2 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)
    valid_mask = idx1 != idx2
    idx1, idx2 = idx1[valid_mask], idx2[valid_mask]

    baseline_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)

    # Filter by baseline range
    mask = (baseline_sims >= POSITIVE_CRITERIA['baseline_range'][0]) & \
           (baseline_sims <= POSITIVE_CRITERIA['baseline_range'][1])

    if mask.sum() == 0:
        continue

    idx1_f, idx2_f = idx1[mask], idx2[mask]
    baseline_sims_f = baseline_sims[mask]

    # Compute TF-IDF for filtered pairs
    tfidf_sims = np.array([(tfidf_matrix[i1] * tfidf_matrix[i2].T).toarray()[0, 0]
                           for i1, i2 in zip(idx1_f, idx2_f)])

    # Filter by TF-IDF
    tfidf_mask = tfidf_sims >= POSITIVE_CRITERIA['tfidf_min']

    for i in np.where(tfidf_mask)[0]:
        positive_pairs.append({
            'idx1': int(idx1_f[i]),
            'idx2': int(idx2_f[i]),
            'label': 1,
            'baseline_sim': float(baseline_sims_f[i]),
            'tfidf_sim': float(tfidf_sims[i]),
            'same_category': categories[idx1_f[i]] == categories[idx2_f[i]],
            'difficulty': abs(baseline_sims_f[i] - 0.5),
        })

print(f"   ✅ Found {len(positive_pairs):,} positive pairs")

# Take top hardest positives
positive_pairs.sort(key=lambda x: x['difficulty'])
if len(positive_pairs) > num_positives_target:
    positive_pairs = positive_pairs[:num_positives_target]
    print(f"   Selected top {len(positive_pairs):,} hardest")

# Mine NEGATIVES
print(f"\n5. Mining NEGATIVE pairs...")
num_negatives_target = TARGET_NUM_PAIRS - len(positive_pairs)
print(f"   Target: {num_negatives_target:,} pairs")

negative_pairs = []

for batch_idx in tqdm(range(NUM_BATCHES), desc="Mining negatives"):
    if len(negative_pairs) >= num_negatives_target:
        break

    idx1 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)
    idx2 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)
    valid_mask = idx1 != idx2
    idx1, idx2 = idx1[valid_mask], idx2[valid_mask]

    baseline_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)

    # Filter by baseline range
    mask = (baseline_sims >= NEGATIVE_CRITERIA['baseline_range'][0]) & \
           (baseline_sims <= NEGATIVE_CRITERIA['baseline_range'][1])

    if mask.sum() == 0:
        continue

    idx1_f, idx2_f = idx1[mask], idx2[mask]
    baseline_sims_f = baseline_sims[mask]

    # Compute TF-IDF for filtered pairs
    tfidf_sims = np.array([(tfidf_matrix[i1] * tfidf_matrix[i2].T).toarray()[0, 0]
                           for i1, i2 in zip(idx1_f, idx2_f)])

    # Filter by TF-IDF
    tfidf_mask = tfidf_sims <= NEGATIVE_CRITERIA['tfidf_max']

    for i in np.where(tfidf_mask)[0]:
        negative_pairs.append({
            'idx1': int(idx1_f[i]),
            'idx2': int(idx2_f[i]),
            'label': 0,
            'baseline_sim': float(baseline_sims_f[i]),
            'tfidf_sim': float(tfidf_sims[i]),
            'same_category': categories[idx1_f[i]] == categories[idx2_f[i]],
            'difficulty': abs(baseline_sims_f[i] - 0.5),
        })

print(f"   ✅ Found {len(negative_pairs):,} negative pairs")

# Take top hardest negatives
negative_pairs.sort(key=lambda x: x['difficulty'])
if len(negative_pairs) > num_negatives_target:
    negative_pairs = negative_pairs[:num_negatives_target]
    print(f"   Selected top {len(negative_pairs):,} hardest")

# Combine and shuffle
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

print(f"\n6. Final dataset: {len(all_pairs):,} pairs")

# Extract data
texts1 = [texts[p['idx1']] for p in all_pairs]
texts2 = [texts[p['idx2']] for p in all_pairs]
labels = [p['label'] for p in all_pairs]
categories1 = [categories[p['idx1']] for p in all_pairs]
categories2 = [categories[p['idx2']] for p in all_pairs]

# Validate
print(f"\n7. Validation...")
baseline_scores = np.array([p['baseline_sim'] for p in all_pairs])
labels_array = np.array(labels)

baseline_roc = roc_auc_score(labels_array, baseline_scores)
baseline_spearman = spearmanr(labels_array, baseline_scores)[0]

print(f"   Baseline ROC-AUC: {baseline_roc:.4f}")
print(f"   Baseline Spearman: {baseline_spearman:.4f}")
print(f"   Positives: {sum(labels):,} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"   Negatives: {len(labels)-sum(labels):,} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

if 0.68 <= baseline_roc <= 0.78:
    print(f"   ✅ EXCELLENT difficulty (ROC in target range 0.68-0.78)")
elif 0.60 <= baseline_roc < 0.68:
    print(f"   ✅ GOOD difficulty (ROC {baseline_roc:.4f} - challenging)")
elif baseline_roc >= 0.78:
    print(f"   ⚠️  Still a bit easy (ROC {baseline_roc:.4f})")
else:
    print(f"   ⚠️  May be too hard (ROC {baseline_roc:.4f})")

# Create curriculum phases
print(f"\n8. Creating curriculum phases...")
sorted_indices = sorted(range(len(all_pairs)), key=lambda i: all_pairs[i]['difficulty'], reverse=True)
phase_size = len(all_pairs) // 3
phase_indicators = [0] * len(all_pairs)

for i, idx in enumerate(sorted_indices):
    if i < phase_size:
        phase_indicators[idx] = 1
    elif i < phase_size * 2:
        phase_indicators[idx] = 2
    else:
        phase_indicators[idx] = 3

print(f"   Phase 1: {phase_indicators.count(1):,} pairs")
print(f"   Phase 2: {phase_indicators.count(2):,} pairs")
print(f"   Phase 3: {phase_indicators.count(3):,} pairs")

# Save
print(f"\n9. Saving...")
output_data = {
    'texts1': texts1,
    'texts2': texts2,
    'labels': labels,
    'categories1': categories1,
    'categories2': categories2,
    'phase_indicators': phase_indicators,
    'metadata': {
        'total_pairs': len(all_pairs),
        'num_positives': sum(labels),
        'num_negatives': len(labels) - sum(labels),
        'baseline_roc_auc': float(baseline_roc),
        'baseline_spearman': float(baseline_spearman),
        'generation_strategy': 'balanced_adversarial_mining',
        'positive_criteria': POSITIVE_CRITERIA,
        'negative_criteria': NEGATIVE_CRITERIA,
        'random_seed': RANDOM_SEED,
    }
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"   ✅ Saved to {OUTPUT_FILE}")

# Summary
print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nDataset: {len(all_pairs):,} pairs")
print(f"  Positives: {sum(labels):,} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"  Negatives: {len(labels)-sum(labels):,} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

print(f"\nBaseline Performance:")
print(f"  ROC-AUC:   {baseline_roc:.4f}")
print(f"  Spearman:  {baseline_spearman:.4f}")

print(f"\nComparison to Original:")
print(f"  Original ROC: 0.9264 (too easy)")
print(f"  New ROC:      {baseline_roc:.4f}")
print(f"  Improvement:  {0.9264 - baseline_roc:+.4f}")

if 0.65 <= baseline_roc <= 0.80:
    print(f"\n✅ SUCCESS! Ready for training")
    print(f"\nRecommended hyperparameters:")
    print(f"  - Learning rate: 1e-6 to 2e-6")
    print(f"  - Epochs: 6-9 (2-3 per phase)")
    print(f"  - Loss: CosineSimilarityLoss")
    print(f"  - Expected improvement: Spearman +0.02 to +0.08")

print(f"\n" + "=" * 80)
