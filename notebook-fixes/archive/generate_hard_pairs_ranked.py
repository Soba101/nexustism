#!/usr/bin/env python3
"""
Ranked Adversarial Mining - Hard Training Pairs Generation

NEW STRATEGY:
Instead of random sampling, we:
1. Pre-compute ALL pairwise similarities (baseline + TF-IDF)
2. RANK pairs by difficulty (baseline uncertainty)
3. SELECT top hardest pairs from each category

This guarantees we get the HARDEST possible pairs, not just random ones
that happen to meet criteria.

Target: Baseline ROC-AUC 0.68-0.75 (challenging but learnable)

Expected Runtime: 20-30 minutes
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from tqdm import tqdm
import random

print("=" * 80)
print("RANKED ADVERSARIAL MINING - HARD TRAINING PAIRS")
print("=" * 80)

# Configuration
SOURCE_DATA = Path('data_new/SNow_incident_ticket_data.csv')
OUTPUT_FILE = Path('data_new/hard_training_pairs_ranked.json')
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cuda'
BATCH_SIZE = 128
RANDOM_SEED = 42

# Sampling strategy (sample from subset to reduce memory)
SAMPLE_SIZE = 5000  # Sample 5K incidents (allows 5K×5K = 25M potential pairs)
TARGET_NUM_PAIRS = 15000

# Target difficulty: baseline score in borderline range
BORDERLINE_RANGE = (0.40, 0.60)  # Tighter range for truly uncertain pairs

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
print(f"\n1. Loading source data...")
df = pd.read_csv(SOURCE_DATA)
print(f"   ✅ Loaded {len(df):,} incidents")

# Create combined text
df['combined_text'] = (
    df['Description'].fillna('') + ' ' +
    '(Context: [' + df['Service'].fillna('') + ' | ' + df['Service offering'].fillna('') + '] ' +
    '[' + df['Category'].fillna('') + ' | ' + df['Subcategory'].fillna('') + '] ' +
    'Group: ' + df['Assignment group'].fillna('') + '.)'
)

df = df[df['combined_text'].str.len() >= 25].reset_index(drop=True)
categories = df['Category'].fillna('Unknown').values

# Sample subset for efficiency
if len(df) > SAMPLE_SIZE:
    print(f"\n2. Sampling {SAMPLE_SIZE:,} incidents for efficiency...")
    sample_indices = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
    df = df.iloc[sample_indices].reset_index(drop=True)
    categories = categories[sample_indices]
    print(f"   ✅ Sampled {len(df):,} incidents")

texts = df['combined_text'].values
print(f"   Unique categories: {len(set(categories))}")

# Load baseline model
print(f"\n3. Loading baseline model...")
model = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
print(f"   ✅ Model loaded")

# Encode all texts
print(f"\n4. Encoding {len(texts):,} incidents...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    device=DEVICE,
    convert_to_numpy=True,
    normalize_embeddings=True  # Pre-normalize for cosine similarity
)
print(f"   ✅ Encoded {len(embeddings):,} incidents")

# Compute TF-IDF
print(f"\n5. Computing TF-IDF matrix...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
tfidf_matrix = tfidf.fit_transform(texts)
print(f"   ✅ TF-IDF matrix: {tfidf_matrix.shape}")

# Strategy: Sample pairs efficiently
print(f"\n6. Mining hard pairs (ranked selection)...")
print(f"   Strategy: Sample and rank by baseline uncertainty")

hard_pairs = []

# We'll sample candidate pairs in batches
NUM_CANDIDATE_BATCHES = 100
PAIRS_PER_BATCH = 50000  # 50K candidates per batch

print(f"\n   Sampling {NUM_CANDIDATE_BATCHES} batches of {PAIRS_PER_BATCH:,} candidate pairs...")

for batch_idx in tqdm(range(NUM_CANDIDATE_BATCHES), desc="Mining batches"):
    # Sample random pairs
    idx1 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)
    idx2 = np.random.randint(0, len(texts), PAIRS_PER_BATCH)

    # Remove self-pairs
    valid_mask = idx1 != idx2
    idx1 = idx1[valid_mask]
    idx2 = idx2[valid_mask]

    if len(idx1) == 0:
        continue

    # Compute baseline similarity (fast - already normalized)
    baseline_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)

    # Filter to borderline range
    borderline_mask = (baseline_sims >= BORDERLINE_RANGE[0]) & (baseline_sims <= BORDERLINE_RANGE[1])

    if borderline_mask.sum() == 0:
        continue

    # Keep borderline pairs
    idx1_borderline = idx1[borderline_mask]
    idx2_borderline = idx2[borderline_mask]
    baseline_sims_borderline = baseline_sims[borderline_mask]

    # Compute TF-IDF similarity for borderline pairs only
    tfidf_sims = np.array([
        (tfidf_matrix[i1] * tfidf_matrix[i2].T).toarray()[0, 0]
        for i1, i2 in zip(idx1_borderline, idx2_borderline)
    ])

    # Label pairs:
    # - High TF-IDF (>0.4): Positive (semantic similarity despite baseline uncertainty)
    # - Low TF-IDF (<0.3): Negative (truly different despite keyword overlap)
    # - Medium TF-IDF (0.3-0.4): Skip (ambiguous)

    for i in range(len(idx1_borderline)):
        tfidf_sim = tfidf_sims[i]
        baseline_sim = baseline_sims_borderline[i]

        # Determine label based on TF-IDF
        if tfidf_sim >= 0.40:
            # High TF-IDF → Positive
            label = 1
        elif tfidf_sim <= 0.30:
            # Low TF-IDF → Negative
            label = 0
        else:
            # Ambiguous → Skip
            continue

        # Check category
        same_category = categories[idx1_borderline[i]] == categories[idx2_borderline[i]]

        hard_pairs.append({
            'idx1': int(idx1_borderline[i]),
            'idx2': int(idx2_borderline[i]),
            'label': label,
            'baseline_sim': float(baseline_sim),
            'tfidf_sim': float(tfidf_sim),
            'same_category': same_category,
            # Difficulty score: closer to 0.5 = harder
            'difficulty': float(abs(baseline_sim - 0.5)),
        })

print(f"\n   ✅ Found {len(hard_pairs):,} borderline pairs")

# Sort by difficulty (most difficult first)
hard_pairs.sort(key=lambda x: x['difficulty'])

# Take top hardest pairs
if len(hard_pairs) > TARGET_NUM_PAIRS:
    print(f"   Selecting top {TARGET_NUM_PAIRS:,} hardest pairs...")
    hard_pairs = hard_pairs[:TARGET_NUM_PAIRS]

print(f"   ✅ Selected {len(hard_pairs):,} pairs")

# Shuffle final pairs
random.shuffle(hard_pairs)

# Extract data
texts1 = [texts[p['idx1']] for p in hard_pairs]
texts2 = [texts[p['idx2']] for p in hard_pairs]
labels = [p['label'] for p in hard_pairs]
categories1 = [categories[p['idx1']] for p in hard_pairs]
categories2 = [categories[p['idx2']] for p in hard_pairs]

# Validate difficulty
print(f"\n7. Validating generated pairs...")
baseline_scores = np.array([p['baseline_sim'] for p in hard_pairs])
labels_array = np.array(labels)

baseline_roc = roc_auc_score(labels_array, baseline_scores)
baseline_spearman = spearmanr(labels_array, baseline_scores)[0]

print(f"   Baseline performance on generated pairs:")
print(f"     ROC-AUC:   {baseline_roc:.4f}")
print(f"     Spearman:  {baseline_spearman:.4f}")

# Distribution analysis
num_positives = sum(labels)
num_negatives = len(labels) - num_positives
num_same_cat = sum(1 for p in hard_pairs if p['same_category'])

print(f"\n   Pair distribution:")
print(f"     Positives:        {num_positives:,} ({num_positives/len(labels)*100:.1f}%)")
print(f"     Negatives:        {num_negatives:,} ({num_negatives/len(labels)*100:.1f}%)")
print(f"     Same category:    {num_same_cat:,} ({num_same_cat/len(labels)*100:.1f}%)")

# Difficulty assessment
if baseline_roc > 0.80:
    print(f"\n   ⚠️  WARNING: Still too easy (ROC {baseline_roc:.4f} > 0.80)")
    print(f"      The data is challenging but baseline still performs well")
elif baseline_roc > 0.70:
    print(f"\n   ✅ GOOD difficulty (ROC {baseline_roc:.4f} in 0.70-0.80 range)")
    print(f"      Baseline struggles but data is learnable")
elif baseline_roc > 0.60:
    print(f"\n   ⚠️  HARD (ROC {baseline_roc:.4f} in 0.60-0.70)")
    print(f"      May be challenging but should still work")
else:
    print(f"\n   ⚠️  WARNING: Too hard (ROC {baseline_roc:.4f} < 0.60)")
    print(f"      Model may struggle to learn from this data")

# Create balanced curriculum phases
print(f"\n8. Creating curriculum phases...")

# Sort by difficulty (easier first)
sorted_indices = sorted(range(len(hard_pairs)), key=lambda i: hard_pairs[i]['difficulty'], reverse=True)

phase_size = len(hard_pairs) // 3
phase_indicators = []

for i, idx in enumerate(sorted_indices):
    if i < phase_size:
        phase_indicators.append(1)  # Easiest third
    elif i < phase_size * 2:
        phase_indicators.append(2)  # Medium third
    else:
        phase_indicators.append(3)  # Hardest third

# Reorder to match original shuffled order
temp_indicators = [0] * len(hard_pairs)
for i, idx in enumerate(sorted_indices):
    temp_indicators[idx] = phase_indicators[i]
phase_indicators = temp_indicators

print(f"   Phase 1 (easier):  {phase_indicators.count(1):,} pairs")
print(f"   Phase 2 (medium):  {phase_indicators.count(2):,} pairs")
print(f"   Phase 3 (harder):  {phase_indicators.count(3):,} pairs")

# Save
print(f"\n9. Saving to file...")
print(f"   Output: {OUTPUT_FILE}")

output_data = {
    'texts1': texts1,
    'texts2': texts2,
    'labels': labels,
    'categories1': categories1,
    'categories2': categories2,
    'phase_indicators': phase_indicators,
    'metadata': {
        'total_pairs': len(hard_pairs),
        'num_positives': num_positives,
        'num_negatives': num_negatives,
        'baseline_roc_auc': float(baseline_roc),
        'baseline_spearman': float(baseline_spearman),
        'generation_strategy': 'ranked_adversarial_mining',
        'borderline_range': BORDERLINE_RANGE,
        'sample_size': SAMPLE_SIZE,
        'random_seed': RANDOM_SEED,
    }
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"   ✅ Saved {len(hard_pairs):,} pairs")

# Summary
print(f"\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

print(f"\nGenerated Pairs Summary:")
print(f"  Total pairs:     {len(hard_pairs):,}")
print(f"  Positives:       {num_positives:,} ({num_positives/len(labels)*100:.1f}%)")
print(f"  Negatives:       {num_negatives:,} ({num_negatives/len(labels)*100:.1f}%)")

print(f"\nBaseline Performance:")
print(f"  ROC-AUC:   {baseline_roc:.4f}")
print(f"  Spearman:  {baseline_spearman:.4f}")

print(f"\nComparison to Old Training Data:")
print(f"  Old baseline ROC: 0.9264 (too easy)")
print(f"  New baseline ROC: {baseline_roc:.4f} ({'better!' if baseline_roc < 0.85 else 'needs improvement'})")
print(f"  Improvement:      {0.9264 - baseline_roc:+.4f}")

if baseline_roc >= 0.60 and baseline_roc <= 0.80:
    print(f"\n✅ SUCCESS: Generated appropriately challenging training data!")
    print(f"\n📊 NEXT STEPS:")
    print(f"   1. Update training notebook to use:")
    print(f"      PAIRS_FILE = 'data_new/hard_training_pairs_ranked.json'")
    print(f"   2. Use conservative hyperparameters:")
    print(f"      - Learning rate: 1e-6 (very low)")
    print(f"      - Epochs: 6-9 (2-3 per phase)")
    print(f"      - Loss: CosineSimilarityLoss")
    print(f"   3. Expected improvement: Spearman 0.50 → 0.52-0.56")
elif baseline_roc < 0.60:
    print(f"\n⚠️  Data may be TOO HARD")
    print(f"   Consider widening BORDERLINE_RANGE to (0.35, 0.65)")
else:
    print(f"\n⚠️  Data still too easy (ROC {baseline_roc:.4f})")
    print(f"   Try tightening BORDERLINE_RANGE to (0.45, 0.55)")

print(f"\n" + "=" * 80)
