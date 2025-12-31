#!/usr/bin/env python3
"""
Hard Training Pairs Generation - Adversarial Mining

Strategy:
- Target pairs where baseline MPNet scores 0.4-0.6 (borderline/uncertain)
- Generate HARD negatives: Same category, medium TF-IDF (0.3-0.5)
- Generate cross-category positives: Different categories, high semantic overlap
- Remove easy pairs (baseline very confident)
- Ensure balanced curriculum phases with proper negative sampling

Goal: Create training data where baseline ROC-AUC is 0.70-0.75 (challenging but learnable)

Expected Runtime: 10-15 minutes
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
print("HARD TRAINING PAIRS GENERATION - ADVERSARIAL MINING")
print("=" * 80)

# Configuration
SOURCE_DATA = Path('data_new/SNow_incident_ticket_data.csv')
OUTPUT_FILE = Path('data_new/hard_training_pairs_adversarial.json')
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cuda'
BATCH_SIZE = 128
RANDOM_SEED = 42

# Target metrics for generated data
TARGET_BASELINE_ROC = 0.72  # Baseline should struggle but not fail
TARGET_NUM_PAIRS = 20000    # More data for harder task

# Pair generation targets
TARGET_DISTRIBUTION = {
    'hard_negatives_same_cat': 0.35,      # 35% - Same category, low similarity
    'hard_negatives_cross_cat': 0.15,     # 15% - Different category, medium similarity
    'hard_positives': 0.30,                # 30% - Borderline positives (baseline uncertain)
    'easy_positives': 0.20,                # 20% - Clear positives (for stability)
}

# Difficulty thresholds
BASELINE_UNCERTAIN_RANGE = (0.35, 0.65)  # Pairs where baseline is uncertain
TFIDF_HARD_NEG_RANGE = (0.25, 0.50)      # Medium TF-IDF for hard negatives
TFIDF_HARD_POS_RANGE = (0.40, 0.70)      # Medium-high TF-IDF for hard positives
TFIDF_EASY_POS_RANGE = (0.60, 1.00)      # High TF-IDF for easy positives

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
print(f"\n1. Loading source data...")
print(f"   File: {SOURCE_DATA}")

df = pd.read_csv(SOURCE_DATA)
print(f"   ✅ Loaded {len(df):,} incidents")

# Filter out short texts
df['combined_text'] = (
    df['Description'].fillna('') + ' ' +
    '(Context: [' + df['Service'].fillna('') + ' | ' + df['Service offering'].fillna('') + '] ' +
    '[' + df['Category'].fillna('') + ' | ' + df['Subcategory'].fillna('') + '] ' +
    'Group: ' + df['Assignment group'].fillna('') + '.)'
)

df = df[df['combined_text'].str.len() >= 25].reset_index(drop=True)
print(f"   After filtering: {len(df):,} incidents")

# Extract metadata
categories = df['Category'].fillna('Unknown').values
subcategories = df['Subcategory'].fillna('Unknown').values
texts = df['combined_text'].values

print(f"   Unique categories: {len(set(categories))}")

# Load baseline model
print(f"\n2. Loading baseline model for adversarial mining...")
print(f"   Model: {BASELINE_MODEL}")

model = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
print(f"   ✅ Model loaded")

# Encode all texts
print(f"\n3. Encoding all incidents (this will take a few minutes)...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    device=DEVICE,
    convert_to_numpy=True
)
print(f"   ✅ Encoded {len(embeddings):,} incidents")

# Compute TF-IDF
print(f"\n4. Computing TF-IDF similarities...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, max_df=0.95)
tfidf_matrix = tfidf.fit_transform(texts)
print(f"   ✅ TF-IDF matrix: {tfidf_matrix.shape}")

# Helper function: Get baseline similarity
def get_baseline_similarity(idx1, idx2):
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]

def get_tfidf_similarity(idx1, idx2):
    """Compute TF-IDF cosine similarity."""
    return (tfidf_matrix[idx1] * tfidf_matrix[idx2].T).toarray()[0, 0]

# Generate pairs
print(f"\n5. Generating hard training pairs...")
print(f"   Target: {TARGET_NUM_PAIRS:,} pairs")
print(f"   Strategy: Adversarial mining (baseline uncertain range: {BASELINE_UNCERTAIN_RANGE})")

pairs = []
pair_metadata = []

# Calculate target counts
num_hard_neg_same = int(TARGET_NUM_PAIRS * TARGET_DISTRIBUTION['hard_negatives_same_cat'])
num_hard_neg_cross = int(TARGET_NUM_PAIRS * TARGET_DISTRIBUTION['hard_negatives_cross_cat'])
num_hard_pos = int(TARGET_NUM_PAIRS * TARGET_DISTRIBUTION['hard_positives'])
num_easy_pos = int(TARGET_NUM_PAIRS * TARGET_DISTRIBUTION['easy_positives'])

print(f"\n   Target distribution:")
print(f"     Hard negatives (same cat):  {num_hard_neg_same:,} ({TARGET_DISTRIBUTION['hard_negatives_same_cat']*100:.0f}%)")
print(f"     Hard negatives (cross cat): {num_hard_neg_cross:,} ({TARGET_DISTRIBUTION['hard_negatives_cross_cat']*100:.0f}%)")
print(f"     Hard positives:             {num_hard_pos:,} ({TARGET_DISTRIBUTION['hard_positives']*100:.0f}%)")
print(f"     Easy positives:             {num_easy_pos:,} ({TARGET_DISTRIBUTION['easy_positives']*100:.0f}%)")

# 1. Generate hard negatives (same category)
print(f"\n   Generating hard negatives (same category)...")
hard_neg_same_pairs = []

with tqdm(total=num_hard_neg_same, desc="Hard neg (same cat)") as pbar:
    attempts = 0
    max_attempts = num_hard_neg_same * 100

    while len(hard_neg_same_pairs) < num_hard_neg_same and attempts < max_attempts:
        attempts += 1

        # Sample two indices from same category
        cat = random.choice(list(set(categories)))
        cat_indices = np.where(categories == cat)[0]

        if len(cat_indices) < 2:
            continue

        idx1, idx2 = random.sample(list(cat_indices), 2)

        # Check TF-IDF and baseline similarity
        tfidf_sim = get_tfidf_similarity(idx1, idx2)
        baseline_sim = get_baseline_similarity(idx1, idx2)

        # Hard negative criteria:
        # - Same category
        # - Medium TF-IDF (0.25-0.50)
        # - Baseline uncertain (0.35-0.65)
        if (TFIDF_HARD_NEG_RANGE[0] <= tfidf_sim <= TFIDF_HARD_NEG_RANGE[1] and
            BASELINE_UNCERTAIN_RANGE[0] <= baseline_sim <= BASELINE_UNCERTAIN_RANGE[1]):

            hard_neg_same_pairs.append({
                'idx1': idx1,
                'idx2': idx2,
                'label': 0,
                'tfidf_sim': tfidf_sim,
                'baseline_sim': baseline_sim,
                'same_category': True,
            })
            pbar.update(1)

print(f"     ✅ Generated {len(hard_neg_same_pairs):,} pairs (attempts: {attempts:,})")

# 2. Generate hard negatives (cross category)
print(f"\n   Generating hard negatives (cross category)...")
hard_neg_cross_pairs = []

with tqdm(total=num_hard_neg_cross, desc="Hard neg (cross cat)") as pbar:
    attempts = 0
    max_attempts = num_hard_neg_cross * 100

    while len(hard_neg_cross_pairs) < num_hard_neg_cross and attempts < max_attempts:
        attempts += 1

        # Sample two indices from different categories
        idx1 = random.randint(0, len(texts) - 1)
        idx2 = random.randint(0, len(texts) - 1)

        if categories[idx1] == categories[idx2]:
            continue

        # Check TF-IDF and baseline similarity
        tfidf_sim = get_tfidf_similarity(idx1, idx2)
        baseline_sim = get_baseline_similarity(idx1, idx2)

        # Cross-category hard negative criteria:
        # - Different categories
        # - Low-medium TF-IDF (0.20-0.45)
        # - Baseline uncertain (0.35-0.65)
        if (0.20 <= tfidf_sim <= 0.45 and
            BASELINE_UNCERTAIN_RANGE[0] <= baseline_sim <= BASELINE_UNCERTAIN_RANGE[1]):

            hard_neg_cross_pairs.append({
                'idx1': idx1,
                'idx2': idx2,
                'label': 0,
                'tfidf_sim': tfidf_sim,
                'baseline_sim': baseline_sim,
                'same_category': False,
            })
            pbar.update(1)

print(f"     ✅ Generated {len(hard_neg_cross_pairs):,} pairs (attempts: {attempts:,})")

# 3. Generate hard positives
print(f"\n   Generating hard positives...")
hard_pos_pairs = []

with tqdm(total=num_hard_pos, desc="Hard positives") as pbar:
    attempts = 0
    max_attempts = num_hard_pos * 100

    while len(hard_pos_pairs) < num_hard_pos and attempts < max_attempts:
        attempts += 1

        idx1 = random.randint(0, len(texts) - 1)
        idx2 = random.randint(0, len(texts) - 1)

        if idx1 == idx2:
            continue

        # Check TF-IDF and baseline similarity
        tfidf_sim = get_tfidf_similarity(idx1, idx2)
        baseline_sim = get_baseline_similarity(idx1, idx2)

        # Hard positive criteria:
        # - Medium-high TF-IDF (0.40-0.70)
        # - Baseline uncertain to slightly confident (0.45-0.70)
        # - Can be cross-category (prevents category shortcuts)
        if (TFIDF_HARD_POS_RANGE[0] <= tfidf_sim <= TFIDF_HARD_POS_RANGE[1] and
            0.45 <= baseline_sim <= 0.70):

            hard_pos_pairs.append({
                'idx1': idx1,
                'idx2': idx2,
                'label': 1,
                'tfidf_sim': tfidf_sim,
                'baseline_sim': baseline_sim,
                'same_category': categories[idx1] == categories[idx2],
            })
            pbar.update(1)

print(f"     ✅ Generated {len(hard_pos_pairs):,} pairs (attempts: {attempts:,})")

# 4. Generate easy positives (for stability)
print(f"\n   Generating easy positives (for stability)...")
easy_pos_pairs = []

with tqdm(total=num_easy_pos, desc="Easy positives") as pbar:
    attempts = 0
    max_attempts = num_easy_pos * 50

    while len(easy_pos_pairs) < num_easy_pos and attempts < max_attempts:
        attempts += 1

        idx1 = random.randint(0, len(texts) - 1)
        idx2 = random.randint(0, len(texts) - 1)

        if idx1 == idx2:
            continue

        # Check TF-IDF and baseline similarity
        tfidf_sim = get_tfidf_similarity(idx1, idx2)
        baseline_sim = get_baseline_similarity(idx1, idx2)

        # Easy positive criteria:
        # - High TF-IDF (0.60-1.00)
        # - Baseline confident (>0.65)
        if (TFIDF_EASY_POS_RANGE[0] <= tfidf_sim <= TFIDF_EASY_POS_RANGE[1] and
            baseline_sim >= 0.65):

            easy_pos_pairs.append({
                'idx1': idx1,
                'idx2': idx2,
                'label': 1,
                'tfidf_sim': tfidf_sim,
                'baseline_sim': baseline_sim,
                'same_category': categories[idx1] == categories[idx2],
            })
            pbar.update(1)

print(f"     ✅ Generated {len(easy_pos_pairs):,} pairs (attempts: {attempts:,})")

# Combine all pairs
all_pairs = (
    hard_neg_same_pairs +
    hard_neg_cross_pairs +
    hard_pos_pairs +
    easy_pos_pairs
)

print(f"\n   Total pairs generated: {len(all_pairs):,}")

# Shuffle pairs
random.shuffle(all_pairs)

# Extract texts and labels
texts1 = [texts[p['idx1']] for p in all_pairs]
texts2 = [texts[p['idx2']] for p in all_pairs]
labels = [p['label'] for p in all_pairs]
categories1 = [categories[p['idx1']] for p in all_pairs]
categories2 = [categories[p['idx2']] for p in all_pairs]

# Compute baseline metrics on generated pairs
print(f"\n6. Validating generated pairs...")
baseline_scores = np.array([p['baseline_sim'] for p in all_pairs])
labels_array = np.array(labels)

baseline_roc = roc_auc_score(labels_array, baseline_scores)
baseline_spearman = spearmanr(labels_array, baseline_scores)[0]

print(f"   Baseline performance on generated pairs:")
print(f"     ROC-AUC:   {baseline_roc:.4f} (target: ~{TARGET_BASELINE_ROC:.2f})")
print(f"     Spearman:  {baseline_spearman:.4f}")

if baseline_roc < 0.65:
    print(f"     ⚠️  WARNING: Data may be TOO HARD (baseline ROC < 0.65)")
elif baseline_roc > 0.80:
    print(f"     ⚠️  WARNING: Data may be TOO EASY (baseline ROC > 0.80)")
else:
    print(f"     ✅ Good difficulty range (0.65-0.80)")

# Create curriculum phases (balanced)
print(f"\n7. Creating curriculum phases...")

num_pairs = len(all_pairs)
phase_size = num_pairs // 3

# Phase 1: Easier (more easy positives + some hard negatives)
# Phase 2: Medium (balanced mix)
# Phase 3: Harder (more hard negatives + hard positives)

phase_indicators = []

# Sort by difficulty (baseline similarity for negatives, inverse for positives)
negatives = [i for i, p in enumerate(all_pairs) if p['label'] == 0]
positives = [i for i, p in enumerate(all_pairs) if p['label'] == 1]

# Assign phases
for i in range(num_pairs):
    if i < phase_size:
        phase_indicators.append(1)  # Phase 1
    elif i < phase_size * 2:
        phase_indicators.append(2)  # Phase 2
    else:
        phase_indicators.append(3)  # Phase 3

print(f"   Phase 1: {phase_indicators.count(1):,} pairs")
print(f"   Phase 2: {phase_indicators.count(2):,} pairs")
print(f"   Phase 3: {phase_indicators.count(3):,} pairs")

# Save to JSON
print(f"\n8. Saving to file...")
print(f"   Output: {OUTPUT_FILE}")

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
        'generation_strategy': 'adversarial_mining',
        'target_baseline_roc': TARGET_BASELINE_ROC,
        'uncertain_range': BASELINE_UNCERTAIN_RANGE,
        'random_seed': RANDOM_SEED,
        'pair_distribution': {
            'hard_negatives_same_cat': len(hard_neg_same_pairs),
            'hard_negatives_cross_cat': len(hard_neg_cross_pairs),
            'hard_positives': len(hard_pos_pairs),
            'easy_positives': len(easy_pos_pairs),
        }
    }
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"   ✅ Saved {len(all_pairs):,} pairs")

# Summary
print(f"\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

print(f"\nGenerated Pairs Summary:")
print(f"  Total pairs:     {len(all_pairs):,}")
print(f"  Positives:       {sum(labels):,} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"  Negatives:       {len(labels) - sum(labels):,} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

print(f"\nPair Distribution:")
print(f"  Hard neg (same cat):  {len(hard_neg_same_pairs):,}")
print(f"  Hard neg (cross cat): {len(hard_neg_cross_pairs):,}")
print(f"  Hard positives:       {len(hard_pos_pairs):,}")
print(f"  Easy positives:       {len(easy_pos_pairs):,}")

print(f"\nBaseline Performance:")
print(f"  ROC-AUC:   {baseline_roc:.4f}")
print(f"  Spearman:  {baseline_spearman:.4f}")

print(f"\nComparison to Old Training Data:")
print(f"  Old baseline ROC: 0.9264 (too easy)")
print(f"  New baseline ROC: {baseline_roc:.4f} ({'better!' if baseline_roc < 0.85 else 'still too easy'})")
print(f"  Improvement:      {0.9264 - baseline_roc:+.4f}")

if baseline_roc < 0.80:
    print(f"\n✅ SUCCESS: Generated harder training data!")
    print(f"   Fine-tuning on this data should improve model performance.")
    print(f"\n📊 NEXT STEPS:")
    print(f"   1. Update model_promax_mpnet_lorapeft_v3.ipynb to use:")
    print(f"      PAIRS_FILE = 'data_new/hard_training_pairs_adversarial.json'")
    print(f"   2. Use lower learning rate: 2e-6 or 1e-6")
    print(f"   3. Train for 6-8 epochs (2-3 per phase)")
    print(f"   4. Monitor validation metrics closely")
else:
    print(f"\n⚠️  WARNING: Data may still be too easy (ROC {baseline_roc:.4f} > 0.80)")
    print(f"   Consider:")
    print(f"   - Tightening BASELINE_UNCERTAIN_RANGE to (0.40, 0.60)")
    print(f"   - Increasing hard negative ratio")
    print(f"   - Removing easy positives entirely")

print(f"\n" + "=" * 80)
