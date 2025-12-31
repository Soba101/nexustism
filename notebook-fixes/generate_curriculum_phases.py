#!/usr/bin/env python3
"""
Generate complete curriculum training dataset with Phases 2 & 3.

Phase 1 (Easy): Already exists in fixed_training_pairs.json
  - Positive: similarity ≥0.52
  - Negative: similarity ≤0.36
  - Separability: 0.374

Phase 2 (Medium): Generate here
  - Positive: 0.40 ≤ similarity < 0.52
  - Negative: 0.36 < similarity ≤ 0.45
  - Target separability: ~0.27

Phase 3 (Hard): Generate here
  - Positive: 0.30 ≤ similarity < 0.40
  - Negative: 0.45 < similarity ≤ 0.50
  - Target separability: ~0.19 (matches test difficulty)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Configuration
DATA_DIR = Path('data_new')
CSV_PATH = DATA_DIR / 'SNow_incident_ticket_data.csv'
PHASE1_PATH = DATA_DIR / 'fixed_training_pairs.json'
OUTPUT_PATH = DATA_DIR / 'curriculum_training_pairs_complete.json'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Phase configuration
PHASE_CONFIGS = {
    2: {
        'name': 'Medium',
        'pos_min': 0.40,
        'pos_max': 0.52,
        'neg_min': 0.36,
        'neg_max': 0.45,
        'target_count': 5000,
        'target_separability': 0.27
    },
    3: {
        'name': 'Hard',
        'pos_min': 0.30,
        'pos_max': 0.40,
        'neg_min': 0.45,
        'neg_max': 0.50,
        'target_count': 5000,
        'target_separability': 0.19
    }
}

def create_combined_text(row):
    """Create combined text same way as notebooks."""
    text_parts = []
    for col in ['Number', 'Description', 'User input', 'Resolution notes']:
        if col in row.index:
            value = str(row.get(col, '')).strip() if pd.notna(row.get(col)) else ''
            if value and value.lower() != 'nan':
                text_parts.append(value)
    return ' '.join(text_parts) if text_parts else ''

def compute_separability(positive_sims, negative_sims):
    """Compute separability metric."""
    pos_mean = np.mean(positive_sims)
    neg_mean = np.mean(negative_sims)
    pos_std = np.std(positive_sims)
    neg_std = np.std(negative_sims)

    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    if pooled_std == 0:
        return 0.0

    return abs(pos_mean - neg_mean) / pooled_std

def compute_overlap(positive_sims, negative_sims):
    """Compute overlap percentage."""
    pos_min = np.min(positive_sims)
    neg_max = np.max(negative_sims)

    overlap_pairs = sum(1 for p in positive_sims if p <= neg_max)
    overlap_pairs += sum(1 for n in negative_sims if n >= pos_min)

    total = len(positive_sims) + len(negative_sims)
    return (overlap_pairs / total) * 100 if total > 0 else 0.0

def generate_phase_pairs(df, embeddings, model, phase_num, config):
    """Generate pairs for a specific phase."""
    print(f"\n{'='*80}")
    print(f"GENERATING PHASE {phase_num}: {config['name']} Difficulty")
    print(f"{'='*80}")

    print(f"\nThresholds:")
    print(f"  Positives: {config['pos_min']:.2f} ≤ similarity < {config['pos_max']:.2f}")
    print(f"  Negatives: {config['neg_min']:.2f} < similarity ≤ {config['neg_max']:.2f}")
    print(f"  Target: {config['target_count']} total pairs")

    # Generate candidates with oversampling
    oversample_factor = 4
    target_per_label = config['target_count'] // 2
    candidates_needed = target_per_label * oversample_factor

    print(f"\nGenerating {candidates_needed * 2} candidate pairs...")

    # Generate positive candidates (same category)
    positive_candidates = []
    categories = df['Category'].fillna('Unknown')

    for category in categories.unique():
        cat_indices = df[categories == category].index.tolist()
        if len(cat_indices) < 2:
            continue

        # Sample pairs within this category
        n_samples = min(candidates_needed // len(categories.unique()),
                       len(cat_indices) * (len(cat_indices) - 1) // 2)

        for _ in range(n_samples):
            idx1, idx2 = random.sample(cat_indices, 2)
            sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]

            if config['pos_min'] <= sim < config['pos_max']:
                positive_candidates.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'similarity': sim,
                    'label': 1
                })

    print(f"  Generated {len(positive_candidates)} positive candidates")

    # Generate negative candidates (different category)
    negative_candidates = []

    cat_list = categories.unique().tolist()
    if len(cat_list) < 2:
        print("  WARNING: Only one category, using random negatives")
        cat_list = [cat_list[0], cat_list[0]]

    for _ in range(candidates_needed * 2):
        cat1, cat2 = random.sample(cat_list, 2) if len(cat_list) > 1 else (cat_list[0], cat_list[0])

        indices1 = df[categories == cat1].index.tolist()
        indices2 = df[categories == cat2].index.tolist()

        if not indices1 or not indices2:
            continue

        idx1 = random.choice(indices1)
        idx2 = random.choice(indices2)

        if idx1 == idx2:
            continue

        sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]

        if config['neg_min'] < sim <= config['neg_max']:
            negative_candidates.append({
                'idx1': idx1,
                'idx2': idx2,
                'similarity': sim,
                'label': 0
            })

    print(f"  Generated {len(negative_candidates)} negative candidates")

    # Select best pairs by similarity distribution
    positive_candidates.sort(key=lambda x: x['similarity'], reverse=True)
    negative_candidates.sort(key=lambda x: x['similarity'])

    selected_positives = positive_candidates[:target_per_label]
    selected_negatives = negative_candidates[:target_per_label]

    print(f"\nSelected {len(selected_positives)} positives, {len(selected_negatives)} negatives")

    # Compute quality metrics
    pos_sims = [p['similarity'] for p in selected_positives]
    neg_sims = [n['similarity'] for n in selected_negatives]

    separability = compute_separability(pos_sims, neg_sims)
    overlap = compute_overlap(pos_sims, neg_sims)

    print(f"\nQuality Metrics:")
    print(f"  Positive similarity: {np.mean(pos_sims):.4f} ± {np.std(pos_sims):.4f}")
    print(f"  Negative similarity: {np.mean(neg_sims):.4f} ± {np.std(neg_sims):.4f}")
    print(f"  Separability: {separability:.4f} (target: {config['target_separability']:.2f})")
    print(f"  Overlap: {overlap:.1f}%")

    # Create pair objects
    pairs = []
    for p in selected_positives:
        pairs.append({
            'texts1': df.iloc[p['idx1']]['combined_text'],
            'texts2': df.iloc[p['idx2']]['combined_text'],
            'labels': 1,
            'phase': phase_num,
            'similarity': float(p['similarity'])
        })

    for n in selected_negatives:
        pairs.append({
            'texts1': df.iloc[n['idx1']]['combined_text'],
            'texts2': df.iloc[n['idx2']]['combined_text'],
            'labels': 0,
            'phase': phase_num,
            'similarity': float(n['similarity'])
        })

    return pairs, separability, overlap

def main():
    print("="*80)
    print("CURRICULUM TRAINING PAIRS GENERATION")
    print("="*80)

    # Load data
    print(f"\n1. Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f"   Loaded {len(df)} incidents")

    # Create combined text
    print(f"\n2. Creating combined text...")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df = df[df['combined_text'].str.len() > 10].reset_index(drop=True)
    df['Category'] = df['Category'].fillna('Unknown')
    print(f"   {len(df)} valid incidents after filtering")

    # Load baseline model
    print(f"\n3. Loading baseline MPNet model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Generate embeddings
    print(f"\n4. Generating embeddings for {len(df)} incidents...")
    texts = df['combined_text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"   Embeddings shape: {embeddings.shape}")

    # Load Phase 1 pairs
    print(f"\n5. Loading Phase 1 (Easy) pairs from {PHASE1_PATH}")
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_data = json.load(f)

    # Convert Phase 1 to new format
    phase1_pairs = []
    for i in range(len(phase1_data['texts1'])):
        phase1_pairs.append({
            'texts1': phase1_data['texts1'][i],
            'texts2': phase1_data['texts2'][i],
            'labels': int(phase1_data['labels'][i]),
            'phase': 1,
            'similarity': None  # Not stored in original
        })

    print(f"   Loaded {len(phase1_pairs)} Phase 1 pairs")

    # Generate Phase 2 & 3
    all_pairs = phase1_pairs.copy()

    for phase_num in [2, 3]:
        config = PHASE_CONFIGS[phase_num]
        pairs, sep, overlap = generate_phase_pairs(df, embeddings, model, phase_num, config)
        all_pairs.extend(pairs)

    # Summary
    print(f"\n{'='*80}")
    print("CURRICULUM DATASET SUMMARY")
    print(f"{'='*80}")

    phase_counts = {}
    for pair in all_pairs:
        phase_counts[pair['phase']] = phase_counts.get(pair['phase'], 0) + 1

    print(f"\nTotal pairs: {len(all_pairs)}")
    for phase in sorted(phase_counts.keys()):
        phase_name = {1: 'Easy', 2: 'Medium', 3: 'Hard'}.get(phase, f'Phase {phase}')
        print(f"  Phase {phase} ({phase_name}): {phase_counts[phase]} pairs")

    # Save
    print(f"\n6. Saving to {OUTPUT_PATH}")

    # Convert to final format
    output_data = {
        'texts1': [p['texts1'] for p in all_pairs],
        'texts2': [p['texts2'] for p in all_pairs],
        'labels': [p['labels'] for p in all_pairs],
        'phases': [p['phase'] for p in all_pairs],
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_pairs': len(all_pairs),
            'phase_counts': phase_counts,
            'seed': SEED
        }
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(all_pairs)} pairs")

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print(f"\nCurriculum dataset ready: {OUTPUT_PATH}")
    print(f"Next: Update model_promax_mpnet_lorapeft.ipynb to use this dataset")

if __name__ == '__main__':
    main()
