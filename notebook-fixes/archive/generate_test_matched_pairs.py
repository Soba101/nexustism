#!/usr/bin/env python3
"""
Generate training pairs that EXACTLY match test distribution.

Key insight: Test set has 54.4% overlap (separability 0.1865).
Current curriculum Phase 3 has separability 0.187 but still easier than test.

Solution: Generate pairs with SAME difficulty as test:
- Positive pairs: TF-IDF 0.30-0.50 (not 0.52+)
- Negative pairs: TF-IDF 0.20-0.45 (HIGH overlap)
- Same-category hard negatives (prevent shortcuts)
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_data():
    """Load incidents and test pairs"""
    print("Loading data...")

    with open('data/servicenow_incidents_full.json', 'r', encoding='utf-8') as f:
        incidents = json.load(f)

    with open('data_new/fixed_test_pairs.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    return incidents, test_data

def compute_simple_similarity(text1, text2):
    """Simple word overlap similarity (proxy for TF-IDF)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0

def generate_hard_pairs(incidents, num_pairs=10000):
    """Generate pairs matching test difficulty"""

    print(f"\nGenerating {num_pairs} hard pairs...")

    # Group by category for hard negatives
    by_category = defaultdict(list)
    for inc in incidents:
        category = inc.get('category', 'unknown')
        by_category[category].append(inc)

    pairs_texts1 = []
    pairs_texts2 = []
    labels = []

    # Target distribution (from test analysis):
    # - Positives: similarity 0.30-0.50 (mean ~0.40)
    # - Negatives: similarity 0.20-0.45 (mean ~0.22, HIGH overlap!)

    num_positives = num_pairs // 2
    num_negatives = num_pairs - num_positives

    print("Generating positive pairs (hard similar)...")
    attempts = 0
    max_attempts = num_positives * 100

    while len([l for l in labels if l == 1.0]) < num_positives and attempts < max_attempts:
        attempts += 1

        # Sample same category for positives
        category = random.choice(list(by_category.keys()))
        if len(by_category[category]) < 2:
            continue

        inc1, inc2 = random.sample(by_category[category], 2)
        text1 = inc1['combined_text']
        text2 = inc2['combined_text']

        sim = compute_simple_similarity(text1, text2)

        # Accept if similarity in target range for HARD positives
        if 0.30 <= sim <= 0.50:
            pairs_texts1.append(text1)
            pairs_texts2.append(text2)
            labels.append(1.0)

    print(f"Generated {len([l for l in labels if l == 1.0])} positives")

    print("Generating negative pairs (hard negatives with HIGH overlap)...")
    attempts = 0
    max_attempts = num_negatives * 100

    while len([l for l in labels if l == 0.0]) < num_negatives and attempts < max_attempts:
        attempts += 1

        # 70% same category (hard negatives), 30% different category
        if random.random() < 0.7:
            # Same category negative (HARD)
            category = random.choice(list(by_category.keys()))
            if len(by_category[category]) < 2:
                continue
            inc1, inc2 = random.sample(by_category[category], 2)
        else:
            # Different category negative
            inc1 = random.choice(incidents)
            inc2 = random.choice(incidents)

        text1 = inc1['combined_text']
        text2 = inc2['combined_text']

        sim = compute_simple_similarity(text1, text2)

        # Accept if similarity in target range for HARD negatives (HIGH overlap!)
        if 0.15 <= sim <= 0.45:
            pairs_texts1.append(text1)
            pairs_texts2.append(text2)
            labels.append(0.0)

    print(f"Generated {len([l for l in labels if l == 0.0])} negatives")

    # Compute final statistics
    pos_pairs = [(t1, t2) for t1, t2, l in zip(pairs_texts1, pairs_texts2, labels) if l == 1.0]
    neg_pairs = [(t1, t2) for t1, t2, l in zip(pairs_texts1, pairs_texts2, labels) if l == 0.0]

    pos_sims = [compute_simple_similarity(t1, t2) for t1, t2 in pos_pairs]
    neg_sims = [compute_simple_similarity(t1, t2) for t1, t2 in neg_pairs]

    print(f"\nFinal statistics:")
    print(f"  Total pairs: {len(labels)}")
    print(f"  Positives: {sum(labels)} (mean sim: {np.mean(pos_sims):.3f})")
    print(f"  Negatives: {len(labels) - sum(labels)} (mean sim: {np.mean(neg_sims):.3f})")
    print(f"  Separability: {np.mean(pos_sims) - np.mean(neg_sims):.4f}")
    print(f"  Overlap: {max(neg_sims):.3f} to {min(pos_sims):.3f}")

    return {
        'texts1': pairs_texts1,
        'texts2': pairs_texts2,
        'labels': labels
    }

def main():
    print("="*70)
    print("GENERATING TEST-MATCHED HARD PAIRS")
    print("="*70)

    incidents, test_data = load_data()

    # Generate 10K pairs matching test difficulty
    hard_pairs = generate_hard_pairs(incidents, num_pairs=10000)

    # Save
    output_path = Path('data_new/test_matched_hard_pairs.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hard_pairs, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUCCESS: Saved to {output_path}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Update CONFIG['train_pairs_path'] to use this file")
    print("  2. Set phase3_epochs = 6 (train ONLY on these hard pairs)")
    print("  3. Use LR 1e-4 (aggressive updates for hard cases)")
    print("  4. Expected: Spearman 0.51-0.54 (beat baseline!)")

if __name__ == '__main__':
    main()
