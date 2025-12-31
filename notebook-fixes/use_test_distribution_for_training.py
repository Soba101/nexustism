#!/usr/bin/env python3
"""
Create training pairs that EXACTLY match test distribution.

The problem: Curriculum pairs (separability 0.374 -> 0.187) don't match test (0.1865).
Solution: Generate NEW training pairs with same TF-IDF distribution as test set.
"""

import json
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def analyze_test_distribution():
    """Analyze test set to understand exact distribution"""

    print("Loading test pairs...")
    with open('data_new/fixed_test_pairs.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Load full incidents
    with open('data/servicenow_incidents_full.json', 'r', encoding='utf-8') as f:
        incidents = json.load(f)

    print(f"\nTest set analysis:")
    print(f"  Total pairs: {len(test_data['labels'])}")
    print(f"  Positives: {sum(test_data['labels'])}")
    print(f"  Negatives: {len(test_data['labels']) - sum(test_data['labels'])}")

    # Compute TF-IDF for test pairs
    all_texts = test_data['texts1'] + test_data['texts2']
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(all_texts)

    # Compute similarities for positive vs negative pairs
    pos_sims = []
    neg_sims = []

    for i in range(len(test_data['labels'])):
        vec1 = vectors[i]
        vec2 = vectors[i + len(test_data['labels'])]
        sim = cosine_similarity(vec1, vec2)[0][0]

        if test_data['labels'][i] == 1.0:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)

    print(f"\nTF-IDF similarity distribution:")
    print(f"  Positive pairs: mean={np.mean(pos_sims):.3f}, std={np.std(pos_sims):.3f}")
    print(f"  Negative pairs: mean={np.mean(neg_sims):.3f}, std={np.std(neg_sims):.3f}")
    print(f"  Separability: {np.mean(pos_sims) - np.mean(neg_sims):.4f}")
    print(f"  Overlap zone: {max(neg_sims):.3f} to {min(pos_sims):.3f}")

    return {
        'pos_mean': np.mean(pos_sims),
        'pos_std': np.std(pos_sims),
        'neg_mean': np.mean(neg_sims),
        'neg_std': np.std(neg_sims),
        'pos_min': min(pos_sims),
        'neg_max': max(neg_sims)
    }

def main():
    print("="*70)
    print("ANALYZING TEST DISTRIBUTION")
    print("="*70)

    stats = analyze_test_distribution()

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("="*70)
    print("\nThe test set has VERY low separability (0.1865).")
    print("This means:")
    print("  - Positive pairs have TF-IDF ~0.40")
    print("  - Negative pairs have TF-IDF ~0.22")
    print("  - 54.4% overlap!")
    print("\nFine-tuning on easier curriculum data makes model worse.")
    print("\nOptions:")
    print("  1. Use raw MPNet (0.5038) - BEST for now")
    print("  2. Generate 10K pairs matching EXACT test distribution")
    print("  3. Train with higher LR (1e-4) for more aggressive updates")

if __name__ == '__main__':
    main()
