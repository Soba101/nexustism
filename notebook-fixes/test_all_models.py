#!/usr/bin/env python3
"""
Test multiple baseline models on your hard test set.

This will show which model performs best WITHOUT fine-tuning.
"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import sys

def test_model(model_name, texts1, texts2, labels):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print('='*70)

    try:
        # Load model
        if 'nomic' in model_name.lower():
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name)

        print("Encoding texts...")
        # Encode
        emb1 = model.encode(texts1, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
        emb2 = model.encode(texts2, show_progress_bar=True, convert_to_numpy=True, batch_size=32)

        # Compute similarities
        print("Computing similarities...")
        sims = np.array([np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
                        for e1, e2 in zip(emb1, emb2)])

        # Metrics
        spearman, _ = spearmanr(labels, sims)
        roc_auc = roc_auc_score(labels, sims)

        # Separability
        pos_sims = sims[np.array(labels) == 1.0]
        neg_sims = sims[np.array(labels) == 0.0]
        separability = np.mean(pos_sims) - np.mean(neg_sims)

        print(f"\nResults:")
        print(f"  Spearman:     {spearman:.4f}")
        print(f"  ROC-AUC:      {roc_auc:.4f}")
        print(f"  Separability: {separability:.4f}")
        print(f"  Pos mean:     {np.mean(pos_sims):.4f}")
        print(f"  Neg mean:     {np.mean(neg_sims):.4f}")

        return {
            'model': model_name,
            'spearman': spearman,
            'roc_auc': roc_auc,
            'separability': separability,
            'success': True
        }

    except Exception as e:
        print(f"\n[ERROR] Failed to test {model_name}: {e}")
        return {
            'model': model_name,
            'spearman': 0.0,
            'roc_auc': 0.0,
            'separability': 0.0,
            'success': False,
            'error': str(e)
        }

def main():
    print("="*70)
    print("TESTING MULTIPLE BASELINE MODELS")
    print("="*70)

    # Load test data
    print("\nLoading test data...")
    with open('data_new/fixed_test_pairs.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    texts1 = test_data['texts1']
    texts2 = test_data['texts2']
    labels = test_data['labels']

    print(f"Loaded {len(labels)} test pairs")
    print(f"  Positives: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Negatives: {len(labels)-sum(labels)}")

    # Models to test (ordered by likelihood of success)
    models_to_try = [
        # Your current baseline
        'sentence-transformers/all-mpnet-base-v2',

        # Nomic (designed for hard negatives)
        'nomic-ai/nomic-embed-text-v1.5',

        # BGE (strong at retrieval)
        'BAAI/bge-base-en-v1.5',

        # GTE (top MTEB performer)
        'Alibaba-NLP/gte-base-en-v1.5',

        # Alternative strong models
        'intfloat/e5-base-v2',
        'sentence-transformers/all-MiniLM-L12-v2',
    ]

    results = []

    # Test each model
    for model_name in models_to_try:
        result = test_model(model_name, texts1, texts2, labels)
        results.append(result)

    # Print final ranking
    print("\n" + "="*70)
    print("FINAL RESULTS (Ranked by Spearman)")
    print("="*70)

    successful_results = [r for r in results if r['success']]
    successful_results.sort(key=lambda x: x['spearman'], reverse=True)

    baseline_spearman = 0.5038

    print(f"\n{'Model':<50} {'Spearman':>10} {'vs Baseline':>12} {'ROC-AUC':>10}")
    print("-"*90)

    for i, r in enumerate(successful_results, 1):
        improvement = ((r['spearman'] - baseline_spearman) / baseline_spearman) * 100
        marker = " ⭐" if r['spearman'] > baseline_spearman else ""
        print(f"{i}. {r['model']:<47} {r['spearman']:>10.4f} {improvement:>+10.1f}% {r['roc_auc']:>10.4f}{marker}")

    # Failed models
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n" + "="*70)
        print("FAILED MODELS")
        print("="*70)
        for r in failed_results:
            print(f"  {r['model']}: {r.get('error', 'Unknown error')}")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if successful_results:
        best = successful_results[0]
        improvement = ((best['spearman'] - baseline_spearman) / baseline_spearman) * 100

        if best['spearman'] > baseline_spearman:
            print(f"\n✅ WINNER: {best['model']}")
            print(f"   Spearman: {best['spearman']:.4f} ({improvement:+.1f}% vs baseline)")
            print(f"\n   Use this model in production!")
        else:
            print(f"\n❌ No model beat the baseline (0.5038)")
            print(f"   Best: {best['model']} ({best['spearman']:.4f})")
            print(f"\n   Recommendation:")
            print(f"   1. Stick with MPNet baseline (0.5038)")
            print(f"   2. Generate test-matched training pairs")
            print(f"   3. Fine-tune BGE or Nomic on matched pairs")

    print("="*70)

if __name__ == '__main__':
    main()
