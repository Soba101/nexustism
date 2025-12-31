#!/usr/bin/env python3
"""
Simple Data Augmentation for ITSM Ticket Pairs (No External Dependencies)

Techniques:
1. Random Word Deletion (10% deletion rate)
2. Random Word Swap (swap adjacent words)
3. Random Word Duplication (repeat random words)
4. Combine original + augmented for 3-5x expansion

No external libraries required - pure Python implementation
"""

import json
import random
import re
from typing import List, Tuple


class SimpleAugmenter:
    """Simple data augmentation without external dependencies"""

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

        # Stop words to avoid deleting (preserve meaning)
        self.stop_words = {
            'error', 'failed', 'success', 'issue', 'problem', 'ticket',
            'incident', 'not', 'no', 'unable', 'cannot', 'can', 'please',
            'urgent', 'critical', 'high', 'low', 'priority'
        }

    def extract_metadata(self, text: str) -> Tuple[str, str]:
        """Extract metadata from ticket text"""
        match = re.search(r'\(Context:.*?\.\)$', text)
        if match:
            metadata = match.group(0)
            clean_text = text[:match.start()].strip()
            return clean_text, metadata
        else:
            return text, ""

    def random_deletion(self, text: str, p=0.10) -> str:
        """
        Randomly delete p% of words (except stop words)

        Args:
            text: Original text
            p: Deletion probability (default 10%)

        Returns:
            Augmented text
        """
        clean_text, metadata = self.extract_metadata(text)
        words = clean_text.split()

        # Don't delete if too few words
        if len(words) <= 5:
            return text

        # Keep important words, randomly delete others
        new_words = []
        for word in words:
            # Keep stop words
            if word.lower() in self.stop_words:
                new_words.append(word)
            # Randomly delete others
            elif random.random() > p:
                new_words.append(word)

        # Ensure at least some words remain
        if len(new_words) < 3:
            new_words = words

        augmented = ' '.join(new_words)

        # Restore metadata
        if metadata:
            return f"{augmented} {metadata}"
        else:
            return augmented

    def random_swap(self, text: str, n=3) -> str:
        """
        Randomly swap n pairs of adjacent words

        Args:
            text: Original text
            n: Number of swaps

        Returns:
            Augmented text
        """
        clean_text, metadata = self.extract_metadata(text)
        words = clean_text.split()

        # Don't swap if too few words
        if len(words) <= 3:
            return text

        new_words = words.copy()

        # Perform n random swaps
        for _ in range(n):
            if len(new_words) >= 2:
                idx = random.randint(0, len(new_words) - 2)
                new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]

        augmented = ' '.join(new_words)

        # Restore metadata
        if metadata:
            return f"{augmented} {metadata}"
        else:
            return augmented

    def random_duplication(self, text: str, p=0.05) -> str:
        """
        Randomly duplicate p% of words

        Args:
            text: Original text
            p: Duplication probability (default 5%)

        Returns:
            Augmented text
        """
        clean_text, metadata = self.extract_metadata(text)
        words = clean_text.split()

        # Don't duplicate if too few words
        if len(words) <= 3:
            return text

        new_words = []
        for word in words:
            new_words.append(word)
            # Randomly duplicate
            if random.random() < p:
                new_words.append(word)

        augmented = ' '.join(new_words)

        # Restore metadata
        if metadata:
            return f"{augmented} {metadata}"
        else:
            return augmented

    def augment_text(self, text: str, num_augments=2) -> List[str]:
        """
        Generate multiple augmentations of a text

        Args:
            text: Original text
            num_augments: Number of augmentations to generate

        Returns:
            List of augmented texts (including original)
        """
        augmented = [text]  # Always include original

        methods = [
            self.random_deletion,
            self.random_swap,
            self.random_duplication
        ]

        for i in range(num_augments):
            # Randomly select method
            method = random.choice(methods)
            aug_text = method(text)

            # Only add if different from original
            if aug_text != text:
                augmented.append(aug_text)

        return augmented[:num_augments + 1]  # Return at most num_augments + original


def augment_dataset(
    input_path='data_new/curriculum_training_pairs_complete.json',
    output_path='data_new/curriculum_training_pairs_augmented_simple.json',
    augments_per_pair=2,
    seed=42
):
    """
    Augment entire dataset

    Args:
        input_path: Input pairs file
        output_path: Output augmented pairs file
        augments_per_pair: Number of augmentations per pair (default 2 -> 3x total)
        seed: Random seed
    """
    print("="*80)
    print("SIMPLE DATA AUGMENTATION (No External Dependencies)")
    print("="*80)
    print()

    # Load original data
    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts1 = data['texts1']
    texts2 = data['texts2']
    labels = data['labels']
    phase_indicators = data['phase_indicators']

    original_count = len(texts1)
    print(f"Original pairs: {original_count:,}")

    # Count positives/negatives
    pos_count = sum(1 for l in labels if l == 1)
    neg_count = original_count - pos_count
    print(f"  Positives: {pos_count:,} ({100*pos_count/original_count:.1f}%)")
    print(f"  Negatives: {neg_count:,} ({100*neg_count/original_count:.1f}%)")
    print()

    # Initialize augmenter
    augmenter = SimpleAugmenter(seed=seed)

    print(f"Augmenting {augments_per_pair} variations per pair...")
    print(f"Expected final count: {original_count * (1 + augments_per_pair):,} pairs")
    print()

    # Augment data
    augmented_texts1 = []
    augmented_texts2 = []
    augmented_labels = []
    augmented_phases = []

    for i, (t1, t2, label, phase) in enumerate(zip(texts1, texts2, labels, phase_indicators)):
        # Show progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{original_count:,} pairs...")

        # For positives: augment both texts
        if label == 1.0:
            # Augment text1
            aug_t1_list = augmenter.augment_text(t1, num_augments=augments_per_pair)

            # Augment text2
            aug_t2_list = augmenter.augment_text(t2, num_augments=augments_per_pair)

            # Create pairs: (original, original) + (aug1, original) + (original, aug2) + ...
            for aug_t1 in aug_t1_list:
                augmented_texts1.append(aug_t1)
                augmented_texts2.append(t2)
                augmented_labels.append(label)
                augmented_phases.append(phase)

            for aug_t2 in aug_t2_list[1:]:  # Skip first (original) to avoid duplicate
                augmented_texts1.append(t1)
                augmented_texts2.append(aug_t2)
                augmented_labels.append(label)
                augmented_phases.append(phase)

        # For negatives: only augment text1 (to save space)
        else:
            aug_t1_list = augmenter.augment_text(t1, num_augments=augments_per_pair)

            for aug_t1 in aug_t1_list:
                augmented_texts1.append(aug_t1)
                augmented_texts2.append(t2)
                augmented_labels.append(label)
                augmented_phases.append(phase)

    final_count = len(augmented_texts1)
    expansion_factor = final_count / original_count

    print()
    print(f"Augmented pairs: {final_count:,} (expansion: {expansion_factor:.1f}x)")

    # Count final positives/negatives
    final_pos = sum(1 for l in augmented_labels if l == 1)
    final_neg = final_count - final_pos
    print(f"  Positives: {final_pos:,} ({100*final_pos/final_count:.1f}%)")
    print(f"  Negatives: {final_neg:,} ({100*final_neg/final_count:.1f}%)")

    # Count per phase
    from collections import Counter
    phase_counts = Counter(augmented_phases)
    print(f"\nPhase distribution:")
    print(f"  Phase 1 (easy): {phase_counts[1]:,} pairs")
    print(f"  Phase 2 (medium): {phase_counts[2]:,} pairs")
    print(f"  Phase 3 (hard): {phase_counts[3]:,} pairs")

    # Save augmented data
    augmented_data = {
        'texts1': augmented_texts1,
        'texts2': augmented_texts2,
        'labels': augmented_labels,
        'phase_indicators': augmented_phases,
        'metadata': {
            'original_count': original_count,
            'augmented_count': final_count,
            'expansion_factor': expansion_factor,
            'augments_per_pair': augments_per_pair,
            'seed': seed,
            'methods': ['random_deletion', 'random_swap', 'random_duplication']
        }
    }

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print()
    print("="*80)
    print("[SUCCESS] Data augmentation complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Update model_promax_mpnet_lorapeft_v3.ipynb CONFIG:")
    print(f"     'train_pairs_path': '{output_path}'")
    print("  2. Adjust epochs (may need to reduce with more data)")
    print("  3. Run training with augmented data")
    print("  4. Expected improvement: Spearman 0.60-0.65 (+19-29%)")
    print()
    print(f"Backup original config before training!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple data augmentation')
    parser.add_argument('--input', default='data_new/curriculum_training_pairs_complete.json')
    parser.add_argument('--output', default='data_new/curriculum_training_pairs_augmented_simple.json')
    parser.add_argument('--augments', type=int, default=2, help='Augmentations per pair')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        augments_per_pair=args.augments,
        seed=args.seed
    )
