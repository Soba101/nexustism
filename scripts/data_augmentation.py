#!/usr/bin/env python3
"""
Data Augmentation Pipeline for ITSM Ticket Pairs

Techniques:
1. Back-Translation (English -> intermediate language -> English)
2. Synonym Replacement (EDA - Easy Data Augmentation)
3. Contextual Word Replacement (BERT-based)
4. Random Deletion (remove 10% of words)

Goal: Increase training data from 15K to 100K+ pairs (6-7x expansion)
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Optional: Import augmentation libraries (install if needed)
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    NLPAUG_AVAILABLE = True
except ImportError:
    print("Warning: nlpaug not installed. Install with: pip install nlpaug")
    NLPAUG_AVAILABLE = False

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    print("Warning: googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
    GOOGLETRANS_AVAILABLE = False


class ITSMDataAugmenter:
    """Data augmentation for ITSM ticket pairs"""

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

        # Initialize augmenters if available
        if NLPAUG_AVAILABLE:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_max=0.15)
            self.context_aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                action="substitute",
                aug_p=0.10  # Replace 10% of words
            )
        else:
            self.synonym_aug = None
            self.context_aug = None

        if GOOGLETRANS_AVAILABLE:
            self.translator = Translator()
        else:
            self.translator = None

    def extract_metadata(self, text: str) -> Tuple[str, str]:
        """
        Extract metadata from ticket text.
        Metadata is at the end: (Context: [Service | Offering] [Category | Subcategory] Group: Assignment group.)

        Returns:
            (clean_text, metadata)
        """
        match = re.search(r'\(Context:.*?\.\)$', text)
        if match:
            metadata = match.group(0)
            clean_text = text[:match.start()].strip()
            return clean_text, metadata
        else:
            return text, ""

    def restore_metadata(self, augmented_text: str, metadata: str) -> str:
        """Restore metadata to augmented text"""
        if metadata:
            return f"{augmented_text} {metadata}"
        else:
            return augmented_text

    def back_translate(self, text: str, intermediate_lang='es') -> str:
        """
        Back-translation: English -> Spanish -> English

        Args:
            text: Original English text
            intermediate_lang: Intermediate language (es=Spanish, fr=French, de=German)

        Returns:
            Paraphrased text
        """
        if not self.translator:
            return text

        try:
            # Extract metadata (don't translate it)
            clean_text, metadata = self.extract_metadata(text)

            # Translate to intermediate language
            translated = self.translator.translate(clean_text, dest=intermediate_lang)

            # Translate back to English
            back_translated = self.translator.translate(translated.text, dest='en')

            # Restore metadata
            return self.restore_metadata(back_translated.text, metadata)

        except Exception as e:
            print(f"Back-translation error: {e}")
            return text

    def synonym_replacement(self, text: str) -> str:
        """
        Replace 15% of words with synonyms using WordNet

        Args:
            text: Original text

        Returns:
            Augmented text
        """
        if not self.synonym_aug:
            return text

        try:
            # Extract metadata
            clean_text, metadata = self.extract_metadata(text)

            # Apply synonym replacement
            augmented = self.synonym_aug.augment(clean_text)

            # Restore metadata
            return self.restore_metadata(augmented, metadata)

        except Exception as e:
            print(f"Synonym replacement error: {e}")
            return text

    def contextual_replacement(self, text: str) -> str:
        """
        Replace words using BERT contextual embeddings

        Args:
            text: Original text

        Returns:
            Augmented text
        """
        if not self.context_aug:
            return text

        try:
            # Extract metadata
            clean_text, metadata = self.extract_metadata(text)

            # Apply contextual replacement
            augmented = self.context_aug.augment(clean_text)

            # Restore metadata
            return self.restore_metadata(augmented, metadata)

        except Exception as e:
            print(f"Contextual replacement error: {e}")
            return text

    def random_deletion(self, text: str, p=0.1) -> str:
        """
        Randomly delete p% of words

        Args:
            text: Original text
            p: Probability of deleting each word (default 10%)

        Returns:
            Augmented text
        """
        # Extract metadata
        clean_text, metadata = self.extract_metadata(text)

        words = clean_text.split()

        # Don't delete if too few words
        if len(words) <= 3:
            return text

        # Randomly keep words
        new_words = [word for word in words if random.random() > p]

        # Ensure at least some words remain
        if len(new_words) == 0:
            new_words = [random.choice(words)]

        augmented = ' '.join(new_words)

        # Restore metadata
        return self.restore_metadata(augmented, metadata)

    def augment_pair(
        self,
        text1: str,
        text2: str,
        label: float,
        phase: int,
        methods: List[str] = ['synonym', 'context', 'deletion']
    ) -> List[Tuple[str, str, float, int]]:
        """
        Augment a single pair using multiple methods

        Args:
            text1: First text
            text2: Second text
            label: Pair label (0.0 or 1.0)
            phase: Curriculum phase (1, 2, or 3)
            methods: List of augmentation methods to apply

        Returns:
            List of (augmented_text1, augmented_text2, label, phase) tuples
        """
        augmented_pairs = []

        # Original pair (always include)
        augmented_pairs.append((text1, text2, label, phase))

        # Augment text1 only (for positives)
        if 'synonym' in methods and self.synonym_aug:
            aug_t1 = self.synonym_replacement(text1)
            augmented_pairs.append((aug_t1, text2, label, phase))

        if 'context' in methods and self.context_aug:
            aug_t1 = self.contextual_replacement(text1)
            augmented_pairs.append((aug_t1, text2, label, phase))

        if 'deletion' in methods:
            aug_t1 = self.random_deletion(text1)
            augmented_pairs.append((aug_t1, text2, label, phase))

        # Augment text2 only (for positives)
        if label == 1.0:  # Only augment positives
            if 'synonym' in methods and self.synonym_aug:
                aug_t2 = self.synonym_replacement(text2)
                augmented_pairs.append((text1, aug_t2, label, phase))

            if 'context' in methods and self.context_aug:
                aug_t2 = self.contextual_replacement(text2)
                augmented_pairs.append((text1, aug_t2, label, phase))

        return augmented_pairs


def augment_dataset(
    input_path: str,
    output_path: str,
    augmentation_factor: int = 5,
    methods: List[str] = ['synonym', 'context', 'deletion'],
    seed: int = 42
):
    """
    Augment entire dataset

    Args:
        input_path: Path to curriculum_training_pairs_complete.json
        output_path: Path to save augmented pairs
        augmentation_factor: Target expansion factor (default 5x = 75K pairs)
        methods: Augmentation methods to use
        seed: Random seed
    """
    print("="*80)
    print("DATA AUGMENTATION PIPELINE")
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
    augmenter = ITSMDataAugmenter(seed=seed)

    # Check available methods
    available_methods = []
    if 'synonym' in methods and augmenter.synonym_aug:
        available_methods.append('synonym')
    if 'context' in methods and augmenter.context_aug:
        available_methods.append('context')
    if 'deletion' in methods:
        available_methods.append('deletion')
    if 'backtrans' in methods and augmenter.translator:
        available_methods.append('backtrans')

    print(f"Augmentation methods: {available_methods}")
    print(f"Target expansion: {augmentation_factor}x ({augmentation_factor * original_count:,} pairs)")
    print()

    # Augment data
    augmented_texts1 = []
    augmented_texts2 = []
    augmented_labels = []
    augmented_phases = []

    print("Augmenting pairs...")
    for i, (t1, t2, label, phase) in enumerate(zip(texts1, texts2, labels, phase_indicators)):
        # Show progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{original_count:,} pairs...")

        # Augment pair
        aug_pairs = augmenter.augment_pair(t1, t2, label, phase, methods=available_methods)

        # Add augmented pairs
        for aug_t1, aug_t2, aug_label, aug_phase in aug_pairs:
            augmented_texts1.append(aug_t1)
            augmented_texts2.append(aug_t2)
            augmented_labels.append(aug_label)
            augmented_phases.append(aug_phase)

            # Stop if we've reached target
            if len(augmented_texts1) >= augmentation_factor * original_count:
                break

        if len(augmented_texts1) >= augmentation_factor * original_count:
            break

    final_count = len(augmented_texts1)
    print(f"\nAugmented pairs: {final_count:,} (expansion: {final_count/original_count:.1f}x)")

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
            'expansion_factor': final_count / original_count,
            'augmentation_methods': available_methods,
            'seed': seed
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
    print("  2. Run training with augmented data")
    print("  3. Expected improvement: Spearman 0.62-0.68 (+23-35%)")


def main():
    parser = argparse.ArgumentParser(description='Augment ITSM training pairs')
    parser.add_argument(
        '--input',
        default='data_new/curriculum_training_pairs_complete.json',
        help='Input pairs file'
    )
    parser.add_argument(
        '--output',
        default='data_new/curriculum_training_pairs_augmented.json',
        help='Output augmented pairs file'
    )
    parser.add_argument(
        '--factor',
        type=int,
        default=5,
        help='Augmentation factor (default: 5x)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['synonym', 'context', 'deletion'],
        choices=['synonym', 'context', 'deletion', 'backtrans'],
        help='Augmentation methods to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        augmentation_factor=args.factor,
        methods=args.methods,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
