#!/usr/bin/env python3
"""
Add category metadata to test pairs for adversarial diagnostic.

This script:
1. Loads the original CSV with Category column
2. Loads existing test pairs JSON
3. Matches each pair's text back to original incidents
4. Adds category1 and category2 fields
5. Saves updated test pairs (with backup)
"""

import pandas as pd
import json
from pathlib import Path
import shutil
from datetime import datetime

# Paths
DATA_DIR = Path('data_new')
CSV_PATH = DATA_DIR / 'SNow_incident_ticket_data.csv'
TEST_PAIRS_PATH = DATA_DIR / 'fixed_test_pairs.json'

def create_combined_text(row):
    """
    Recreate combined_text the same way as notebooks do.
    Must match exactly for successful matching.
    """
    text_parts = []
    for col in ['Number', 'Description', 'User input', 'Resolution notes']:
        if col in row.index:
            value = str(row.get(col, '')).strip() if pd.notna(row.get(col)) else ''
            if value and value.lower() != 'nan':
                text_parts.append(value)
    return ' '.join(text_parts) if text_parts else ''

def main():
    print("="*80)
    print("ADD CATEGORY METADATA TO TEST PAIRS")
    print("="*80)

    # 1. Load CSV with Category column
    print(f"\n1. Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f"   Loaded {len(df)} incidents")

    # Check if Category column exists
    if 'Category' not in df.columns:
        print("   ERROR: 'Category' column not found in CSV!")
        print(f"   Available columns: {list(df.columns)}")
        return

    print(f"   Category column found")

    # Create combined_text for matching
    print(f"\n2. Creating combined_text for matching...")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df['combined_text'] = df['combined_text'].astype(str)
    df = df[df['combined_text'].str.len() > 10].reset_index(drop=True)
    print(f"   {len(df)} valid incidents after filtering")

    # Fill NaN categories with "Unknown"
    df['Category'] = df['Category'].fillna('Unknown')

    # Create lookup dictionary: combined_text -> category
    text_to_category = dict(zip(df['combined_text'], df['Category']))
    print(f"   Created lookup with {len(text_to_category)} entries")

    # Show category distribution
    category_counts = df['Category'].value_counts()
    print(f"\n   Category distribution:")
    for cat, count in category_counts.head(10).items():
        print(f"      {cat:30s}: {count:5d}")

    # 2. Load test pairs
    print(f"\n3. Loading test pairs: {TEST_PAIRS_PATH}")
    with open(TEST_PAIRS_PATH, 'r', encoding='utf-8') as f:
        test_pairs = json.load(f)

    print(f"   Loaded {len(test_pairs)} pairs")

    # Check if already has categories
    if test_pairs and 'category1' in test_pairs[0]:
        print("   WARNING: Test pairs already have category metadata")
        response = input("   Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("   Aborted")
            return

    # 3. Backup original
    backup_path = TEST_PAIRS_PATH.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    print(f"\n4. Creating backup: {backup_path}")
    shutil.copy(TEST_PAIRS_PATH, backup_path)
    print(f"   Backup created")

    # 4. Add category metadata
    print(f"\n5. Adding category metadata...")

    updated_pairs = []
    matched = 0
    unmatched = 0

    for i, pair in enumerate(test_pairs):
        text1 = pair['text1']
        text2 = pair['text2']
        label = pair['label']

        # Try to find categories
        category1 = text_to_category.get(text1, None)
        category2 = text_to_category.get(text2, None)

        # If exact match fails, try finding best match (first 100 chars)
        if category1 is None:
            text1_prefix = text1[:100]
            for full_text, cat in text_to_category.items():
                if full_text.startswith(text1_prefix):
                    category1 = cat
                    break

        if category2 is None:
            text2_prefix = text2[:100]
            for full_text, cat in text_to_category.items():
                if full_text.startswith(text2_prefix):
                    category2 = cat
                    break

        # If still no match, use "Unknown"
        if category1 is None:
            category1 = "Unknown"
            unmatched += 1
        else:
            matched += 1

        if category2 is None:
            category2 = "Unknown"
            if category1 != "Unknown":  # Only count if text1 matched
                unmatched += 1

        # Create updated pair
        updated_pair = {
            'text1': text1,
            'text2': text2,
            'label': label,
            'category1': category1,
            'category2': category2
        }
        updated_pairs.append(updated_pair)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(test_pairs)} pairs...")

    print(f"\n   Matching summary:")
    print(f"      Matched:   {matched}")
    print(f"      Unmatched: {unmatched}")
    print(f"      Unknown rate: {unmatched/(matched+unmatched)*100:.1f}%")

    # 5. Analyze updated pairs
    print(f"\n6. Analyzing updated pairs...")

    cross_category = sum(1 for p in updated_pairs if p['category1'] != p['category2'])
    same_category = sum(1 for p in updated_pairs if p['category1'] == p['category2'])

    print(f"   Cross-category pairs: {cross_category} ({cross_category/len(updated_pairs)*100:.1f}%)")
    print(f"   Same-category pairs:  {same_category} ({same_category/len(updated_pairs)*100:.1f}%)")

    # Check adversarial potential
    cross_cat_positive = sum(1 for p in updated_pairs
                             if p['category1'] != p['category2'] and p['label'] == 1)
    same_cat_negative = sum(1 for p in updated_pairs
                            if p['category1'] == p['category2'] and p['label'] == 0)

    print(f"\n   Adversarial diagnostic potential:")
    print(f"      Cross-category positives: {cross_cat_positive}")
    print(f"      Same-category negatives:  {same_cat_negative}")

    if cross_cat_positive < 10 or same_cat_negative < 10:
        print(f"      WARNING: May not have enough adversarial pairs!")
    else:
        print(f"      Good - sufficient adversarial pairs for testing")

    # 6. Save updated pairs
    print(f"\n7. Saving updated test pairs...")
    with open(TEST_PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_pairs, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(updated_pairs)} pairs to {TEST_PAIRS_PATH}")

    # 7. Sample output
    print(f"\n8. Sample pairs with categories:")
    for i, pair in enumerate(updated_pairs[:3]):
        print(f"\n   Pair {i+1}:")
        print(f"      Text1: {pair['text1'][:80]}...")
        print(f"      Category1: {pair['category1']}")
        print(f"      Text2: {pair['text2'][:80]}...")
        print(f"      Category2: {pair['category2']}")
        print(f"      Label: {pair['label']}")
        print(f"      Cross-category: {pair['category1'] != pair['category2']}")

    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}")
    print(f"\nCategory metadata added to test pairs.")
    print(f"Backup saved to: {backup_path}")
    print(f"\nNext step: Run evaluate_model_v2.ipynb")
    print(f"  - Adversarial diagnostic will auto-enable")

if __name__ == '__main__':
    main()
