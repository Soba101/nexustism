#!/usr/bin/env python3
"""
Add category metadata to test pairs for adversarial diagnostic.

Format: {texts1: [...], texts2: [...], labels: [...]} → adds categories1: [...], categories2: [...]
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
    """Recreate combined_text the same way as notebooks do."""
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

    if 'Category' not in df.columns:
        print("   ERROR: 'Category' column not found!")
        return

    # Create combined_text for matching
    print(f"\n2. Creating combined_text for matching...")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df['combined_text'] = df['combined_text'].astype(str)
    df = df[df['combined_text'].str.len() > 10].reset_index(drop=True)
    df['Category'] = df['Category'].fillna('Unknown')

    # Create lookup
    text_to_category = dict(zip(df['combined_text'], df['Category']))
    print(f"   Created lookup with {len(text_to_category)} entries")

    # Show categories
    print(f"\n   Top categories:")
    for cat, count in df['Category'].value_counts().head(10).items():
        print(f"      {cat:30s}: {count:5d}")

    # 2. Load test pairs
    print(f"\n3. Loading test pairs: {TEST_PAIRS_PATH}")
    with open(TEST_PAIRS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check format
    if 'texts1' not in data:
        print(f"   ERROR: Expected 'texts1' key, got: {list(data.keys())}")
        return

    texts1 = data['texts1']
    texts2 = data['texts2']
    labels = data['labels']
    print(f"   Loaded {len(texts1)} pairs")

    # Check if already has categories
    if 'categories1' in data:
        print("   WARNING: Already has category metadata")
        response = input("   Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("   Aborted")
            return

    # 3. Backup
    backup_path = TEST_PAIRS_PATH.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    print(f"\n4. Creating backup: {backup_path.name}")
    shutil.copy(TEST_PAIRS_PATH, backup_path)

    # 4. Add categories
    print(f"\n5. Adding category metadata...")

    categories1 = []
    categories2 = []
    matched = 0
    unmatched = 0

    for i, (text1, text2) in enumerate(zip(texts1, texts2)):
        # Try exact match first
        cat1 = text_to_category.get(text1, None)
        cat2 = text_to_category.get(text2, None)

        # If no exact match, try prefix match (first 100 chars)
        if cat1 is None:
            prefix = text1[:100]
            for full_text, cat in text_to_category.items():
                if full_text.startswith(prefix):
                    cat1 = cat
                    break

        if cat2 is None:
            prefix = text2[:100]
            for full_text, cat in text_to_category.items():
                if full_text.startswith(prefix):
                    cat2 = cat
                    break

        # Default to Unknown
        if cat1 is None:
            cat1 = "Unknown"
            unmatched += 1
        else:
            matched += 1

        if cat2 is None:
            cat2 = "Unknown"
            if cat1 != "Unknown":
                unmatched += 1

        categories1.append(cat1)
        categories2.append(cat2)

        if (i + 1) % 200 == 0:
            print(f"   Processed {i+1}/{len(texts1)} pairs...")

    print(f"\n   Matching summary:")
    print(f"      Matched:   {matched}")
    print(f"      Unmatched: {unmatched}")
    print(f"      Match rate: {matched/(matched+unmatched)*100:.1f}%")

    # 5. Analyze
    print(f"\n6. Analyzing categories...")

    cross_cat = sum(1 for c1, c2 in zip(categories1, categories2) if c1 != c2)
    same_cat = len(categories1) - cross_cat

    print(f"   Cross-category: {cross_cat} ({cross_cat/len(categories1)*100:.1f}%)")
    print(f"   Same-category:  {same_cat} ({same_cat/len(categories1)*100:.1f}%)")

    # Adversarial potential
    cross_cat_pos = sum(1 for c1, c2, lbl in zip(categories1, categories2, labels)
                        if c1 != c2 and lbl == 1)
    same_cat_neg = sum(1 for c1, c2, lbl in zip(categories1, categories2, labels)
                       if c1 == c2 and lbl == 0)

    print(f"\n   Adversarial diagnostic potential:")
    print(f"      Cross-category positives: {cross_cat_pos}")
    print(f"      Same-category negatives:  {same_cat_neg}")

    if cross_cat_pos < 10 or same_cat_neg < 10:
        print(f"      WARNING: May have insufficient adversarial pairs!")
    else:
        print(f"      GOOD - Sufficient pairs for adversarial testing")

    # 6. Save
    print(f"\n7. Saving updated test pairs...")

    data['categories1'] = categories1
    data['categories2'] = categories2

    with open(TEST_PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"   Saved to: {TEST_PAIRS_PATH}")

    # 7. Sample
    print(f"\n8. Sample pairs:")
    for i in range(min(3, len(texts1))):
        print(f"\n   Pair {i+1}:")
        print(f"      Text1:     {texts1[i][:70]}...")
        print(f"      Category1: {categories1[i]}")
        print(f"      Text2:     {texts2[i][:70]}...")
        print(f"      Category2: {categories2[i]}")
        print(f"      Label:     {labels[i]}")
        print(f"      Cross-cat: {categories1[i] != categories2[i]}")

    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}")
    print(f"\nCategory metadata added ({len(categories1)} pairs)")
    print(f"Backup: {backup_path.name}")
    print(f"\nNext: Run evaluate_model_v2.ipynb")

if __name__ == '__main__':
    main()
