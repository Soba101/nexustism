import pandas as pd
import json
import random
from itertools import combinations

def generate_relationship_pairs(input_csv_path, output_json_path, num_pairs_per_category=50):
    """
    Generates a relationship pairs file from a given CSV of incident data.

    Args:
        input_csv_path (str): Path to the source CSV file.
        output_json_path (str): Path to save the generated JSON file.
        num_pairs_per_category (int): Number of pairs to generate for each label.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # Ensure required columns exist
    required_columns = ['Number', 'Short Description', 'Description', 'Category', 'Subcategory']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain the columns: {required_columns}")
        return

    # Combine Short Description and Description for full text
    df['text'] = df['Short Description'].fillna('') + ". " + df['Description'].fillna('')
    
    all_pairs = []
    generated_ids = set()
 
    # 1. Generate "Duplicate" pairs
    print("Generating 'duplicate' pairs...")
    duplicate_groups = df[df.duplicated(subset=['Short Description'], keep=False)].groupby('Short Description')
    for _, group in duplicate_groups:
        if len(all_pairs) >= num_pairs_per_category:
            break
        if len(group) > 1:
            for i, j in combinations(group.index, 2):
                pair_id = tuple(sorted((df.at[i, 'Number'], df.at[j, 'Number'])))
                if pair_id not in generated_ids:
                    all_pairs.append({
                        "ticket_a_id": df.at[i, 'Number'],
                        "ticket_b_id": df.at[j, 'Number'],
                        "text_a": df.at[i, 'text'],
                        "text_b": df.at[j, 'text'],
                        "label": "duplicate"
                    })
                    generated_ids.add(pair_id)
                    if len(all_pairs) % num_pairs_per_category == 0:
                        break
        if len(all_pairs) >= num_pairs_per_category:
            break
    
    # 2. Generate "Related" pairs
    print("Generating 'related' pairs...")
    related_count_start = len(all_pairs)
    subcategory_groups = df.groupby('Subcategory')
    for _, group in subcategory_groups:
        if len(all_pairs) >= related_count_start + num_pairs_per_category:
            break
        if len(group) > 1:
            for i, j in combinations(group.index, 2):
                # Avoid re-using duplicate pairs
                if df.at[i, 'Short Description'] == df.at[j, 'Short Description']:
                    continue
                
                pair_id = tuple(sorted((df.at[i, 'Number'], df.at[j, 'Number'])))
                if pair_id not in generated_ids:
                    all_pairs.append({
                        "ticket_a_id": df.at[i, 'Number'],
                        "ticket_b_id": df.at[j, 'Number'],
                        "text_a": df.at[i, 'text'],
                        "text_b": df.at[j, 'text'],
                        "label": "related"
                    })
                    generated_ids.add(pair_id)
        if len(all_pairs) >= related_count_start + num_pairs_per_category:
            break
            
    # 3. Generate "Causal" pairs (Synthetic examples)
    print("Generating 'causal' pairs (synthetic)...")
    causal_count_start = len(all_pairs)
    # Find a disruptive event, like an "Outage"
    outage_tickets = df[df['Short Description'].str.contains("Outage", case=False, na=False)]
    if not outage_tickets.empty:
        for i in outage_tickets.index:
            if len(all_pairs) >= causal_count_start + num_pairs_per_category:
                break
            
            outage_ticket = df.loc[i]
            # Find another ticket in the same category that is not an outage
            potential_effects = df[
                (df['Category'] == outage_ticket['Category']) &
                (~df['Short Description'].str.contains("Outage", case=False, na=False)) &
                (df.index != i)
            ]
            
            if not potential_effects.empty:
                effect_ticket_row = potential_effects.sample(1)
                j = effect_ticket_row.index[0]
                
                pair_id = tuple(sorted((df.at[i, 'Number'], df.at[j, 'Number'])))
                if pair_id not in generated_ids:
                    all_pairs.append({
                        "ticket_a_id": df.at[i, 'Number'], # The cause
                        "ticket_b_id": df.at[j, 'Number'], # The effect
                        "text_a": df.at[i, 'text'],
                        "text_b": df.at[j, 'text'],
                        "label": "causal"
                    })
                    generated_ids.add(pair_id)

    # 4. Generate "None" pairs
    print("Generating 'none' pairs...")
    none_count_start = len(all_pairs)
    while len(all_pairs) < none_count_start + num_pairs_per_category:
        i, j = random.sample(range(len(df)), 2)
        
        # Ensure they don't share a category for a clear "none" signal
        if df.at[i, 'Category'] != df.at[j, 'Category']:
            pair_id = tuple(sorted((df.at[i, 'Number'], df.at[j, 'Number'])))
            if pair_id not in generated_ids:
                all_pairs.append({
                    "ticket_a_id": df.at[i, 'Number'],
                    "ticket_b_id": df.at[j, 'Number'],
                    "text_a": df.at[i, 'text'],
                    "text_b": df.at[j, 'text'],
                    "label": "none"
                })
                generated_ids.add(pair_id)

    print(f"\nGenerated a total of {len(all_pairs)} pairs.")
    # Shuffle for randomness
    random.shuffle(all_pairs)

    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=4)
        
    print(f"Successfully saved new relationship pairs to {output_json_path}")



if __name__ == '__main__':
    # Configuration
    INPUT_CSV = 'data/dummy_data_promax.csv'
    OUTPUT_JSON = 'data/new_relationship_pairs.json'
    
    generate_relationship_pairs(INPUT_CSV, OUTPUT_JSON)