import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CAUSAL_CATEGORY_PAIRS = {
    'Network': ['Application', 'Database', 'Email'],
    'Server': ['Application', 'Database'],
    'Database': ['Application'],
    'Power': ['Server', 'Network', 'Application'],
    'Storage': ['Database', 'Application'],
}


def normalize_embeddings(emb):
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def category_dependency(cat_a, cat_b):
    if not cat_a or not cat_b:
        return False
    return cat_b in CAUSAL_CATEGORY_PAIRS.get(cat_a, [])


def label_temporal(row, sim, delta_hours, cfg):
    label_duplicate = ''
    label_related = ''
    label_causal = ''
    confidence = 0.0

    if sim >= cfg['duplicate_sim_min']:
        label_duplicate = 'yes'
    if sim >= cfg['related_sim_min']:
        label_related = 'yes'

    if delta_hours is not None and delta_hours > 0 and sim >= cfg['causal_sim_min']:
        cat_dep = category_dependency(row.get('category_a'), row.get('category_b'))
        if cat_dep or row.get('category_a') == row.get('category_b'):
            label_causal = 'A->B'
            confidence = min(0.99, 0.5 + sim * 0.5)
        else:
            confidence = min(0.9, 0.3 + sim * 0.5)
    else:
        confidence = min(0.7, sim * 0.7)

    if label_duplicate == 'yes':
        label_causal = ''

    if confidence < cfg['causal_conf_min']:
        label_causal = ''

    return label_duplicate, label_related, label_causal, round(confidence, 3)


def label_embedding(sim, cfg):
    label_duplicate = ''
    label_related = ''
    confidence = 0.0

    if sim >= cfg['duplicate_sim_min']:
        label_duplicate = 'yes'
        confidence = min(0.99, 0.6 + sim * 0.4)
    elif sim >= cfg['related_sim_min']:
        label_related = 'yes'
        confidence = min(0.9, 0.4 + sim * 0.4)
    else:
        confidence = min(0.6, sim * 0.6)

    return label_duplicate, label_related, round(confidence, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='nexustism/data_new/SNow_incident_ticket_data.csv')
    parser.add_argument('--emb', default='nexustism/embeddings_v4_cache.npy')
    parser.add_argument('--temporal_in', default='nexustism/docs/labeling/labeling_pairs_temporal_24h_5000.csv')
    parser.add_argument('--embedding_in', default='nexustism/docs/labeling/labeling_pairs_embedding_neighbors_5000.csv')
    parser.add_argument('--temporal_out', default='nexustism/docs/labeling/silver_pairs_temporal_24h_5000.csv')
    parser.add_argument('--embedding_out', default='nexustism/docs/labeling/silver_pairs_embedding_neighbors_5000.csv')
    parser.add_argument('--keep_embedding_positives', type=int, default=0)
    parser.add_argument('--duplicate_sim_min', type=float, default=0.88)
    parser.add_argument('--related_sim_min', type=float, default=0.75)
    parser.add_argument('--causal_sim_min', type=float, default=0.70)
    parser.add_argument('--causal_conf_min', type=float, default=0.85)
    parser.add_argument('--temporal_non_causal_sim_max', type=float, default=0.45)
    parser.add_argument('--embed_neg_sim_max', type=float, default=0.55)
    parser.add_argument('--temporal_neg_same_category', type=int, default=1)
    parser.add_argument('--neg_pos_ratio', type=float, default=2.0)
    parser.add_argument('--embedding_neg_limit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df['Created_dt'] = pd.to_datetime(df['Created'], format='%d/%m/%y %H:%M', errors='coerce')
    id_to_idx = {num: i for i, num in enumerate(df['Number'])}

    emb = normalize_embeddings(np.load(args.emb))

    temporal = pd.read_csv(args.temporal_in)
    embedding_pairs = pd.read_csv(args.embedding_in)

    def compute_sim(row):
        idx_a = id_to_idx.get(row['incident_a_id'])
        idx_b = id_to_idx.get(row['incident_b_id'])
        if idx_a is None or idx_b is None:
            return np.nan
        return float(np.dot(emb[idx_a], emb[idx_b]))

    if 'similarity' not in temporal.columns:
        temporal['similarity'] = temporal.apply(compute_sim, axis=1)
    if 'similarity' not in embedding_pairs.columns:
        embedding_pairs['similarity'] = embedding_pairs.apply(compute_sim, axis=1)

    # Fill delta_hours if missing
    if 'delta_hours' not in temporal.columns:
        temporal['delta_hours'] = np.nan
    if temporal['delta_hours'].isna().any():
        created_map = dict(zip(df['Number'], df['Created_dt']))
        def compute_delta(row):
            a_time = created_map.get(row['incident_a_id'])
            b_time = created_map.get(row['incident_b_id'])
            if pd.isna(a_time) or pd.isna(b_time):
                return np.nan
            return (b_time - a_time).total_seconds() / 3600
        temporal['delta_hours'] = temporal.apply(compute_delta, axis=1)

    # Apply labels
    cfg = {
        'duplicate_sim_min': args.duplicate_sim_min,
        'related_sim_min': args.related_sim_min,
        'causal_sim_min': args.causal_sim_min,
        'causal_conf_min': args.causal_conf_min,
    }

    temporal_labels = temporal.apply(
        lambda r: label_temporal(r, r['similarity'], r['delta_hours'], cfg),
        axis=1,
        result_type='expand',
    )
    temporal[['label_duplicate', 'label_related', 'label_causal_direction', 'label_confidence']] = temporal_labels
    temporal['label_notes'] = 'weak_label'

    emb_labels = embedding_pairs['similarity'].apply(lambda s: label_embedding(s, cfg))
    embedding_pairs[['label_duplicate', 'label_related', 'label_confidence']] = pd.DataFrame(
        emb_labels.tolist(), index=embedding_pairs.index
    )
    embedding_pairs['label_causal_direction'] = ''
    embedding_pairs['label_notes'] = 'weak_label'

    # Filter non-causal candidates to reduce noisy negatives
    causal_rows = temporal[temporal['label_causal_direction'].isin(['A->B', 'B->A'])]
    non_causal_rows = temporal[~temporal.index.isin(causal_rows.index)]
    non_causal_rows = non_causal_rows[non_causal_rows['similarity'] <= args.temporal_non_causal_sim_max]

    if args.temporal_neg_same_category:
        non_causal_rows = non_causal_rows[
            non_causal_rows['category_a'].astype(str) == non_causal_rows['category_b'].astype(str)
        ]

    max_non_causal = int(len(causal_rows) * args.neg_pos_ratio)
    if len(non_causal_rows) > max_non_causal:
        non_causal_rows = non_causal_rows.sample(max_non_causal, random_state=args.seed)

    temporal = pd.concat([causal_rows, non_causal_rows], ignore_index=True)

    embedding_pairs['label_duplicate'] = embedding_pairs['label_duplicate'].fillna('')
    embedding_pairs['label_related'] = embedding_pairs['label_related'].fillna('')
    pos_mask = (embedding_pairs['label_duplicate'] == 'yes') | (embedding_pairs['label_related'] == 'yes')
    if args.keep_embedding_positives:
        embedding_neg = embedding_pairs[~pos_mask & (embedding_pairs['similarity'] <= args.embed_neg_sim_max)]
        embedding_pos = embedding_pairs[pos_mask]
        embedding_pairs = pd.concat([embedding_pos, embedding_neg], ignore_index=True)
    else:
        embedding_pairs = embedding_pairs[
            (~pos_mask) & (embedding_pairs['similarity'] <= args.embed_neg_sim_max)
        ].reset_index(drop=True)

    if len(embedding_pairs) > args.embedding_neg_limit:
        embedding_pairs = embedding_pairs.sample(args.embedding_neg_limit, random_state=args.seed)

    Path(args.temporal_out).parent.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(args.temporal_out, index=False)
    embedding_pairs.to_csv(args.embedding_out, index=False)

    print(f'Wrote silver temporal: {args.temporal_out}')
    print(f'Wrote silver embedding: {args.embedding_out}')


if __name__ == '__main__':
    main()
