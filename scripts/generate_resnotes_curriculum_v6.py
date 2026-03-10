#!/usr/bin/env python3
"""
generate_resnotes_curriculum_v6.py

Generates curriculum training pairs for semantic similarity fine-tuning.
Labels = TF-IDF cosine similarity of Resolution notes (distant supervision).

Why resolution notes?
  The model is trained on Description text but labelled by resolution-note
  similarity.  Two tickets resolved the same way → high label, even if their
  descriptions were phrased differently.  This teaches the model "same problem"
  rather than "same words", matching the grounded benchmark exactly.

Output format  (matches what training notebooks expect):
  {
    "texts1": [...],        # build_text formatted anchor strings
    "texts2": [...],        # build_text formatted positive/negative strings
    "labels": [...],        # float TF-IDF cosine of resolution notes (0.0–1.0)
    "phase_indicators": [...],  # 1, 2 or 3
    "metadata": {...}
  }

Text format (same as benchmark_v4_semantic_resnotes.json):
  "Category: X | Service: Y | Priority: Z | Description: ..."

Usage:
    conda run -n itsm python nexustism/scripts/generate_resnotes_curriculum_v6.py
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_NEW   = BASE_DIR / 'data_new'
CSV_PATH   = DATA_NEW / 'SNow_incident_ticket_data.csv'
BENCH_PATH = DATA_NEW / 'benchmark_v4_semantic_resnotes.json'
OUT_PATH   = DATA_NEW / 'resnotes_curriculum_training_pairs_v6.json'

# ── Curriculum phase thresholds (TF-IDF cosine of resolution notes) ────────────
# Mirroring the V4 curriculum philosophy: easy → medium → hard
PHASES = [
    {'phase': 1, 'pos_min': 0.50, 'neg_max': 0.10, 'name': 'Easy',   'target_per_side': 3000},
    {'phase': 2, 'pos_min': 0.35, 'neg_max': 0.20, 'name': 'Medium', 'target_per_side': 4000},
    {'phase': 3, 'pos_min': 0.22, 'neg_max': 0.30, 'name': 'Hard',   'target_per_side': 5000},
]

# ── Filtering ──────────────────────────────────────────────────────────────────
MIN_RESNOTE_LEN  = 20   # chars — skip trivially short resolution notes
MIN_DESC_LEN     = 10   # chars
TFIDF_MAX_FEATURES = 15_000
RANDOM_SEED = 42

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Text formatter (identical to benchmark notebook) ──────────────────────────
def build_text(row: pd.Series) -> str:
    parts = [
        f"Category: {row.get('Category', '')}",
        f"Service: {row.get('Service', '')}",
        f"Priority: {row.get('Priority', '')}",
    ]
    desc = str(row.get('Description', '')).strip()
    if desc:
        parts.append(f"Description: {desc}")
    return ' | '.join(p for p in parts if p.split(': ', 1)[1].strip())


# ── 1. Load & filter data ─────────────────────────────────────────────────────
log.info(f'Loading CSV: {CSV_PATH}')
df = pd.read_csv(CSV_PATH, encoding='utf-8', encoding_errors='replace')
df['Description']      = df['Description'].fillna('').str.strip()
df['Resolution notes'] = df['Resolution notes'].fillna('').str.strip()

mask = (
    (df['Resolution notes'].str.len() >= MIN_RESNOTE_LEN) &
    (df['Description'].str.len() >= MIN_DESC_LEN)
)
df = df[mask].reset_index(drop=True)
log.info(f'Rows with usable description + resolution notes: {len(df)}')

df['text'] = df.apply(build_text, axis=1)
df = df[df['text'].str.len() >= 20].reset_index(drop=True)
log.info(f'Rows after text-length filter: {len(df)}')


# ── 2. Load benchmark → blacklist pair texts ───────────────────────────────────
log.info(f'Loading benchmark blacklist: {BENCH_PATH}')
bench = json.load(open(BENCH_PATH, encoding='utf-8'))
blacklist: set[tuple[str, str]] = set()
for p in bench:
    blacklist.add((p['text1'], p['text2']))
    blacklist.add((p['text2'], p['text1']))
log.info(f'Blacklisted {len(bench)} benchmark pairs (both orderings)')


# ── 3. Build TF-IDF on resolution notes ───────────────────────────────────────
log.info('Fitting TF-IDF vectorizer on Resolution notes …')
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
X = tfidf.fit_transform(df['Resolution notes'].tolist())
log.info(f'TF-IDF matrix: {X.shape}  nnz={X.nnz:,}')


# ── 4. Compute full pairwise cosine similarity matrix ─────────────────────────
# n=9.2K → n²×float32 ≈ 340 MB — fits in RAM
n = X.shape[0]
log.info(f'Computing pairwise cosine similarity ({n}×{n}) …')

# Normalise rows once, then use dot-product for speed
from sklearn.preprocessing import normalize
X_norm = normalize(X, norm='l2')
# Compute in batches of 1 000 rows to keep peak memory low
BATCH = 1_000
sim_matrix = np.empty((n, n), dtype=np.float32)
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    sim_matrix[start:end] = (X_norm[start:end] @ X_norm.T).toarray()
    if start % (BATCH * 5) == 0:
        log.info(f'  … row {start}/{n}')

log.info('Similarity matrix ready.')

# Upper-triangle indices (no self-pairs, no duplicates)
triu_i, triu_j = np.triu_indices(n, k=1)
sim_flat = sim_matrix[triu_i, triu_j]
log.info(f'  Candidate pairs in upper triangle: {len(sim_flat):,}')

# Quick distribution check
for thr in [0.10, 0.20, 0.30, 0.35, 0.50]:
    log.info(f'    sim >= {thr:.2f}: {(sim_flat >= thr).sum():,}')


# ── 5. Sample curriculum pairs ─────────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)
texts = df['text'].tolist()

all_texts1:    list[str]   = []
all_texts2:    list[str]   = []
all_labels:    list[float] = []
all_phases:    list[int]   = []

for cfg in PHASES:
    phase     = cfg['phase']
    pos_min   = cfg['pos_min']
    neg_max   = cfg['neg_max']
    name      = cfg['name']
    target    = cfg['target_per_side']

    log.info(f'\n── Phase {phase} · {name} (pos≥{pos_min}, neg≤{neg_max}, target={target}/side) ──')

    pos_idx_all = np.where(sim_flat >= pos_min)[0]
    neg_idx_all = np.where(sim_flat <= neg_max)[0]
    log.info(f'  Positive pool: {len(pos_idx_all):,}   Negative pool: {len(neg_idx_all):,}')

    pos_sample = rng.sample(list(pos_idx_all), min(target, len(pos_idx_all)))
    neg_sample = rng.sample(list(neg_idx_all), min(target, len(neg_idx_all)))

    added = 0
    for flat_idx in pos_sample:
        i, j = int(triu_i[flat_idx]), int(triu_j[flat_idx])
        t1, t2 = texts[i], texts[j]
        if (t1, t2) in blacklist:
            continue
        all_texts1.append(t1)
        all_texts2.append(t2)
        all_labels.append(float(sim_flat[flat_idx]))
        all_phases.append(phase)
        added += 1

    for flat_idx in neg_sample:
        i, j = int(triu_i[flat_idx]), int(triu_j[flat_idx])
        t1, t2 = texts[i], texts[j]
        if (t1, t2) in blacklist:
            continue
        all_texts1.append(t1)
        all_texts2.append(t2)
        all_labels.append(float(sim_flat[flat_idx]))
        all_phases.append(phase)
        added += 1

    log.info(f'  → Added {added} pairs (≈{added//2} pos + {added//2} neg)')


# ── 6. Shuffle ─────────────────────────────────────────────────────────────────
log.info('\nShuffling …')
combined = list(zip(all_texts1, all_texts2, all_labels, all_phases))
rng.shuffle(combined)
s_t1, s_t2, s_lb, s_ph = zip(*combined)


# ── 7. Save ────────────────────────────────────────────────────────────────────
output = {
    'texts1':           list(s_t1),
    'texts2':           list(s_t2),
    'labels':           list(s_lb),
    'phase_indicators': list(s_ph),
    'metadata': {
        'generated_at':   datetime.now().isoformat(),
        'total_pairs':    len(s_lb),
        'label_source':   'tfidf_cosine_resolution_notes',
        'text_format':    'build_text(Category | Service | Priority | Description)',
        'phases':         PHASES,
        'tfidf_vocab':    TFIDF_MAX_FEATURES,
        'n_docs':         n,
        'blacklisted':    len(bench),
        'source_csv':     CSV_PATH.name,
    },
}

log.info(f'Saving {len(s_lb):,} pairs → {OUT_PATH}')
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f)
log.info('Saved.')

# ── 8. Summary stats ───────────────────────────────────────────────────────────
arr = np.array(s_lb)
log.info('\n── Label distribution ──────────────────────────────────────────')
log.info(f'  count : {len(arr)}')
log.info(f'  mean  : {arr.mean():.4f}')
log.info(f'  std   : {arr.std():.4f}')
log.info(f'  min   : {arr.min():.4f}')
log.info(f'  max   : {arr.max():.4f}')
log.info(f'  ≥0.22 : {(arr >= 0.22).sum()} (positives)')
log.info(f'  ≤0.30 : {(arr <= 0.30).sum()} (negatives)')
log.info(f'\n[OK] Done — {OUT_PATH.name}')
