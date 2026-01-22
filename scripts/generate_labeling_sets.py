import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def truncate_text(text, max_chars):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_temporal_pairs(df, num_pairs, window_hours, rng, max_pairs_per_target=3, max_chars=500):
    df = df.copy()
    if "Created_dt" not in df.columns:
        df["Created_dt"] = pd.to_datetime(
            df["Created"],
            format="%d/%m/%y %H:%M",
            errors="coerce",
        )
    df = df.dropna(subset=["Created_dt"]).sort_values("Created_dt").reset_index(drop=True)
    times = df["Created_dt"].values

    pairs = []
    seen = set()

    target_indices = rng.permutation(len(df))
    for idx in target_indices:
        target_time = times[idx]
        window_start = target_time - np.timedelta64(window_hours, "h")
        start = np.searchsorted(times, window_start, side="left")
        end = np.searchsorted(times, target_time, side="left")
        if end <= start:
            continue

        candidate_pool = np.arange(start, end)
        rng.shuffle(candidate_pool)
        for cand in candidate_pool[:max_pairs_per_target]:
            row_a = df.iloc[cand]
            row_b = df.iloc[idx]
            if row_a["Created_dt"] >= row_b["Created_dt"]:
                continue
            key = (row_a["Number"], row_b["Number"])
            if key in seen:
                continue
            seen.add(key)
            delta_hours = (row_b["Created_dt"] - row_a["Created_dt"]).total_seconds() / 3600

            pairs.append(
                {
                    "pair_id": f"TEMP-{len(pairs)+1:06d}",
                    "pair_type": f"temporal_{window_hours}h",
                    "incident_a_id": row_a["Number"],
                    "incident_b_id": row_b["Number"],
                    "created_a": row_a["Created_dt"].isoformat(),
                    "created_b": row_b["Created_dt"].isoformat(),
                    "delta_hours": round(delta_hours, 3),
                    "category_a": row_a.get("Category", ""),
                    "category_b": row_b.get("Category", ""),
                    "description_a": truncate_text(row_a.get("Description", ""), max_chars),
                    "description_b": truncate_text(row_b.get("Description", ""), max_chars),
                    "label_duplicate": "",
                    "label_related": "",
                    "label_causal_direction": "",
                    "label_confidence": "",
                    "notes": "",
                }
            )
            if len(pairs) >= num_pairs:
                return pairs
    return pairs


def build_embedding_pairs(df, embeddings, num_pairs, neighbors_per_query, rng, max_chars=500):
    emb = embeddings.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    pairs = []
    seen = set()
    num_queries = max(1, int(np.ceil(num_pairs / neighbors_per_query)))
    query_pool = rng.permutation(len(df))

    for q in query_pool[: len(query_pool)]:
        sims = emb @ emb[q]
        top_k = min(neighbors_per_query + 1, len(df))
        idxs = np.argpartition(-sims, range(top_k))[:top_k]
        idxs = [i for i in idxs if i != q]
        idxs = sorted(idxs, key=lambda i: sims[i], reverse=True)[:neighbors_per_query]

        row_q = df.iloc[q]
        for rank, n in enumerate(idxs, start=1):
            key = (row_q["Number"], df.iloc[n]["Number"])
            if key in seen:
                continue
            seen.add(key)
            row_n = df.iloc[n]

            created_a = row_q.get("Created_dt")
            created_b = row_n.get("Created_dt")
            delta_hours = ""
            if pd.notna(created_a) and pd.notna(created_b):
                delta_hours = round((created_b - created_a).total_seconds() / 3600, 3)

            pairs.append(
                {
                    "pair_id": f"EMB-{len(pairs)+1:06d}",
                    "pair_type": "embedding_neighbors",
                    "incident_a_id": row_q["Number"],
                    "incident_b_id": row_n["Number"],
                    "created_a": created_a.isoformat() if pd.notna(created_a) else "",
                    "created_b": created_b.isoformat() if pd.notna(created_b) else "",
                    "delta_hours": delta_hours,
                    "category_a": row_q.get("Category", ""),
                    "category_b": row_n.get("Category", ""),
                    "neighbor_rank": rank,
                    "similarity": round(float(sims[n]), 6),
                    "description_a": truncate_text(row_q.get("Description", ""), max_chars),
                    "description_b": truncate_text(row_n.get("Description", ""), max_chars),
                    "label_duplicate": "",
                    "label_related": "",
                    "label_causal_direction": "",
                    "label_confidence": "",
                    "notes": "",
                }
            )
            if len(pairs) >= num_pairs:
                return pairs
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="nexustism/data_new/SNow_incident_ticket_data.csv")
    parser.add_argument("--emb", default="nexustism/embeddings_v4_cache.npy")
    parser.add_argument("--out_dir", default="nexustism/docs/labeling")
    parser.add_argument("--num_temporal", type=int, default=1000)
    parser.add_argument("--num_embedding", type=int, default=1000)
    parser.add_argument("--window_hours", type=int, default=24)
    parser.add_argument("--neighbors_per_query", type=int, default=10)
    parser.add_argument("--max_chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    df["Created_dt"] = pd.to_datetime(
        df["Created"],
        format="%d/%m/%y %H:%M",
        errors="coerce",
    )
    emb = np.load(args.emb)

    temporal_pairs = build_temporal_pairs(
        df,
        num_pairs=args.num_temporal,
        window_hours=args.window_hours,
        rng=rng,
        max_pairs_per_target=3,
        max_chars=args.max_chars,
    )

    embedding_pairs = build_embedding_pairs(
        df,
        emb,
        num_pairs=args.num_embedding,
        neighbors_per_query=args.neighbors_per_query,
        rng=rng,
        max_chars=args.max_chars,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal_path = out_dir / f"labeling_pairs_temporal_{args.window_hours}h_{args.num_temporal}.csv"
    embedding_path = out_dir / f"labeling_pairs_embedding_neighbors_{args.num_embedding}.csv"

    pd.DataFrame(temporal_pairs).to_csv(temporal_path, index=False)
    pd.DataFrame(embedding_pairs).to_csv(embedding_path, index=False)

    print(f"Wrote temporal pairs: {temporal_path} ({len(temporal_pairs)})")
    print(f"Wrote embedding pairs: {embedding_path} ({len(embedding_pairs)})")


if __name__ == "__main__":
    main()
