import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

ID_RE = re.compile(r"\b(PRB|CHG|KB|KEDB|RFC|RITM|REQ|SR)\d+\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\\S+", re.IGNORECASE)
KB_LINE_RE = re.compile(r"knowledge object\\s*:\\s*([^\\n\\r]+)", re.IGNORECASE)


def extract_signals(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set()
    text = str(text)
    if not text.strip():
        return set()

    signals = set()
    for match in URL_RE.findall(text):
        cleaned = match.rstrip(").,;\\\"' ")
        if cleaned:
            signals.add(f"url:{cleaned}")

    for match in ID_RE.finditer(text):
        signals.add(f"id:{match.group(0).upper()}")

    for match in KB_LINE_RE.finditer(text):
        raw = match.group(1).strip()
        if not raw:
            continue
        token = raw.split()[0].strip(").,;\\\"' ")
        if token.upper() in {"NA", "N/A", "NONE", "NOT"}:
            continue
        signals.add(f"kb:{token.upper()}")

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="nexustism/data_new/SNow_incident_ticket_data.csv")
    parser.add_argument(
        "--columns",
        default="Resolution notes,Comments and Work notes",
        help="Comma-separated columns to scan for independent signals.",
    )
    parser.add_argument("--out", default="nexustism/docs/labeling/independent_signal_pairs.csv")
    parser.add_argument("--max_incidents_per_signal", type=int, default=50)
    parser.add_argument("--max_pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]

    df = pd.read_csv(args.csv, dtype=str, encoding="utf-8", encoding_errors="replace")
    if "Number" not in df.columns:
        raise ValueError("CSV is missing Number column")

    signal_to_ids = {}
    for _, row in df.iterrows():
        incident_id = str(row.get("Number", "")).strip()
        if not incident_id:
            continue
        signals = set()
        for col in cols:
            if col not in df.columns:
                continue
            signals.update(extract_signals(row.get(col)))
        if not signals:
            continue
        for sig in signals:
            signal_to_ids.setdefault(sig, []).append(incident_id)

    pairs = []
    for sig, ids in signal_to_ids.items():
        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) < 2:
            continue
        if args.max_incidents_per_signal and len(unique_ids) > args.max_incidents_per_signal:
            unique_ids = rng.choice(unique_ids, size=args.max_incidents_per_signal, replace=False).tolist()
        for a_id, b_id in itertools.combinations(sorted(unique_ids), 2):
            pairs.append(
                {
                    "pair_id": f"SIG-{len(pairs)+1:06d}",
                    "pair_type": "independent_signal",
                    "incident_a_id": a_id,
                    "incident_b_id": b_id,
                    "signal": sig,
                    "signal_type": sig.split(":", 1)[0],
                }
            )
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(out_path, index=False)
    print(f"Wrote independent signal pairs: {out_path} ({len(pairs)})")


if __name__ == "__main__":
    main()
