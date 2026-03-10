#!/usr/bin/env python3
"""
validate_training_pairs.py
==========================

Pre-training data quality gate. Run this before any training notebook to catch
label imbalance, duplicate pairs, text length issues, and benchmark leakage.

Usage:
    python validate_training_pairs.py \
        --pairs ../data_new/resnotes_curriculum_training_pairs_v6.json \
        --benchmark ../data_new/benchmark_v4_semantic_resnotes.json

Exit codes:
    0 = All checks pass (warnings printed but not fatal)
    1 = Hard failure: benchmark leakage detected OR duplicate rate > 1%
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def _hash_pair(t1: str, t2: str) -> str:
    """Canonical hash for a text pair (order-independent)."""
    key = "\x00".join(sorted([t1.strip(), t2.strip()]))
    return hashlib.sha256(key.encode()).hexdigest()


def _token_count(text: str) -> int:
    """Whitespace-based proxy for token count."""
    return len(text.split())


def check_label_distribution(texts1, texts2, labels, phase_indicators=None) -> list[str]:
    """Warn if pos:neg ratio is outside 1:3 – 3:1 for any phase."""
    warnings = []

    def _check_phase(phase_labels, phase_name):
        if not phase_labels:
            return
        pos = sum(1 for l in phase_labels if float(l) >= 0.5)
        neg = len(phase_labels) - pos
        ratio = pos / max(neg, 1)
        print(f"   {phase_name}: {pos} pos / {neg} neg  (ratio={ratio:.2f})")
        if ratio < 1/3 or ratio > 3.0:
            warnings.append(
                f"Label imbalance in {phase_name}: pos:neg ratio={ratio:.2f} "
                f"(expected 0.33–3.0)"
            )

    if phase_indicators is not None:
        phase_map: dict[str, list] = {}
        for label, phase in zip(labels, phase_indicators):
            phase_map.setdefault(str(phase), []).append(label)
        for phase_id in sorted(phase_map):
            _check_phase(phase_map[phase_id], f"Phase {phase_id}")
    else:
        _check_phase(labels, "All pairs")

    return warnings


def check_duplicates(texts1, texts2) -> tuple[list[str], int]:
    """Hard fail if >1% of pairs are duplicates."""
    failures = []
    hashes = [_hash_pair(t1, t2) for t1, t2 in zip(texts1, texts2)]
    counts = Counter(hashes)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    total = len(hashes)
    dup_pct = dup_count / max(total, 1) * 100
    print(f"   Duplicate pairs: {dup_count}/{total} ({dup_pct:.2f}%)")
    if dup_pct > 1.0:
        failures.append(
            f"Duplicate rate {dup_pct:.2f}% exceeds 1% threshold "
            f"({dup_count} duplicate pairs)"
        )
    return failures, dup_count


def check_text_lengths(texts1, texts2) -> list[str]:
    """Warn if p95 token count exceeds 512 (model max_seq_length)."""
    warnings = []
    all_texts = texts1 + texts2
    lengths = sorted(_token_count(t) for t in all_texts)
    n = len(lengths)
    p50 = lengths[n // 2]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99)]
    print(f"   Token counts  — p50={p50}  p95={p95}  p99={p99}  max={lengths[-1]}")
    if p95 > 512:
        warnings.append(
            f"p95 token count={p95} exceeds model max_seq_length=512. "
            f"Long texts will be silently truncated during training."
        )
    return warnings


def check_benchmark_leakage(
    texts1, texts2, benchmark_path: Path
) -> tuple[list[str], int]:
    """Hard fail if ANY training pair appears in the benchmark set."""
    failures = []

    if not benchmark_path.exists():
        print(f"   [SKIP] Benchmark file not found: {benchmark_path}")
        return [], 0

    with open(benchmark_path, encoding="utf-8") as f:
        bench = json.load(f)

    # Benchmark may be a list of dicts or a dict with keys
    bench_pairs: set[str] = set()
    if isinstance(bench, list):
        for item in bench:
            t1 = item.get("text1") or item.get("query") or item.get("sentence1") or ""
            t2 = item.get("text2") or item.get("candidate") or item.get("sentence2") or ""
            if t1 and t2:
                bench_pairs.add(_hash_pair(t1, t2))
    elif isinstance(bench, dict):
        b1 = bench.get("texts1") or bench.get("text1", [])
        b2 = bench.get("texts2") or bench.get("text2", [])
        for t1, t2 in zip(b1, b2):
            bench_pairs.add(_hash_pair(t1, t2))

    print(f"   Benchmark set : {len(bench_pairs)} unique pairs")

    leaked = 0
    for t1, t2 in zip(texts1, texts2):
        if _hash_pair(t1, t2) in bench_pairs:
            leaked += 1

    print(f"   Leakage       : {leaked} training pairs found in benchmark")
    if leaked > 0:
        failures.append(
            f"BENCHMARK LEAKAGE: {leaked} training pair(s) are present in the benchmark set. "
            f"Remove them from training data before proceeding."
        )

    return failures, leaked


def check_phase_sizes(labels, phase_indicators, min_pairs: int = 2000) -> list[str]:
    """Warn if any curriculum phase has fewer than min_pairs pairs."""
    warnings = []
    if phase_indicators is None:
        return warnings

    phase_map: dict[str, int] = {}
    for phase in phase_indicators:
        phase_map[str(phase)] = phase_map.get(str(phase), 0) + 1

    for phase_id, count in sorted(phase_map.items()):
        if count < min_pairs:
            warnings.append(
                f"Phase {phase_id} has only {count} pairs (minimum recommended: {min_pairs})"
            )

    return warnings


def _load_pairs(path: Path) -> tuple:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts1 = data.get("texts1") or data.get("text1", [])
    texts2 = data.get("texts2") or data.get("text2", [])
    labels = data.get("labels", [])
    phase_indicators = data.get("phase_indicators") or data.get("phases")

    if not texts1 or not texts2:
        print(f"[ERROR] Could not parse texts1/texts2 from {path}")
        sys.exit(1)

    if len(texts1) != len(texts2):
        print(f"[ERROR] texts1 ({len(texts1)}) and texts2 ({len(texts2)}) length mismatch")
        sys.exit(1)

    return texts1, texts2, labels, phase_indicators


def main():
    parser = argparse.ArgumentParser(
        description="Validate training pairs before ML training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pairs", required=True, type=Path,
        help="Path to training pairs JSON file"
    )
    parser.add_argument(
        "--benchmark", required=True, type=Path,
        help="Path to benchmark JSON file (for leakage check)"
    )
    parser.add_argument(
        "--min-phase-size", type=int, default=2000,
        help="Minimum pairs per curriculum phase (default: 2000)"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Training Data Validation Report")
    print(f"  Pairs    : {args.pairs}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"{'='*60}\n")

    texts1, texts2, labels, phase_indicators = _load_pairs(args.pairs)
    n = len(texts1)
    print(f"Loaded {n:,} pairs\n")

    all_warnings = []
    all_failures = []

    # 1. Label distribution
    print("[ 1/5 ] Label distribution")
    all_warnings += check_label_distribution(texts1, texts2, labels, phase_indicators)
    print()

    # 2. Duplicate pairs
    print("[ 2/5 ] Duplicate detection")
    failures, _ = check_duplicates(texts1, texts2)
    all_failures += failures
    print()

    # 3. Text length
    print("[ 3/5 ] Text length distribution")
    all_warnings += check_text_lengths(texts1, texts2)
    print()

    # 4. Benchmark leakage (CRITICAL)
    print("[ 4/5 ] Benchmark leakage check")
    failures, _ = check_benchmark_leakage(texts1, texts2, args.benchmark)
    all_failures += failures
    print()

    # 5. Phase sizes
    print("[ 5/5 ] Curriculum phase sizes")
    all_warnings += check_phase_sizes(labels, phase_indicators, args.min_phase_size)
    if phase_indicators is None:
        print("   [SKIP] No phase_indicators found in file")
    print()

    # Summary
    print("="*60)
    if all_warnings:
        print(f"  WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    ⚠  {w}")
        print()

    if all_failures:
        print(f"  FAILURES ({len(all_failures)}) — training blocked:")
        for f in all_failures:
            print(f"    ✗  {f}")
        print()
        print("  Result: FAIL (exit code 1)")
        print("="*60)
        sys.exit(1)
    else:
        print("  Result: PASS — all hard checks passed")
        if all_warnings:
            print("  (review warnings above before proceeding)")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
