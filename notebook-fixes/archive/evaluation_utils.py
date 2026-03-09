"""
Utility functions for model evaluation.

This module provides common utilities used across evaluation notebooks.
"""

import gc
import random
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
from scipy.stats import spearmanr
import sys


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"✓ All random seeds set to {seed}")


# ============================================================================
# DEVICE & MEMORY MANAGEMENT
# ============================================================================

def detect_device() -> Tuple[str, Optional[float]]:
    """
    Auto-detect best available device and GPU memory.

    Returns:
        (device_name, gpu_memory_gb) tuple
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        return device, gpu_memory

    elif torch.backends.mps.is_available():
        device = 'mps'
        print("✓ Using Apple MPS device")
        return device, None

    else:
        device = 'cpu'
        print("⚠ Using CPU (no GPU detected)")
        return device, None


def cleanup_gpu_memory(device: str = None) -> None:
    """
    Clean up GPU/MPS memory to prevent fragmentation.

    Args:
        device: Device type ('cuda', 'mps', or None for auto-detect)
    """
    gc.collect()

    if device is None:
        device, _ = detect_device()

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_gpu_memory_usage(device: str = 'cuda') -> Dict[str, float]:
    """
    Get current GPU memory usage statistics.

    Args:
        device: Device type ('cuda' or 'mps')

    Returns:
        Dictionary with memory stats in GB
    """
    if device == 'cuda' and torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / (1024**3),
            'reserved': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated': torch.cuda.max_memory_allocated() / (1024**3),
        }
    else:
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def print_gpu_memory_summary(device: str = 'cuda') -> None:
    """Print GPU memory usage summary."""
    stats = get_gpu_memory_usage(device)
    if stats['allocated'] > 0:
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated:     {stats['allocated']:.2f} GB")
        print(f"  Reserved:      {stats['reserved']:.2f} GB")
        print(f"  Peak Allocated: {stats['max_allocated']:.2f} GB")


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame has required structure.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name for error messages

    Raises:
        AssertionError: If validation fails
    """
    assert df is not None, f"{name} is None"
    assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
    assert len(df) >= min_rows, f"{name} has {len(df)} rows, expected >= {min_rows}"

    missing_cols = set(required_columns) - set(df.columns)
    assert not missing_cols, f"{name} missing columns: {missing_cols}"

    print(f"✓ {name} validated: {len(df)} rows, {len(df.columns)} columns")


def validate_test_pairs(
    texts1: List[str],
    texts2: List[str],
    labels: np.ndarray,
    name: str = "Test pairs"
) -> None:
    """
    Validate test pair data.

    Args:
        texts1: First texts in pairs
        texts2: Second texts in pairs
        labels: Binary labels
        name: Name for error messages

    Raises:
        AssertionError: If validation fails
    """
    assert len(texts1) == len(texts2) == len(labels), \
        f"{name}: Mismatched lengths (texts1={len(texts1)}, texts2={len(texts2)}, labels={len(labels)})"

    assert len(texts1) > 0, f"{name}: Empty test set"

    assert set(labels).issubset({0, 1}), \
        f"{name}: Labels must be binary (0/1), got {set(labels)}"

    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    print(f"✓ {name} validated:")
    print(f"  Total pairs: {len(labels)}")
    print(f"  Positive: {num_pos} ({num_pos/len(labels)*100:.1f}%)")
    print(f"  Negative: {num_neg} ({num_neg/len(labels)*100:.1f}%)")


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        labels: True binary labels
        scores: Predicted similarity scores
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    predictions = (scores >= threshold).astype(int)

    metrics = {
        'spearman': spearmanr(labels, scores)[0] if len(set(scores)) > 1 else 0.0,
        'roc_auc': roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0,
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'accuracy': accuracy_score(labels, predictions),
    }

    return metrics


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score.

    Args:
        labels: True binary labels
        scores: Predicted similarity scores
        num_thresholds: Number of thresholds to test

    Returns:
        (best_threshold, best_f1) tuple
    """
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def compute_confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        labels: True binary labels
        scores: Predicted similarity scores
        threshold: Classification threshold

    Returns:
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    predictions = (scores >= threshold).astype(int)
    return confusion_matrix(labels, predictions)


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(
    texts1: List[str],
    texts2: List[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    num_examples: int = 5
) -> Dict[str, Any]:
    """
    Analyze false positives and false negatives.

    Args:
        texts1: First texts in pairs
        texts2: Second texts in pairs
        labels: True binary labels
        scores: Predicted similarity scores
        threshold: Classification threshold
        num_examples: Number of examples to return

    Returns:
        Dictionary with error analysis
    """
    predictions = (scores >= threshold).astype(int)

    fp_mask = (predictions == 1) & (labels == 0)
    fn_mask = (predictions == 0) & (labels == 1)

    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    result = {
        'false_positives': {
            'count': len(fp_indices),
            'percentage': len(fp_indices) / len(labels) * 100,
            'mean_score': scores[fp_mask].mean() if len(fp_indices) > 0 else 0,
            'std_score': scores[fp_mask].std() if len(fp_indices) > 0 else 0,
            'examples': []
        },
        'false_negatives': {
            'count': len(fn_indices),
            'percentage': len(fn_indices) / len(labels) * 100,
            'mean_score': scores[fn_mask].mean() if len(fn_indices) > 0 else 0,
            'std_score': scores[fn_mask].std() if len(fn_indices) > 0 else 0,
            'examples': []
        }
    }

    # Add examples
    for idx in fp_indices[:num_examples]:
        result['false_positives']['examples'].append({
            'text1': texts1[idx][:100],
            'text2': texts2[idx][:100],
            'score': float(scores[idx]),
        })

    for idx in fn_indices[:num_examples]:
        result['false_negatives']['examples'].append({
            'text1': texts1[idx][:100],
            'text2': texts2[idx][:100],
            'score': float(scores[idx]),
        })

    return result


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_inference_speed(
    model,
    texts: List[str],
    batch_size: int = 32,
    num_runs: int = 3,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Args:
        model: SentenceTransformer model
        texts: Sample texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of timing runs
        device: Device to use

    Returns:
        Dictionary with timing statistics
    """
    times = []

    # Warmup run
    _ = model.encode(texts[:min(10, len(texts))], batch_size=batch_size, show_progress_bar=False)
    cleanup_gpu_memory(device)

    # Timed runs
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

        cleanup_gpu_memory(device)

    return {
        'total_ms_mean': float(np.mean(times)),
        'total_ms_std': float(np.std(times)),
        'per_sample_ms_mean': float(np.mean(times) / len(texts)),
        'per_sample_ms_std': float(np.std(times) / len(texts)),
        'throughput_samples_per_sec': float(len(texts) / (np.mean(times) / 1000)),
    }


# ============================================================================
# METADATA TRACKING
# ============================================================================

def get_environment_metadata() -> Dict[str, Any]:
    """
    Get environment metadata for reproducibility.

    Returns:
        Dictionary with environment information
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
    }

    try:
        import transformers
        metadata['transformers_version'] = transformers.__version__
    except ImportError:
        pass

    try:
        import sentence_transformers
        metadata['sentence_transformers_version'] = sentence_transformers.__version__
    except ImportError:
        pass

    if torch.cuda.is_available():
        metadata['cuda_available'] = True
        metadata['cuda_version'] = torch.version.cuda
        metadata['device_name'] = torch.cuda.get_device_name(0)
        metadata['device_count'] = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        metadata['mps_available'] = True
        metadata['device_name'] = 'Apple MPS'
    else:
        metadata['device_name'] = 'CPU'

    return metadata


def print_environment_info() -> None:
    """Print environment information."""
    metadata = get_environment_metadata()

    print("="*80)
    print("ENVIRONMENT INFORMATION")
    print("="*80)
    for key, value in metadata.items():
        print(f"  {key:30s}: {value}")
    print("="*80)


# ============================================================================
# FILE I/O
# ============================================================================

def save_results(
    results: Dict[str, Any],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Results dictionary
        output_path: Output file path
        overwrite: Whether to overwrite existing file
    """
    import json

    if output_path.exists() and not overwrite:
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_stem(f"{output_path.stem}_{timestamp}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Results saved to: {output_path}")


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_metrics_table(results: Dict[str, Dict[str, float]], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted table.

    Args:
        results: Dictionary mapping model names to metrics
        title: Table title
    """
    if not results:
        print("No results to display")
        return

    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)

    # Get all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    metric_names = sorted(all_metrics)

    # Header
    print(f"{'Model':<40} " + " ".join(f"{m:>10s}" for m in metric_names))
    print("-"*80)

    # Rows
    for model_name, metrics in results.items():
        values = [f"{metrics.get(m, 0):.4f}" for m in metric_names]
        print(f"{model_name:<40} " + " ".join(f"{v:>10s}" for v in values))

    print("="*80)


if __name__ == '__main__':
    # Test utilities
    print("Testing evaluation utilities...")

    set_all_seeds(42)
    device, gpu_memory = detect_device()
    print_environment_info()

    print("\n✓ All utilities loaded successfully")
