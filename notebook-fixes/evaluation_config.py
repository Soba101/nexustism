"""
Centralized configuration for model evaluation.

This module contains all configuration parameters for evaluate_model_v2.ipynb.
Modify settings here instead of in the notebook cells.
"""

from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data_new'
MODELS_DIR = PROJECT_ROOT / 'models'

# Data files
INCIDENT_DATA_FILE = 'SNow_incident_ticket_data.csv'
TEST_PAIRS_FILE = 'fixed_test_pairs.json'
RESULTS_OUTPUT_DIR = MODELS_DIR / 'results'

# ============================================================================
# DEVICE & MEMORY CONFIGURATION
# ============================================================================

# Auto-detect device (CUDA > MPS > CPU)
AUTO_DETECT_DEVICE = True

# Batch size configuration (auto-adjusted based on GPU memory)
BATCH_SIZE_CONFIG = {
    'auto_detect': True,  # Automatically adjust based on GPU memory
    'default': 32,        # Default for most models
    'large_model': 8,     # For models with large context (Nomic, JinaBERT)

    # GPU memory-based batch sizes (when auto_detect=True)
    'gpu_memory_thresholds': {
        24: 128,  # >= 24GB VRAM
        16: 64,   # >= 16GB VRAM
        8: 32,    # >= 8GB VRAM
        4: 16,    # >= 4GB VRAM
    },

    # MPS (Apple Silicon) uses smaller batches
    'mps_batch_size': 8,
}

# Memory management
MEMORY_CONFIG = {
    'enable_expandable_segments': True,  # Prevent fragmentation
    'cleanup_between_models': True,      # Clear GPU cache between models
}

# ============================================================================
# DATA PROCESSING
# ============================================================================

DATA_CONFIG = {
    # Text columns to combine from incident data
    'text_columns': ['Number', 'Description', 'User input', 'Resolution notes'],

    # Minimum text length for filtering
    'min_text_length': 10,

    # Random seed for reproducibility
    'random_seed': 42,

    # Test pair generation (if generating on the fly)
    'num_positives': 500,
    'num_negatives': 500,
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Baseline model
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'

# Fine-tuned models to evaluate
FINETUNED_MODELS: List[str] = [
    'v6_refactored_finetuned/v6_refactored_finetuned_20251204_1424',
    'real_servicenow_finetuned_mpnet/real_servicenow_v2_20251210_1939',
    'real_servicenow_finetuned_mpnet_lora',
]

# Additional baseline models to compare
ADDITIONAL_BASELINES: List[Dict[str, Any]] = [
    {
        'name': 'Nomic-Embed-v1.5',
        'model_id': 'nomic-ai/nomic-embed-text-v1.5',
        'trust_remote_code': True,
        'install_deps': ['einops'],
        'batch_size': 8,  # Large context model
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'BGE-base-en-v1.5',
        'model_id': 'BAAI/bge-base-en-v1.5',
        'trust_remote_code': False,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'GTE-base-en-v1.5',
        'model_id': 'Alibaba-NLP/gte-base-en-v1.5',
        'trust_remote_code': True,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'E5-base-v2',
        'model_id': 'intfloat/e5-base-v2',
        'trust_remote_code': False,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'MiniLM-L12-v2',
        'model_id': 'sentence-transformers/all-MiniLM-L12-v2',
        'trust_remote_code': False,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'JinaBERT-v2-base',
        'model_id': 'jinaai/jina-embeddings-v2-base-en',
        'trust_remote_code': True,
        'install_deps': [],
        'batch_size': 8,  # Large context model
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'UAE-Large-v1',
        'model_id': 'WhereIsAI/UAE-Large-V1',
        'trust_remote_code': True,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'EmbeddingGemma-300M',
        'model_id': 'google/embeddinggemma-300m',
        'trust_remote_code': True,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {},
        'tokenizer_kwargs': {},
        'encode_kwargs': {},
    },
    {
        'name': 'Qwen3-Embed-0.6B-768d',
        'model_id': 'Qwen/Qwen3-Embedding-0.6B',
        'trust_remote_code': True,
        'install_deps': [],
        'batch_size': 32,
        'model_kwargs': {'device_map': 'auto'},
        'tokenizer_kwargs': {'padding_side': 'left'},
        'encode_kwargs': {'output_value': None, 'output_precision': 'float32'},
    },
]

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    # Number of thresholds to test for optimal F1
    'num_thresholds': 100,

    # Metrics to compute
    'metrics': ['spearman', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy'],

    # Adversarial diagnostic thresholds (per CLAUDE.md requirements)
    'adversarial_diagnostic': {
        'enabled': True,
        'min_roc_auc': 0.70,  # Minimum ROC-AUC to pass
        'min_f1': 0.70,       # Minimum F1 to pass
        'tfidf_threshold_high': 0.5,  # High similarity threshold
        'tfidf_threshold_low': 0.3,   # Low similarity threshold
    },

    # Per-category evaluation
    'per_category_analysis': True,

    # Error analysis
    'error_analysis': {
        'enabled': True,
        'num_examples': 5,  # Number of error examples to show
    },

    # Inference benchmarking
    'benchmarking': {
        'enabled': True,
        'num_runs': 3,  # Number of timing runs
    },

    # Statistical significance testing
    'statistical_testing': {
        'enabled': True,
        'num_bootstrap_samples': 1000,
        'confidence_level': 0.95,
    },
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',

    # Color palette
    'colors': {
        'baseline': '#1f77b4',
        'finetuned': '#ff7f0e',
        'positive': '#2ca02c',
        'negative': '#d62728',
    },

    # Save figures
    'save_figures': True,
    'figure_output_dir': RESULTS_OUTPUT_DIR / 'figures',
}

# ============================================================================
# HUGGINGFACE AUTHENTICATION
# ============================================================================

HUGGINGFACE_CONFIG = {
    # Use environment variable for token (more secure)
    'use_env_token': True,
    'env_var_name': 'HUGGINGFACE_TOKEN',

    # Fallback to interactive login if no token found
    'interactive_login_fallback': True,
}

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

LOGGING_CONFIG = {
    'verbose': True,
    'log_file': RESULTS_OUTPUT_DIR / 'evaluation_log.txt',

    # Track metadata
    'track_metadata': True,
    'metadata_fields': [
        'timestamp',
        'python_version',
        'torch_version',
        'transformers_version',
        'device_name',
        'cuda_version',
        'random_seed',
    ],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_batch_size(device: str, gpu_memory_gb: float = None) -> int:
    """
    Get appropriate batch size based on device and GPU memory.

    Args:
        device: Device type ('cuda', 'mps', or 'cpu')
        gpu_memory_gb: GPU memory in GB (if known)

    Returns:
        Recommended batch size
    """
    if not BATCH_SIZE_CONFIG['auto_detect']:
        return BATCH_SIZE_CONFIG['default']

    if device == 'mps':
        return BATCH_SIZE_CONFIG['mps_batch_size']

    if device == 'cuda' and gpu_memory_gb is not None:
        thresholds = BATCH_SIZE_CONFIG['gpu_memory_thresholds']
        for threshold, batch_size in sorted(thresholds.items(), reverse=True):
            if gpu_memory_gb >= threshold:
                return batch_size

    return BATCH_SIZE_CONFIG['default']


def validate_config() -> None:
    """Validate configuration settings."""
    # Check paths exist
    assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"

    # Check data files
    data_file = DATA_DIR / INCIDENT_DATA_FILE
    if not data_file.exists():
        print(f"Warning: Incident data file not found: {data_file}")

    test_pairs_file = DATA_DIR / TEST_PAIRS_FILE
    if not test_pairs_file.exists():
        print(f"Warning: Test pairs file not found: {test_pairs_file}")

    # Create output directories
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if VISUALIZATION_CONFIG['save_figures']:
        VISUALIZATION_CONFIG['figure_output_dir'].mkdir(parents=True, exist_ok=True)

    print("✓ Configuration validated successfully")


if __name__ == '__main__':
    # Test configuration
    validate_config()
    print(f"\nConfiguration loaded:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Results directory: {RESULTS_OUTPUT_DIR}")
    print(f"  Baseline model: {BASELINE_MODEL}")
    print(f"  Fine-tuned models: {len(FINETUNED_MODELS)}")
    print(f"  Additional baselines: {len(ADDITIONAL_BASELINES)}")
