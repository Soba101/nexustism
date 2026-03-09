#!/usr/bin/env python3
"""
Create model_promax_mpnet_lorapeft-v2.ipynb from scratch with all fixes integrated.
"""

import json
from pathlib import Path

def create_markdown_cell(source):
    """Create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def create_code_cell(source):
    """Create a code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

def create_notebook():
    """Create the complete notebook structure"""
    cells = []

    # ====================================================================================
    # CELL 0: MARKDOWN - Title
    # ====================================================================================
    cells.append(create_markdown_cell([
        "# MPNet LoRA Fine-tuning for ServiceNow Incident Similarity (v2)\n",
        "\n",
        "**Model:** `sentence-transformers/all-mpnet-base-v2` with LoRA/PEFT\n",
        "**Training:** Curriculum learning (3 phases: easy → medium → hard)\n",
        "**Data:** 15,000 pre-generated curriculum pairs + 1,000 test pairs\n",
        "**Objective:** Improve Spearman correlation from 0.4885 to 0.52+ (beat baseline 0.5038)\n"
    ]))

    # ====================================================================================
    # CELL 1: CODE - Environment Setup & Package Installation
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 1: Environment Setup & Package Installation\n",
        "\n",
        "import os, sys, subprocess\n",
        "from pathlib import Path\n",
        "\n",
        "# Suppress warnings\n",
        "os.environ['WANDB_DISABLED'] = 'true'\n",
        "os.environ['WANDB_MODE'] = 'offline'\n",
        "os.environ['WANDB_SILENT'] = 'true'\n",
        "os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'\n",
        "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n",
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'\n",
        "\n",
        "# Install required packages\n",
        "def ensure_packages():\n",
        "    try:\n",
        "        import importlib.metadata as importlib_metadata\n",
        "    except ImportError:\n",
        "        import importlib_metadata\n",
        "    \n",
        "    required = {\n",
        "        'sentence-transformers': 'sentence-transformers>=2.2.2',\n",
        "        'torch': 'torch',\n",
        "        'scikit-learn': 'scikit-learn>=1.3.0',\n",
        "        'pandas': 'pandas',\n",
        "        'numpy': 'numpy>=1.24.0',\n",
        "        'tqdm': 'tqdm',\n",
        "        'matplotlib': 'matplotlib',\n",
        "        'seaborn': 'seaborn',\n",
        "        'peft': 'peft>=0.4.0',\n",
        "        'transformers': 'transformers>=4.30.0',\n",
        "        'datasets': 'datasets>=2.13.1',\n",
        "    }\n",
        "    \n",
        "    missing = []\n",
        "    for name, spec in required.items():\n",
        "        try:\n",
        "            importlib_metadata.version(name)\n",
        "        except importlib_metadata.PackageNotFoundError:\n",
        "            missing.append(spec)\n",
        "    \n",
        "    if missing:\n",
        "        print(f'[INSTALL] Installing: {\", \".join(missing)}')\n",
        "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])\n",
        "    else:\n",
        "        print('[OK] All packages installed')\n",
        "\n",
        "ensure_packages()\n"
    ]))

    # ====================================================================================
    # CELL 2: CODE - Core Imports
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 2: Core Imports\n",
        "\n",
        "import random\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import json\n",
        "import logging\n",
        "from datetime import datetime\n",
        "from pathlib import Path\n",
        "from tqdm.auto import tqdm\n",
        "\n",
        "# SentenceTransformers\n",
        "from sentence_transformers import (\n",
        "    SentenceTransformer,\n",
        "    InputExample,\n",
        "    losses,\n",
        "    evaluation,\n",
        "    util\n",
        ")\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "# LoRA/PEFT\n",
        "from peft import (\n",
        "    get_peft_model,\n",
        "    LoraConfig,\n",
        "    TaskType,\n",
        "    PeftModel\n",
        ")\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import (\n",
        "    roc_auc_score,\n",
        "    f1_score,\n",
        "    precision_score,\n",
        "    recall_score\n",
        ")\n",
        "from scipy.stats import spearmanr\n",
        "\n",
        "# Visualization\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "print('[OK] All imports successful')\n"
    ]))

    # ====================================================================================
    # CELL 3: CODE - Configuration
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 3: Configuration\n",
        "\n",
        "CONFIG = {\n",
        "    # Model\n",
        "    'model_name': 'sentence-transformers/all-mpnet-base-v2',\n",
        "    \n",
        "    # Training\n",
        "    'epochs': 6,  # 2 epochs per curriculum phase\n",
        "    'batch_size': 16,  # Will auto-adjust based on device\n",
        "    'lr': 5e-5,  # Increased from 2e-5 for LoRA\n",
        "    'warmup_steps': 100,\n",
        "    'max_seq_length': 256,  # Reduced from 384 to match baseline\n",
        "    'seed': 42,\n",
        "    \n",
        "    # LoRA\n",
        "    'use_lora': True,\n",
        "    'lora_r': 16,  # Rank\n",
        "    'lora_alpha': 32,  # Scaling factor\n",
        "    'lora_dropout': 0.1,\n",
        "    \n",
        "    # Data - CURRICULUM LEARNING\n",
        "    'use_pre_generated_pairs': True,\n",
        "    'train_pairs_path': 'data_new/curriculum_training_pairs_20251224_065436.json',\n",
        "    'test_pairs_path': 'data_new/fixed_test_pairs.json',\n",
        "    'use_curriculum': True,\n",
        "    'epochs_per_phase': 2,\n",
        "    \n",
        "    # Output\n",
        "    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',\n",
        "    'save_best_model': True,\n",
        "    \n",
        "    # Evaluation\n",
        "    'eval_steps': 500,\n",
        "    'threshold_cv_folds': 5,\n",
        "}\n",
        "\n",
        "# Set random seeds\n",
        "random.seed(CONFIG['seed'])\n",
        "np.random.seed(CONFIG['seed'])\n",
        "torch.manual_seed(CONFIG['seed'])\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(CONFIG['seed'])\n",
        "\n",
        "print('[OK] Configuration loaded')\n",
        "print(f\"  Model: {CONFIG['model_name']}\")\n",
        "print(f\"  Learning rate: {CONFIG['lr']}\")\n",
        "print(f\"  Curriculum learning: {CONFIG['use_curriculum']}\")\n",
        "print(f\"  Training pairs: {CONFIG['train_pairs_path']}\")\n"
    ]))

    # ====================================================================================
    # CELL 4: CODE - Logging Utilities
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 4: Logging Utilities\n",
        "\n",
        "logging_buffer = []\n",
        "\n",
        "def log(message):\n",
        "    \"\"\"Print and buffer log messages\"\"\"\n",
        "    timestamp = datetime.now().strftime('%H:%M:%S')\n",
        "    formatted = f\"[{timestamp}] {message}\"\n",
        "    print(formatted)\n",
        "    logging_buffer.append(formatted)\n",
        "\n",
        "log('Logging initialized')\n"
    ]))

    # ====================================================================================
    # CELL 5: MARKDOWN - Data Loading
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 1. Load Pre-Generated Pairs\n",
        "\n",
        "Load curriculum training pairs (15K) and test pairs (1K) from pre-generated JSON files.\n"
    ]))

    # ====================================================================================
    # CELL 6: CODE - Load Curriculum & Test Pairs
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 6: Load Curriculum & Test Pairs\n",
        "\n",
        "def load_curriculum_pairs(pairs_path):\n",
        "    \"\"\"Load pre-generated curriculum pairs from JSON\"\"\"\n",
        "    log(f\"Loading curriculum pairs from: {pairs_path}\")\n",
        "    \n",
        "    with open(pairs_path, 'r', encoding='utf-8') as f:\n",
        "        data = json.load(f)\n",
        "    \n",
        "    texts1 = data['texts1']\n",
        "    texts2 = data['texts2']\n",
        "    labels = data['labels']\n",
        "    phase_indicators = data.get('phase_indicators', [1] * len(labels))\n",
        "    \n",
        "    # Separate by phase\n",
        "    phase1_pairs = []\n",
        "    phase2_pairs = []\n",
        "    phase3_pairs = []\n",
        "    \n",
        "    for i in range(len(labels)):\n",
        "        example = InputExample(texts=[texts1[i], texts2[i]], label=float(labels[i]))\n",
        "        phase = phase_indicators[i]\n",
        "        \n",
        "        if phase == 1:\n",
        "            phase1_pairs.append(example)\n",
        "        elif phase == 2:\n",
        "            phase2_pairs.append(example)\n",
        "        elif phase == 3:\n",
        "            phase3_pairs.append(example)\n",
        "    \n",
        "    log(f\"Loaded {len(labels):,} total pairs\")\n",
        "    log(f\"  Phase 1 (Easy):   {len(phase1_pairs):,} pairs\")\n",
        "    log(f\"  Phase 2 (Medium): {len(phase2_pairs):,} pairs\")\n",
        "    log(f\"  Phase 3 (Hard):   {len(phase3_pairs):,} pairs\")\n",
        "    \n",
        "    return phase1_pairs, phase2_pairs, phase3_pairs\n",
        "\n",
        "def load_test_pairs(pairs_path):\n",
        "    \"\"\"Load test pairs from JSON\"\"\"\n",
        "    log(f\"Loading test pairs from: {pairs_path}\")\n",
        "    \n",
        "    with open(pairs_path, 'r', encoding='utf-8') as f:\n",
        "        data = json.load(f)\n",
        "    \n",
        "    test_pairs = [\n",
        "        InputExample(texts=[t1, t2], label=float(label))\n",
        "        for t1, t2, label in zip(data['texts1'], data['texts2'], data['labels'])\n",
        "    ]\n",
        "    \n",
        "    log(f\"Loaded {len(test_pairs):,} test pairs\")\n",
        "    pos_count = sum(1 for ex in test_pairs if ex.label == 1.0)\n",
        "    log(f\"  Positives: {pos_count:,} ({100*pos_count/len(test_pairs):.1f}%)\")\n",
        "    log(f\"  Negatives: {len(test_pairs)-pos_count:,}\")\n",
        "    \n",
        "    return test_pairs\n",
        "\n",
        "# Load pairs\n",
        "log(\"=\"*70)\n",
        "log(\"LOADING PRE-GENERATED CURRICULUM PAIRS\")\n",
        "log(\"=\"*70)\n",
        "\n",
        "phase1_train, phase2_train, phase3_train = load_curriculum_pairs(CONFIG['train_pairs_path'])\n",
        "eval_examples = load_test_pairs(CONFIG['test_pairs_path'])\n",
        "\n",
        "# Combine for total count\n",
        "train_examples = phase1_train + phase2_train + phase3_train\n",
        "log(f\"\\nTotal training examples: {len(train_examples):,}\")\n",
        "\n",
        "# Store phases for curriculum training\n",
        "CURRICULUM_PHASES = {\n",
        "    'phase1': phase1_train,\n",
        "    'phase2': phase2_train,\n",
        "    'phase3': phase3_train\n",
        "}\n"
    ]))

    # ====================================================================================
    # CELL 7: MARKDOWN - Device Detection
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 2. Device Detection\n",
        "\n",
        "Auto-detect CUDA/MPS/CPU and configure training accordingly.\n"
    ]))

    # ====================================================================================
    # CELL 8: CODE - Device Detection
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 8: Device Detection\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    DEVICE = 'cuda'\n",
        "    log(f\"[OK] Using CUDA: {torch.cuda.get_device_name(0)}\")\n",
        "    log(f\"   CUDA version: {torch.version.cuda}\")\n",
        "    log(f\"   Device capability: {torch.cuda.get_device_capability(0)}\")\n",
        "    CONFIG['batch_size'] = 32  # Larger batch for GPU\n",
        "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
        "    DEVICE = 'mps'\n",
        "    log(\"[OK] Using MPS (Apple Silicon)\")\n",
        "    CONFIG['batch_size'] = 8  # Smaller batch for MPS\n",
        "else:\n",
        "    DEVICE = 'cpu'\n",
        "    log(\"[OK] Using CPU\")\n",
        "    CONFIG['batch_size'] = 8\n",
        "\n",
        "log(f\"\\nDevice: {DEVICE}\")\n",
        "log(f\"Batch size: {CONFIG['batch_size']}\")\n"
    ]))

    # ====================================================================================
    # CELL 9: MARKDOWN - Model Setup
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 3. Model Setup\n",
        "\n",
        "Initialize MPNet with LoRA adapters for parameter-efficient fine-tuning.\n"
    ]))

    # ====================================================================================
    # CELL 10: CODE - Model Initialization
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 10: Model Initialization with LoRA\n",
        "\n",
        "def init_model_with_lora(config, device):\n",
        "    \"\"\"Initialize SentenceTransformer with LoRA adapters\"\"\"\n",
        "    log(f\"Initializing model: {config['model_name']}\")\n",
        "    \n",
        "    # Load base model\n",
        "    model = SentenceTransformer(config['model_name'], device=device)\n",
        "    model.max_seq_length = config['max_seq_length']\n",
        "    \n",
        "    if config['use_lora']:\n",
        "        log(\"Applying LoRA adapters...\")\n",
        "        \n",
        "        # Configure LoRA\n",
        "        lora_config = LoraConfig(\n",
        "            r=config['lora_r'],\n",
        "            lora_alpha=config['lora_alpha'],\n",
        "            target_modules=['query', 'key', 'value', 'dense'],\n",
        "            lora_dropout=config['lora_dropout'],\n",
        "            bias='none',\n",
        "            task_type=TaskType.FEATURE_EXTRACTION\n",
        "        )\n",
        "        \n",
        "        # Apply LoRA to the transformer\n",
        "        model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)\n",
        "        \n",
        "        # Print trainable parameters\n",
        "        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
        "        total_params = sum(p.numel() for p in model.parameters())\n",
        "        log(f\"  LoRA rank: {config['lora_r']}\")\n",
        "        log(f\"  LoRA alpha: {config['lora_alpha']}\")\n",
        "        log(f\"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\")\n",
        "    \n",
        "    return model\n",
        "\n",
        "# Initialize model\n",
        "model = init_model_with_lora(CONFIG, DEVICE)\n",
        "log(\"[OK] Model initialized\")\n"
    ]))

    # ====================================================================================
    # CELL 11: MARKDOWN - Training
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 4. Training\n",
        "\n",
        "Train with curriculum learning: 3 phases from easy → medium → hard pairs.\n"
    ]))

    # ====================================================================================
    # CELL 12: CODE - Training Loop
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 12: Curriculum Training\n",
        "\n",
        "from sentence_transformers import losses\n",
        "\n",
        "# Setup output directory\n",
        "save_path = Path(CONFIG['output_dir'])\n",
        "save_path.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "log(\"\\n\" + \"=\"*70)\n",
        "log(\"STARTING CURRICULUM TRAINING\")\n",
        "log(\"=\"*70)\n",
        "\n",
        "# Define loss function\n",
        "train_loss = losses.CosineSimilarityLoss(model)\n",
        "\n",
        "# Create evaluator\n",
        "evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(\n",
        "    eval_examples[:100],  # Use subset for faster evaluation during training\n",
        "    name='eval_subset'\n",
        ")\n",
        "\n",
        "# Curriculum training: 3 phases\n",
        "phases = [\n",
        "    ('Phase 1: Easy', CURRICULUM_PHASES['phase1']),\n",
        "    ('Phase 2: Medium', CURRICULUM_PHASES['phase2']),\n",
        "    ('Phase 3: Hard', CURRICULUM_PHASES['phase3'])\n",
        "]\n",
        "\n",
        "for phase_num, (phase_name, phase_data) in enumerate(phases, 1):\n",
        "    log(f\"\\n{'='*70}\")\n",
        "    log(f\"TRAINING {phase_name}\")\n",
        "    log(f\"{'='*70}\")\n",
        "    log(f\"Pairs: {len(phase_data):,}\")\n",
        "    log(f\"Epochs: {CONFIG['epochs_per_phase']}\")\n",
        "    \n",
        "    # Create DataLoader\n",
        "    train_dataloader = DataLoader(\n",
        "        phase_data,\n",
        "        shuffle=True,\n",
        "        batch_size=CONFIG['batch_size'],\n",
        "        num_workers=0  # Required for InputExample\n",
        "    )\n",
        "    \n",
        "    # Train\n",
        "    model.fit(\n",
        "        train_objectives=[(train_dataloader, train_loss)],\n",
        "        epochs=CONFIG['epochs_per_phase'],\n",
        "        warmup_steps=CONFIG['warmup_steps'],\n",
        "        evaluator=evaluator,\n",
        "        evaluation_steps=CONFIG['eval_steps'],\n",
        "        output_path=str(save_path),\n",
        "        save_best_model=True,\n",
        "        show_progress_bar=True,\n",
        "        optimizer_params={'lr': CONFIG['lr']}\n",
        "    )\n",
        "    \n",
        "    log(f\"[OK] {phase_name} complete\")\n",
        "\n",
        "log(\"\\n\" + \"=\"*70)\n",
        "log(\"TRAINING COMPLETE\")\n",
        "log(\"=\"*70)\n",
        "\n",
        "# Load best model\n",
        "log(\"Loading best model...\")\n",
        "best_model = SentenceTransformer(str(save_path), device=DEVICE)\n",
        "log(\"[OK] Best model loaded\")\n"
    ]))

    # ====================================================================================
    # CELL 13: MARKDOWN - Evaluation
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 5. Evaluation\n",
        "\n",
        "Evaluate the trained model on the test set.\n"
    ]))

    # ====================================================================================
    # CELL 14: CODE - Evaluation
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 14: Final Evaluation\n",
        "\n",
        "def comprehensive_eval(pairs, model, name=\"Eval\"):\n",
        "    \"\"\"Comprehensive evaluation on a set of pairs\"\"\"\n",
        "    log(f\"\\nEvaluating on {name} ({len(pairs):,} pairs)...\")\n",
        "    \n",
        "    # Extract texts and labels\n",
        "    texts1 = [ex.texts[0] for ex in pairs]\n",
        "    texts2 = [ex.texts[1] for ex in pairs]\n",
        "    labels = np.array([ex.label for ex in pairs])\n",
        "    \n",
        "    # Encode\n",
        "    emb1 = model.encode(texts1, show_progress_bar=True, convert_to_numpy=True)\n",
        "    emb2 = model.encode(texts2, show_progress_bar=True, convert_to_numpy=True)\n",
        "    \n",
        "    # Compute cosine similarity\n",
        "    scores = np.array([np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8) \n",
        "                       for e1, e2 in zip(emb1, emb2)])\n",
        "    \n",
        "    # Spearman correlation\n",
        "    spearman, _ = spearmanr(labels, scores)\n",
        "    \n",
        "    # Binary classification metrics (threshold = 0.5)\n",
        "    predictions = (scores > 0.5).astype(int)\n",
        "    labels_binary = labels.astype(int)\n",
        "    \n",
        "    roc_auc = roc_auc_score(labels_binary, scores)\n",
        "    f1 = f1_score(labels_binary, predictions)\n",
        "    precision = precision_score(labels_binary, predictions)\n",
        "    recall = recall_score(labels_binary, predictions)\n",
        "    \n",
        "    # Separability\n",
        "    pos_scores = scores[labels == 1.0]\n",
        "    neg_scores = scores[labels == 0.0]\n",
        "    separability = np.mean(pos_scores) - np.mean(neg_scores) if len(pos_scores) > 0 and len(neg_scores) > 0 else 0.0\n",
        "    \n",
        "    # Print results\n",
        "    log(f\"\\n{name} Results:\")\n",
        "    log(f\"  Spearman:     {spearman:.4f}\")\n",
        "    log(f\"  ROC-AUC:      {roc_auc:.4f}\")\n",
        "    log(f\"  F1:           {f1:.4f}\")\n",
        "    log(f\"  Precision:    {precision:.4f}\")\n",
        "    log(f\"  Recall:       {recall:.4f}\")\n",
        "    log(f\"  Separability: {separability:.4f}\")\n",
        "    log(f\"  Pos mean:     {np.mean(pos_scores):.4f}\")\n",
        "    log(f\"  Neg mean:     {np.mean(neg_scores):.4f}\")\n",
        "    \n",
        "    return {\n",
        "        'spearman': spearman,\n",
        "        'roc_auc': roc_auc,\n",
        "        'f1': f1,\n",
        "        'precision': precision,\n",
        "        'recall': recall,\n",
        "        'separability': separability\n",
        "    }\n",
        "\n",
        "# Run evaluation\n",
        "log(\"=\"*70)\n",
        "log(\"FINAL EVALUATION\")\n",
        "log(\"=\"*70)\n",
        "\n",
        "results = comprehensive_eval(eval_examples, best_model, \"Test Set\")\n",
        "\n",
        "log(\"\\n\" + \"=\"*70)\n",
        "log(\"EVALUATION COMPLETE\")\n",
        "log(\"=\"*70)\n",
        "\n",
        "# Compare to baseline\n",
        "baseline_spearman = 0.5038\n",
        "improvement = ((results['spearman'] - baseline_spearman) / baseline_spearman) * 100\n",
        "log(f\"\\nBaseline Spearman: {baseline_spearman:.4f}\")\n",
        "log(f\"New Model Spearman: {results['spearman']:.4f}\")\n",
        "log(f\"Improvement: {improvement:+.1f}%\")\n",
        "\n",
        "if results['spearman'] > baseline_spearman:\n",
        "    log(\"\\n[SUCCESS] Model beats baseline!\")\n",
        "else:\n",
        "    log(\"\\n[NOTICE] Model did not beat baseline. Consider additional training.\")\n"
    ]))

    # ====================================================================================
    # CELL 15: MARKDOWN - Save
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## 6. Save Model & Metadata\n",
        "\n",
        "Save the trained model and training metadata.\n"
    ]))

    # ====================================================================================
    # CELL 16: CODE - Save Model
    # ====================================================================================
    cells.append(create_code_cell([
        "# CELL 16: Save Model & Metadata\n",
        "\n",
        "# Model is already saved during training to CONFIG['output_dir']\n",
        "log(f\"\\nModel saved to: {CONFIG['output_dir']}\")\n",
        "\n",
        "# Save training metadata\n",
        "metadata = {\n",
        "    'config': CONFIG,\n",
        "    'results': results,\n",
        "    'training_date': datetime.now().isoformat(),\n",
        "    'device': DEVICE,\n",
        "    'total_training_pairs': len(train_examples),\n",
        "    'total_test_pairs': len(eval_examples),\n",
        "    'curriculum_phases': {\n",
        "        'phase1': len(phase1_train),\n",
        "        'phase2': len(phase2_train),\n",
        "        'phase3': len(phase3_train)\n",
        "    }\n",
        "}\n",
        "\n",
        "metadata_path = save_path / 'training_metadata.json'\n",
        "with open(metadata_path, 'w', encoding='utf-8') as f:\n",
        "    json.dump(metadata, f, indent=2)\n",
        "\n",
        "log(f\"Metadata saved to: {metadata_path}\")\n",
        "log(\"\\n[OK] All files saved successfully\")\n"
    ]))

    # ====================================================================================
    # CELL 17: MARKDOWN - Summary
    # ====================================================================================
    cells.append(create_markdown_cell([
        "## Summary\n",
        "\n",
        "Training complete! The model has been trained using curriculum learning and evaluated on the test set.\n",
        "\n",
        "### Next Steps:\n",
        "1. Review the evaluation metrics above\n",
        "2. If performance is satisfactory, deploy the model from `models/real_servicenow_finetuned_mpnet_lora/`\n",
        "3. Use `evaluate_model.ipynb` for more detailed analysis\n",
        "4. Generate embeddings for production use with `supabase/embed_incidents.py`\n"
    ]))

    return cells

def main():
    """Create the v2 notebook"""
    print("Creating model_promax_mpnet_lorapeft-v2.ipynb...")

    cells = create_notebook()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = Path('model_promax_mpnet_lorapeft-v2.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("SUCCESS: Created model_promax_mpnet_lorapeft-v2.ipynb")
    print(f"{'='*70}")
    print(f"\nNotebook structure:")
    print(f"  Total cells: {len(cells)}")
    print(f"  Code cells: {len([c for c in cells if c['cell_type'] == 'code'])}")
    print(f"  Markdown cells: {len([c for c in cells if c['cell_type'] == 'markdown'])}")
    print(f"\nKey features:")
    print(f"  - Clean structure from scratch")
    print(f"  - All fixes integrated")
    print(f"  - Curriculum learning (3 phases)")
    print(f"  - LoRA fine-tuning")
    print(f"  - Comprehensive evaluation")
    print(f"  - No Unicode issues")
    print(f"  - All variables properly defined")
    print(f"\nTo run:")
    print(f"  1. Open model_promax_mpnet_lorapeft-v2.ipynb")
    print(f"  2. Run All Cells")
    print(f"  3. Wait for training (~30-60 min)")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
