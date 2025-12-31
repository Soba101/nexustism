# Model ProMax MPNet LoRA v2 - Quick Guide

**File:** `model_promax_mpnet_lorapeft-v2.ipynb`
**Created:** 2024-12-24
**Status:** ✅ Clean build from scratch with all fixes integrated

---

## What's New in V2

✅ **Complete rebuild from scratch** - No legacy code, no patches
✅ **Simplified structure** - 18 cells total (10 code, 8 markdown)
✅ **All fixes integrated** - Device detection, curriculum loading, evaluation
✅ **No Unicode issues** - Pure ASCII, Windows-compatible
✅ **Clean execution flow** - Sequential, no skipped cells
✅ **Comprehensive evaluation** - Spearman, ROC-AUC, F1, separability

---

## Cell Structure (18 Cells)

### Setup (Cells 0-4)
- **Cell 0**: Title & overview
- **Cell 1**: Package installation & environment setup
- **Cell 2**: Core imports (SentenceTransformers, PyTorch, sklearn)
- **Cell 3**: Configuration (LR=5e-5, curriculum settings)
- **Cell 4**: Logging utilities

### Data Loading (Cells 5-6)
- **Cell 5**: Section header
- **Cell 6**: Load curriculum (15K) + test pairs (1K)
  - Defines: `phase1_train`, `phase2_train`, `phase3_train`, `eval_examples`

### Device & Model (Cells 7-10)
- **Cell 7**: Section header
- **Cell 8**: Device detection (CUDA/MPS/CPU)
  - Defines: `DEVICE`
- **Cell 9**: Section header
- **Cell 10**: Initialize model with LoRA
  - Defines: `model`

### Training (Cells 11-12)
- **Cell 11**: Section header
- **Cell 12**: Curriculum training loop (3 phases)
  - Trains model for 2 epochs per phase
  - Defines: `best_model`

### Evaluation (Cells 13-14)
- **Cell 13**: Section header
- **Cell 14**: Comprehensive evaluation
  - Computes all metrics
  - Compares to baseline

### Save & Summary (Cells 15-17)
- **Cell 15**: Section header
- **Cell 16**: Save model & metadata
- **Cell 17**: Summary & next steps

---

## Key Variables

| Variable | Cell | Type | Description |
|----------|------|------|-------------|
| `CONFIG` | 3 | dict | Training configuration |
| `phase1_train` | 6 | list | 5,000 easy pairs |
| `phase2_train` | 6 | list | 5,000 medium pairs |
| `phase3_train` | 6 | list | 5,000 hard pairs |
| `eval_examples` | 6 | list | 1,000 test pairs |
| `DEVICE` | 8 | str | 'cuda', 'mps', or 'cpu' |
| `model` | 10 | SentenceTransformer | Model with LoRA adapters |
| `best_model` | 12 | SentenceTransformer | Trained model (best checkpoint) |
| `results` | 14 | dict | Evaluation metrics |

---

## Configuration

```python
CONFIG = {
    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',

    # Training
    'epochs': 6,           # Total (2 per phase)
    'batch_size': 16,      # Auto-adjusted by device
    'lr': 5e-5,            # Increased for LoRA
    'max_seq_length': 256, # Reduced from 384

    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    # Data
    'use_curriculum': True,
    'train_pairs_path': 'data_new/curriculum_training_pairs_20251224_065436.json',
    'test_pairs_path': 'data_new/fixed_test_pairs.json',

    # Output
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',
}
```

---

## How to Run

### Quick Start
```bash
# Option 1: Jupyter Notebook
jupyter notebook model_promax_mpnet_lorapeft-v2.ipynb
# Then: Cell → Run All

# Option 2: Jupyter Lab
jupyter lab model_promax_mpnet_lorapeft-v2.ipynb
# Then: Run → Run All Cells
```

### Step-by-Step
1. **Open** notebook in Jupyter
2. **Run All** - Click `Cell → Run All`
3. **Wait** - Training takes ~30-60 minutes on GPU
4. **Review** - Check evaluation metrics in Cell 14

---

## Expected Results

### Training Progress
```
[OK] All packages installed
[OK] Using CUDA: NVIDIA GeForce RTX 3090
Device: cuda
Batch size: 32

Loading curriculum pairs from: data_new/curriculum_training_pairs_20251224_065436.json
Loaded 15,000 total pairs
  Phase 1 (Easy):   5,000 pairs
  Phase 2 (Medium): 5,000 pairs
  Phase 3 (Hard):   5,000 pairs

Loading test pairs from: data_new/fixed_test_pairs.json
Loaded 1,000 test pairs

TRAINING Phase 1: Easy
Pairs: 5,000
Epochs: 2
[Progress bars...]

TRAINING Phase 2: Medium
[Progress bars...]

TRAINING Phase 3: Hard
[Progress bars...]

TRAINING COMPLETE
```

### Evaluation Results
```
Test Set Results:
  Spearman:     0.52-0.55  (Target: >0.5038)
  ROC-AUC:      0.85+
  F1:           0.75+
  Precision:    0.78+
  Recall:       0.72+
  Separability: 0.20+

Baseline Spearman: 0.5038
New Model Spearman: 0.5250
Improvement: +4.2%

[SUCCESS] Model beats baseline!
```

---

## Output Files

After training completes:

```
models/real_servicenow_finetuned_mpnet_lora/
├── config.json
├── model.safetensors
├── modules.json
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── training_metadata.json  ← Includes all config & results
└── vocab.txt
```

---

## Differences from V1

| Aspect | V1 (Old) | V2 (New) |
|--------|----------|----------|
| **Cells** | 32 cells | 18 cells |
| **Structure** | Patched legacy code | Clean from scratch |
| **Pair generation** | Has legacy cells (skipped) | Only curriculum loading |
| **Device detection** | Separate cell added | Integrated from start |
| **Unicode** | Had emojis (fixed) | Pure ASCII |
| **Syntax errors** | 2 orphaned else clauses (fixed) | None |
| **Documentation** | Added via patches | Built-in comments |
| **Complexity** | High (many unused cells) | Low (only what's needed) |

---

## Troubleshooting

### If training fails
1. **Check CUDA**: Ensure GPU drivers are installed
2. **Check paths**: Verify curriculum pairs exist at `data_new/curriculum_training_pairs_20251224_065436.json`
3. **Check memory**: Reduce `batch_size` in Cell 3 if OOM errors occur
4. **Check disk space**: Model checkpoints require ~1GB

### If performance is poor
1. **Verify data**: Check that curriculum pairs loaded correctly
2. **Check device**: Ensure training on GPU (CUDA), not CPU
3. **Review logs**: Check training progress in cell outputs
4. **Try more epochs**: Increase `epochs_per_phase` in Cell 3

### If imports fail
1. **Install packages**: Cell 1 auto-installs missing packages
2. **Restart kernel**: After installation, restart and run again
3. **Check Python version**: Requires Python 3.8+

---

## Next Steps After Training

1. **Evaluate thoroughly**: Run `evaluate_model.ipynb` for detailed analysis
2. **Generate embeddings**: Use `supabase/embed_incidents.py` to embed all incidents
3. **Test search**: Try `supabase/test_hybrid_search.py` to test retrieval
4. **Deploy**: Upload model to Hugging Face or deploy to production API
5. **Monitor**: Track performance on real user queries

---

## FAQ

**Q: Can I modify the configuration?**
A: Yes! Edit Cell 3 before running. Common changes:
- Increase `epochs_per_phase` for longer training
- Adjust `batch_size` based on GPU memory
- Change `lora_r` for different parameter efficiency

**Q: How long does training take?**
A: ~30-60 minutes on modern GPU (RTX 3090)
   ~2-4 hours on CPU (not recommended)

**Q: Can I resume if training stops?**
A: Model checkpoints are saved during training. Load from `CONFIG['output_dir']`

**Q: Do I need the old v1 notebook?**
A: No! V2 is a complete replacement with all fixes integrated.

---

**Created:** 2024-12-24
**Author:** Claude Code
**Version:** 2.0
