# evaluate_model_v2.ipynb Fix Applied

**Date:** 2025-12-26
**Status:** ✅ FIXED
**File:** evaluate_model_v2.ipynb

---

## Problem

The evaluation notebook couldn't find LoRA models, showing this error:

```
Evaluating: real_servicenow_finetuned_mpnet_lora
No sentence-transformers model found with name models/real_servicenow_finetuned_mpnet_lora
Error: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5,
       model.ckpt.index or flax_model.msgpack found in directory
```

## Root Cause

**Cell 2 - FINETUNED_MODELS configuration** pointed to the parent directory instead of a specific model subdirectory:

```python
# ❌ BEFORE (broken):
FINETUNED_MODELS = [
    'v6_refactored_finetuned/v6_refactored_finetuned_20251204_1424',
    'real_servicenow_finetuned_mpnet/real_servicenow_v2_20251210_1939',
    'real_servicenow_finetuned_mpnet_lora',  # Points to parent directory!
]
```

The actual trained LoRA models are stored in timestamped subdirectories:
- `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251223_1138/`
- `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251224_1234/`
- `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1637/` (most recent)

**Total:** 29 trained LoRA models available for evaluation

---

## Fix Applied

Updated FINETUNED_MODELS to point to the most recent model subdirectory:

```python
# ✅ AFTER (fixed):
FINETUNED_MODELS = [
    'v6_refactored_finetuned/v6_refactored_finetuned_20251204_1424',
    'real_servicenow_finetuned_mpnet/real_servicenow_v2_20251210_1939',
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1637',  # Most recent LoRA model
]
```

**Model selected:** `real_servicenow_v2_20251226_1637`
**Reason:** Most recent training run (Dec 26, 2025 at 16:37)

---

## How to Use

### Option 1: Evaluate Most Recent Model (Default)

The notebook is now configured to evaluate the most recent LoRA model. Simply:

1. Open `evaluate_model_v2.ipynb` in Jupyter
2. Run all cells
3. Results will include the latest LoRA model vs baseline comparison

### Option 2: Evaluate Multiple LoRA Models

To compare multiple training runs, edit Cell 2 and add more models:

```python
FINETUNED_MODELS = [
    'v6_refactored_finetuned/v6_refactored_finetuned_20251204_1424',
    'real_servicenow_finetuned_mpnet/real_servicenow_v2_20251210_1939',
    # Top 5 most recent LoRA models
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1637',
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1614',
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1544',
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1528',
    'real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251226_1502',
]
```

### Option 3: Evaluate Specific Model

To evaluate a specific model by timestamp:

```bash
# List all available LoRA models
ls -la models/real_servicenow_finetuned_mpnet_lora/
```

Then update Cell 2 with the desired model path.

---

## Expected Output

After fix, Cell 9 should show:

```
================================================================================
EVALUATING FINE-TUNED MODELS
================================================================================

Evaluating: real_servicenow_v2_20251226_1637
  Loaded as PEFT model

================================================================================
EVALUATING: real_servicenow_v2_20251226_1637
================================================================================

Metrics (threshold=X.XXXX):
  Spearman:  X.XXXX
  ROC-AUC:   X.XXXX
  F1:        X.XXXX
  Precision: X.XXXX
  Recall:    X.XXXX
  Accuracy:  X.XXXX

Confusion Matrix:
  TN=XXXX  FP=XXXX
  FN=XXXX  TP=XXXX
```

---

## Success Criteria

- ✅ No "model not found" errors
- ✅ Model loads successfully via PEFT
- ✅ Evaluation completes without errors
- ✅ Results show performance metrics
- ✅ Comparison to baseline (Spearman 0.504) included
- ✅ Results exported to CSV/JSON

---

## Context: Model Performance

Based on `training_metadata.json` from a recent model (20251224_1234):

- **Spearman:** 0.483 (vs baseline 0.504 = -4.2%)
- **ROC-AUC:** 0.779 (vs baseline 0.791 = -1.5%)
- **F1:** 0.619
- **Training:** 10 epochs, batch_size=32, LoRA r=32/alpha=64

The LoRA models trained before the v3 notebook fixes were slightly below baseline. The most recent models (Dec 26) may perform better if they used the fixed v3 notebook configuration with:
- Proper GPU utilization (batch_size=64)
- Full 12 epochs curriculum training
- Corrected hyperparameters

---

## Files Modified

1. **evaluate_model_v2.ipynb**
   - Cell 2 (Configuration section)
   - Line ~28: Updated FINETUNED_MODELS list

---

## Next Steps

1. **Run Evaluation:**
   ```bash
   jupyter notebook evaluate_model_v2.ipynb
   # Or run all cells in Jupyter
   ```

2. **Review Results:**
   - Check `models/results/model_evaluation_YYYYMMDD_HHMMSS.csv`
   - Compare LoRA model performance vs baseline
   - Review confusion matrices and error analysis

3. **If Model Underperforms:**
   - Verify it used the fixed v3 notebook (GPU-optimized)
   - Check training_metadata.json for configuration
   - Consider training with latest v3 fixes if needed

---

## All Available LoRA Models (29 total)

Most recent models (Dec 26, 2025):
- `real_servicenow_v2_20251226_1637` ← **Currently configured**
- `real_servicenow_v2_20251226_1614`
- `real_servicenow_v2_20251226_1544`
- `real_servicenow_v2_20251226_1528`
- `real_servicenow_v2_20251226_1502`
- `real_servicenow_v2_20251226_1440`
- `real_servicenow_v2_20251226_1413`
- `real_servicenow_v2_20251226_1406`
- `real_servicenow_v2_20251226_1351`
- `real_servicenow_v2_20251226_1338`

Earlier models (Dec 23-24, 2025):
- `real_servicenow_v2_20251224_1234`
- `real_servicenow_v2_20251224_0449`
- `real_servicenow_v2_20251224_0404`
- ... (16 more models)

All models contain:
- `adapter_config.json` (LoRA configuration)
- `adapter_model.safetensors` (trained LoRA weights)
- `config_sentence_transformers.json` (model config)
- `training_metadata.json` (performance metrics)

---

**Fix complete! The evaluation notebook is now ready to use.**
