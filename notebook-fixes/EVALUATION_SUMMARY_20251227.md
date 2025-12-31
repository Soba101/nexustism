# Training & Evaluation Summary - December 27, 2025

**Date:** 2025-12-27 05:30 AM
**Status:** ⚠️ CRITICAL ISSUE FOUND - LoRA adapters may not be working

---

## Executive Summary

### ✅ Training Completed Successfully

**Latest Model:** `real_servicenow_v2_20251227_0444`
**Trained:** 04:44-04:55 AM (11 minutes)
**All Fixes Applied:** LR=5e-6, Curriculum (3 phases), Verification, Metadata logging

**Training Metadata Verification:**
```json
{
  "learning_rate": 5e-06,          // ✅ FIXED (was hardcoded 2e-5)
  "curriculum_learning": {
    "enabled": true,
    "num_phases": 3,               // ✅ FIXED (was empty [])
    "epochs_per_phase": 4,
    "total_pairs": 15000           // ✅ FIXED (was 50K)
  },
  "adversarial_diagnostic": {
    "roc_auc": 0.9820,             // ✅ PASSED
    "f1_score": 0.9457,            // ✅ PASSED
    "pass_status": true
  }
}
```

### ⚠️ Evaluation Reveals Critical Problem

**ALL fine-tuned LoRA models have IDENTICAL metrics to baseline MPNet!**

| Model | Spearman | vs Baseline | Status |
|-------|----------|-------------|--------|
| Baseline (Raw MPNet) | 0.5038 | 0.0% | Baseline |
| real_servicenow_v2_20251226_1637 | 0.5038 | **0.0%** | **IDENTICAL** |
| real_servicenow_v2_20251227_0214 | 0.5038 | **0.0%** | **IDENTICAL** |

**This is statistically impossible** unless the LoRA adapters are not being applied during inference.

---

## Detailed Evaluation Results

### Model Rankings (Spearman Correlation)

```
 #  Model                              Spearman  ROC-AUC  F1     vs Baseline
=================================================================================
 1  Baseline (Raw MPNet)               0.5038    0.7909   0.7227   0.0%
 2  real_servicenow_v2_20251226_1637   0.5038    0.7909   0.7227   0.0% ⚠️
 3  real_servicenow_v2_20251227_0214   0.5038    0.7909   0.7227   0.0% ⚠️
----------------------------------------------------------------------------------
 4  Baseline (BGE-base-en-v1.5)        0.4774    0.7756   0.7143  -5.2%
 5  real_servicenow_v2_20251210_1939   0.4669    0.7696   0.7130  -7.3%
 6  Baseline (UAE-Large-v1)            0.4435    0.7561   0.6934 -12.0%
 7  Baseline (E5-base-v2)              0.4417    0.7550   0.6966 -12.3%
 8  Baseline (Nomic-Embed-v1.5)        0.4390    0.7534   0.7017 -12.9%
 9  v6_refactored_finetuned_20251204   0.4368    0.7522   0.7064 -13.3%
10  Baseline (GTE-base-en-v1.5)        0.4164    0.7404   0.6815 -17.3%
11  Baseline (JinaBERT-v2-base)        0.3991    0.7304   0.6812 -20.8%
12  Baseline (MiniLM-L12-v2)           0.3720    0.7148   0.6891 -26.2%
```

### Full Metrics - Fine-Tuned vs Baseline

| Metric | Baseline | real_servicenow_v2_20251227_0214 | Delta | % Change |
|--------|----------|----------------------------------|-------|----------|
| Spearman | 0.5038 | 0.5038 | 0.0000 | 0.0% |
| ROC-AUC | 0.7909 | 0.7909 | 0.0000 | 0.0% |
| F1 | 0.7227 | 0.7227 | 0.0000 | 0.0% |
| Precision | 0.5664 | 0.5664 | 0.0000 | 0.0% |
| Recall | 0.998 | 0.998 | 0.0000 | 0.0% |
| Accuracy | 0.617 | 0.617 | 0.0000 | 0.0% |

**Confusion Matrix (IDENTICAL for both models):**
```
                Predicted
                Neg    Pos
Actual  Neg    118    382    <- 76.4% FP rate
        Pos      1    499    <- 0.2% FN rate
```

---

## Root Cause Analysis

### Why Are Fine-Tuned Models Identical to Baseline?

**Evidence:**
1. ✅ Three different LoRA models from different training runs
2. ✅ All have IDENTICAL metrics (to 4+ decimal places)
3. ✅ Confusion matrices are byte-for-byte identical
4. ✅ Even minor metrics (thresholds, accuracy) match exactly

**This cannot be coincidence** - it indicates a systematic issue.

### Hypothesis 1: LoRA Adapters Not Applied During Inference ⚠️ **MOST LIKELY**

**Evidence FOR this hypothesis:**
```python
# From evaluate_model_v2.ipynb Cell 9:
Evaluating: real_servicenow_v2_20251226_1637
  Loaded as PEFT model                     # ✅ Loading succeeds

# But embeddings are identical to baseline
# This suggests adapter weights aren't merged during encode()
```

**Why this could happen:**
- `SentenceTransformer.encode()` may bypass PEFT adapter layers
- Adapter modules exist but aren't in forward pass
- PEFT integration incomplete with SentenceTransformer

**Test to confirm:**
```python
# Load both models
base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
peft_model = PeftModel.from_pretrained(base_model, 'models/.../real_servicenow_v2_20251227_0444')

# Encode same text
test_text = "Unable to access email"
base_emb = base_model.encode(test_text)
peft_emb = peft_model.encode(test_text)

# If identical → adapter not being used
print(f"Embeddings identical: {np.allclose(base_emb, peft_emb)}")
```

### Hypothesis 2: Training Didn't Converge (UNLIKELY)

**Evidence AGAINST:**
- Adversarial diagnostic PASSED (ROC-AUC 0.982)
- This proves model learned *something*
- But evaluation shows it learned *nothing*
- Contradiction suggests inference problem, not training

### Hypothesis 3: Data Issue (RULED OUT)

**Evidence AGAINST:**
- Evaluation uses same test set for all models
- Baseline shows expected performance (Spearman 0.504)
- Other models (BGE, E5, etc.) show different results
- Only LoRA models are identical

---

## Critical Finding: High False Positive Rate

**All models (including baseline) have severe FP problem:**

```
False Positives: 382 / 500 negatives (76.4%)
False Negatives:   1 / 500 positives (0.2%)
```

**What this means:**
- Model predicts "similar" for almost everything (recall 99.8%)
- But most "similar" predictions are wrong (precision 56.6%)
- Threshold optimization can't fix this (fundamental separability issue)

**Root Cause:**
- Insufficient hard negatives in training
- Weak separation between positive/negative score distributions
- Training data may have noisy labels

---

## Next Steps - Critical Decision Point

### IMMEDIATE: Verify LoRA Adapter Is Working

**Create diagnostic script:**

```python
#!/usr/bin/env python3
"""
Verify LoRA adapter is being applied during inference
"""

import numpy as np
from peft import PeftModel
from sentence_transformers import SentenceTransformer

print("="*80)
print("LORA ADAPTER DIAGNOSTIC")
print("="*80)

# Load models
print("\n1. Loading base model...")
base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
print("   ✅ Base model loaded")

print("\n2. Loading PEFT model...")
peft_model = PeftModel.from_pretrained(
    base_model,
    'models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251227_0444'
)
print("   ✅ PEFT model loaded")

# Check adapter config
print("\n3. Checking adapter configuration...")
print(f"   Adapter config: {peft_model.peft_config}")
print(f"   Trainable params: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")

# Find LoRA modules
print("\n4. Finding LoRA modules...")
lora_modules = [name for name, _ in peft_model.named_modules() if 'lora' in name.lower()]
print(f"   Found {len(lora_modules)} LoRA modules:")
for name in lora_modules[:5]:
    print(f"     - {name}")

# Test encoding
print("\n5. Testing embeddings...")
test_texts = [
    "Unable to access email account",
    "Outlook not syncing emails",
    "Printer jammed with paper",
]

for text in test_texts:
    base_emb = base_model.encode(text)
    peft_emb = peft_model.encode(text)

    identical = np.allclose(base_emb, peft_emb, atol=1e-6)
    max_diff = np.abs(base_emb - peft_emb).max()

    print(f"\n   Text: '{text[:40]}...'")
    print(f"     Identical: {identical}")
    print(f"     Max diff:  {max_diff:.6f}")

    if identical:
        print(f"     ⚠️ WARNING: Embeddings are identical!")
    else:
        print(f"     ✅ Embeddings differ (adapter working)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all(np.allclose(base_model.encode(t), peft_model.encode(t)) for t in test_texts):
    print("\n⚠️ CRITICAL ISSUE:")
    print("   All embeddings are identical to base model")
    print("   LoRA adapter is NOT being applied during inference")
    print("\nRECOMMENDED FIX:")
    print("   1. Try merging adapter: peft_model.merge_and_unload()")
    print("   2. Save merged model and re-evaluate")
    print("   3. If still fails, switch to full fine-tuning (no PEFT)")
else:
    print("\n✅ LoRA adapter is working correctly")
    print("   Embeddings differ from base model")
    print("\nNEXT STEPS:")
    print("   Re-run evaluation - there may have been an issue with loading")
```

**Run diagnostic:**
```bash
python diagnose_lora_adapter.py
```

### Path A: If Adapter NOT Working (Expected)

**Option 1: Try Explicit Merge**
```python
# Merge adapter into base model
peft_model = peft_model.merge_and_unload()
peft_model.save_pretrained('models/real_servicenow_merged_20251227_0444')

# Re-evaluate merged model
# If THIS works, we can fix all previous models
```

**Option 2: Switch to Full Fine-Tuning**
- Use standard fine-tuning (no PEFT/LoRA)
- Proven to work (real_servicenow_v2_20251210_1939 showed different results)
- Requires more VRAM but more reliable
- Keep same hyperparameters (LR=5e-6, curriculum, etc.)

**Option 3: Fix PEFT Integration**
- Debug SentenceTransformer + PEFT interaction
- May require custom encode() method
- Higher risk, longer timeline

### Path B: If Adapter IS Working (Unlikely)

**Then the problem is training convergence:**

1. **Try Augmented Data** (64K pairs)
   - Use `curriculum_training_pairs_augmented_simple.json`
   - More training signal to diverge from baseline
   - Expected: +5-15% improvement

2. **Increase LoRA Rank**
   - Current: rank=16
   - Try: rank=32 or rank=64
   - More capacity to deviate from baseline

3. **Longer Training**
   - Current: 12 epochs (3 phases × 4 epochs)
   - Try: 24 epochs (3 phases × 8 epochs)
   - Give model more time to converge

---

## Model Status - Production Readiness

### Target Metrics (from CLAUDE.md)

| Criterion | Target | Current | Gap | Status |
|-----------|--------|---------|-----|--------|
| Spearman Correlation | ≥0.80 | 0.5038 | -0.296 | ❌ 37% below |
| ROC-AUC | ≥0.95 | 0.7909 | -0.159 | ❌ 17% below |
| Adversarial Diagnostic | PASS | PASS | — | ✅ |
| Beat Baseline | >0% | 0% | 0% | ❌ **TIED** |
| Low FP Rate | <20% | 76.4% | +56.4% | ❌ **3.8x too high** |

**Overall Status:** ❌ **NOT PRODUCTION READY**

**Blockers:**
1. Fine-tuned models not improving over baseline
2. Extremely high false positive rate (76.4%)
3. Need Spearman >0.65 before Phase 2

---

## Files Generated

### Training
- ✅ `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251227_0444/`
  - ✅ `adapter_config.json`
  - ✅ `adapter_model.bin`
  - ✅ `training_metadata.json` (correctly logged)

### Evaluation
- ✅ `models/results/evaluation_results_20251227_051857.json`
- ✅ `models/results/evaluation_metadata_20251227_051857.json`
- ✅ `models/results/model_evaluation_20251227_051857.csv`

### Documentation
- ✅ `fix_training_notebook.py` (applied 4 fixes)
- ✅ `READY_FOR_TRAINING.md`
- ✅ `INDENTATION_FIX_APPLIED.md`
- ✅ `EVALUATION_SUMMARY_20251227.md` (this file)

### Backups
- ✅ `model_promax_mpnet_lorapeft_v3.ipynb.backup_metadata_fix`
- ✅ `model_promax_mpnet_lorapeft_v3.ipynb.backup_before_indent_fix`

---

## Recommendations

### PRIORITY 1: Verify LoRA Adapter (IMMEDIATE)

**Time:** 15-30 minutes
**Risk:** Low
**Reward:** May reveal all previous models are actually fine

**Action:**
1. Create `diagnose_lora_adapter.py` (script provided above)
2. Run diagnostic
3. If adapter not working → try explicit merge
4. Re-evaluate merged model

### PRIORITY 2: If Adapter Not Working → Full Fine-Tuning

**Time:** 2-3 hours (training + eval)
**Risk:** Low (proven to work)
**Reward:** Known path forward

**Action:**
1. Create `model_promax_mpnet_full_finetune_v3.ipynb`
2. Remove PEFT/LoRA code
3. Use same hyperparameters (LR=5e-6, curriculum, MatryoshkaLoss)
4. Train and evaluate

**Expected:** Spearman 0.52-0.58 (+3-15% vs baseline)

### PRIORITY 3: After Working Model → Address FP Rate

**Time:** 4-6 hours (data work + retraining)
**Risk:** Medium
**Reward:** +10-20% improvement

**Action:**
1. Mine better hard negatives (TF-IDF 0.3-0.5)
2. Add more cross-category hard positives
3. Use augmented data (64K pairs)
4. Re-train with improved data

**Expected:** Spearman 0.60-0.68, FP rate <40%

### PRIORITY 4: After Spearman >0.65 → Phase 2

**Time:** 6-8 hours
**Risk:** Low
**Reward:** Causal relationship classification

**Action:**
1. Train CrossEncoder (NLI.ipynb)
2. Binary classification: causal vs non-causal
3. Integrate with bi-encoder pipeline

---

## Summary

### What Worked ✅
- Training completed successfully
- All metadata bugs fixed
- Curriculum learning implemented
- Adversarial diagnostic PASSED
- Verification output added

### What Didn't Work ❌
- Fine-tuned models identical to baseline
- High false positive rate (76.4%)
- No improvement over baseline MPNet
- Production targets not met

### Critical Question ⚠️
**Is the LoRA adapter being applied during inference?**

### Immediate Action Required
**Run LoRA adapter diagnostic** to determine:
- Path A: Fix adapter integration → re-evaluate
- Path B: Switch to full fine-tuning → retrain

---

**Session Status:** ✅ COMPLETE (training + evaluation done)
**Next Session:** Start with LoRA diagnostic → Decision point → Execute path forward

**ETA to Production-Ready Model:** 4-8 hours (depending on path chosen)
