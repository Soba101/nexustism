# 🚀 Start Training - Quick Guide

**ALL BUGS FIXED** - Ready to train immediately!

---

## Quick Start (3 Steps)

### 1. Verify Fixes Applied

```bash
python final_verification.py
```

**Expected output:**
```
[SUCCESS] All checks passed!
Cell 16 is ready to execute!
```

✅ If you see this, proceed to step 2.

---

### 2. Open Jupyter

```bash
jupyter notebook model_promax_mpnet_lorapeft_v3.ipynb
```

---

### 3. Run All Cells in Order

**Setup (5 minutes):**
- Cell 0-11: Imports, config, data loading

**Load Curriculum (1 minute):**
- **Cell 12:** Should output:
  ```
  Phase 1 (Easy): 5,000 pairs
  Phase 2 (Medium): 5,000 pairs
  Phase 3 (Hard): 5,000 pairs
  Total: 15,000
  ```

**Initialize Training (1 minute):**
- Cell 13-15: Model, LoRA, loss function

**TRAIN (2-4 hours):**
- **Cell 16:** Should immediately show:
  ```
  🚀 Starting Training (V2)...

  [CURRICULUM] Training in 3 phases (easy -> medium -> hard)

  ============================================================
  [PHASE 1] PHASE1: 4 epochs
     Training examples: 5,000
  ============================================================
     Batches per epoch: 313 (batch_size=16)
     Total steps this phase: 1,252

  [TRAINING] phase1...
  Epoch: 100%|████████████████| 4/4 [XX:XX<00:00]

  [OK] phase1 complete!

  [PHASE 2] PHASE2: 4 epochs
  ...

  [PHASE 3] PHASE3: 4 epochs
  ...

  ============================================================
  [SUCCESS] All curriculum phases complete!
  ============================================================
  ```

**Evaluate (5 minutes):**
- Cell 17-28: Metrics, plots, error analysis

---

## What to Expect

### Training Progress
- **Phase 1 (Easy):** 4 epochs, ~30-40 min
- **Phase 2 (Medium):** 4 epochs, ~30-40 min
- **Phase 3 (Hard):** 4 epochs, ~30-40 min
- **Total:** 12 epochs, 2-4 hours

### Performance Target
- **Baseline:** Spearman 0.504
- **Conservative:** Spearman 0.55-0.58 (+9-15%)
- **Target:** Spearman 0.60+ (+19%)

### Model Output
- Saved to: `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_YYYYMMDD_HHMM/`

---

## Bugs Fixed (This Session)

✅ **Bug 1:** Cell 16 silent skip (inverted guard condition)
✅ **Bug 2:** KeyError 'curriculum_phases'
✅ **Bug 3:** Epochs per phase mismatch (2 → 4)
✅ **Bug 4:** train_dataloader referenced before creation
✅ **Bug 5:** generate_training_pairs() called (doesn't exist)

**Total fixes:** 5 critical bugs + rewrote curriculum section

---

## If Something Goes Wrong

### Cell 12 fails - FileNotFoundError
```bash
# Check file exists (should be 22MB)
ls -lh data_new/curriculum_training_pairs_complete.json

# If missing, copy:
cp data_new/curriculum_training_pairs_20251224_065436.json data_new/curriculum_training_pairs_complete.json
```

### Cell 16 shows error
```bash
# Re-run all fixes
python fix_v3_training_cell.py
python complete_cell16_curriculum_fix.py
python fix_epochs_per_phase.py
python fix_cell16_final_issues.py

# Verify
python final_verification.py
```

### Out of Memory
In Cell 6, edit CONFIG:
```python
'batch_size': 8,  # Reduce from 16 (CUDA) or 8 (MPS/CPU)
```
Restart kernel, re-run all cells.

---

## After Training

1. **Check Cell 20** - ROC/PR curves, confusion matrix
2. **Compare to baseline:**
   - Your model: Spearman = ?
   - Baseline: Spearman = 0.504
   - Improvement: ? %
3. **Run Cell 24** - Adversarial diagnostic (check for category shortcuts)
4. **If Spearman ≥ 0.60:** Ready for production! 🎉

---

## Files Reference

### Main Notebook
- `model_promax_mpnet_lorapeft_v3.ipynb` - **RUN THIS**

### Verification
- `final_verification.py` - Run before training
- `verify_v3_fixes.py` - Verify all hyperparameters
- `test_v3_cell12.py` - Test curriculum loading

### Documentation
- `V3_FINAL_STATUS.md` - Complete status report
- `RUN_V3_TRAINING.md` - Detailed guide
- `START_TRAINING_NOW.md` - This file

---

## Summary

**Status:** ✅ All bugs fixed, ready to train

**Time:** 2-4 hours training + 5 min setup + 5 min eval = ~2.5-4.5 hours total

**Goal:** Beat baseline Spearman 0.504 by 10-20%

**Command:** `jupyter notebook model_promax_mpnet_lorapeft_v3.ipynb`

---

**Go train! 🚀**
