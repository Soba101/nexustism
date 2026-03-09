# Restart Kernel and Re-run

## What Happened

Training started but hit a `NameError: name 'average_precision_score' is not defined` in the evaluator.

**Root cause:** Missing import in Cell 2.

**Fix applied:** Added `average_precision_score` to sklearn.metrics imports.

---

## How to Continue Training

### Option 1: Restart from Beginning (RECOMMENDED)

**In Jupyter:**

1. **Kernel → Restart Kernel** (or press `00` in command mode)
2. **Cell → Run All Above** (when on Cell 16)
3. Wait for training to complete

**Why restart:** Ensures all imports are loaded correctly and all variables are properly initialized.

**Time:** 5-10 minutes setup + 2-4 hours training

---

### Option 2: Quick Fix Without Restart

**In Jupyter, add a new cell BEFORE Cell 16:**

```python
# Quick fix: Import missing function
from sklearn.metrics import average_precision_score
print("[OK] Added missing import")
```

**Run that cell, then run Cell 16 again.**

**Risk:** May have other stale state issues. If training fails again, use Option 1.

---

## Verification

After restarting kernel and running Cell 2, verify the import:

**Add a temporary cell after Cell 2:**
```python
# Verify import
print("Checking imports...")
print(f"average_precision_score: {average_precision_score}")
print("[OK] All imports loaded!")
```

Should output:
```
Checking imports...
average_precision_score: <function average_precision_score at 0x...>
[OK] All imports loaded!
```

---

## What Was Fixed

**Cell 2 sklearn.metrics imports (line ~90-99):**

```python
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,  # <-- ADDED THIS
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
```

---

## Training Should Now Work

After restart + re-run, Cell 16 should execute successfully:

```
🚀 Starting Training (V2)...

[CURRICULUM] Training in 3 phases (easy -> medium -> hard)

[PHASE 1] PHASE1: 4 epochs
   Training examples: 5,000
   Batches per epoch: 313

[TRAINING] phase1...
Epoch 1/4: 100%|████████| 313/313 [XX:XX<00:00]
...
Evaluation: Spearman: 0.XXX, ROC-AUC: 0.XXX  <-- Should not error here anymore
...

[OK] phase1 complete!
[PHASE 2] PHASE2: 4 epochs
...
```

---

## Summary

**Issue:** Missing import `average_precision_score`

**Fix:** Added to Cell 2 imports

**Action needed:**
1. Restart kernel
2. Run cells 0-15 again (5-10 min)
3. Run Cell 16 (training - 2-4 hours)

**File updated:** `model_promax_mpnet_lorapeft_v3.ipynb` (Cell 2)

---

**Restart kernel and try again! Should work now. 🚀**
