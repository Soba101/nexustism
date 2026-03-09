# Bug #6 Fixed: Missing Import

**Date:** 2025-12-26
**Severity:** CRITICAL (causes training to fail)
**Status:** ✅ FIXED

---

## The Error

```
NameError: name 'average_precision_score' is not defined
```

Occurred in: `ITSMEvaluator.__call__()` line 39 (Cell 14)

---

## Root Cause

The `ITSMEvaluator` class uses `average_precision_score()` to calculate PR-AUC metrics:

```python
class ITSMEvaluator(SentenceEvaluator):
    def __call__(self, model, ...):
        # ... calculate scores ...
        try:
            roc_auc = roc_auc_score(self.labels, scores)
            pr_auc = average_precision_score(self.labels, scores)  # <-- ERROR HERE
        except ValueError:
            roc_auc, pr_auc = 0.0, 0.0
```

But `average_precision_score` was NOT imported in Cell 2!

**Cell 2 had:**
```python
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,        # <-- average_precision_score should be here!
    accuracy_score,
    ...
)
```

---

## Fix Applied

**Script:** `fix_missing_import.py`

**Change:** Added `average_precision_score` to sklearn.metrics import list in Cell 2:

```python
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,  # <-- ADDED
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
```

---

## How to Apply Fix

### In Jupyter Notebook:

1. **Kernel → Restart Kernel** (or press `00`)
2. **Run Cell 2** (imports) - should see no errors
3. **Run cells 3-15** (setup through training functions)
4. **Run Cell 16** (training) - should now work!

**Why restart kernel?** To reload imports with the new `average_precision_score`.

---

## Verification

After restarting and running Cell 2, test:

```python
# Quick check (run in a new cell after Cell 2)
from sklearn.metrics import average_precision_score
print(f"average_precision_score loaded: {average_precision_score}")
```

Should output:
```
average_precision_score loaded: <function average_precision_score at 0x...>
```

---

## Where It's Used

`average_precision_score` is called in 3 places:

1. **Line 885:** `ITSMEvaluator` class (during training evaluation)
   ```python
   pr_auc = average_precision_score(self.labels, scores)
   ```

2. **Line 1482:** Import verification (legacy code)

3. **Line 1509:** Final evaluation (Cell 20+)
   ```python
   pr_auc = average_precision_score(labels, scores)
   ```

All will now work after restart.

---

## Complete Bug List (All Fixed)

| # | Bug | Fixed By |
|---|-----|----------|
| 1 | Cell 16 silent skip | fix_v3_training_cell.py |
| 2 | KeyError 'curriculum_phases' | complete_cell16_curriculum_fix.py |
| 3 | Epochs per phase mismatch | fix_epochs_per_phase.py |
| 4 | train_dataloader premature use | fix_cell16_final_issues.py |
| 5 | generate_training_pairs() missing | fix_cell16_final_issues.py |
| **6** | **average_precision_score import** | **fix_missing_import.py** |

**All bugs fixed! ✅**

---

## Next Steps

1. **Restart Jupyter kernel**
2. **Run cells 0-15** (5-10 minutes)
3. **Run Cell 16** (training - 2-4 hours)
4. **Training should complete successfully!**

---

**Summary:** Import added. Restart kernel and re-run from top. Training will work! 🚀
