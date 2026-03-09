# LoRA Adapter Diagnostic - Next Steps

**Date:** 2025-12-27
**Status:** ✅ Diagnostic script created, ready to run

---

## Summary

I've created a comprehensive diagnostic script ([diagnose_lora_adapter.py](diagnose_lora_adapter.py)) that will definitively identify why all fine-tuned LoRA models produce IDENTICAL metrics to baseline MPNet.

**Root Cause Hypothesis:** PEFT wrapper is applied at the wrong level in the model hierarchy during evaluation.

- **Training (CORRECT):** `model[0].auto_model = get_peft_model(...)`
- **Evaluation (WRONG):** `PeftModel.from_pretrained(SentenceTransformer, ...)`

---

## How to Run the Diagnostic

### Option 1: In Jupyter Notebook (Recommended)

Since your Python CLI environment doesn't have `peft` installed, run the diagnostic in Jupyter where all packages are available:

1. Open a new Jupyter notebook or use existing one
2. Copy and paste the code from [diagnose_lora_adapter.py](diagnose_lora_adapter.py)
3. Run the cells
4. Review the output

### Option 2: Fix Python Environment (Alternative)

```bash
pip install peft safetensors
python diagnose_lora_adapter.py
```

---

## What the Diagnostic Tests

### Test 1: Embedding Difference (Smoking Gun) ⭐ CRITICAL
- Encodes same text with baseline vs PEFT models
- Compares using **TWO** loading methods:
  - **Current** (evaluate_model_v2.ipynb): Wrapper at SentenceTransformer level
  - **Correct** (training code): Wrapper at transformer component level
- **Expected:** Current method produces IDENTICAL embeddings (adapter not applied)
- **Expected:** Correct method produces DIFFERENT embeddings (adapter working)

### Test 2: Weight Inspection
- Checks if `type(model[0].auto_model)` == PeftModel
- Verifies LoRA weights (lora_A, lora_B) exist and non-zero
- Counts trainable parameters (~1-5% of total)

### Test 3: Module Structure
- Inspects SentenceTransformer hierarchy
- Verifies where PEFT wrapper is attached
- Compares current vs correct loading methods

### Test 4: Forward Pass Comparison
- Compares `model.encode()` vs manual forward pass
- Determines if encode() bypasses PEFT wrapper

### Test 5: Adapter File Validation
- Verifies adapter_config.json and adapter_model.safetensors exist
- Validates configuration (LoRA rank, target modules, etc.)
- Checks if weights are non-zero

---

## Expected Output

### If Root Cause is Confirmed (90% confidence)

```
================================================================================
FINAL DIAGNOSIS
================================================================================

Summary of Test Results:
  Test 1 (Embedding Difference):
    Current method: ❌ FAIL (cosine similarity > 0.9999)
    Correct method: ✅ PASS (cosine similarity < 0.99)
  Test 2 (Weight Inspection): ✅ PASS
  Test 3 (Module Structure): ✅ PASS (correct method has PEFT wrapper)
  Test 4 (Forward Pass): ✅ PASS
  Test 5 (Adapter Files): ✅ PASS

--------------------------------------------------------------------------------
DIAGNOSIS:
--------------------------------------------------------------------------------

✅ ROOT CAUSE IDENTIFIED:
  The LoRA adapter IS NOT applied during inference with current method.
  The adapter DOES work with the correct loading method.

  Current (WRONG): PeftModel.from_pretrained(SentenceTransformer, ...)
  Correct (RIGHT): model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, ...)

RECOMMENDED FIX:
  Modify evaluate_model_v2.ipynb Cell 9, lines 1700-1703
  Change PEFT loading to apply at transformer component level, not wrapper level

EXPECTED RESULT:
  All 14 existing models will produce DIFFERENT metrics from baseline
  No retraining required - just fix evaluation loading!
```

---

## After Running Diagnostic

### If Root Cause Confirmed (Expected)

**IMMEDIATE ACTION:** Fix [evaluate_model_v2.ipynb](evaluate_model_v2.ipynb) Cell 9

**Current Code (WRONG):**
```python
# Lines 1700-1703
from peft import PeftModel
base_model = SentenceTransformer(BASELINE_MODEL, device=device)
model = PeftModel.from_pretrained(base_model, str(full_path))
print('  Loaded as PEFT model')
```

**Fixed Code (CORRECT):**
```python
# Lines 1700-1710
from peft import PeftModel
base_model = SentenceTransformer(BASELINE_MODEL, device=device)

# Apply PEFT to transformer component, not full wrapper
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,  # ← Apply to inner transformer
    str(full_path)
)

model = base_model  # Now properly wrapped
print('  Loaded as PEFT model (transformer component level)')
```

**Then:**
1. Re-run diagnostic → should show all tests PASS
2. Re-run [evaluate_model_v2.ipynb](evaluate_model_v2.ipynb) → metrics should DIFFER from baseline
3. Expected: Spearman 0.50 → 0.55-0.65 (+10-30%)

**TIME:** 30 minutes total (15 mins fix + 10 mins evaluation + 5 mins validation)

### If Adapter IS Working (Unlikely - 5% chance)

If Test 1 shows **both** methods produce different embeddings:

**Then the problem is NOT the loading method.**

**Possible causes:**
- Evaluation metrics calculation bug
- Threshold optimization issue
- Training didn't actually converge (contradicts adversarial diagnostic passing)

**Next steps:**
1. Check evaluation code for bugs
2. Manually compare embeddings for sample pairs
3. Consider hard negative mining (Phase 4 of plan)

### If Adapter Files Missing/Invalid (Very Unlikely - 1% chance)

If Test 5 fails:

**Then adapter files are corrupted or missing.**

**Next steps:**
1. Check training logs for save errors
2. Re-train latest model
3. Verify adapter files are saved correctly

---

## Fix Implementation Details

### Cell 9 in evaluate_model_v2.ipynb

**Location:** Around line 1700

**Find this block:**
```python
print(f'\nEvaluating: {model_name}')

try:
    # Try PEFT adapter first
    try:
        from peft import PeftModel
        base_model = SentenceTransformer(BASELINE_MODEL, device=device)
        model = PeftModel.from_pretrained(base_model, str(full_path))  # ← WRONG!
        print('  Loaded as PEFT model')
    except:
        # Fallback to standard
        model = SentenceTransformer(str(full_path), device=device)
        print('  Loaded as standard model')
```

**Replace with:**
```python
print(f'\nEvaluating: {model_name}')

try:
    # Try PEFT adapter first
    try:
        from peft import PeftModel
        base_model = SentenceTransformer(BASELINE_MODEL, device=device)

        # Apply PEFT to transformer component, not full wrapper
        base_model[0].auto_model = PeftModel.from_pretrained(
            base_model[0].auto_model,  # ← CORRECT: inner transformer
            str(full_path)
        )

        model = base_model
        print('  Loaded as PEFT model (transformer component level)')
    except:
        # Fallback to standard
        model = SentenceTransformer(str(full_path), device=device)
        print('  Loaded as standard model')
```

---

## Timeline

| Step | Task | Time |
|------|------|------|
| 1 | Run diagnostic in Jupyter | 5 min |
| 2 | Review results | 5 min |
| 3 | Fix evaluate_model_v2.ipynb Cell 9 | 10 min |
| 4 | Re-run diagnostic | 5 min |
| 5 | Re-run evaluation (all models) | 10 min |
| **Total** | | **35 min** |

---

## Success Criteria

### After Fix Applied

- ✅ Diagnostic Test 1 shows **BOTH** methods produce different embeddings
- ✅ Evaluation metrics **DIFFER** from baseline:
  - Spearman: 0.5038 → 0.55-0.65 (anything ≠ 0.5038 proves fix worked)
  - ROC-AUC: 0.7909 → different value
  - Confusion matrix: NOT identical to baseline
- ✅ All 14 models can be re-evaluated without retraining

### If Metrics Still Identical After Fix

**Then we proceed to Phase 3:** Merge adapters with `merge_and_unload()`

See [Plan](C:\Users\donov\.claude\plans\woolly-bubbling-cupcake.md) Phase 3 for details.

---

## Files Created

1. **[diagnose_lora_adapter.py](diagnose_lora_adapter.py)** ✅
   - 5 comprehensive diagnostic tests
   - ~400 lines
   - Ready to run in Jupyter

2. **[LORA_DIAGNOSTIC_NEXT_STEPS.md](LORA_DIAGNOSTIC_NEXT_STEPS.md)** (this file) ✅
   - Step-by-step guide
   - Expected outputs
   - Fix implementation

3. **[Plan](C:\Users\donov\.claude\plans\woolly-bubbling-cupcake.md)** ✅
   - Complete implementation plan
   - All 4 phases documented
   - Backup strategies

---

## Quick Start

### Fastest Path to Working Model (65 minutes)

1. **Run diagnostic** (5 min)
   - Open Jupyter
   - Copy/paste [diagnose_lora_adapter.py](diagnose_lora_adapter.py) code
   - Run all cells

2. **Confirm root cause** (1 min)
   - Look for "ROOT CAUSE IDENTIFIED" in output
   - Verify Test 1 shows current=FAIL, correct=PASS

3. **Fix evaluate_model_v2.ipynb** (15 min)
   - Open notebook
   - Find Cell 9, line ~1700
   - Replace PEFT loading code (see above)

4. **Re-run diagnostic** (5 min)
   - Should show all tests PASS
   - Both methods produce different embeddings

5. **Re-run evaluation** (10 min)
   - Run all cells in evaluate_model_v2.ipynb
   - Check latest results JSON

6. **Verify success** (2 min)
   - Spearman ≠ 0.5038
   - Confusion matrix differs from baseline
   - ✅ Fix worked!

**Total:** ~38 minutes to working solution

---

## Contact Points

**If diagnostic shows unexpected results:** Review output and check which tests failed

**If fix doesn't work:** Proceed to Phase 3 (merge adapters) in the plan

**If adapter files missing:** Re-train latest model with fixed training notebook

**Questions?** Review the comprehensive plan at [woolly-bubbling-cupcake.md](C:\Users\donov\.claude\plans\woolly-bubbling-cupcake.md)

---

**Status:** ✅ Ready to run diagnostic

**Next:** Run [diagnose_lora_adapter.py](diagnose_lora_adapter.py) in Jupyter environment
