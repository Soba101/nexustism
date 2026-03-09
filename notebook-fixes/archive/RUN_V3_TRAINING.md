# How to Run V3 Training - Quick Guide

**Status:** ✅ Ready to train (bug fixed)
**File:** `model_promax_mpnet_lorapeft_v3.ipynb`

---

## Quick Start

```bash
jupyter notebook model_promax_mpnet_lorapeft_v3.ipynb
```

**Then in Jupyter:**
1. Run cells 0-15 (setup, configuration, data loading)
2. Run Cell 16 (training execution) - **This will take 2-4 hours**
3. Run cells 17-28 (evaluation and analysis)

---

## What Cell 16 Does

**Training Configuration:**
- Dataset: 15,000 curriculum pairs (3 phases × 5,000 pairs)
- Epochs: 12 total (4 per phase)
- Learning rate: 5e-5
- LoRA: rank=16, alpha=32
- Device: Auto-detected (CUDA/MPS/CPU)

**Phase 1 (Easy):** pos ≥ 0.52, neg ≤ 0.36
**Phase 2 (Medium):** pos 0.40-0.52, neg 0.36-0.45
**Phase 3 (Hard):** pos 0.30-0.40, neg 0.45-0.50

---

## Expected Output

When Cell 16 runs, you should see:

```
================================================================================
USING PRE-GENERATED CURRICULUM PAIRS
================================================================================

Loading curriculum pairs from: data_new/curriculum_training_pairs_complete.json

Phase 1 (Easy): 5,000 pairs
Phase 2 (Medium): 5,000 pairs
Phase 3 (Hard): 5,000 pairs

Total training examples: 15,000

[TRAINING] Phase 1: 4 epochs
Epoch: 100%|███████████| 1/1 [XX:XX<00:00]
...

[TRAINING] Phase 2: 4 epochs
...

[TRAINING] Phase 3: 4 epochs
...

[OK] Training complete!
```

**If you see NO output:** The fix may not have been applied. Run:
```bash
python fix_v3_training_cell.py
```

---

## After Training

**Model saved to:** `models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_TIMESTAMP/`

**Evaluation (Cells 18-28):**
- Cell 18: Score distribution analysis
- Cell 20: ROC/PR curves, confusion matrix
- Cell 22: Error analysis (false positives/negatives)
- Cell 24: Adversarial diagnostic (category leakage test)
- Cell 26: Save training metadata

---

## Success Criteria

**Minimum Success:**
- Spearman ≥ 0.55 (beat baseline 0.504 by 9%)
- ROC-AUC ≥ 0.80
- Training completes without OOM errors

**Good Success:**
- Spearman ≥ 0.58 (+15%)
- ROC-AUC ≥ 0.85

**Excellent Success:**
- Spearman ≥ 0.60 (+19%)
- ROC-AUC ≥ 0.90
- Adversarial diagnostic passes (ROC-AUC ≥ 0.70)

---

## Troubleshooting

**Problem:** Cell 16 produces no output
**Solution:** Run `python fix_v3_training_cell.py` then restart kernel

**Problem:** Out of memory (OOM)
**Solution:** Reduce batch_size in CONFIG (Cell 6):
- CUDA: Try batch_size=8 (default 16)
- MPS: Try batch_size=4 (default 8)
- CPU: Try batch_size=4 (default 8)

**Problem:** Training very slow
**Solution:** Check DEVICE in Cell 4 output - should be 'cuda' or 'mps', not 'cpu'

**Problem:** ImportError
**Solution:** Install requirements: `pip install -r requirements.txt`

---

## Verification Before Training

Run these checks:

```bash
# Check Cell 16 is fixed
python verify_v3_training_fix.py

# Check curriculum data exists
python test_v3_cell12.py

# Check all v3 fixes preserved
python verify_v3_fixes.py
```

All should show `[PASS]` or `[OK]`.

---

## Time Estimates

**Setup (Cells 0-15):** 2-5 minutes
**Training (Cell 16):** 2-4 hours (varies by GPU)
**Evaluation (Cells 17-28):** 5-10 minutes

**Total:** ~2.5-4.5 hours

---

## What's Different from Original

**v3 Improvements:**
- ✅ All imports consolidated (Cell 2)
- ✅ Sections logically ordered (1-13)
- ✅ Dead code removed (TF-IDF pair generation)
- ✅ Training cell fixed (no inverted guard)
- ✅ Hyperparameters correct (epochs=12, lr=5e-5)
- ✅ Curriculum data path correct

**Everything preserved:**
- ✅ All training functionality
- ✅ All evaluation code
- ✅ Curriculum learning (3 phases)
- ✅ LoRA fine-tuning with PEFT

---

## After Training Completes

1. **Check results** in Cell 20 (ROC/PR curves)
2. **Compare to baseline:**
   - Baseline: Spearman 0.504
   - Your model: Should be ≥ 0.55
3. **Run adversarial test** (Cell 24)
4. **Save metadata** (Cell 26)

If results look good:
- Model is ready for production testing
- Can integrate with Supabase vector search
- Can deploy to ServiceNow

---

**Ready to train! Open Jupyter and run Cell 16! 🚀**
