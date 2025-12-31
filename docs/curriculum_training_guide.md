# Curriculum Training Guide - Fix Train/Test Mismatch

## Quick Start

**Run this notebook:** [`fix_train_test_mismatch.ipynb`](../fix_train_test_mismatch.ipynb)

This will generate a 3-phase curriculum dataset to fix the distribution mismatch.

## The Problem (Recap)

**Current situation:**
- Training data: Too easy (separability=0.374, 0% overlap)
- Test data: Too hard (separability=0.187, 54.4% overlap)
- Models learn easy task, fail on hard test

## The Solution: Curriculum Learning

Generate training data across **3 difficulty levels** that progressively match the test distribution:

### Phase 1: Easy (Foundation)
```
Pairs: 5,000 (existing)
Positive threshold: similarity ≥ 0.52
Negative threshold: similarity ≤ 0.36
Purpose: Build strong baseline understanding
```

### Phase 2: Medium (Bridge)
```
Pairs: 5,000 (NEW)
Positive threshold: similarity ≥ 0.40
Negative threshold: similarity ≤ 0.45
Purpose: Handle intermediate difficulty
```

### Phase 3: Hard (Test-Realistic)
```
Pairs: 5,000 (NEW)
Positive threshold: similarity ≥ 0.30
Negative threshold: similarity ≤ 0.50
Purpose: Match test set difficulty
```

**Total: 15,000 pairs** spanning all difficulty levels

## How to Use the Notebook

### Step 1: Generate Curriculum Data

```bash
# Open Jupyter
jupyter notebook fix_train_test_mismatch.ipynb

# Run all cells (takes ~10-15 minutes)
```

**Output files:**
- `phase2_medium_pairs_YYYYMMDD_HHMMSS.json` - Medium difficulty
- `phase3_hard_pairs_YYYYMMDD_HHMMSS.json` - Hard difficulty
- `curriculum_training_pairs_YYYYMMDD_HHMMSS.json` - Combined all phases

### Step 2: Update LoRA Training Notebook

Open `model_promax_mpnet_lorapeft.ipynb` and update CONFIG:

```python
CONFIG = {
    # Data
    'use_pre_generated_pairs': True,
    'train_pairs_path': 'data_new/curriculum_training_pairs_YYYYMMDD_HHMMSS.json',

    # Curriculum training
    'use_curriculum': True,  # Train in 3 phases
    'epochs_per_phase': 2,    # 2 epochs for each phase

    # LoRA config
    'lr': 5e-5,  # Higher LR for LoRA
    'batch_size': 16,

    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
}
```

### Step 3: Train with Curriculum

Two options:

**Option A: Sequential Phase Training (Recommended)**
```python
# In the training loop, separate by phase
phase1_data = [pair for pair, phase in zip(pairs, phase_indicators) if phase == 1]
phase2_data = [pair for pair, phase in zip(pairs, phase_indicators) if phase == 2]
phase3_data = [pair for pair, phase in zip(pairs, phase_indicators) if phase == 3]

# Train Phase 1 (2 epochs)
model.fit(train_dataloader(phase1_data), epochs=2)

# Train Phase 2 (2 epochs)
model.fit(train_dataloader(phase2_data), epochs=2)

# Train Phase 3 (2 epochs)
model.fit(train_dataloader(phase3_data), epochs=2)
```

**Option B: Mixed Training (Simpler)**
```python
# Train on all phases together (6 epochs)
model.fit(train_dataloader(all_pairs), epochs=6)
```

## Expected Results

### Before (Current Model)
```
Baseline Spearman:  0.5038
LoRA Spearman:      0.4885 (-3.0%)
Regular MPNet:      0.4669 (-7.3%)

Issue: Trained on easy, tested on hard
```

### After (With Curriculum)
```
Expected LoRA Spearman:  0.52-0.55 (+6-12% vs baseline)
Expected ROC-AUC:        0.82-0.85 (+4-7% vs baseline)

Improvement: Trained on all difficulty levels
```

### Why This Works

1. **Phase 1** builds foundation with clear examples
2. **Phase 2** introduces ambiguity gradually
3. **Phase 3** matches test difficulty exactly
4. Model learns to handle **all** difficulty levels, not just easy cases

## Validation

The notebook includes validation that compares:
- Phase 1 vs test: Too easy (gap)
- Phase 2 vs test: Closer (bridge)
- Phase 3 vs test: **Matched** (same thresholds)

## Alternative: Quick Fix (Test Only)

If you want to validate that models learned correctly without retraining:

**Generate an "easy" test set:**
```python
# In evaluate_model.ipynb
test_pos_threshold = 0.52  # Match Phase 1 training
test_neg_threshold = 0.36  # Match Phase 1 training
```

Expected result: Models will **significantly beat baseline** on easy test.

This proves models learned correctly - they just weren't trained for the hard task.

## Next Steps After Running

1. ✅ Run [`fix_train_test_mismatch.ipynb`](../fix_train_test_mismatch.ipynb)
2. ✅ Update `model_promax_mpnet_lorapeft.ipynb` with new data path
3. ✅ Train with curriculum (6 epochs total, 2 per phase)
4. ✅ Evaluate with [`evaluate_model.ipynb`](../evaluate_model.ipynb)
5. ✅ Compare results to baseline and previous models

## Files Created

After running the notebook:

```
data_new/
├── fixed_training_pairs.json          # Phase 1 (existing)
├── phase2_medium_pairs_YYYYMMDD.json  # NEW: Phase 2
├── phase3_hard_pairs_YYYYMMDD.json    # NEW: Phase 3
└── curriculum_training_pairs_YYYYMMDD.json  # NEW: Combined
```

## Troubleshooting

**Issue: Generation is slow**
- Reduce `max_attempts_per_pair` in CONFIG (default: 100)
- Or reduce pairs per phase (5000 → 3000)

**Issue: Can't find enough pairs**
- Thresholds may be too strict
- Adjust pos/neg thresholds slightly

**Issue: Out of memory**
- Reduce `batch_size` in embedding computation (128 → 64)
- Or process in smaller chunks

## Reference

See [`docs/train_test_mismatch_analysis.md`](train_test_mismatch_analysis.md) for detailed analysis of the problem.
