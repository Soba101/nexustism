# Training Optimization Guide for MPNet LoRA

**Current Results:** Spearman 0.4970 (baseline: 0.5038, -1.3%)
**Goal:** Beat baseline (>0.5038)

---

## Current Setup Analysis

### What You're Using Now (V2 Notebook)

```python
CONFIG = {
    'lr': 5e-5,                    # Learning rate
    'epochs_per_phase': 2,         # Too short!
    'max_seq_length': 256,         # Good
    'batch_size': 32,              # Good (auto-adjusted for CUDA)
    'lora_r': 16,                  # Standard
    'lora_alpha': 32,              # Standard
    'lora_dropout': 0.1,           # Standard
    'warmup_steps': 100,           # Low
}
```

### Issues Identified

1. **❌ Too Few Epochs Per Phase**
   - Current: 2 epochs per phase = 6 total
   - Problem: LoRA needs more iterations to converge
   - Baseline was pre-trained on billions of examples

2. **❌ Low Warmup Steps**
   - Current: 100 steps
   - With 5K pairs/phase and batch_size=32: ~156 steps/epoch
   - Warmup ends in first epoch, learning rate jumps too fast

3. **⚠️ Learning Rate May Be Too High**
   - Current: 5e-5
   - For LoRA, this might be aggressive
   - Could cause instability or poor convergence

4. **⚠️ Equal Time on Each Phase**
   - Phase 1 (easy): 2 epochs
   - Phase 2 (medium): 2 epochs
   - Phase 3 (hard): 2 epochs
   - **Problem:** Should spend MORE time on hard phase (what we're tested on!)

---

## Recommended Improvements

### Option 1: Conservative (Safest)

**Best for:** First try, minimize overfitting risk

```python
CONFIG = {
    # Training - MORE EPOCHS
    'epochs_per_phase': 4,         # 12 total (was 2, now 4)
    'lr': 3e-5,                    # Lower LR (was 5e-5)
    'warmup_steps': 500,           # More warmup (was 100)
    'max_seq_length': 256,         # Keep

    # LoRA - Same
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    # Progressive curriculum
    'progressive_epochs': {
        'phase1': 3,                # Easy: 3 epochs
        'phase2': 4,                # Medium: 4 epochs
        'phase3': 5,                # Hard: 5 epochs (most time!)
    }
}
```

**Expected improvement:** Spearman 0.51-0.52 (+2-3% vs baseline)

---

### Option 2: Aggressive (Higher Potential)

**Best for:** If conservative doesn't work

```python
CONFIG = {
    # Training - MUCH MORE EPOCHS
    'epochs_per_phase': 6,         # 18 total
    'lr': 2e-5,                    # Even lower (more stable)
    'warmup_steps': 800,           # Longer warmup
    'max_seq_length': 256,

    # LoRA - HIGHER RANK (more capacity)
    'lora_r': 32,                  # Was 16, now 32
    'lora_alpha': 64,              # Scale with rank
    'lora_dropout': 0.05,          # Less dropout

    # Progressive curriculum
    'progressive_epochs': {
        'phase1': 4,                # Easy: 4 epochs
        'phase2': 6,                # Medium: 6 epochs
        'phase3': 8,                # Hard: 8 epochs (most important!)
    }
}
```

**Expected improvement:** Spearman 0.52-0.54 (+3-7% vs baseline)
**Risk:** Possible overfitting, longer training time

---

### Option 3: Focus on Hard Phase (Targeted)

**Best for:** If you're time-constrained

```python
CONFIG = {
    # Training - UNEQUAL DISTRIBUTION
    'lr': 4e-5,                    # Moderate LR
    'warmup_steps': 300,
    'max_seq_length': 256,

    # LoRA - Standard
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    # Heavily weighted toward hard phase
    'progressive_epochs': {
        'phase1': 2,                # Easy: 2 epochs (quick)
        'phase2': 3,                # Medium: 3 epochs
        'phase3': 10,               # Hard: 10 epochs (dominate training!)
    }
}
```

**Rationale:** Test set difficulty matches Phase 3, so spend most time there
**Expected improvement:** Spearman 0.51-0.53 (+1-5% vs baseline)

---

## Additional Optimizations

### 1. **Loss Function Improvement**

Current: `CosineSimilarityLoss`
Better: `MultipleNegativesRankingLoss` or hybrid

```python
# In Cell 12, replace:
train_loss = losses.CosineSimilarityLoss(model)

# With:
from sentence_transformers import losses
train_loss = losses.MultipleNegativesRankingLoss(model)
```

**Why:** MNRL is better for ranking/similarity tasks, uses in-batch negatives

---

### 2. **Better Evaluation During Training**

Current: Uses 100 eval examples during training
Better: Use full eval set

```python
# In Cell 12, replace:
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples[:100],  # Only 100
    name='eval_subset'
)

# With:
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,  # All 1,000
    name='eval_full'
)
```

**Why:** Better visibility into actual test performance

---

### 3. **Add Weight Decay**

Prevent overfitting:

```python
# In model.fit() call, add:
optimizer_params={
    'lr': CONFIG['lr'],
    'weight_decay': 0.01  # NEW
}
```

---

### 4. **Dynamic Learning Rate Schedule**

Current: Constant LR after warmup
Better: Cosine decay

```python
# In model.fit() call, add:
scheduler='WarmupCosine'  # NEW parameter
```

---

## My Recommendation

**Start with Option 1 (Conservative) + Loss Function Change**

### Updated Cell 3 (Configuration):

```python
CONFIG = {
    # Model
    'model_name': 'sentence-transformers/all-mpnet-base-v2',

    # Training - IMPROVED
    'epochs': 12,                  # Total (was 6)
    'batch_size': 16,
    'lr': 3e-5,                    # Lower (was 5e-5)
    'warmup_steps': 500,           # Higher (was 100)
    'max_seq_length': 256,
    'seed': 42,

    # LoRA - Same
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    # Data
    'use_pre_generated_pairs': True,
    'train_pairs_path': 'data_new/curriculum_training_pairs_20251224_065436.json',
    'test_pairs_path': 'data_new/fixed_test_pairs.json',
    'use_curriculum': True,

    # Progressive curriculum - SPEND MORE TIME ON HARD
    'phase1_epochs': 3,            # Easy
    'phase2_epochs': 4,            # Medium
    'phase3_epochs': 5,            # Hard (most important!)

    # Output
    'output_dir': 'models/real_servicenow_finetuned_mpnet_lora',
    'save_best_model': True,
    'eval_steps': 500,
    'threshold_cv_folds': 5,
}
```

### Updated Cell 12 (Training):

```python
# Use MultipleNegativesRankingLoss instead of CosineSimilarity
train_loss = losses.MultipleNegativesRankingLoss(model)

# Full evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,  # Full set
    name='eval_full'
)

# Progressive epochs
phases = [
    ('Phase 1: Easy', CURRICULUM_PHASES['phase1'], CONFIG['phase1_epochs']),
    ('Phase 2: Medium', CURRICULUM_PHASES['phase2'], CONFIG['phase2_epochs']),
    ('Phase 3: Hard', CURRICULUM_PHASES['phase3'], CONFIG['phase3_epochs'])
]

for phase_num, (phase_name, phase_data, phase_epochs) in enumerate(phases, 1):
    # ... rest of training loop ...
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=phase_epochs,  # Use phase-specific epochs
        warmup_steps=CONFIG['warmup_steps'],
        evaluator=evaluator,
        evaluation_steps=CONFIG['eval_steps'],
        output_path=str(save_path),
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={
            'lr': CONFIG['lr'],
            'weight_decay': 0.01  # Add weight decay
        },
        scheduler='WarmupCosine'  # Add LR scheduler
    )
```

---

## Expected Results

With these changes:

| Metric | Current | Expected | Change |
|--------|---------|----------|--------|
| **Spearman** | 0.4970 | 0.51-0.52 | +2.6-4.6% |
| **ROC-AUC** | 0.7870 | 0.82-0.85 | +4-8% |
| **F1** | 0.6552 | 0.70-0.72 | +7-10% |
| **vs Baseline** | -1.3% | +1.2-3.2% | **BEATS** |

---

## If That Doesn't Work

Try in order:

1. **Option 2 (Aggressive)** - More epochs, higher LoRA rank
2. **Increase Phase 3 epochs to 10** - Spend even more time on hard cases
3. **Try different loss function**: `ContrastiveLoss` or `TripletLoss`
4. **Check data quality**: Verify curriculum pairs are actually progressive
5. **Try without LoRA**: Full fine-tuning (last resort, much slower)

---

## Training Time Estimates

- **Current (2 epochs/phase):** ~4.5 minutes (you saw this)
- **Option 1 (3-4-5 epochs):** ~13-15 minutes
- **Option 2 (4-6-8 epochs):** ~20-25 minutes
- **Option 3 (2-3-10 epochs):** ~17-20 minutes

On RTX 5090, all options are fast enough to try multiple times!

---

## Quick Wins to Try First

**Easiest changes with high impact:**

1. Change `epochs_per_phase` from 2 to 4 (double training time)
2. Change `lr` from 5e-5 to 3e-5 (more stable)
3. Change loss from `CosineSimilarityLoss` to `MultipleNegativesRankingLoss`
4. Increase `warmup_steps` from 100 to 500

**These 4 changes alone should get you above baseline!**

---

**Bottom line:** No, the current setup is NOT optimal. You're undertraining (only 6 epochs total) and using a suboptimal loss function. Try the recommended changes above! 🎯
