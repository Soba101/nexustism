# Train/Test Distribution Mismatch Analysis

**Date:** December 24, 2024
**Finding:** Critical distribution mismatch explains why all fine-tuned models underperform baseline

## Executive Summary

**All fine-tuned models (LoRA and regular) underperform the baseline NOT because of training bugs, but because of a fundamental train/test distribution mismatch.**

- **Training data:** 10,000 easy, well-separated pairs (separability=0.374, 0% overlap)
- **Test data:** 1,000 hard, ambiguous pairs (separability=0.187, 54.4% overlap)
- **Result:** Models learn to solve an easy task, then fail on a much harder evaluation

## The Numbers

### Training Data (fixed_training_pairs.json)

```
Total pairs: 10,000
Positive/Negative: 50/50 split

Thresholds:
  Positive: similarity ≥ 0.52 (strict - only clear matches)
  Negative: similarity ≤ 0.36 (strict - only clear non-matches)

Quality Metrics:
  Baseline separability: 0.3742 (EXCELLENT)
  Baseline overlap: 0.0% (PERFECT - no ambiguity)
  Positive mean similarity: 0.6616
  Negative mean similarity: 0.2874
  Gap between pos/neg: 0.374 (huge separation)

Status: "EXCELLENT" - validation passed
```

### Test Data (fixed_test_pairs.json)

```
Total pairs: 1,000
Positive/Negative: 50/50 split

Thresholds:
  Positive: similarity ≥ 0.30 (loose - includes marginal cases)
  Negative: similarity ≤ 0.50 (loose - includes confusing pairs)

Quality Metrics:
  Baseline separability: 0.1865 (POOR - half of training)
  Baseline overlap: 54.4% (TERRIBLE - massive ambiguity)
  Positive mean score: 0.5412
  Negative mean score: 0.3547
  Gap between pos/neg: 0.187 (weak separation)

Status: Challenging - many edge cases
```

### The Mismatch

| Metric | Training | Test | Delta |
|--------|----------|------|-------|
| **Separability** | 0.3742 | 0.1865 | **-50%** |
| **Overlap** | 0.0% | 54.4% | **+54.4pp** |
| **Positive threshold** | 0.52 | 0.30 | -0.22 |
| **Negative threshold** | 0.36 | 0.50 | +0.14 |
| **Difficulty** | Easy | Hard | **2x harder** |

## Why This Matters

### What Happens During Training

1. Model sees 10,000 pairs with clear separation (pos: 0.66, neg: 0.29)
2. Model learns: "High similarity (>0.52) = similar, Low similarity (<0.36) = different"
3. Model achieves excellent performance on this distribution
4. **Model never sees the ambiguous 0.36-0.50 similarity range!**

### What Happens During Testing

1. Test includes many pairs in the 0.30-0.50 range (the "gray zone")
2. 54.4% of test pairs have overlapping score distributions
3. Model is forced to make decisions on cases it never trained on
4. **Model underperforms because it's solving a different, harder problem**

### Why Baseline Performs Better

The baseline (raw MPNet) was pre-trained on 1 billion+ diverse sentence pairs from the internet, including:
- Easy cases (cat vs refrigerator)
- Medium cases (cat vs dog)
- Hard cases (cat vs kitten)
- Edge cases (similar but different documents)

**The baseline has seen ALL difficulty levels, while fine-tuned models only saw easy cases.**

## Evaluation Results Explained

### Model Performance

| Model | Spearman | Separability | vs Baseline |
|-------|----------|--------------|-------------|
| **Baseline (raw MPNet)** | 0.5038 | 0.1865 | - |
| **LoRA fine-tuned** | 0.4885 | 0.1566 | -3.0% |
| **Regular MPNet fine-tuned** | 0.4669 | 0.1493 | -7.3% |

### Why Fine-Tuned Models Perform Worse

1. **Training on easy data** → Model learns simple decision boundary
2. **Testing on hard data** → Simple boundary fails on edge cases
3. **Reduced separability** → Model's scores are less discriminative than baseline
4. **Lower positive scores** → Model is too conservative (trained on high-threshold positives)

### Why LoRA Is Actually Doing Well

Despite the mismatch, **LoRA is the BEST fine-tuned model** (-3.0% vs baseline compared to -7.3% for regular training).

LoRA's lower learning capacity (rank-16 adapters) may actually HELP by:
- Preventing overfitting to the easy training distribution
- Retaining more of the baseline's diverse knowledge
- Being more robust to distribution shift

## Concrete Examples

### Training Data - Positive Pair (Clear Match)
```
Text 1: "Error occurred while processing the EDI transaction. Interface: AS2,
         Subsidiary: PIDSAP, API Name: ext-partners-ord..."

Text 2: "Error occurred while processing the EDI transaction. Interface:
         attendance-swipes, Subsidiary: PIDMY, API Name: pan..."
```
**Similarity: 0.66** → Obviously similar (same error pattern, similar structure)

### Training Data - Negative Pair (Clear Non-Match)
```
Text 1: "User's sales group changed on this month to C82 from CG1 and
         This user have eQuote document with CG1..."

Text 2: "Error occurred while processing the EDI transaction. Interface:
         attendance-swipes, Subsidiary: PIDMY..."
```
**Similarity: 0.29** → Obviously different (completely different issues)

### Test Data - Positive Pair (Subtle Match)
```
Text 1: "Error occurred while processing the EDI transaction. Please find
         the details below and attached is the file..."

Text 2: "Error occurred while processing the EDI transaction. Please find
         the details below and attached is the file..."
```
**Similarity: 0.54** → Similar but not as obvious (less context, more generic)

### Test Data - Negative Pair (Confusingly Similar)
```
Text 1: "PSV CS changed printer from HP LaserJet Pro MFP M225 (10.92.194.149)
         to ApeosPort-V C4476 (10.92.194.146) at Binh Duong warehouse..."

Text 2: "Report Name: PL RBD + BR (DIV) MTH, PL RBD + BR (DIV) YTD.
         Difference is showing for the YTD. please check and rectify..."
```
**Similarity: 0.35** → Different topics but similar IT/business language

## Solution Options

### Option 1: Align Test to Training (Quick Fix)

**Change test data thresholds to match training:**
- Positive: similarity ≥ 0.52
- Negative: similarity ≤ 0.36

**Expected result:**
- Models will significantly BEAT baseline
- Separability improves from 0.187 to ~0.35
- Overlap drops from 54.4% to near 0%

**Pros:**
- Immediate improvement in metrics
- Validates that models learned correctly

**Cons:**
- Only testing easy cases
- Doesn't reflect real-world difficulty
- Not solving the actual problem

### Option 2: Align Training to Test (Proper Fix)

**Regenerate training data with test-realistic thresholds:**
- Positive: similarity ≥ 0.30
- Negative: similarity ≤ 0.50
- Include challenging pairs in 0.30-0.50 range

**Expected result:**
- Models learn to handle edge cases
- Performance on test set improves
- Better real-world generalization

**Pros:**
- Solves the root cause
- Models learn the hard task
- Realistic evaluation

**Cons:**
- Requires regenerating 10,000 pairs
- Longer training time (harder task)
- May need more training data

### Option 3: Curriculum Learning with Hard Negatives (Best)

**Multi-phase training with progressive difficulty:**

**Phase 1 - Easy pairs (current):**
- Positive: ≥ 0.52, Negative: ≤ 0.36
- 5,000 pairs, 2 epochs
- Build strong foundation

**Phase 2 - Medium pairs:**
- Positive: ≥ 0.40, Negative: ≤ 0.45
- 5,000 pairs, 2 epochs
- Bridge the gap

**Phase 3 - Hard pairs (test-realistic):**
- Positive: ≥ 0.30, Negative: ≤ 0.50
- 5,000 pairs, 2 epochs
- Match evaluation difficulty

**Expected result:**
- Best of both worlds
- Models learn progressively
- Strong performance on both easy and hard cases

**Pros:**
- Curriculum learning proven effective
- Gradual adaptation to difficulty
- Likely to beat baseline

**Cons:**
- Most complex to implement
- Requires 15,000 total pairs
- Longer total training time

### Option 4: Just Fix the Test Set (Pragmatic)

**Keep training as-is, but be realistic about test metrics:**

**Acknowledge that:**
- Test set is artificially hard (54.4% overlap is extreme)
- Fine-tuned models ARE learning (LoRA -3% vs regular -7%)
- Baseline advantage comes from broader pre-training, not better task fit

**Report metrics separately:**
- Easy task (training-like): Fine-tuned models win
- Hard task (test-like): Baseline wins
- Real-world: Somewhere in between

**Pros:**
- No code changes needed
- Honest about trade-offs
- Helps set realistic expectations

**Cons:**
- Doesn't improve actual performance
- Still have train/test mismatch

## Recommendation

**Implement Option 3 (Curriculum Learning) with the following tweaks:**

1. **Keep current training as Phase 1** (already done, models learned it well)
2. **Add Phase 2: Hard negative mining**
   - Generate 5,000 pairs with similarity in 0.36-0.50 range
   - This is the "missing" difficulty level
   - 2 epochs of focused hard negative training
3. **Evaluate on both easy and hard test sets**
   - Easy test: pos≥0.52, neg≤0.36 (should beat baseline)
   - Hard test: pos≥0.30, neg≤0.50 (current test set)

This approach:
- Builds on existing work (don't throw away trained models)
- Adds missing difficulty level gradually
- Likely to close the gap with baseline or exceed it
- Realistic training time (~4 epochs total for Phase 2)

## Implementation Priority

For immediate improvement of LoRA model specifically:

1. **Short-term (learning rate fix):** Still worth trying 5e-5 LR - may help with hard cases
2. **Medium-term (hard negatives):** Generate Phase 2 data (similarity 0.36-0.50)
3. **Long-term (full curriculum):** Complete 3-phase training pipeline

The learning rate change alone won't fix the distribution mismatch, but it may help the model adapt better to edge cases it does see in training.

## Conclusion

**The LoRA model is NOT broken - it's a victim of distribution shift.**

The evaluation results show that:
- Training data is too easy (0% overlap, 0.374 separability)
- Test data is too hard (54.4% overlap, 0.187 separability)
- Models are penalized for learning what they were taught

To truly improve performance, we need to either:
1. Make test easier (quick validation)
2. Make training harder (proper fix)
3. Add curriculum learning (best approach)

The learning rate increase will help incrementally, but won't solve the fundamental mismatch.
