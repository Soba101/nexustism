# Notebook Comparison: V1 vs V2

## Quick Decision

**Use V2** - It's a clean rebuild with all fixes integrated from the start.

---

## Side-by-Side Comparison

| Feature | V1 (model_promax_mpnet_lorapeft.ipynb) | V2 (model_promax_mpnet_lorapeft-v2.ipynb) |
|---------|----------------------------------------|-------------------------------------------|
| **Total Cells** | 32 | 18 |
| **Code Cells** | 18 | 10 |
| **Markdown Cells** | 14 | 8 |
| **File Size** | ~250 KB | ~85 KB |
| **Created** | Original (patched 9 times) | 2024-12-24 (clean build) |
| **Syntax Errors** | 0 (after fixes) | 0 |
| **Unicode Issues** | None (sanitized) | None (never had) |
| **Legacy Code** | Yes (skipped via guards) | No |
| **Complexity** | High | Low |
| **Recommended** | ❌ No (use for reference) | ✅ **Yes** (use this) |

---

## Detailed Breakdown

### V1: model_promax_mpnet_lorapeft.ipynb

**History:**
- Original notebook
- Patched 9 times to fix issues
- Has 9 backup files (.ipynb.backup through .ipynb.backup9)

**Structure:**
```
32 cells total:
- Setup & imports (4 cells)
- Data loading (4 cells)
- Data splitting (3 cells)
- Pair loading (1 cell) ← CRITICAL
- Legacy pair generation (2 cells) ← SKIPPED
- Device detection (1 cell) ← CRITICAL
- Model setup (1 cell)
- Training (1 cell) ← CRITICAL
- Evaluation (6 cells)
- Borderline test (1 cell) ← SKIPPED
- Visualization (3 cells)
- Save (2 cells)
- Summary (2 cells)
```

**Pros:**
- ✅ All fixes applied and verified
- ✅ Works correctly
- ✅ Comprehensive evaluation
- ✅ Detailed visualization

**Cons:**
- ❌ Complex structure (32 cells)
- ❌ Contains legacy code (though skipped)
- ❌ Multiple patches applied
- ❌ Harder to understand flow
- ❌ Has 2 skipped cells (14, 29)

---

### V2: model_promax_mpnet_lorapeft-v2.ipynb

**History:**
- Built from scratch 2024-12-24
- All fixes integrated from the start
- No patches or modifications

**Structure:**
```
18 cells total:
- Title (1 cell)
- Setup & imports (2 cells)
- Config (1 cell)
- Logging (1 cell)
- Data loading (2 cells) ← Load curriculum + test
- Device detection (2 cells)
- Model setup (2 cells)
- Training (2 cells) ← Curriculum learning
- Evaluation (2 cells)
- Save (2 cells)
- Summary (1 cell)
```

**Pros:**
- ✅ Clean build from scratch
- ✅ Simple structure (18 cells)
- ✅ No legacy code
- ✅ All fixes integrated
- ✅ Easy to understand
- ✅ No skipped cells
- ✅ Comprehensive comments
- ✅ Pure ASCII (no Unicode)

**Cons:**
- ⚠️ Less detailed visualization (can be added if needed)
- ⚠️ Fewer evaluation metrics (but covers essentials)

---

## Functional Differences

### Both Notebooks Do:
✅ Load 15,000 curriculum pairs (3 phases)
✅ Load 1,000 test pairs
✅ Auto-detect device (CUDA/MPS/CPU)
✅ Initialize MPNet with LoRA
✅ Train with curriculum learning
✅ Evaluate on test set
✅ Save model and metadata

### V1 Also Has:
- Legacy on-the-fly pair generation (skipped)
- Borderline test evaluation (skipped)
- More extensive visualization
- Score distribution diagnostics
- Cross-validation threshold selection

### V2 Focuses On:
- Essential functionality only
- Clean execution flow
- Minimal complexity
- Core metrics (Spearman, ROC-AUC, F1)

---

## Which Should You Use?

### Use V2 if you want:
- ✅ **Clean, simple notebook** (recommended)
- ✅ Quick training with core metrics
- ✅ Easy to understand and modify
- ✅ No legacy code to worry about
- ✅ Fresh start

### Use V1 if you want:
- Detailed visualization and diagnostics
- More comprehensive evaluation metrics
- To reference the original implementation
- Additional analysis features

---

## Migration from V1 to V2

If you're currently using V1, switching to V2 is easy:

1. **No data changes needed** - Both use same curriculum pairs
2. **Same output** - Both save to `models/real_servicenow_finetuned_mpnet_lora/`
3. **Same config** - CONFIG dict is nearly identical
4. **Compatible** - Models are interchangeable

**To switch:**
```bash
# Simply open V2 instead of V1
jupyter notebook model_promax_mpnet_lorapeft-v2.ipynb

# Your data files remain the same
data_new/curriculum_training_pairs_20251224_065436.json  ← Used by both
data_new/fixed_test_pairs.json                           ← Used by both
```

---

## Recommendation

**Use model_promax_mpnet_lorapeft-v2.ipynb**

Reasons:
1. Clean build = less chance of hidden issues
2. Simpler = easier to debug
3. Smaller = faster to load and run
4. Modern = built with all lessons learned
5. Maintainable = easier to modify later

Keep V1 for reference or if you need the extra visualization features.

---

## Files Overview

```
Current directory:
├── model_promax_mpnet_lorapeft.ipynb           ← V1 (32 cells, patched)
├── model_promax_mpnet_lorapeft-v2.ipynb        ← V2 (18 cells, clean) ⭐ USE THIS
├── model_promax_mpnet_lorapeft.ipynb.backup    ← V1 backups (9 files)
├── ...
│
├── docs/
│   ├── NOTEBOOK_CELL_MAP.md                    ← V1 cell guide
│   ├── V2_NOTEBOOK_GUIDE.md                    ← V2 cell guide ⭐
│   ├── VERIFICATION_REPORT.md                  ← V1 verification
│   └── ...
│
└── data_new/
    ├── curriculum_training_pairs_20251224_065436.json  ← 15K pairs
    └── fixed_test_pairs.json                           ← 1K test pairs
```

---

**Recommendation:** Start using **model_promax_mpnet_lorapeft-v2.ipynb** for all new training runs.

**Documentation:** See [docs/V2_NOTEBOOK_GUIDE.md](docs/V2_NOTEBOOK_GUIDE.md) for complete guide.
