# Notebook Cell Map: model_promax_mpnet_lorapeft.ipynb

Complete guide to all 32 cells in the LoRA training notebook.

## Quick Reference

**Critical cells that must run in order:**
1. **Cell 11** - Loads training & test pairs (defines `train_examples`, `eval_examples`)
2. **Cell 16** - Device detection (defines `DEVICE`)
3. **Cell 18** - Training (defines `best_model`)
4. **Cells 20-23** - Evaluation (requires `best_model`)

**Cells that are SKIPPED in curriculum mode:**
- Cell 14: On-the-fly pair generation (legacy)
- Cell 29: Borderline test evaluation (legacy)

---

## Cell-by-Cell Breakdown

### Section 1: Setup & Configuration (Cells 0-3)

**Cell 0** - `[MARKDOWN]` Setup: Environment variables & package installation
- Not shown in IDE (metadata cell)

**Cell 1** - `[MARKDOWN]` Imports
- Section header

**Cell 2** - `[MARKDOWN]` Project Overview
- Description of the notebook's purpose

**Cell 3** - `[CODE]` **Config: Training parameters**
- Defines `CONFIG` dictionary
- Key settings:
  - `model_name`: 'sentence-transformers/all-mpnet-base-v2'
  - `lr`: 5e-5 (increased from 2e-5)
  - `max_seq_length`: 256
  - `use_pre_generated_pairs`: True
  - `use_curriculum`: True
  - `train_pairs_path`: 'data_new/curriculum_training_pairs_20251224_065436.json'
- **Variables defined:** `CONFIG`

---

### Section 2: Data Loading (Cells 4-8)

**Cell 4** - `[MARKDOWN]` Data Loading Section
- Section header

**Cell 5** - `[CODE]` **Logging: Setup logging utilities**
- Defines `log()` function for printing with timestamps
- **Variables defined:** `log()`, `logging_buffer`

**Cell 6** - `[CODE]` **Data Loading: Functions to load ServiceNow incidents**
- Defines `load_incidents()` function
- Handles JSON/CSV loading
- **Variables defined:** `load_incidents()`

**Cell 7** - `[MARKDOWN]` Data Preprocessing
- Section header

**Cell 8** - `[CODE]` **Data: Load incident data from JSON**
- Calls `load_incidents(CONFIG['source_data'])`
- Creates combined text field for training
- **Variables defined:** `df_incidents`

---

### Section 3: Data Splitting (Cells 9-11)

**Cell 9** - `[MARKDOWN]` Data Splitting
- Section header

**Cell 10** - `[CODE]` **Data Split: Split into train/eval/holdout sets**
- Defines `split_data()` function
- Uses stratified splitting by category
- Splits: 76.5% train, 13.5% eval, 10% holdout
- **Variables defined:** `train_df`, `eval_df`, `holdout_df`

**Cell 11** - `[CODE]` ⚠️ **CRITICAL: Load curriculum training pairs & test pairs**
- Loads pre-generated curriculum pairs (15,000 pairs)
- Loads test pairs from `fixed_test_pairs.json` (1,000 pairs)
- Separates into 3 curriculum phases
- **Variables defined:**
  - `train_examples` (15,000 InputExample objects)
  - `eval_examples` (1,000 test pairs)
  - `holdout_examples` (same as eval)
  - `borderline_examples` (empty list)
  - `CURRICULUM_PHASES` (dict with phase1/phase2/phase3)
  - `SKIP_PAIR_GENERATION` (True)

---

### Section 4: Pair Generation - LEGACY (Cells 12-14)

**Cell 12** - `[MARKDOWN]` Pair Generation
- Section header

**Cell 13** - `[CODE]` **Pair Generation: TF-IDF classes (legacy, SKIPPED)**
- Defines `TFIDFSimilarityCalculator` class
- Defines `generate_training_pairs()` function
- Only used in legacy mode

**Cell 14** - `[CODE]` **Pair Generation: Generate pairs on-the-fly (legacy, SKIPPED)**
- Builds TF-IDF matrices and generates pairs
- **SKIPPED** when `use_pre_generated_pairs = True`
- Would define: `train_examples`, `eval_examples`, `holdout_examples`, `borderline_examples`

---

### Section 5: Model Training (Cells 15-18)

**Cell 15** - `[MARKDOWN]` Model Training
- Section header

**Cell 16** - `[CODE]` ⚠️ **CRITICAL: Device detection**
- Auto-detects CUDA/MPS/CPU
- **Variables defined:** `DEVICE` ('cuda', 'mps', or 'cpu')

**Cell 17** - `[CODE]` **Model Setup: LoRA initialization & loss functions**
- Defines `init_model_with_lora()` function
- Defines `HybridLoss` class (MNRL + Cosine)
- Configures LoRA adapters (rank=16, alpha=32)

**Cell 18** - `[CODE]` ⚠️ **CRITICAL: Training loop**
- Initializes model: `model = init_model_with_lora(CONFIG, DEVICE)`
- Trains with curriculum learning (3 phases, 2 epochs each)
- Saves checkpoints to `models/real_servicenow_finetuned_mpnet_lora/`
- Loads best model after training
- **Variables defined:** `best_model` (SentenceTransformer)
- **Dependencies:** Requires `DEVICE`, `train_examples`, `eval_examples`

---

### Section 6: Evaluation (Cells 19-23)

**Cell 19** - `[MARKDOWN]` Evaluation
- Section header

**Cell 20** - `[MARKDOWN]` Score Distribution Diagnostic
- Subsection header

**Cell 21** - `[CODE]` **Evaluation: Score distribution diagnostic**
- Computes similarity scores on sample pairs
- Analyzes separation between positive/negative pairs
- **Dependencies:** Requires `best_model`, `eval_examples`

**Cell 22** - `[CODE]` **Evaluation: Cross-validation threshold function**
- Defines `get_cv_threshold()` for optimal threshold selection
- Uses k-fold cross-validation

**Cell 23** - `[CODE]` **Evaluation: Comprehensive evaluation function**
- Defines `comprehensive_eval()` function
- Computes: Spearman, ROC-AUC, F1, Precision, Recall, Separability
- **Dependencies:** Requires `best_model`, `eval_examples`, `holdout_examples`

**Cell 24** - `[CODE]` **Evaluation: Run final evaluation on all test sets**
- Calls `comprehensive_eval()` on eval and holdout sets
- Prints final metrics
- **Dependencies:** Requires `best_model`, `eval_examples`, `holdout_examples`

---

### Section 7: Visualization (Cells 24-25)

**Cell 24** - `[MARKDOWN]` Visualization
- Section header

**Cell 25** - `[CODE]` **Visualization: Plot training metrics**
- Creates plots for score distributions
- Visualizes model performance

---

### Section 8: Save Model (Cells 26-27)

**Cell 26** - `[MARKDOWN]` Save Model
- Section header

**Cell 27** - `[CODE]` **Save: Save trained model & metadata**
- Saves model to disk
- Saves training metadata (config, metrics, timestamps)
- **Dependencies:** Requires `best_model`

---

### Section 9: Borderline Test - LEGACY (Cells 28-29)

**Cell 28** - `[MARKDOWN]` Borderline Test
- Section header

**Cell 29** - `[CODE]` **Evaluation: Borderline test (SKIPPED in curriculum mode)**
- Tests model on borderline pairs (hard negatives)
- **SKIPPED** when using curriculum training (borderline_examples is empty)

---

### Section 10: Summary (Cells 30-31)

**Cell 30** - `[MARKDOWN]` Summary
- Section header

**Cell 31** - `[CODE]` **Summary: Print final results & next steps**
- Prints summary of training results
- Shows next steps for deployment

---

## Variable Dependencies

| Variable | Defined In | Required By |
|----------|------------|-------------|
| `CONFIG` | Cell 3 | All cells |
| `log()` | Cell 5 | All cells |
| `df_incidents` | Cell 8 | Cell 10 |
| `train_df`, `eval_df`, `holdout_df` | Cell 10 | Cell 11 (legacy mode) |
| `train_examples` | Cell 11 | Cell 18 |
| `eval_examples` | Cell 11 | Cells 18, 21, 23, 24 |
| `holdout_examples` | Cell 11 | Cells 23, 24 |
| `borderline_examples` | Cell 11 | Cell 29 |
| `DEVICE` | Cell 16 | Cell 18 |
| `best_model` | Cell 18 | Cells 21, 23, 24, 27 |

---

## Common Errors & Solutions

### Error: `NameError: name 'DEVICE' is not defined`
- **Cause:** Cell 16 hasn't been run
- **Solution:** Run Cell 16 before Cell 18

### Error: `NameError: name 'eval_examples' is not defined`
- **Cause:** Cell 11 hasn't been run
- **Solution:** Run Cell 11 before Cell 18

### Error: `NameError: name 'best_model' is not defined`
- **Cause:** Cell 18 hasn't been run (training hasn't completed)
- **Solution:** Run Cell 18 and wait for training to complete before running evaluation cells

### Error: `KeyError: 'hard_neg_ratio'`
- **Cause:** Trying to run Cell 14 (legacy pair generation) when it should be skipped
- **Solution:** Ensure `use_pre_generated_pairs = True` in Cell 3, then restart kernel and run all cells

---

## Recommended Execution

**Method 1: Run All (Recommended)**
1. Restart kernel: `Kernel → Restart & Clear Output`
2. Run all cells: `Cell → Run All`
3. Wait for training to complete (~30-60 minutes depending on GPU)

**Method 2: Run Sequentially**
1. Restart kernel
2. Run cells 0-3 (Setup & Config)
3. Run cells 5-11 (Data loading & pair loading) ⚠️ CRITICAL
4. Run cell 16 (Device detection) ⚠️ CRITICAL
5. Run cells 17-18 (Model & Training) ⚠️ CRITICAL - Wait for completion!
6. Run cells 21-24 (Evaluation)
7. Run cell 27 (Save)

**Do NOT:**
- Skip cells or run out of order
- Run evaluation cells (21-24) before training completes
- Run Cell 14 (it will be automatically skipped)

---

## Cell Types Summary

- **Markdown cells:** 14 (section headers and documentation)
- **Code cells:** 18 (executable Python code)
- **Critical cells:** 3 (cells 11, 16, 18)
- **Skipped cells:** 2 (cells 14, 29 in curriculum mode)
- **Total cells:** 32
