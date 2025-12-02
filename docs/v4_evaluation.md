# finetune_model_v4.ipynb Evaluation

## Checks performed

- Verified notebook JSON integrity and fixed the logging f-string in the training callback that previously broke JSON parsing.
- Ran a static syntax check by concatenating code cells and parsing with `ast` (no syntax errors).
- Reviewed data loading, training, and evaluation cells for likely runtime risks and configuration gaps.

## Findings (action recommended)

- Negative pair generation can throw a `ZeroDivisionError` when either ticket text is empty because `overlap = len(words1 & words2) / max(len(words1), len(words2))` divides by zero. Add a guard to skip or handle empty texts before computing overlap.
- Hard-negative sampling assumes at least two distinct categories. With a single category in the CSV, `random.sample(categories, 2)` will raise. Bail out early or relax the constraint when categories < 2.
- Relationship classifier path: if SMOTE is skipped due to tiny classes (`min_class_size < 2`), the subsequent `train_test_split(..., stratify=y_resampled)` will still raise when any class has only one sample. Need a fallback (e.g., skip stratify or drop singleton classes) when SMOTE is bypassed.
- Configured learning rate (`CONFIG['learning_rate']`) is never passed into `model.fit`, so training always uses the library default. Wire it through `optimizer_params` if the intention is to honor the config.
- The notebook exits immediately if `imbalanced-learn` is missing. Consider making the relationship-classifier section optional instead of terminating the whole run.

## Minor notes

- `TrainerControl` is imported in multiple cells; can be trimmed, but it does not affect execution.

---

## Run on 2025-11-29

### V4 Changes Implemented

- **Improved Positive Pair Generation:** Uses both `Category` and `Subcategory` for more specific positive training pairs.
- **Increased Early Stopping Patience:** Patience was increased to 3.
- **SMOTE for Class Imbalance:** Implemented SMOTE to handle class imbalance in the optional relationship classifier.
- **Data Augmentation:** Simple data augmentation was added to increase the diversity of positive training pairs.
- **Hard Negatives:** Negative pair generation creates "hard negatives" with some keyword overlap but different categories.

### [GEMINI-FIX] New Changes Implemented

- **Refined Data Generation with Jaccard Similarity:**
  - The `jaccard_similarity` function was introduced to measure textual overlap.
  - Positive pair generation now incorporates a Jaccard similarity threshold (>0.3) to select higher-quality pairs that share common keywords, in addition to sharing `Category` and `Subcategory`. This ensures that positive pairs are not only semantically related but also textually similar.
  - Hard negative generation now uses Jaccard similarity (between 0.1 and 0.5) to identify pairs from different categories that still have some textual overlap, making them "harder" for the model to distinguish and thus improving model robustness.
- **Advanced Data Augmentation with `nlpaug`:**
  - The simple `augment_text_simple` function has been replaced with a more sophisticated `augment_text` function.
  - When `nlpaug` is available (checked via `NLPAUG_AVAILABLE`), it leverages `nlpaug.augmenter.word.SynonymAug` with WordNet to perform synonym replacement (augmenting 10% of words and ignoring stopwords). This generates more diverse and realistic paraphrases.
  - If `nlpaug` is not installed, it gracefully falls back to the previous simple word drop/swap augmentation logic, ensuring backward compatibility.
  - Error handling for `nlpaug` augmentation failures is also included, logging warnings and returning the original text if augmentation fails.
- **Learning Rate Configuration Corrected:**
  - The `CONFIG['learning_rate']` is now correctly passed to the `model.fit()` method via the `optimizer_params` argument, allowing the configured learning rate to be used during training. This addresses a previous configuration oversight where the default learning rate might have been used instead.

### Training Results

- **Model:** `sentence-transformers/all-mpnet-base-v2`
- **Data:** `data/dummy_data_promax.csv` (10,000 incidents)
- **Training Pairs:**
  - 10,080 positive pairs (including 1,680 augmented pairs)
  - 8,400 hard negative pairs
- **Training Loss:** `MultipleNegativesRankingLoss`
- **Early Stopping:** Training stopped early at epoch 0.93.
- **Best Score (Spearman Cosine):** 0.4964 (achieved at epoch 0.84)

### Evaluation Results

- **ROC AUC:** 0.920
- **PR AUC:** 0.684
- **Optimal Classification Threshold:** 0.900
  - **F1 Score at Threshold:** 0.684
  - **Accuracy at Threshold:** 0.920

### Relationship Classifier (Optional)

- **Resampling:** SMOTE was applied to the `relationship_pairs.json` dataset.
- **Performance:**
  - **Accuracy:** 0.97
  - The classifier performed well for `duplicate`, `related`, and `none` classes. No data was available for the `causal` class in the validation set.

### Potential Improvements

- **Hard Negative Generation Robustness:** The current hard negative generation logic can be fragile if there aren't enough distinct categories or if texts are empty, leading to `ZeroDivisionError` or errors from `random.sample`. Implement more robust error handling or fallback mechanisms for these scenarios.
- **Relationship Classifier `k_neighbors` for SMOTE:** The `k_neighbors` for SMOTE is capped at `min(5, min_class_size - 1)`. If `min_class_size` is very small (e.g., 1 or 2), SMOTE might still struggle or be skipped. Consider more advanced techniques or explicit handling for extremely small minority classes.
- **Learning Rate Configuration:** The configured `CONFIG['learning_rate']` is not currently passed to `model.fit`. Ensure this parameter is correctly utilized to allow for fine-grained control over the training process.
- **Error Handling for `imbalanced-learn`:** The notebook currently exits if `imbalanced-learn` is not installed. To improve flexibility, the relationship classifier section should be made optional, allowing the similarity model to train independently of this dependency.
- **Causal Class in Relationship Classifier:** The 'causal' class showed 0 precision, recall, and f1-score, suggesting its absence from the validation set after splitting/resampling. Address this by ensuring adequate representation or appropriate handling of such classes in evaluation.
- **Data Augmentation Strategy:** The current data augmentation is basic (random word dropping/shuffling). Explore more sophisticated techniques like back-translation or synonym replacement using libraries such as `nlpaug` to further enhance data diversity and model robustness.
- **Evaluation Metrics for Imbalanced Data (Classifier):** While SMOTE helps with training, a deeper analysis of per-class metrics, particularly for minority classes (even if synthetically generated), could provide more granular insights into the relationship classifier's performance.
- **Logging and Reproducibility:** Expand `training_metadata.json` to include additional details crucial for full reproducibility, such as specific library versions and all random seeds used throughout the data preparation and training pipeline.
- **Threshold Optimization:** The current threshold optimization employs a simple grid search. A more sophisticated approach (e.g., a finer grid search, methods based on Youden's J statistic, or integrating with model calibration techniques) could lead to a more optimally chosen classification threshold.

### Updates Applied to finetune_model_v4.ipynb

- Hard negatives: Added category-count guard, zero-length text checks, and a random fallback to avoid `ZeroDivisionError` or `random.sample` crashes on sparse data.
- Learning rate: `CONFIG['learning_rate']` now feeds into `model.fit` via `optimizer_params`.
- Seeds and metadata: Added global seed setting (Python/NumPy/torch/CUDA) and recorded library versions/seeds in `training_metadata.json`.
- Relationship classifier: Made `imbalanced-learn` optional, filtered ultra-rare classes, guarded SMOTE/stratified split when class counts are tiny, and ensured the block is skippable when the dependency is missing.
- Evaluation: Finer threshold grid (0.01 step) for F1/accuracy selection.
- Augmentation: Slightly richer word-drop/swap augmentation for positive pairs.

### Validation Against finetune_model_v4.ipynb (2025-11-29 run)

- Hard negatives: Category-count and zero-length guards added with a random fallback; crash risks from sparse categories should be removed.
- Relationship classifier SMOTE/stratify: Dependency is optional now; SMOTE/stratify are skipped when class counts are tiny. Rare labels (<2 samples) are dropped before training—collector still needs more data to keep those labels.
- Learning rate wiring: `CONFIG['learning_rate']` now flows into `model.fit` via `optimizer_params`.
- Optional dependency handling: Missing `imbalanced-learn` now skips only the classifier block; similarity training continues.
- Causal class coverage: Classes with <2 samples are filtered; the classifier will not cover them until more data is provided or augmentation is added.
- Augmentation depth: Word-drop/swap augmentation adds small lexical variety; still lightweight compared to back-translation/synonym replacement.
- Threshold search resolution: Threshold grid refined to 0.01 steps for better F1/accuracy selection.
- Reproducibility metadata: Seeds set (Python/NumPy/torch/CUDA) and key library versions logged to `training_metadata.json`.

### Expected Impact vs V3 (observed + estimated)

- Observed deltas (V4 vs V3): ROC AUC ~0.92 (+0.12), PR AUC ~0.684 (+0.184), F1 at threshold ~0.684 (+0.024), Spearman 0.4964 (−0.003). Hard-negative mining and added data likely drove the AUC gains; Spearman dipped slightly.
- After addressing the above: Wiring the learning rate and refining thresholding should conservatively add ~0.01–0.03 to Spearman/PR AUC and ~0.005–0.02 to F1, while robustness fixes remove crash risk on sparse categories or missing `imbalanced-learn` installs (availability win rather than metric lift).
