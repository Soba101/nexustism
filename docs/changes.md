## Tuning Change Log

### 2025-?? — finetune_model.ipynb (v1 baseline)

- Initial ticket-similarity fine-tune on `all-mpnet-base-v2` with cosine loss and generated positive/negative pairs.
- Limited guards for cell execution order; no relationship classifier; minimal metrics/plots; CONFIG handling fragile.

### 2025-?? — finetune_model_v2.ipynb (incremental fixes + classifier)

- Added relationship-classifier workflow on relationship_pairs with logistic regression over embedding features.
- Patched execution-order gaps: defaults for missing CONFIG fields, reloading pairs if generation cell skipped, ensured imports where used.
- Stabilized classification reporting by passing explicit labels/target_names to avoid crashes when classes are absent.
- Reduced batch size and added default eval_split/base_model guards to keep runs reproducible.

### 2025-?? — finetune_model_v3.ipynb (rebuild + hardening)

- Rebuilt pipeline with explicit CONFIG defaults, output path normalization, and required `data/training_pairs.json` guard.
- Training loop uses DataLoader + CosineSimilarityLoss, evaluator on eval split, and warmup derived from loader length.
- Added similarity/ROC/PR plots; optional relationship classifier with pairwise features and reporting.
- Captured results: Pearson/Spearman improved markedly by epoch 12; relationship classifier remained imbalanced with weak duplicate/causal support.

### 2025-11-29 — finetune_model_v4.ipynb hardening and reproducibility

- Made `imbalanced-learn` optional; relationship classifier now skips cleanly when missing.
- Added global seeding (Python/NumPy/torch/CUDA) and recorded library versions/seeds in `training_metadata.json`.
- Hardened hard-negative creation with category-count guards and zero-length handling plus safe fallback.
- Wired `CONFIG['learning_rate']` into `model.fit` via `optimizer_params`.
- Enhanced augmentation with word drop/swap variety for positive pairs.
- Relationship classifier now filters ultra-rare classes, guards SMOTE/stratify when class counts are tiny, and keeps training skippable.
- Threshold grid refined to 0.01 steps for better F1 tuning.
