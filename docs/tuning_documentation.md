# Tuning Documentation (SentenceTransformer ITSM Project)

Detailed history of the three notebook iterations, what was tried, issues encountered, fixes applied, and current recommendations based on the data available.

## Notebook Summaries

### finetune_model.ipynb (v1, baseline)

- Objective: fine-tune a SentenceTransformer for ticket similarity using positive/negative pairs.
- Base model: sentence-transformers/all-mpnet-base-v2.
- Loss: cosine similarity (SentenceTransformer fit loop).
- Data: positive/negative pairs generated from incidents.
- Gaps discovered later: no guards when cells run out of order, no relationship classifier, limited visibility into metrics/plots, fragile CONFIG handling.

### finetune_model_v2.ipynb (v2, incremental fixes + relationship classifier)

- Added relationship-classifier workflow on relationship_pairs.json/CSV using fine-tuned embeddings + logistic regression.
- Execution-order issues and fixes:
  - Missing positive_pairs/negative_pairs: added guard to load data/training_pairs.json if the generation cell was skipped.
  - Missing CONFIG fields: defaulted eval_split; later defaulted base_model if absent.
  - Missing imports when earlier cells not run: added local “from sentence_transformers import InputExample” in the training-examples cell; ensured DataLoader is imported where used.
  - Classification report crashes when a class is absent in validation: passed explicit labels and target_names to classification_report and confusion_matrix.
- Outcome: runnable end-to-end if cells are run, more resilient to missing state; still affected by class imbalance in relationship labels.

### finetune_model_v3.ipynb (v3, rebuilt and hardened)

- Self-contained pipeline with explicit defaults, guards, and plotting.
- CONFIG defaults: base_model=all-mpnet-base-v2, output_dir=models/all-mpnet-finetuned, epochs=12, batch_size=32, learning_rate=2e-5, warmup_steps=100, eval_split=0.2, max_seq_length=256.
- Output path handling: coerce output_dir to string, resolve absolute vs relative, os.makedirs ensured.
- Data loading: requires data/training_pairs.json; clear error if missing. Current run used 291 positive / 118 negative → train 327 / eval 82.
- Training: builds InputExamples, shuffles/splits, DataLoader + CosineSimilarityLoss, evaluator on eval set, warmup computed from loader length.
- Results (logged):
  - Eval Pearson/Spearman climbed from ~0.26/0.26 early to ~0.90/0.79 by epoch 12.
  - Reload smoke test similarity ~0.80 on example pair.
- Visualization: similarity histograms, ROC + AUC, PR + AUC; confusion-matrix plot for relationship classifier if trained in-session.
- Relationship classifier (optional):
  - Uses pairwise features (concat, diff, product) + logistic regression.
  - Validation was highly imbalanced: zero duplicate and zero causal samples; none strongest, related moderate. Micro ~0.69, macro ~0.29 due to imbalance.

## Issues Observed

- Relationship classifier: severe class imbalance and zero support for duplicate/causal in validation lead to near-zero metrics for those classes; model biases to none.
- Small eval set for similarity (82 pairs) makes metrics somewhat noisy; single split can vary.
- Execution-order fragility mitigated in v2/v3 but v1 remains order-sensitive.

## Recommendations (with current data)

- Balance/stratify relationship data:
  - Oversample duplicate and causal in training.
  - Enforce stratified splits that guarantee at least one sample per class in validation; consider repeated stratified splits or k-fold to stabilize metrics.
- Feature tweaks for the classifier:
  - Add explicit cosine similarity as an extra feature alongside concat/diff/product of embeddings.
  - Try alternative linear models: tuned Logistic Regression (vary C, keep class_weight), linear SVM, or a shallow MLP.
- Evaluation hygiene:
  - Fix the random seed and verify per-class counts before training.
  - If eval remains small, average results over multiple splits instead of relying on one.
- Similarity model:
  - Training curve shows plateau around epoch ~10–12; keep monitoring Pearson/Spearman and stop when it stabilizes.
  - Keep max_seq_length moderate (256) unless you have evidence longer text is needed.

## Alternative 768d Base Models (swap into CONFIG)

- Quality-focused: all-mpnet-base-v2, paraphrase-mpnet-base-v2, intfloat/e5-base-v2 (use query/passage prefixes if following e5 guidance).
- Faster/smaller: all-MiniLM-L12-v2, paraphrase-MiniLM-L12-v2.
- Retrieval-leaning: msmarco-distilbert-base-v4, multi-qa-MiniLM-L12-v1, e5-base-v2.

## File Paths of Interest

- Notebooks: finetune_model.ipynb, finetune_model_v2.ipynb, finetune_model_v3.ipynb.
- Training pairs: data/training_pairs.json.
- Relationship pairs: data/relationship_pairs.json.
- Model output: models/all-mpnet-finetuned/ (and relationship_classifier/ subdir for classifier artifacts).
