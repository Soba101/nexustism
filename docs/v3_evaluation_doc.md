# Evaluation and Improvement Plan for finetune_model_v3.ipynb

## 1. Executive Summary

The `finetune_model_v3.ipynb` notebook provides a solid pipeline for fine-tuning a SentenceTransformer model for IT service management (ITSM) ticket similarity. The model shows decent performance, but there is significant room for improvement. This document outlines the current performance, identifies key areas for enhancement, and proposes a set of actionable changes to boost the model's effectiveness.

The key recommendations are:

- Enhance the training data quality with more specific positive/negative pairs and advanced data augmentation.
- Fine-tune the training process by adjusting hyperparameters and exploring alternative loss functions.
- Address the class imbalance issue in the relationship classifier more robustly.

## 2. Performance Evaluation

### 2.1. Similarity Model

- **Base Model:** `sentence-transformers/all-mpnet-base-v2`
- **Training Data:** 1428 positive training pairs, with an additional 280 augmented pairs. 1400 negative pairs.
- **Training:** The model was trained with `MultipleNegativesRankingLoss`. Training stopped early at epoch 2.2 due to the early stopping mechanism (patience=2), with the best Spearman correlation score of **0.4993** achieved at epoch 1.6. This suggests that the model might be overfitting or the learning rate could be further optimized.
- **Evaluation Metrics:**
- **Spearman Correlation:** ~0.50. This is a reasonable score, indicating a positive correlation between the model's similarity scores and the ground truth labels. However, for a production system, a higher score is desirable.
- **ROC AUC:** ~0.8. The model is fairly good at distinguishing between positive and negative pairs.
- **PR AUC:** ~0.5. This score, which is more informative on imbalanced datasets, suggests that there is room for improvement in precision and recall.
- **Optimal Threshold:** The optimal classification threshold was found to be **0.9**, with an F1 score of **0.66**. A high threshold like this often indicates that the model produces relatively low similarity scores, even for positive pairs, and a high threshold is needed to filter out false positives.

### 2.2. Relationship Classifier

- **Issue:** The initial attempt to train the classifier failed with a `ValueError` because some classes had too few samples for a stratified train-test split. The 'duplicate' class, for instance, had only 2 samples.
- **Fix:** A subsequent cell corrected this by filtering out classes with fewer than 5 samples. While this allows the code to run, it's not an ideal solution as it means the classifier is not trained on all possible relationship types.
- **Performance:** For the classes it was trained on ('none' and 'related'), the classifier achieved an accuracy of **0.93**. This is a good result, but it's on a simplified, balanced version of the problem.

## 3. Proposed Improvements

### 3.1. Data-Centric Improvements

1. **Refine Positive/Negative Pair Generation:**
    - Currently, positive pairs are generated from the same 'Category'. To create higher-quality positive pairs, we can use a combination of **'Category' and 'Subcategory'**. This will ensure that the paired tickets are more semantically similar.
    - The "hard" negative mining is a good technique. We can enhance it by being more selective. For instance, we could look for pairs with high lexical overlap (e.g., using TF-IDF or Jaccard similarity) but different root causes or resolutions.

2. **Advanced Data Augmentation:**
    - The current augmentation is simple (randomly dropping/shuffling words). We can introduce more sophisticated techniques using the `nlpaug` library (which is mentioned in the notebook but not used).
    - **Back-translation:** Translate the text to another language and then back to English. This often paraphrases the sentence while preserving its meaning.
    - **Synonym Replacement:** Use a thesaurus (like WordNet) to replace words with their synonyms.

### 3.2. Training-Centric Improvements

1. **Hyperparameter Tuning:**
    - **Patience:** The early stopping patience of 2 seems too aggressive. I recommend increasing it to **3 or 4** to allow the model more time to converge, especially if we are using a lower learning rate.
    - **Learning Rate:** While `1e-5` is a good starting point, we should experiment with a **learning rate scheduler**, such as `cosine annealing`, which can help the model converge more effectively.
    - **Batch Size:** The batch size was reduced to 16. We can experiment with different batch sizes, as this can affect the gradient updates and generalization.

2. **Loss Function:**
    - `MultipleNegativesRankingLoss` is a strong choice. We could also experiment with other loss functions like `ContrastiveLoss` or `MegaBatchMarginLoss` to see if they yield better performance on our specific dataset.

### 3.3. Model-Centric Improvements

1. **Explore Alternative Pre-trained Models:**
    - While `all-mpnet-base-v2` is a great general-purpose model, it might be beneficial to explore models pre-trained on more technical or domain-specific corpora if available.

### 3.4. Relationship Classifier Improvements

1. **Handle Class Imbalance:**
    - Instead of dropping classes with few samples, we should use techniques to handle the class imbalance.
    - **Oversampling:** For minority classes like 'duplicate', we can use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples.
    - **Weighted Loss:** When training the classifier, we can use class weights to give more importance to the minority classes. The `LogisticRegression` model already has a `class_weight='balanced'` parameter, but if we switch to a different model (e.g., a neural network), we would need to implement this.

2. **Collect More Data:**
    - The most effective way to improve the relationship classifier is to **collect more labeled data**, especially for the underrepresented classes.

## 4. Conclusion

The `finetune_model_v3.ipynb` notebook is a good starting point, but the model's performance can be significantly improved. By focusing on data quality, refining the training process, and addressing the class imbalance in the relationship classifier, we can build a much more robust and accurate model for ITSM ticket similarity and relationship detection. I recommend implementing the proposed changes in a new version of the notebook, `finetune_model_v4.ipynb`, to track the improvements.
