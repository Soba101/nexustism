# V5 Implementation Plan

## Overview

Create a new V5 notebook that systematically addresses identified performance bottlenecks to target 0.8+ scores across F1, PR AUC, and improved Spearman correlation.

## Problem-Specific Solutions

### 1. Training Data Quality Enhancement

**Current issue:** Only ~10,080 positive pairs from 10K incidents (sparse coverage).

-**V5 solutions:**

- **Data expansion:** Generate 50k+ training pairs using multiple strategies.
- **Semantic filtering:** Use sentence-transformers embeddings and keep positives in the 0.4–0.8 similarity range.
- **Multi-level positives:**
  - Level 1 — Exact category + subcategory matches (high similarity).
  - Level 2 — Category match + semantic similarity (medium similarity).
  - Level 3 — Cross-category but high text similarity (challenging positives).
- **Stratified negatives:** Ensure negatives span easy → hard difficulty levels.
- **Validation pipeline:** Remove duplicates/invalid pairs and enforce quality checks.

### 2. Training Configuration Optimization

**Current issues:** Short training (8 epochs), low LR (1e-5), small batch size (16), premature early stopping.

-**V5 solutions:**

- **Learning rate:** Base to 2e-5 with cosine annealing and peaks up to 5e-5.
- **Training duration:** 20 epochs with flexible early stopping (patience=7, min_delta=0.005).
- **Batch size:** Target 64 with gradient accumulation if memory constrained.
- **Warmup:** 10% warmup steps followed by cosine decay.
- **Multi-phase training:**
  - Phase 1: General similarity (≈15 epochs).
  - Phase 2: Fine-grained ranking (≈5 epochs, lower LR).

### 3. Data Generation Strategy Overhaul

**Current issues:** Simple category/subcategory matching and weak hard negatives.

-**V5 solutions:**

- **Advanced positives:** TF-IDF cosine (0.3–0.7), Jaccard with stemming/lemmatization, named-entity overlap, temporal proximity for incident sequences.
- **Hard negatives:** Cross-category lexical overlap, similar error messages with different root causes, same component but different issue type, and dynamic hard-negative mining during training.
- **Quality scoring:** Rate pair quality and use weighted sampling for training.

### 4. Model Architecture Improvements

**Current issue:** Generic `all-mpnet-base-v2` not optimized for the ITSM domain.

-**V5 solutions:**

- **Ensemble approach:**
  - Primary: fine-tuned `all-mpnet-base-v2`.
  - Secondary: `microsoft/DialoGPT-medium` (technical/conversational text).
  - Tertiary: `sentence-transformers/all-MiniLM-L12-v2` (fast inference).
- **Domain pre-training:** Additional MLM on ITSM corpus and vocabulary expansion.
- **Loss engineering:** Combine `MultipleNegativesRankingLoss`, `TripletLoss` (with tuned margin), and `CosineSimilarityLoss` for calibration.
- **Pooling:** Use advanced pooling (mean + max + CLS) instead of mean-only.

## Implementation Strategy

### Phase 1 — Enhanced Data Pipeline (Days 1–2)

- Smart pair generation: semantic scoring, multi-level positives, advanced hard-negative mining, data validation.
- Augmentations: synonym replacement (WordNet + domain lexicon), back-translation paraphrases, technical-term substitution, noise injection.

### Phase 2 — Training Optimization (Days 3–4)

- Configuration: learning-rate scheduling, gradient accumulation, multi-phase training, advanced early-stopping logic.
- Loss engineering: weighted loss combinations, margin-based triplet loss, dynamic loss weighting.

### Phase 3 — Architecture Enhancement (Days 5–7)

- Model improvements: domain vocabulary injection, advanced pooling, ensemble composition, model compression for inference.
- Evaluation: cross-validation, metric-tracking dashboard, A/B testing framework, benchmarking.

## Expected Performance Gains

**Conservative estimates (90% confidence):**

- F1: 0.684 → 0.78–0.82
- PR AUC: 0.684 → 0.76–0.80
- Spearman: 0.497 → 0.62–0.68
- ROC AUC: 0.920 → 0.94–0.96

**Optimistic targets (60% confidence):**

- F1: 0.684 → 0.82–0.87
- PR AUC: 0.684 → 0.80–0.85
- Spearman: 0.497 → 0.68–0.75
- ROC AUC: 0.920 → 0.95–0.97

## Risk Mitigation

- **Modularity:** Toggle components on/off for incremental testing.
- **Baseline comparison:** Keep v4 as the canonical baseline for every experiment.
- **Progressive rollout:** Implement improvements incrementally with validation checks.
- **Fallbacks:** Graceful degradation if advanced features fail.
- **Resource monitoring:** Track memory and compute during development and training.

## Key Innovations in V5

1. Intelligent data generation with quality-scored semantic filtering.
2. Dynamic multi-phase training with adaptive learning-rate schedules.
3. Ensemble architecture for robustness and complementary strengths.
4. Advanced evaluation and metric tracking for reliable benchmarking.
5. Production-ready inference pipeline with model compression and efficiency.
