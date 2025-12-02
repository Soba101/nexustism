# V6 Implementation Plan: Feature Selection & Contextual Embedding

## Kaggle Environment Setup (Critical)

If running this notebook on a remote Kaggle kernel, you **must** run the following setup code at the very beginning of your session to handle dependencies and NLTK data:

```python
# [Setup] Install dependencies and download NLTK data (Run this first!)
!pip install --upgrade sentence-transformers imbalanced-learn "protobuf<=3.20.1" --quiet

# Fix TensorFlow/Protobuf conflicts
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download NLTK data immediately
import nltk
for res in ['wordnet', 'omw-1.4', 'stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(res, quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download {res}: {e}")
```

## Overview

This document outlines the strategy for the V6 iteration of the ITSM similarity model. The primary focus is to move beyond the simple `Short Description + Description` text concatenation used in V5 and incorporate high-value structured fields from the ServiceNow data to create a richer, "context-aware" text representation.

## Findings from V5 Analysis

Analysis of `finetune_model_v5.ipynb` and the available data (`data/dummy_data_promax.csv`) revealed:

1. **Underutilized Features:** The V5 model currently ignores several critical columns that contain strong signals for ticket similarity:

   - `Service` & `Service offering`: High-level indicators of the affected system (e.g., "CRM", "SAP").
     - `Assignment group`: A strong proxy for the technical domain and resolution team.
     - `Category` & `Subcategory`: Currently used only for sampling pairs, but not for the actual text embedding.

2. **Exclusion of "Resolution Notes":**

- **Decision:** We will explicitly **EXCLUDE** `Resolution notes` (and similar fields like `Resolution code`) from the training embedding.
- **Reasoning (Data Leakage):** The primary use case is to match a _new, incoming_ ticket (which has no resolution yet) to _historical_ tickets. If the historical tickets' embeddings are heavily influenced by their resolution text (e.g., "Rebooted server", "User training provided"), they will drift away from the "Problem-only" embedding of the new ticket.
- **False Positives:** Two unrelated problems (e.g., "Printer broken" and "VPN down") might share the resolution "Replaced hardware" or "User training". Training on this would incorrectly cluster them together.
- **Best Practice:** Match on _Symptoms_ (Description, Service, Category), then _retrieve_ the Resolution to show to the user.

## Proposed Changes: Contextual Prefixing

We will implement "Contextual Prefixing" to inject structured metadata directly into the text input string. This forces the model to learn embeddings that are sensitive to the technical context of the ticket.

### New Features to Include

- `Service`
- `Service offering`
- `Category`
- `Subcategory`
- `Assignment group`

### Text Construction Strategy

Instead of:

```python
text = Short Description + " " + Description
```

We will use a structured format:

```python
text = [Service | Service Offering] [Category | Subcategory] Group: Assignment Group. Short Description. Description
```

**Example:**

- **Before:** "Unable to log in. User gets timeout error."
- **After:** `[SAP | S4HANA] [Inquiry | Help] Group: PISCAP L2 BI. Unable to log in. User gets timeout error.`

## Implementation Details

The `load_and_clean_data` function will be updated to:

1. **Fill Missing Values:** Ensure `Service`, `Service offering`, and `Assignment group` have default values (e.g., empty string or "Unknown") to prevent format breakage.
2. **Construct Rich Text:**

```python
df["text"] = (
    "[" + df["Service"].fillna("") + " | " + df["Service offering"].fillna("") + "] " +
    "[" + df["Category"].fillna("") + " | " + df["Subcategory"].fillna("") + "] " +
    "Group: " + df["Assignment group"].fillna("") + ". " +
    df["Short Description"] + ". " +
    df["Description"]
).str.strip()
```

**Clean Formatting:** Remove empty brackets `[]` or generic placeholders if fields are missing, ensuring a clean text string.

## Expected Benefits

1. **Disambiguation:** "Password reset" tickets for different systems (e.g., SAP vs. AD) will be pushed further apart in vector space.
2. **Cluster Purity:** Tickets belonging to the same `Assignment group` or `Service` will naturally cluster more tightly.
3. **Improved Hard Negatives:** The model will better learn to distinguish between tickets that share words ("timeout") but differ in context (Network vs. Database).
