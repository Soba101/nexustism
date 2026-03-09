#!/usr/bin/env python3
"""
Fix evaluation errors in v3 notebook:
1. Skip borderline results in error summary if None
2. Add missing TFIDFSimilarityCalculator class
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

print("="*80)
print("FIXING EVALUATION ERRORS")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_made = []

# Fix 1: Skip None borderline_results in error summary
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find the error summary loop
        if 'for name, errors, results in [' in source and 'borderline_results' in source:
            print(f"\n[FIX 1] Found error summary loop in Cell {cell_idx}")

            new_lines = []
            i = 0
            while i < len(cell['source']):
                line = cell['source'][i]

                # Find the loop line
                if 'for name, errors, results in [' in line:
                    new_lines.append(line)
                    i += 1

                    # Copy lines until we hit the closing bracket
                    paren_depth = 1
                    loop_lines = []
                    while i < len(cell['source']) and paren_depth > 0:
                        line = cell['source'][i]
                        paren_depth += line.count('[') - line.count(']')
                        loop_lines.append(line)
                        i += 1

                    # Add the loop lines
                    new_lines.extend(loop_lines)

                    # Find the next line (should be accessing results)
                    if i < len(cell['source']):
                        next_line = cell['source'][i]

                        # Add None check before accessing results
                        indent = len(next_line) - len(next_line.lstrip())
                        new_lines.append(' ' * indent + '# Skip if results is None (e.g., borderline not available)\n')
                        new_lines.append(' ' * indent + 'if results is None:\n')
                        new_lines.append(' ' * (indent + 4) + 'continue\n')
                        new_lines.append(' ' * indent + '\n')
                        changes_made.append(f"Cell {cell_idx}: Added None check for results in error summary")

                    continue

                new_lines.append(line)
                i += 1

            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell

# Fix 2: Add TFIDFSimilarityCalculator class before adversarial diagnostic
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find adversarial diagnostic cell (uses TFIDFSimilarityCalculator)
        if 'diag_tfidf = TFIDFSimilarityCalculator(' in source and 'class TFIDFSimilarityCalculator' not in source:
            print(f"\n[FIX 2] Found adversarial diagnostic in Cell {cell_idx}")

            # Insert TFIDFSimilarityCalculator class at the beginning of the cell
            tfidf_class = '''# Define TFIDFSimilarityCalculator for adversarial diagnostic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSimilarityCalculator:
    """Calculate TF-IDF based similarity for text pairs."""

    def __init__(self, texts, max_features=5000):
        """Initialize with corpus of texts."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def get_similarity(self, idx1, idx2):
        """Get cosine similarity between two texts by index."""
        vec1 = self.tfidf_matrix[idx1]
        vec2 = self.tfidf_matrix[idx2]
        return cosine_similarity(vec1, vec2)[0, 0]

'''

            new_lines = [tfidf_class + '\n'] + cell['source']
            cell['source'] = new_lines
            nb['cells'][cell_idx] = cell
            changes_made.append(f"Cell {cell_idx}: Added TFIDFSimilarityCalculator class")

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
if changes_made:
    print(f"[SUCCESS] Applied {len(changes_made)} fixes!")
    print(f"{'='*80}")
    for change in changes_made:
        print(f"  [OK] {change}")
    print(f"\n{'='*80}")
    print("Evaluation errors fixed!")
    print("\nNext steps:")
    print("  1. Restart Jupyter kernel (if needed)")
    print("  2. Re-run evaluation cells")
    print(f"{'='*80}")
else:
    print("[INFO] No changes made")
    print(f"{'='*80}")
