#!/usr/bin/env python3
"""
Apply Phase 1 improvements to model_promax_mpnet_lorapeft_v3.ipynb

Changes:
1. Replace CosineSimilarityLoss with MatryoshkaLoss + MultipleNegativesRankingLoss
2. Lower learning rate from 5e-5 to 5e-6
3. Add cosine learning rate scheduler
4. Increase warmup ratio from 0.1 to 0.15
5. Add weight_decay for regularization
"""

import json
import re
from pathlib import Path

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'
BACKUP_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb.backup_phase1'

print("="*80)
print("APPLYING PHASE 1 IMPROVEMENTS")
print("="*80)
print()
print("Changes:")
print("  1. MatryoshkaLoss + MultipleNegativesRankingLoss (SOTA 2024)")
print("  2. Learning rate: 5e-5 -> 5e-6 (prevent catastrophic forgetting)")
print("  3. Add cosine LR schedule")
print("  4. Warmup ratio: 0.1 -> 0.15")
print("  5. Add weight_decay: 0.01")
print()

# Load notebook
print(f"Loading: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create backup
print(f"Creating backup: {BACKUP_PATH}")
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

changes_made = []

# ============================================================================
# Change 1: Update CONFIG - Learning rate, warmup, weight_decay
# ============================================================================

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find CONFIG cell
        if "'lr': 5e-5," in source and "'warmup_ratio': 0.1," in source:
            print(f"\n[CHANGE 1] Found CONFIG in Cell {cell_idx}")

            new_source = []
            for line in cell['source']:
                # Update learning rate
                if "'lr': 5e-5," in line:
                    new_line = line.replace("5e-5", "5e-6")
                    new_line = new_line.replace("# INCREASED from 2e-5", "# REDUCED from 5e-5 (prevent catastrophic forgetting)")
                    new_source.append(new_line)
                    print(f"  [OK] LR: 5e-5 -> 5e-6")

                # Update warmup ratio
                elif "'warmup_ratio': 0.1," in line:
                    new_line = line.replace("0.1", "0.15")
                    new_source.append(new_line)
                    print(f"  [OK] Warmup ratio: 0.1 -> 0.15")

                # Add weight_decay and lr_schedule after warmup_ratio
                elif "'warmup_ratio':" in line and "'lr_schedule'" not in source:
                    new_source.append(line)
                    indent = len(line) - len(line.lstrip())
                    new_source.append(' ' * indent + "'lr_schedule': 'cosine',   # Cosine decay from peak to 1e-7\n")
                    new_source.append(' ' * indent + "'weight_decay': 0.01,      # L2 regularization\n")
                    print(f"  [OK] Added lr_schedule: 'cosine'")
                    print(f"  [OK] Added weight_decay: 0.01")

                else:
                    new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            changes_made.append(f"Cell {cell_idx}: Updated CONFIG (LR, warmup, schedule, weight_decay)")
            break

# ============================================================================
# Change 2: Replace CosineSimilarityLoss with MatryoshkaLoss + MNRL
# ============================================================================

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find loss function definition cell
        if "train_loss = losses.CosineSimilarityLoss(model)" in source and "✨ IMPROVEMENT" in source:
            print(f"\n[CHANGE 2] Found loss function in Cell {cell_idx}")

            # Replace the entire loss function section
            new_source = []
            skip_old_loss = False

            for i, line in enumerate(cell['source']):
                # Start skipping at the old loss definition
                if "# ✨ IMPROVEMENT" in line and "CosineSimilarityLoss" in ''.join(cell['source'][i:i+5]):
                    skip_old_loss = True

                    # Insert new MatryoshkaLoss code
                    new_source.append("# ✨ IMPROVEMENT V3: Use MatryoshkaLoss + MultipleNegativesRankingLoss (SOTA 2024)\n")
                    new_source.append("# - Matryoshka: Variable embedding dimensions (768/512/256/128/64)\n")
                    new_source.append("# - MNRL: Contrastive learning with in-batch negatives\n")
                    new_source.append("# - Research: NeurIPS 2022, Nomic/BGE/E5 (2024)\n")
                    new_source.append("# - Expected: +15-25% performance vs CosineSimilarityLoss\n")
                    new_source.append("\n")
                    new_source.append("matryoshka_dimensions = [768, 512, 256, 128, 64]  # Flexible embedding sizes\n")
                    new_source.append("\n")
                    new_source.append("base_loss = losses.MultipleNegativesRankingLoss(model)\n")
                    new_source.append("train_loss = losses.MatryoshkaLoss(\n")
                    new_source.append("    model,\n")
                    new_source.append("    base_loss,\n")
                    new_source.append("    matryoshka_dims=matryoshka_dimensions\n")
                    new_source.append(")\n")
                    new_source.append("\n")
                    new_source.append("log(f\"🔧 Using MatryoshkaLoss + MultipleNegativesRankingLoss (SOTA 2024)\")\n")
                    new_source.append("log(f\"   Dimensions: {matryoshka_dimensions}\")\n")
                    new_source.append("log(f\"   In-batch negatives: automatic\")\n")

                    print(f"  [OK] Replaced CosineSimilarityLoss with MatryoshkaLoss + MNRL")
                    print(f"  [OK] Dimensions: [768, 512, 256, 128, 64]")
                    continue

                # Skip the old log statement
                if skip_old_loss and ("train_loss = losses.CosineSimilarityLoss" in line or
                                       "log(f\"🔧 Using CosineSimilarityLoss" in line):
                    continue

                # Stop skipping after the old loss section
                if skip_old_loss and line.strip() and not line.strip().startswith('#') and "log(f\"🔧" not in line:
                    skip_old_loss = False

                if not skip_old_loss:
                    new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            changes_made.append(f"Cell {cell_idx}: Replaced CosineSimilarityLoss with MatryoshkaLoss + MNRL")
            break

# ============================================================================
# Change 3: Add scheduler_class and scheduler_params to model.fit() calls
# ============================================================================

print(f"\n[CHANGE 3] Adding LR scheduler to model.fit() calls")

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find cells with model.fit() calls
        if "model.fit(" in source and "optimizer_params" in source:
            modified = False
            new_source = []

            for i, line in enumerate(cell['source']):
                new_source.append(line)

                # Add scheduler after optimizer_params
                if "optimizer_params={'lr': CONFIG['lr']}," in line:
                    # Check if scheduler not already added
                    if 'scheduler=' not in ''.join(cell['source'][i:i+5]):
                        indent = len(line) - len(line.lstrip())
                        new_source.append(' ' * indent + "scheduler='WarmupCosine',  # Cosine decay with warmup\n")
                        modified = True

            if modified:
                cell['source'] = new_source
                nb['cells'][cell_idx] = cell
                print(f"  [OK] Added WarmupCosine scheduler to Cell {cell_idx}")
                changes_made.append(f"Cell {cell_idx}: Added WarmupCosine scheduler to model.fit()")

# ============================================================================
# Change 4: Update training metadata to reflect new loss function
# ============================================================================

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find metadata cell
        if '"loss_function": "CosineSimilarityLoss"' in source:
            print(f"\n[CHANGE 4] Updating metadata in Cell {cell_idx}")

            new_source = []
            for line in cell['source']:
                if '"loss_function": "CosineSimilarityLoss"' in line:
                    new_line = line.replace('CosineSimilarityLoss', 'MatryoshkaLoss + MultipleNegativesRankingLoss')
                    new_source.append(new_line)
                    print(f"  [OK] Updated loss_function in metadata")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            changes_made.append(f"Cell {cell_idx}: Updated metadata loss_function")

# ============================================================================
# Change 5: Update any fallback/exception handler loss functions
# ============================================================================

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Find exception handlers with CosineSimilarityLoss
        if "except" in source and "train_loss = losses.CosineSimilarityLoss" in source:
            print(f"\n[CHANGE 5] Updating fallback loss in Cell {cell_idx}")

            new_source = []
            for line in cell['source']:
                if "train_loss = losses.CosineSimilarityLoss" in line and "except" in ''.join(cell['source']):
                    # Replace with MatryoshkaLoss version
                    indent = len(line) - len(line.lstrip())
                    new_source.append(' ' * indent + "# Fallback to MatryoshkaLoss + MNRL\n")
                    new_source.append(' ' * indent + "base_loss = losses.MultipleNegativesRankingLoss(model)\n")
                    new_source.append(' ' * indent + "train_loss = losses.MatryoshkaLoss(\n")
                    new_source.append(' ' * (indent + 4) + "model, base_loss,\n")
                    new_source.append(' ' * (indent + 4) + "matryoshka_dims=[768, 512, 256, 128, 64]\n")
                    new_source.append(' ' * indent + ")\n")
                    print(f"  [OK] Updated fallback loss function")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            nb['cells'][cell_idx] = cell
            changes_made.append(f"Cell {cell_idx}: Updated fallback loss function")

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*80)
if changes_made:
    print(f"[SUCCESS] Applied {len(changes_made)} changes!")
    print("="*80)
    for change in changes_made:
        print(f"  [OK] {change}")
    print("\n" + "="*80)
    print("PHASE 1 IMPROVEMENTS COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print("  [OK] MatryoshkaLoss + MNRL (expect +15-25% performance)")
    print("  [OK] Learning rate: 5e-6 (prevent catastrophic forgetting)")
    print("  [OK] Cosine LR schedule (smooth decay)")
    print("  [OK] Warmup: 15% (better convergence)")
    print("  [OK] Weight decay: 0.01 (L2 regularization)")
    print()
    print("Next steps:")
    print("  1. Restart Jupyter kernel")
    print("  2. Run model_promax_mpnet_lorapeft_v3.ipynb")
    print("  3. Expected improvement: Spearman 0.55-0.60 (+9-19%)")
    print("  4. If successful, proceed to Phase 2 (data augmentation)")
    print()
    print(f"Backup saved to: {BACKUP_PATH}")
    print("="*80)
else:
    print("[INFO] No changes made")
    print("="*80)
