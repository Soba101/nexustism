#!/usr/bin/env python3
"""
Complete fix for Cell 16 curriculum training section.

Replaces the broken curriculum training loop with a working version that uses:
- CURRICULUM_PHASES dictionary from Cell 12
- CONFIG['epochs_per_phase'] instead of phase['epochs']
- phase_examples for DataLoader creation
"""
import json

NOTEBOOK_PATH = 'model_promax_mpnet_lorapeft_v3.ipynb'

# The correct curriculum training code
CURRICULUM_TRAINING_CODE = """
    if CONFIG['use_curriculum']:
        # V2: Curriculum Learning - train in phases
        log("\\n[CURRICULUM] Training in 3 phases (easy -> medium -> hard)")

        for phase_idx, (phase_name, phase_examples) in enumerate(sorted(CURRICULUM_PHASES.items())):
            log(f"\\n{'='*60}")
            log(f"[PHASE {phase_idx + 1}] {phase_name.upper()}: {CONFIG['epochs_per_phase']} epochs")
            log(f"   Training examples: {len(phase_examples):,}")
            log(f"{'='*60}")

            # Create DataLoader for this phase
            phase_dataloader = DataLoader(
                phase_examples,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=0
            )

            log(f"   Batches per epoch: {len(phase_dataloader)}")
            log(f"   Total steps this phase: {len(phase_dataloader) * CONFIG['epochs_per_phase']}")

            # Train this phase
            log(f"\\n[TRAINING] {phase_name}...")
            model.fit(
                train_objectives=[(phase_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=CONFIG['epochs_per_phase'],
                warmup_steps=warmup_steps,
                optimizer_params={'lr': CONFIG['lr']},
                output_path=str(save_path),
                evaluation_steps=eval_steps,
                save_best_model=True,
                show_progress_bar=True,
                use_amp=use_amp,
            )

            log(f"[OK] {phase_name} complete!")

        log(f"\\n{'='*70}")
        log("[SUCCESS] All curriculum phases complete!")
        log(f"{'='*70}")

    else:
        # Non-curriculum: train on all examples together
        log("\\n[TRAINING] Standard training (no curriculum)")
        train_dataloader = DataLoader(
            train_examples,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0
        )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=CONFIG['epochs'],
            warmup_steps=warmup_steps,
            optimizer_params={'lr': CONFIG['lr']},
            output_path=str(save_path),
            evaluation_steps=eval_steps,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=use_amp,
        )
"""

print("="*80)
print("COMPLETE FIX FOR CELL 16 CURRICULUM TRAINING")
print("="*80)

# Load notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell16 = nb['cells'][16]
lines = cell16['source']

print(f"\\nOriginal Cell 16: {len(lines)} lines")

# Find where curriculum training starts and ends
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if "if CONFIG['use_curriculum']:" in line and start_idx is None:
        start_idx = i
        print(f"Found curriculum training start at line {i+1}")

    if start_idx is not None and "except RuntimeError" in line:
        end_idx = i
        print(f"Found curriculum training end at line {i}")
        break

if start_idx is None:
    print("[ERROR] Could not find curriculum training section!")
    exit(1)

if end_idx is None:
    print("[WARNING] Could not find end marker, using end of cell")
    end_idx = len(lines)

print(f"\\nReplacing lines {start_idx+1} to {end_idx}")
print(f"Old section: {end_idx - start_idx} lines")

# Build new cell
new_lines = []

# Keep everything before curriculum training
new_lines.extend(lines[:start_idx])

# Add new curriculum training code
new_curriculum_lines = CURRICULUM_TRAINING_CODE.split('\\n')
new_lines.extend([line + '\\n' for line in new_curriculum_lines])

# Keep everything after (error handlers, etc.)
new_lines.extend(lines[end_idx:])

print(f"New section: {len(new_curriculum_lines)} lines")
print(f"New Cell 16: {len(new_lines)} lines")

# Update cell
cell16['source'] = new_lines
nb['cells'][16] = cell16

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\\n{'='*80}")
print("[SUCCESS] Cell 16 curriculum training rewritten!")
print(f"{'='*80}")
print("\\nNew training flow:")
print("  1. Iterates over sorted(CURRICULUM_PHASES.items())")
print("  2. Creates phase_dataloader for each phase")
print("  3. Trains each phase for CONFIG['epochs_per_phase'] epochs")
print("  4. Uses CONFIG['lr'] for learning rate")
print("  5. Saves best model at each phase")
print(f"\\n{'='*80}")
print("Ready to train! Run Cell 16 in Jupyter.")
print(f"{'='*80}")
