# Cell 16 Curriculum Training - Manual Fix Required

## Problem

Cell 16 has a `KeyError: 'curriculum_phases'` because it's trying to access `CONFIG['curriculum_phases']` which doesn't exist.

The curriculum data is actually loaded in Cell 12 into the `CURRICULUM_PHASES` dictionary:

```python
CURRICULUM_PHASES = {
    'phase1': phase1_train,  # List of InputExample objects
    'phase2': phase2_train,
    'phase3': phase3_train
}
```

## What Cell 16 Currently Does (BROKEN)

```python
if CONFIG['use_curriculum']:
    for phase_idx, phase in enumerate(CONFIG['curriculum_phases']):  # <-- ERROR HERE
        # phase['epochs'] doesn't exist
        # phase['hard_neg_ratio'] doesn't exist
```

## What Cell 16 SHOULD Do

```python
if CONFIG['use_curriculum']:
    # Use pre-loaded CURRICULUM_PHASES from Cell 12
    for phase_idx, (phase_name, phase_examples) in enumerate(sorted(CURRICULUM_PHASES.items())):
        log(f"\n{'='*60}")
        log(f"Phase {phase_idx + 1} ({phase_name}): {CONFIG['epochs_per_phase']} epochs")
        log(f"Training examples: {len(phase_examples):,}")
        log(f"{'='*60}")

        # Create DataLoader for this phase
        phase_dataloader = DataLoader(
            phase_examples,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0
        )

        # Train this phase
        log(f"\nTraining {phase_name}...")
        model.fit(
            train_objectives=[(phase_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=CONFIG['epochs_per_phase'],  # 4 epochs per phase
            warmup_steps=warmup_steps,
            optimizer_params={'lr': CONFIG['lr']},
            output_path=str(save_path),
            evaluation_steps=eval_steps,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=use_amp,
        )
```

## Automated Fix Scripts Created

1. ✅ `fix_v3_curriculum_simple.py` - Fixed iteration and CONFIG references
2. ✅ `fix_v3_phase_training_loop.py` - Changed to phase_dataloader

## Manual Steps Still Needed

The automated scripts made partial fixes, but Cell 16 still needs manual editing in Jupyter:

### Step 1: Find the Curriculum Training Section

Around line 27 in Cell 16, you'll see:

```python
for phase_idx, (phase_name, phase_examples) in enumerate(sorted(CURRICULUM_PHASES.items())):
    log(f"\n{'='*60}")
    log(f"Phase {phase_idx + 1}: {CONFIG['epochs_per_phase']} epochs")
    # COMMENTED: log(f"   Hard neg ratio: {phase['hard_neg_ratio']*100:.0f}%")
```

### Step 2: Add DataLoader Creation

Right after the logging lines (around line 31), add:

```python
        # Create DataLoader for this phase
        phase_dataloader = DataLoader(
            phase_examples,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0
        )

        log(f"Phase {phase_idx + 1} DataLoader: {len(phase_dataloader)} batches")
```

### Step 3: Verify model.fit() Uses phase_dataloader

Find the `model.fit()` call (should be around line 50-70 in the curriculum section).

Make sure the `train_objectives` line uses `phase_dataloader`:

```python
model.fit(
    train_objectives=[(phase_dataloader, train_loss)],  # <-- phase_dataloader, not train_dataloader
    evaluator=evaluator,
    epochs=CONFIG['epochs_per_phase'],  # <-- not phase['epochs']
    warmup_steps=warmup_steps,
    optimizer_params={'lr': CONFIG['lr']},
    ...
)
```

## Alternative: Complete Cell Replacement

If manual editing is too complex, I can generate a complete replacement for the curriculum training section of Cell 16.

This would involve:
1. Reading the original notebook
2. Extracting Cell 16
3. Completely rewriting the curriculum training block (lines ~25-95)
4. Preserving the error handler and fallback code (lines 96+)

Let me know if you want me to create a complete Cell 16 replacement script.

## Quick Test

After fixing, test with:

```python
# In Jupyter, before running Cell 16, check:
print("CURRICULUM_PHASES keys:", list(CURRICULUM_PHASES.keys()))
print("Phase 1 size:", len(CURRICULUM_PHASES['phase1']))
print("CONFIG epochs_per_phase:", CONFIG['epochs_per_phase'])
```

Should show:
```
CURRICULUM_PHASES keys: ['phase1', 'phase2', 'phase3']
Phase 1 size: 5000
CONFIG epochs_per_phase: 2
```

If this works, then Cell 16 should work after the manual fixes above.
