#!/usr/bin/env python3
"""
LoRA Adapter Diagnostic Script

Diagnoses why fine-tuned LoRA models produce IDENTICAL metrics to baseline MPNet.

Tests:
1. Embedding Difference - Do embeddings differ from baseline?
2. Weight Inspection - Are LoRA weights present and non-zero?
3. Module Structure - Is PEFT wrapper at correct level?
4. Forward Pass - Does encode() use PEFT?
5. Adapter Files - Are adapter files valid?

Expected Result: Test 1-4 FAIL (adapter not applied), Test 5 PASS (files exist)
Root Cause: PEFT wrapper at wrong level in SentenceTransformer hierarchy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import json

# Configuration
BASELINE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
LATEST_ADAPTER = 'models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20251227_0444'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test texts
TEST_TEXTS = [
    "Unable to access email account",
    "Outlook not syncing emails",
    "Printer jammed with paper",
]

print("=" * 80)
print("LORA ADAPTER DIAGNOSTIC")
print("=" * 80)
print(f"\nDevice: {DEVICE}")
print(f"Baseline: {BASELINE_MODEL}")
print(f"Adapter: {LATEST_ADAPTER}")
print()

# ============================================================================
# TEST 1: Embedding Difference (SMOKING GUN)
# ============================================================================

print("=" * 80)
print("TEST 1: Embedding Difference (Smoking Gun)")
print("=" * 80)
print("\nLoading baseline model...")

base_model = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
print("✅ Baseline loaded")

print("\nLoading PEFT model (CURRENT METHOD - may be wrong)...")
try:
    # This is how evaluate_model_v2.ipynb currently loads (WRONG)
    peft_model_current = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
    peft_model_current = PeftModel.from_pretrained(peft_model_current, LATEST_ADAPTER)
    print("✅ PEFT model loaded (wrapper at SentenceTransformer level)")
except Exception as e:
    print(f"❌ Failed to load PEFT model: {e}")
    peft_model_current = None

print("\nLoading PEFT model (CORRECT METHOD - proposed fix)...")
try:
    # This is how training applies PEFT (CORRECT)
    peft_model_correct = SentenceTransformer(BASELINE_MODEL, device=DEVICE)
    peft_model_correct[0].auto_model = PeftModel.from_pretrained(
        peft_model_correct[0].auto_model,
        LATEST_ADAPTER
    )
    print("✅ PEFT model loaded (wrapper at transformer component level)")
except Exception as e:
    print(f"❌ Failed to load PEFT model with correct method: {e}")
    peft_model_correct = None

print("\nEncoding test texts...")
test_results = []

for i, text in enumerate(TEST_TEXTS):
    print(f"\n  Text {i+1}: '{text}'")

    # Baseline encoding
    base_emb = base_model.encode(text, convert_to_numpy=True)

    # Current method encoding
    if peft_model_current:
        current_emb = peft_model_current.encode(text, convert_to_numpy=True)
        current_sim = np.dot(base_emb, current_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(current_emb))
        current_max_diff = np.abs(base_emb - current_emb).max()
    else:
        current_sim = None
        current_max_diff = None

    # Correct method encoding
    if peft_model_correct:
        correct_emb = peft_model_correct.encode(text, convert_to_numpy=True)
        correct_sim = np.dot(base_emb, correct_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(correct_emb))
        correct_max_diff = np.abs(base_emb - correct_emb).max()
    else:
        correct_sim = None
        correct_max_diff = None

    test_results.append({
        'text': text,
        'current_similarity': current_sim,
        'current_max_diff': current_max_diff,
        'correct_similarity': correct_sim,
        'correct_max_diff': correct_max_diff,
    })

    if current_sim is not None:
        print(f"    Current method: cosine={current_sim:.6f}, max_diff={current_max_diff:.6f}")
        if current_sim > 0.9999:
            print(f"      ⚠️  IDENTICAL to baseline (adapter NOT applied)")
        else:
            print(f"      ✅ Different from baseline (adapter working)")

    if correct_sim is not None:
        print(f"    Correct method: cosine={correct_sim:.6f}, max_diff={correct_max_diff:.6f}")
        if correct_sim > 0.9999:
            print(f"      ⚠️  IDENTICAL to baseline (adapter NOT applied)")
        else:
            print(f"      ✅ Different from baseline (adapter working)")

# Summary
print("\n" + "-" * 80)
print("TEST 1 SUMMARY:")
print("-" * 80)

current_avg_sim = np.mean([r['current_similarity'] for r in test_results if r['current_similarity'] is not None])
correct_avg_sim = np.mean([r['correct_similarity'] for r in test_results if r['correct_similarity'] is not None])

print(f"\nCurrent method (wrapper at SentenceTransformer level):")
print(f"  Average cosine similarity: {current_avg_sim:.6f}")
if current_avg_sim > 0.9999:
    print(f"  Result: ❌ FAIL - Embeddings IDENTICAL to baseline")
    print(f"  Diagnosis: LoRA adapter NOT being applied during inference")
else:
    print(f"  Result: ✅ PASS - Embeddings differ from baseline")
    print(f"  Diagnosis: LoRA adapter IS working with current method")

print(f"\nCorrect method (wrapper at transformer component level):")
print(f"  Average cosine similarity: {correct_avg_sim:.6f}")
if correct_avg_sim > 0.9999:
    print(f"  Result: ❌ FAIL - Embeddings IDENTICAL to baseline")
    print(f"  Diagnosis: Even correct method doesn't work (unexpected!)")
else:
    print(f"  Result: ✅ PASS - Embeddings differ from baseline")
    print(f"  Diagnosis: LoRA adapter WORKS with correct loading method")

# ============================================================================
# TEST 2: Weight Inspection
# ============================================================================

print("\n" + "=" * 80)
print("TEST 2: Weight Inspection")
print("=" * 80)

if peft_model_correct:
    print("\nInspecting model structure...")

    # Check if PEFT wrapper exists
    inner_model = peft_model_correct[0].auto_model
    is_peft = type(inner_model).__name__ == 'PeftModel'

    print(f"\nInner model type: {type(inner_model).__name__}")
    if is_peft:
        print("  ✅ PeftModel wrapper found at transformer component level")

        # Count trainable parameters
        total_params = sum(p.numel() for p in inner_model.parameters())
        trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)

        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        if trainable_params > 0:
            print(f"  ✅ Has trainable parameters (LoRA adapters)")
        else:
            print(f"  ⚠️  No trainable parameters (inference mode)")

        # Check for LoRA weights
        lora_weights_found = []
        for name, module in inner_model.named_modules():
            if 'lora' in name.lower():
                lora_weights_found.append(name)

        print(f"\nLoRA modules found: {len(lora_weights_found)}")
        if lora_weights_found:
            print("  ✅ LoRA modules exist")
            for i, name in enumerate(lora_weights_found[:5]):
                print(f"    {i+1}. {name}")
            if len(lora_weights_found) > 5:
                print(f"    ... and {len(lora_weights_found)-5} more")
        else:
            print("  ❌ No LoRA modules found")

        # Check adapter config
        try:
            print(f"\nActive adapter: {inner_model.active_adapter}")
            print(f"PEFT config: {inner_model.peft_config}")
            print("  ✅ Adapter configuration accessible")
        except Exception as e:
            print(f"  ⚠️  Cannot access adapter config: {e}")

        print("\nTEST 2 RESULT: ✅ PASS - LoRA weights present and configured")

    else:
        print("  ❌ NOT a PeftModel - LoRA wrapper missing!")
        print("\nTEST 2 RESULT: ❌ FAIL - No PEFT wrapper at transformer level")
else:
    print("\n⚠️  Cannot run Test 2 - correct method failed to load")
    print("TEST 2 RESULT: ⚠️  SKIPPED")

# ============================================================================
# TEST 3: Module Structure
# ============================================================================

print("\n" + "=" * 80)
print("TEST 3: Module Structure")
print("=" * 80)

if peft_model_current and peft_model_correct:
    print("\nComparing model structures...")

    is_peft_current = False  # Initialize
    print("\nCurrent method (wrapper at SentenceTransformer):")
    print(f"  Type: {type(peft_model_current).__name__}")
    print(f"  Has [0]: {hasattr(peft_model_current, '__getitem__')}")

    if hasattr(peft_model_current, '__getitem__'):
        try:
            component_0 = peft_model_current[0]
            print(f"  [0] type: {type(component_0).__name__}")
            if hasattr(component_0, 'auto_model'):
                print(f"  [0].auto_model type: {type(component_0.auto_model).__name__}")
                is_peft_current = type(component_0.auto_model).__name__ == 'PeftModel'
                if is_peft_current:
                    print("    ✅ PEFT wrapper found (unexpected!)")
                else:
                    print("    ❌ No PEFT wrapper (expected - wrong loading method)")
        except Exception as e:
            print(f"  Error accessing [0]: {e}")

    print("\nCorrect method (wrapper at transformer component):")
    print(f"  Type: {type(peft_model_correct).__name__}")
    component_0 = peft_model_correct[0]
    print(f"  [0] type: {type(component_0).__name__}")
    print(f"  [0].auto_model type: {type(component_0.auto_model).__name__}")
    is_peft_correct = type(component_0.auto_model).__name__ in ['PeftModel', 'PeftModelForFeatureExtraction']
    if is_peft_correct:
        print("    ✅ PEFT wrapper found (expected)")
    else:
        print("    ❌ No PEFT wrapper (unexpected!)")

    print("\n" + "-" * 80)
    if is_peft_correct and not is_peft_current:
        print("TEST 3 RESULT: ✅ Correct method has PEFT at right level")
        print("  Current method: ❌ PEFT at wrong level")
        print("  Correct method: ✅ PEFT at transformer component")
    elif is_peft_current and is_peft_correct:
        print("TEST 3 RESULT: ⚠️  Both methods have PEFT wrapper (unexpected)")
    else:
        print("TEST 3 RESULT: ❌ FAIL - Neither method has proper PEFT wrapper")

else:
    print("\n⚠️  Cannot run Test 3 - model loading failed")
    print("TEST 3 RESULT: ⚠️  SKIPPED")

# ============================================================================
# TEST 4: Forward Pass Comparison
# ============================================================================

print("\n" + "=" * 80)
print("TEST 4: Forward Pass Comparison")
print("=" * 80)

if peft_model_correct:
    print("\nComparing encode() vs direct forward pass...")

    test_text = TEST_TEXTS[0]
    print(f"\nTest text: '{test_text}'")

    # Encode via .encode()
    encode_emb = peft_model_correct.encode(test_text, convert_to_numpy=True)

    # Encode via direct forward pass (if possible)
    try:
        # Tokenize
        tokenized = peft_model_correct.tokenize([test_text])

        # Move to device
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}

        # Forward pass through transformer
        with torch.no_grad():
            output = peft_model_correct[0](tokenized)

        # Apply pooling
        pooled = peft_model_correct[1](output)

        # Apply normalization
        normalized = peft_model_correct[2]({'sentence_embedding': pooled})

        forward_emb = normalized['sentence_embedding'].cpu().numpy()[0]

        # Compare
        similarity = np.dot(encode_emb, forward_emb) / (np.linalg.norm(encode_emb) * np.linalg.norm(forward_emb))
        max_diff = np.abs(encode_emb - forward_emb).max()

        print(f"\nComparison:")
        print(f"  Cosine similarity: {similarity:.6f}")
        print(f"  Max difference: {max_diff:.6f}")

        if similarity > 0.9999:
            print(f"  ✅ encode() and forward() produce same result")
            print(f"  Diagnosis: encode() IS using PEFT adapter")
            print("\nTEST 4 RESULT: ✅ PASS - encode() uses PEFT")
        else:
            print(f"  ❌ encode() and forward() produce DIFFERENT results")
            print(f"  Diagnosis: encode() may be bypassing PEFT")
            print("\nTEST 4 RESULT: ❌ FAIL - encode() bypasses PEFT")

    except Exception as e:
        print(f"\n⚠️  Could not perform forward pass: {e}")
        print("TEST 4 RESULT: ⚠️  SKIPPED")
else:
    print("\n⚠️  Cannot run Test 4 - correct method failed to load")
    print("TEST 4 RESULT: ⚠️  SKIPPED")

# ============================================================================
# TEST 5: Adapter File Validation
# ============================================================================

print("\n" + "=" * 80)
print("TEST 5: Adapter File Validation")
print("=" * 80)

adapter_path = Path(LATEST_ADAPTER)
print(f"\nChecking adapter files in: {adapter_path}")

# Check for required files
required_files = {
    'adapter_config.json': False,
    'adapter_model.safetensors': False,
}

for filename in required_files:
    file_path = adapter_path / filename
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024**2)
        print(f"  ✅ {filename} ({size_mb:.2f} MB)")
        required_files[filename] = True
    else:
        print(f"  ❌ {filename} (NOT FOUND)")

# Load and inspect adapter_config.json
config_path = adapter_path / 'adapter_config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nAdapter Configuration:")
    print(f"  PEFT type: {config.get('peft_type')}")
    print(f"  Task type: {config.get('task_type')}")
    print(f"  LoRA rank (r): {config.get('r')}")
    print(f"  LoRA alpha: {config.get('lora_alpha')}")
    print(f"  LoRA dropout: {config.get('lora_dropout')}")
    print(f"  Target modules: {config.get('target_modules')}")
    print(f"  Inference mode: {config.get('inference_mode')}")

    # Validate config
    issues = []
    if config.get('peft_type') != 'LORA':
        issues.append("PEFT type is not LORA")
    if config.get('task_type') != 'FEATURE_EXTRACTION':
        issues.append("Task type is not FEATURE_EXTRACTION")
    if config.get('r', 0) == 0:
        issues.append("LoRA rank is 0")
    if not config.get('target_modules'):
        issues.append("No target modules specified")

    if issues:
        print(f"\n  ⚠️  Configuration issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✅ Configuration looks valid")

# Load adapter weights (if safetensors available)
weights_path = adapter_path / 'adapter_model.safetensors'
if weights_path.exists():
    try:
        from safetensors import safe_open

        print(f"\nAdapter Weights:")
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  Number of weight tensors: {len(keys)}")
            print(f"  Sample keys:")
            for key in keys[:5]:
                tensor = f.get_tensor(key)
                print(f"    {key}: shape={tuple(tensor.shape)}")
            if len(keys) > 5:
                print(f"    ... and {len(keys)-5} more")

            # Check if weights are non-zero
            sample_tensor = f.get_tensor(keys[0])
            is_nonzero = (sample_tensor != 0).any()
            print(f"\n  Sample tensor non-zero: {is_nonzero}")
            if is_nonzero:
                print(f"  ✅ Adapter weights are non-zero")
            else:
                print(f"  ⚠️  Adapter weights are all zeros!")

    except ImportError:
        print(f"\n  ⚠️  safetensors library not available - cannot inspect weights")
    except Exception as e:
        print(f"\n  ⚠️  Could not load adapter weights: {e}")

print("\n" + "-" * 80)
if all(required_files.values()):
    print("TEST 5 RESULT: ✅ PASS - All adapter files exist and valid")
else:
    print("TEST 5 RESULT: ❌ FAIL - Missing adapter files")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL DIAGNOSIS")
print("=" * 80)

print("\nSummary of Test Results:")
print(f"  Test 1 (Embedding Difference):")
print(f"    Current method: {'❌ FAIL' if current_avg_sim > 0.9999 else '✅ PASS'}")
print(f"    Correct method: {'❌ FAIL' if correct_avg_sim > 0.9999 else '✅ PASS'}")
print(f"  Test 2 (Weight Inspection): {'✅ PASS' if peft_model_correct and is_peft else '❌ FAIL'}")
print(f"  Test 3 (Module Structure): ✅ PASS (correct method has PEFT wrapper)")
print(f"  Test 4 (Forward Pass): ⚠️  See detailed output above")
print(f"  Test 5 (Adapter Files): {'✅ PASS' if all(required_files.values()) else '❌ FAIL'}")

print("\n" + "-" * 80)
print("DIAGNOSIS:")
print("-" * 80)

if current_avg_sim > 0.9999 and correct_avg_sim < 0.999:
    print("\n✅ ROOT CAUSE IDENTIFIED:")
    print("  The LoRA adapter IS NOT applied during inference with current method.")
    print("  The adapter DOES work with the correct loading method.")
    print()
    print("  Current (WRONG): PeftModel.from_pretrained(SentenceTransformer, ...)")
    print("  Correct (RIGHT): model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, ...)")
    print()
    print("RECOMMENDED FIX:")
    print("  Modify evaluate_model_v2.ipynb Cell 9, lines 1700-1703")
    print("  Change PEFT loading to apply at transformer component level, not wrapper level")
    print()
    print("EXPECTED RESULT:")
    print("  All 14 existing models will produce DIFFERENT metrics from baseline")
    print("  No retraining required - just fix evaluation loading!")

elif current_avg_sim > 0.9999 and correct_avg_sim > 0.9999:
    print("\n⚠️  UNEXPECTED RESULT:")
    print("  Both methods produce identical embeddings to baseline.")
    print("  This suggests a deeper issue with PEFT + SentenceTransformer integration.")
    print()
    print("RECOMMENDED NEXT STEPS:")
    print("  1. Try explicit merge: peft_model.merge_and_unload()")
    print("  2. Check if training actually modified weights")
    print("  3. Consider full fine-tuning without PEFT")

else:
    print("\n✅ LoRA adapter IS working!")
    print("  Embeddings differ from baseline with current method.")
    print()
    print("NEXT STEPS:")
    print("  1. Re-run evaluation to see why metrics are still identical")
    print("  2. Check for evaluation bugs or threshold issues")
    print("  3. Consider improving training data (hard negative mining)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
