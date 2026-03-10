"""
Patch training notebooks to use V6 resnotes-based curriculum pairs.

Changes:
  Nomic notebook  (model_promax_nomic_lorapeft_v4_semantic_mnrl.ipynb):
    - CONFIG: train_pairs_path → resnotes_curriculum_training_pairs_v6.json
    - CONFIG: eval_pairs_path  → benchmark_v4_semantic_resnotes.json
    - loader: detect flat format + prepend "search_document: " prefix for Nomic
    - eval  : add "search_document: " prefix for Nomic
    - labels: update V5.3 version strings → V6.0

  MPNet notebook  (model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb):
    - CONFIG: train_pairs_path → resnotes_curriculum_training_pairs_v6.json
    - CONFIG: eval_pairs_path  → benchmark_v4_semantic_resnotes.json
    - loader: detect flat format (no prefix needed for MPNet)
    - eval  : (no prefix change needed)
    - labels: update V5.2 version strings → V6.0
"""

import json
import re
import shutil
from pathlib import Path

NEXUSTISM = Path(__file__).parent.parent

NEW_TRAIN = "data_new/resnotes_curriculum_training_pairs_v6.json"
NEW_EVAL  = "data_new/benchmark_v4_semantic_resnotes.json"


def patch_cell_source(source_lines: list[str], replacements: list[tuple[str, str]]) -> list[str]:
    """Apply (old, new) string replacements to notebook cell source lines."""
    full = "".join(source_lines)
    for old, new in replacements:
        if old in full:
            full = full.replace(old, new, 1)
        else:
            print(f"  [WARN] Pattern not found:\n    {repr(old[:80])}")
    return list(full.splitlines(keepends=True)) if source_lines else []


def patch_notebook(nb_path: Path, per_cell_patches: list[dict]) -> None:
    """
    per_cell_patches is a list of dicts:
        {
            "match": str          # unique substring to identify the right cell
            "replacements": [(old, new), ...]
        }
    """
    shutil.copy(nb_path, nb_path.with_suffix(".ipynb.bak"))

    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    for patch in per_cell_patches:
        match_str = patch["match"]
        replacements = patch["replacements"]

        found = False
        for cell in nb["cells"]:
            src = cell.get("source", [])
            if isinstance(src, list):
                full = "".join(src)
            else:
                full = src

            if match_str in full:
                found = True
                new_src = patch_cell_source(
                    src if isinstance(src, list) else [src],
                    replacements
                )
                cell["source"] = new_src
                print(f"  [OK] Patched cell matching: {repr(match_str[:60])}")
                break

        if not found:
            print(f"  [WARN] No cell matched: {repr(match_str[:60])}")

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Saved: {nb_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Nomic notebook patches
# ─────────────────────────────────────────────────────────────────────────────

NOMIC_NB = NEXUSTISM / "model_promax_nomic_lorapeft_v4_semantic_mnrl.ipynb"

nomic_patches = [
    # 1. CONFIG cell: update paths + version label
    {
        "match": "'train_pairs_path': 'data_new/group_based_training_pairs_v5_3_nomic.json'",
        "replacements": [
            (
                "'train_pairs_path': 'data_new/group_based_training_pairs_v5_3_nomic.json',",
                "'train_pairs_path': 'data_new/resnotes_curriculum_training_pairs_v6.json',  # V6: resnotes distant supervision",
            ),
            (
                "'eval_pairs_path':  'data_new/semantic_test_pairs_v3.json',",
                "'eval_pairs_path':  'data_new/benchmark_v4_semantic_resnotes.json',",
            ),
            # Update version description in log strings
            (
                'log("CONFIGURATION (V5.3 - Nomic Base, Group-Based Training)")',
                'log("CONFIGURATION (V6.0 - Nomic - ResNotes Distant Supervision)")',
            ),
            (
                'log(f"\\nData (V5.3 \u2014 Group-Based + Category Labels, gate=0.30):")',
                'log(f"\\nData (V6.0 \u2014 ResNotes Distant Supervision):")',
            ),
        ],
    },
    # 2. Loader function: add flat-format + Nomic prefix handling
    {
        "match": "def load_or_generate_group_pairs(config: dict, train_df):",
        "replacements": [
            (
                "    if pairs_path.exists():\n"
                "        log(f\"[LOAD] Loading group-based pairs \u2192 {pairs_path}\")\n"
                "        with open(pairs_path) as f:\n"
                "            raw = json.load(f)\n"
                "        phases: dict = {}\n"
                "        for key in ('phase1', 'phase2', 'phase3'):\n"
                "            if key in raw:\n"
                "                d = raw[key]\n"
                "                phases[key] = [\n"
                "                    InputExample(texts=[t1, t2], label=float(lb))\n"
                "                    for t1, t2, lb in zip(d['texts1'], d['texts2'], d['labels'])\n"
                "                ]\n"
                "                log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
                "        return phases\n",

                "    if pairs_path.exists():\n"
                "        log(f\"[LOAD] Loading training pairs \u2192 {pairs_path}\")\n"
                "        with open(pairs_path) as f:\n"
                "            raw = json.load(f)\n"
                "        phases: dict = {}\n"
                "        # V6+ flat format: {texts1, texts2, labels, phase_indicators}\n"
                "        if 'phase_indicators' in raw:\n"
                "            NOMIC_PREFIX = 'search_document: '  # Nomic asymmetric embedding\n"
                "            for phase_num, key in ((1, 'phase1'), (2, 'phase2'), (3, 'phase3')):\n"
                "                idxs = [i for i, p in enumerate(raw['phase_indicators']) if p == phase_num]\n"
                "                phases[key] = [\n"
                "                    InputExample(\n"
                "                        texts=[NOMIC_PREFIX + raw['texts1'][i], NOMIC_PREFIX + raw['texts2'][i]],\n"
                "                        label=float(raw['labels'][i])\n"
                "                    )\n"
                "                    for i in idxs\n"
                "                ]\n"
                "                log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
                "        else:\n"
                "            # Legacy phased format (V5.x)\n"
                "            for key in ('phase1', 'phase2', 'phase3'):\n"
                "                if key in raw:\n"
                "                    d = raw[key]\n"
                "                    phases[key] = [\n"
                "                        InputExample(texts=[t1, t2], label=float(lb))\n"
                "                        for t1, t2, lb in zip(d['texts1'], d['texts2'], d['labels'])\n"
                "                    ]\n"
                "                    log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
                "        return phases\n",
            ),
        ],
    },
    # 3. Eval pairs loader: update comment + default path + add Nomic prefix
    {
        "match": "# 2. Evaluation pairs \u2014 semantic_test_pairs_v3",
        "replacements": [
            (
                "# 2. Evaluation pairs \u2014 semantic_test_pairs_v3 (group-based, NOT MPNet-cosine)",
                "# 2. Evaluation pairs \u2014 benchmark_v4_semantic_resnotes (grounded benchmark)",
            ),
            (
                "_eval_path = Path(CONFIG.get('eval_pairs_path', 'data_new/semantic_test_pairs_v3.json'))",
                "_eval_path = Path(CONFIG.get('eval_pairs_path', 'data_new/benchmark_v4_semantic_resnotes.json'))",
            ),
            (
                'log(f"\\n[LOAD] Eval pairs (group-based): {_eval_path}")',
                'log(f"\\n[LOAD] Eval pairs (benchmark): {_eval_path}")',
            ),
            # Add Nomic prefix to benchmark list format (text1/text2 keys)
            (
                "        InputExample(texts=[p['text1'], p['text2']], label=float(p['label']))\n"
                "        for p in _v3_raw\n",
                "        InputExample(texts=['search_document: ' + p['text1'],\n"
                "                           'search_document: ' + p['text2']], label=float(p['label']))\n"
                "        for p in _v3_raw\n",
            ),
            # Add Nomic prefix to dict format fallback (texts1/texts2 keys)
            (
                "        InputExample(texts=[t1, t2], label=float(lb))\n"
                "        for t1, t2, lb in zip(_v3_raw['texts1'], _v3_raw['texts2'], _v3_raw['labels'])\n",
                "        InputExample(texts=['search_document: ' + t1, 'search_document: ' + t2], label=float(lb))\n"
                "        for t1, t2, lb in zip(_v3_raw['texts1'], _v3_raw['texts2'], _v3_raw['labels'])\n",
            ),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MPNet notebook patches
# ─────────────────────────────────────────────────────────────────────────────

MPNET_NB = NEXUSTISM / "model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb"

mpnet_patches = [
    # 1. CONFIG cell: update paths
    {
        "match": "'train_pairs_path': 'data_new/group_based_training_pairs_v5_2.json'",
        "replacements": [
            (
                "'train_pairs_path': 'data_new/group_based_training_pairs_v5_2.json',  # V5.2: gate 0.30 + category",
                "'train_pairs_path': 'data_new/resnotes_curriculum_training_pairs_v6.json',  # V6: resnotes distant supervision",
            ),
        ],
    },
]


def patch_mpnet_eval_path(nb_path: Path) -> None:
    """Patch eval_pairs_path in MPNet CONFIG (may be on a different line from train path)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = cell.get("source", [])
        full = "".join(src) if isinstance(src, list) else src
        if "'eval_pairs_path'" in full and "semantic_test_pairs" in full and "model_name" in full:
            full_new = re.sub(
                r"'eval_pairs_path'\s*:\s*'data_new/[^']+\.json'",
                f"'eval_pairs_path':  '{NEW_EVAL}'",
                full,
            )
            if full_new != full:
                cell["source"] = list(full_new.splitlines(keepends=True))
                print(f"  [OK] Patched MPNet eval_pairs_path")
            break
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_mpnet_loader(nb_path: Path) -> None:
    """Add flat-format support to MPNet's load_or_generate_group_pairs (no prefix needed)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    target_old = (
        "    if pairs_path.exists():\n"
        "        log(f\"[LOAD] Loading group-based pairs \u2192 {pairs_path}\")\n"
        "        with open(pairs_path) as f:\n"
        "            raw = json.load(f)\n"
        "        phases: dict = {}\n"
        "        for key in ('phase1', 'phase2', 'phase3'):\n"
        "            if key in raw:\n"
        "                d = raw[key]\n"
        "                phases[key] = [\n"
        "                    InputExample(texts=[t1, t2], label=float(lb))\n"
        "                    for t1, t2, lb in zip(d['texts1'], d['texts2'], d['labels'])\n"
        "                ]\n"
        "                log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
        "        return phases\n"
    )
    target_new = (
        "    if pairs_path.exists():\n"
        "        log(f\"[LOAD] Loading training pairs \u2192 {pairs_path}\")\n"
        "        with open(pairs_path) as f:\n"
        "            raw = json.load(f)\n"
        "        phases: dict = {}\n"
        "        # V6+ flat format: {texts1, texts2, labels, phase_indicators}\n"
        "        if 'phase_indicators' in raw:\n"
        "            for phase_num, key in ((1, 'phase1'), (2, 'phase2'), (3, 'phase3')):\n"
        "                idxs = [i for i, p in enumerate(raw['phase_indicators']) if p == phase_num]\n"
        "                phases[key] = [\n"
        "                    InputExample(\n"
        "                        texts=[raw['texts1'][i], raw['texts2'][i]],\n"
        "                        label=float(raw['labels'][i])\n"
        "                    )\n"
        "                    for i in idxs\n"
        "                ]\n"
        "                log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
        "        else:\n"
        "            # Legacy phased format (V5.x)\n"
        "            for key in ('phase1', 'phase2', 'phase3'):\n"
        "                if key in raw:\n"
        "                    d = raw[key]\n"
        "                    phases[key] = [\n"
        "                        InputExample(texts=[t1, t2], label=float(lb))\n"
        "                        for t1, t2, lb in zip(d['texts1'], d['texts2'], d['labels'])\n"
        "                    ]\n"
        "                    log(f\"   {key}: {len(phases[key]):,} pairs\")\n"
        "        return phases\n"
    )

    for cell in nb["cells"]:
        src = cell.get("source", [])
        full = "".join(src) if isinstance(src, list) else src
        if "def load_or_generate_group_pairs" in full and target_old in full:
            new_full = full.replace(target_old, target_new, 1)
            cell["source"] = list(new_full.splitlines(keepends=True))
            print("  [OK] Patched MPNet loader function")
            break
    else:
        print("  [WARN] MPNet loader pattern not found — checking alternate format")

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    print("\n=== Patching Nomic notebook ===")
    patch_notebook(NOMIC_NB, nomic_patches)

    print("\n=== Patching MPNet notebook ===")
    patch_notebook(MPNET_NB, mpnet_patches)
    patch_mpnet_eval_path(MPNET_NB)
    patch_mpnet_loader(MPNET_NB)

    print("\nDone. Backups saved as *.ipynb.bak")
