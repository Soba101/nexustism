"""
Add HuggingFace authentication cell to evaluate_model.ipynb
"""

import json
from pathlib import Path

def main():
    notebook_path = Path('evaluate_model.ipynb')

    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell after 'f1977da6' (baseline evaluation)
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == 'f1977da6':
            insert_index = i + 1
            break

    if insert_index is None:
        print("ERROR: Could not find cell 'f1977da6'")
        return

    print(f"Found insertion point at index {insert_index}")

    # Create markdown header cell
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6a. HuggingFace Authentication\n",
            "\n",
            "**Required for gated models** like `EmbeddingGemma-300M`.\n",
            "\n",
            "**One-time setup steps:**\n",
            "1. Visit https://huggingface.co/google/embeddinggemma-300m\n",
            "2. Click \"Agree and access repository\" (accept terms)\n",
            "3. Get your token: https://huggingface.co/settings/tokens\n",
            "4. Run the cell below and paste your token when prompted\n",
            "\n",
            "**Note:** This only needs to be done once. After authentication, the token is cached."
        ]
    }

    # Create code cell for authentication
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Authenticate with HuggingFace to access gated models\n",
            "from huggingface_hub import login\n",
            "\n",
            "# Option 1: Interactive login (RECOMMENDED for first time)\n",
            "# Uncomment and run this to be prompted for your token:\n",
            "# login()\n",
            "\n",
            "# Option 2: Direct token (if you already have it)\n",
            "# Replace 'hf_xxxxx' with your actual token:\n",
            "# login(token=\"hf_xxxxxxxxxxxxxxxxxxxxx\")\n",
            "\n",
            "print(\"\\nTo authenticate:\")\n",
            "print(\"  1. Uncomment one of the login() lines above\")\n",
            "print(\"  2. Re-run this cell\")\n",
            "print(\"  3. Follow the prompts to paste your token\")\n",
            "print(\"\\nAfter authentication, re-run cell 6b to load EmbeddingGemma\")"
        ]
    }

    # Insert cells
    nb['cells'].insert(insert_index, markdown_cell)
    nb['cells'].insert(insert_index + 1, code_cell)

    # Write back
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Successfully added HuggingFace authentication cells!")
    print(f"\nNext steps:")
    print(f"  1. Open evaluate_model.ipynb")
    print(f"  2. Find the new section '6a. HuggingFace Authentication'")
    print(f"  3. Follow the instructions in that cell")
    print(f"  4. After authentication, re-run cell 6b")

if __name__ == '__main__':
    main()
