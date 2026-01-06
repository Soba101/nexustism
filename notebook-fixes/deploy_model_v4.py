"""
V4 Cosine Model Deployment Script
==================================

Production deployment script for the V4 Curriculum Cosine model.

Model: real_servicenow_v2_20260104_2321
Performance: Spearman=0.4949, ROC-AUC=0.7857, Adversarial=0.967
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Tuple
import json

class V4CosineModelDeployment:
    """Production-ready deployment wrapper for V4 Cosine model."""

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize V4 Cosine model for production use.

        Args:
            model_path: Path to model directory (default: auto-detect latest)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        # Auto-detect model path if not provided
        if model_path is None:
            # Get absolute path relative to project root
            project_root = Path(__file__).parent.parent
            model_path = project_root / "models" / "real_servicenow_finetuned_mpnet_lora" / "real_servicenow_v2_20260104_2321"

        self.model_path = Path(model_path)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Loading V4 Cosine model from: {self.model_path}")
        print(f"Using device: {self.device}")

        # Load model with LoRA adapters
        self.model = self._load_model()

        # Load metadata
        self.metadata = self._load_metadata()

        print(f"[OK] Model loaded successfully")
        print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {self.model.max_seq_length}")

    def _load_model(self) -> SentenceTransformer:
        """Load the fine-tuned model with LoRA adapters."""
        try:
            from peft import PeftModel

            # Load base model
            base_model = SentenceTransformer(
                'sentence-transformers/all-mpnet-base-v2',
                device=self.device
            )

            # Apply PEFT to transformer component
            base_model[0].auto_model = PeftModel.from_pretrained(
                base_model[0].auto_model,
                str(self.model_path)
            )

            print("   Loaded as PEFT model with LoRA adapters")
            return base_model

        except Exception as e:
            print(f"   Warning: Failed to load as PEFT model: {e}")
            print("   Falling back to standard SentenceTransformer loading")

            # Fallback: load as standard model
            return SentenceTransformer(str(self.model_path), device=self.device)

    def _load_metadata(self) -> dict:
        """Load training metadata if available."""
        metadata_path = self.model_path / "training_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   Loaded metadata: {metadata.get('timestamp', 'unknown')}")
            return metadata
        else:
            print("   No metadata file found")
            return {}

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to model.encode()

        Returns:
            numpy array of embeddings (shape: [len(texts), 768])
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=self.device,
            **kwargs
        )

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1, emb2 = self.encode([text1, text2])

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def find_similar(self, query: str, candidates: List[str],
                    top_k: int = 5, threshold: float = 0.3784) -> List[Tuple[int, str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (default from eval)

        Returns:
            List of (index, text, score) tuples, sorted by score descending
        """
        # Encode query and candidates
        query_emb = self.encode([query])[0]
        candidate_embs = self.encode(candidates)

        # Compute similarities
        similarities = np.array([
            np.dot(query_emb, cand_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb))
            for cand_emb in candidate_embs
        ])

        # Filter by threshold and get top-k
        above_threshold = similarities >= threshold
        filtered_indices = np.where(above_threshold)[0]
        filtered_scores = similarities[above_threshold]

        # Sort by score descending
        sorted_indices = np.argsort(filtered_scores)[::-1][:top_k]

        results = [
            (int(filtered_indices[i]), candidates[filtered_indices[i]], float(filtered_scores[i]))
            for i in sorted_indices
        ]

        return results

    def get_model_info(self) -> dict:
        """Get model information and performance metrics."""
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'embedding_dim': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length,
            'performance': {
                'spearman': 0.4949,
                'roc_auc': 0.7857,
                'f1': 0.7134,
                'precision': 0.6290,
                'recall': 0.8240,
                'adversarial_roc_auc': 0.967,
            },
            'metadata': self.metadata
        }


def main():
    """Example usage of the V4 Cosine model deployment."""
    print("="*80)
    print("V4 COSINE MODEL - PRODUCTION DEPLOYMENT")
    print("="*80)

    # Initialize model
    model = V4CosineModelDeployment()

    # Display model info
    print("\nModel Information:")
    info = model.get_model_info()
    print(json.dumps(info, indent=2, default=str))

    # Example 1: Compute similarity between two texts
    print("\n" + "="*80)
    print("EXAMPLE 1: Similarity Comparison")
    print("="*80)

    text1 = "User cannot login to SAP system. Error message: authentication failed."
    text2 = "SAP login issue - getting access denied error when trying to connect."
    text3 = "Outlook keeps crashing when opening large attachments."

    sim_12 = model.compute_similarity(text1, text2)
    sim_13 = model.compute_similarity(text1, text3)

    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {sim_12:.4f}")

    print(f"\nText 1: {text1}")
    print(f"Text 3: {text3}")
    print(f"Similarity: {sim_13:.4f}")

    # Example 2: Find similar tickets
    print("\n" + "="*80)
    print("EXAMPLE 2: Find Similar Tickets")
    print("="*80)

    query = "Cannot access email, Outlook not responding"

    candidates = [
        "Email client crashes randomly. Users report Outlook freezing.",
        "User cannot login to SAP system. Authentication errors.",
        "Outlook keeps crashing when opening attachments.",
        "Printer not working in Building A.",
        "Request to provision new laptop for incoming employee.",
    ]

    print(f"\nQuery: {query}")
    print("\nCandidates:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")

    results = model.find_similar(query, candidates, top_k=3)

    print(f"\nTop 3 Similar Tickets (threshold=0.3784):")
    for idx, text, score in results:
        print(f"  [{idx}] Score={score:.4f}: {text}")

    print("\n" + "="*80)
    print("Deployment test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
