"""
Nomic Embed Text v1.5 Deployment Script
========================================

Zero-shot production deployment of nomic-ai/nomic-embed-text-v1.5.

Benchmark performance (zero-shot, grounded resolution-notes):
  Spearman = 0.4476   ROC-AUC = 0.7584

Replaces V4 Cosine (fine-tuned MPNet LoRA, Spearman=0.2949 on same benchmark).

Usage:
    Queries  → encode_query(texts)   — prepends 'search_query: '
    Documents → encode(texts)        — prepends 'search_document: '

Nomic requires trust_remote_code=True (custom pooling layer).
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Tuple

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX    = "search_query: "
DOCUMENT_PREFIX = "search_document: "
EMBEDDING_DIM   = 768


class NomicModelDeployment:
    """
    Production-ready deployment wrapper for nomic-embed-text-v1.5.

    Drop-in replacement for V4CosineModelDeployment.  Main differences:
      - No LoRA adapters — loaded directly from HuggingFace
      - Requires prefix injection: 'search_query: ' / 'search_document: '
      - encode()       → document prefix  (used when re-embedding incidents)
      - encode_query() → query prefix     (used at search time in the API)
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.query_prefix    = QUERY_PREFIX
        self.document_prefix = DOCUMENT_PREFIX

        print(f"Loading Nomic model: {self.model_id}")
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=True,
        )

        print("[OK] Nomic model loaded")
        print(f"   Embedding dimension : {self.model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length : {self.model.max_seq_length}")

    # ------------------------------------------------------------------
    # Core encode methods
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """
        Encode documents (incidents).  Prepends 'search_document: ' to each text.
        Use this when building / updating the embedding index.
        """
        prefixed = [self.document_prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            device=self.device,
            **kwargs,
        )

    def encode_query(self, texts: List[str], batch_size: int = 32,
                     show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """
        Encode search queries.  Prepends 'search_query: ' to each text.
        Use this at search time in the API.
        """
        prefixed = [self.query_prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            device=self.device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience helpers (preserve API parity with V4CosineModelDeployment)
    # ------------------------------------------------------------------

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two documents."""
        emb1, emb2 = self.encode([text1, text2])
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def find_similar(self, query: str, candidates: List[str],
                     top_k: int = 5, threshold: float = 0.3) -> List[Tuple[int, str, float]]:
        """Find most similar documents to a query."""
        query_emb = self.encode_query([query])[0]
        cand_embs = self.encode(candidates)
        sims = cand_embs @ query_emb  # dot product of L2-normalised vectors = cosine sim
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            (int(i), candidates[i], float(sims[i]))
            for i in top_indices
            if sims[i] >= threshold
        ]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = NomicModelDeployment()
    docs = [
        "Network switch unresponsive after power outage in Building C",
        "VPN client fails to authenticate for remote users",
        "Email delivery delayed to external recipients",
    ]
    query = "network connectivity problem"
    results = m.find_similar(query, docs, top_k=3, threshold=0.0)
    print("\nSelf-test results:")
    for rank, (idx, text, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {text}")
