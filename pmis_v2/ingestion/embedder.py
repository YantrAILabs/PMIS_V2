"""
Dual Embedding Generator for PMIS v2.

Takes raw text → calls embedding model (Ollama or OpenAI) →
produces triple embedding: euclidean + hyperbolic + temporal.
"""

import numpy as np
import httpx
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.poincare import ProjectionManager, assign_hyperbolic_coords
from core.temporal import temporal_encode, compute_era
from core import config


class Embedder:
    """Generates all three embedding types for a text input."""

    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hp = hyperparams or config.get_all()
        self.use_local = self.hp.get("use_local", True)

        # Projection manager for euclidean → hyperbolic
        self.projection = ProjectionManager(
            input_dim=self._embedding_dim(),
            output_dim=self.hp.get("poincare_dimensions", 32),
        )
        # Try to load saved projection matrix
        self._load_projection()

    def _embedding_dim(self) -> int:
        if self.use_local:
            return self.hp.get("local_embedding_dimensions", 768)
        return self.hp.get("embedding_dimensions", 1536)

    def _load_projection(self):
        """Load projection matrix if it exists."""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "projection_matrix.npy")
        if os.path.exists(path):
            self.projection.load(path)

    def save_projection(self):
        """Persist projection matrix to disk."""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "projection_matrix.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.projection.save(path)

    def embed_text(self, text: str) -> np.ndarray:
        """Get euclidean embedding from the configured model."""
        if self.use_local:
            return self._embed_ollama(text)
        else:
            return self._embed_openai(text)

    def batch_embed_texts(self, texts: List[str], batch_size: int = 20) -> List[np.ndarray]:
        """
        P1b: Batch embedding for migration speed.
        Processes texts in batches to reduce API overhead.
        Falls back to sequential if batch mode unavailable.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.use_local:
                batch_results = self._batch_embed_ollama(batch)
            else:
                batch_results = self._batch_embed_openai(batch)
            results.extend(batch_results)
        return results

    def _batch_embed_ollama(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed via Ollama. Ollama doesn't support true batching, so we pipeline."""
        results = []
        for text in texts:
            results.append(self._embed_ollama(text))
        return results

    def _batch_embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed via OpenAI API (supports native batching)."""
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = self.hp.get("embedding_model", "text-embedding-3-small")
        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"input": texts, "model": model},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
        except Exception as e:
            print(f"[Embedder] Batch OpenAI error: {e}. Falling back to sequential.")
            return [self._embed_openai(t) for t in texts]

    def get_model_name(self) -> str:
        """Return the active embedding model identifier for version tracking."""
        if self.use_local:
            return self.hp.get("local_embedding_model", "nomic-embed-text")
        return self.hp.get("embedding_model", "text-embedding-3-small")

    def _embed_ollama(self, text: str) -> np.ndarray:
        """Call Ollama's local embedding endpoint."""
        model = self.hp.get("local_embedding_model", "nomic-embed-text")
        try:
            response = httpx.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[Embedder] Ollama error: {e}. Returning zero vector.")
            return np.zeros(self._embedding_dim(), dtype=np.float32)

    def _embed_openai(self, text: str) -> np.ndarray:
        """Call OpenAI's embedding API."""
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = self.hp.get("embedding_model", "text-embedding-3-small")
        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"input": text, "model": model},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[Embedder] OpenAI error: {e}. Returning zero vector.")
            return np.zeros(self._embedding_dim(), dtype=np.float32)

    def generate_triple_embedding(
        self,
        text: str,
        level: str,
        parent_coords: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate all three embeddings for a memory node.

        Returns: {
            "euclidean": float[N],
            "hyperbolic": float[32],
            "temporal": float[16],
        }
        """
        ts = timestamp or datetime.now()

        # 1. Euclidean embedding (semantic)
        euclidean = self.embed_text(text)

        # 2. Hyperbolic coordinates (hierarchy)
        hyperbolic = assign_hyperbolic_coords(
            euclidean_embedding=euclidean,
            level=level,
            projection_manager=self.projection,
            parent_coords=parent_coords,
            hyperparams=self.hp,
        )

        # 3. Temporal embedding (time)
        temporal_dim = self.hp.get("temporal_embedding_dim", 16)
        temporal = temporal_encode(ts, dim=temporal_dim)

        return {
            "euclidean": euclidean,
            "hyperbolic": hyperbolic,
            "temporal": temporal,
        }
