"""
ChromaDB embedding store — handles vector storage for memory hierarchy.
Supports both productivity-specific and central PMIS v2 collections.
"""

import json
import logging
import os
from pathlib import Path

import chromadb
from openai import OpenAI

logger = logging.getLogger("tracker.chromadb_store")


class ChromaDBStore:
    """Vector store for productivity memory embeddings."""

    COLLECTION_HOURLY = "productivity_hourly"
    COLLECTION_DAILY = "productivity_daily"
    COLLECTION_DELIVERABLES = "productivity_deliverables"

    def __init__(self, config: dict = None):
        db_path = os.environ.get(
            "CHROMADB_PATH",
            str(Path.home() / ".productivity-tracker" / "chromadb"),
        )
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)
        self.openai = OpenAI()
        self.embedding_model = (
            config["memory"]["embedding_model"]
            if config else "text-embedding-3-small"
        )

        # Initialize collections
        self.hourly = self.client.get_or_create_collection(self.COLLECTION_HOURLY)
        self.daily = self.client.get_or_create_collection(self.COLLECTION_DAILY)
        self.deliverables = self.client.get_or_create_collection(self.COLLECTION_DELIVERABLES)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding via OpenAI."""
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def store_hourly(
        self,
        entry_id: str,
        text: str,
        metadata: dict,
    ) -> str:
        """Store an hourly memory entry with embedding."""
        embedding = self.embed_text(text)
        self.hourly.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return entry_id

    def store_daily(
        self,
        entry_id: str,
        text: str,
        metadata: dict,
    ) -> str:
        """Store a daily memory entry."""
        embedding = self.embed_text(text)
        self.daily.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return entry_id

    def store_deliverable(
        self,
        deliverable_id: str,
        text: str,
        metadata: dict,
    ) -> str:
        """Store a deliverable for semantic matching."""
        embedding = self.embed_text(text)
        self.deliverables.upsert(
            ids=[deliverable_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return deliverable_id

    def search_daily(self, query: str, n_results: int = 5, date_filter: str = None) -> list[dict]:
        """Search daily memory by semantic similarity."""
        embedding = self.embed_text(query)
        where = {"date": date_filter} if date_filter else None
        results = self.daily.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )
        return self._format_results(results)

    def search_deliverables(self, query: str, n_results: int = 5) -> list[dict]:
        """Find matching deliverables by semantic similarity."""
        embedding = self.embed_text(query)
        results = self.deliverables.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        return self._format_results(results)

    def match_to_deliverable(self, text: str, threshold: float = 0.8) -> dict | None:
        """Find the best matching deliverable above threshold."""
        embedding = self.embed_text(text)
        results = self.deliverables.query(
            query_embeddings=[embedding],
            n_results=1,
        )
        if results["distances"] and results["distances"][0]:
            # ChromaDB returns L2 distance; convert to cosine similarity
            distance = results["distances"][0][0]
            similarity = 1 - (distance / 2)  # Approximate for normalized embeddings
            if similarity >= threshold:
                return {
                    "id": results["ids"][0][0],
                    "document": results["documents"][0][0],
                    "metadata": results["metadatas"][0][0],
                    "similarity": similarity,
                }
        return None

    def delete_hourly_for_date(self, target_date: str):
        """Remove hourly embeddings after daily rollup."""
        try:
            results = self.hourly.get(where={"date": target_date})
            if results["ids"]:
                self.hourly.delete(ids=results["ids"])
        except Exception as e:
            logger.warning(f"Hourly cleanup failed: {e}")

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return formatted
