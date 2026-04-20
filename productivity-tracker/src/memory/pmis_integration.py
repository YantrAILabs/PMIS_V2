"""
PMIS v2 integration — bridges productivity tracker with the central memory system.

ISOLATION PRINCIPLE:
- Tracker data lives in ~/.productivity-tracker/chromadb (hourly, daily, deliverables collections)
- Central PMIS memory lives in ~/.pmis-v2/chromadb (configured in settings.yaml → pmis.chromadb_path)
- This module ONLY reads/writes to central memory during the final daily merge
- It NEVER touches tracker's own collections
"""

import logging
import os
from pathlib import Path

import chromadb
import httpx

logger = logging.getLogger("tracker.pmis")


class PMISIntegration:
    """
    Interface to PMIS v2 central memory.
    Uses a SEPARATE ChromaDB instance from the tracker's own storage.
    """

    def __init__(self, config: dict):
        self.config = config

        # Central memory path — completely separate from tracker storage
        pmis_config = config.get("pmis", {})
        central_path = os.path.expanduser(
            pmis_config.get("chromadb_path", "~/.pmis-v2/chromadb")
        )
        Path(central_path).mkdir(parents=True, exist_ok=True)

        # Separate ChromaDB client pointing to PMIS storage
        self._central_client = chromadb.PersistentClient(path=central_path)
        self._collection_name = pmis_config.get("collection_name", "pmis_central_memory")
        self.central = self._central_client.get_or_create_collection(self._collection_name)

        self.merge_threshold = pmis_config.get("merge_threshold", 0.75)
        self.merge_levels = pmis_config.get("merge_levels", ["SC", "context"])

        # Embedding provider — Ollama by default, OpenAI optional
        mem_cfg = config.get("memory", {}) if config else {}
        self._embedding_provider = mem_cfg.get("embedding_provider", "ollama").lower()
        if self._embedding_provider == "openai":
            from openai import OpenAI  # lazy
            self._openai = OpenAI()
            self._embedding_model = mem_cfg.get("embedding_model", "text-embedding-3-small")
        else:
            self._openai = None
            self._embedding_model = mem_cfg.get("embedding_model", "nomic-embed-text")

        ollama_cfg = config.get("ollama", {}) if config else {}
        self._ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")
        self._ollama_timeout = float(ollama_cfg.get("timeout", 60))

        logger.info(
            "PMIS central memory: %s / %s (embed=%s/%s)",
            central_path, self._collection_name,
            self._embedding_provider, self._embedding_model,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding via the configured provider."""
        if self._embedding_provider == "openai":
            response = self._openai.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            return response.data[0].embedding
        # Ollama
        r = httpx.post(
            f"{self._ollama_url}/api/embeddings",
            json={"model": self._embedding_model, "prompt": text},
            timeout=self._ollama_timeout,
        )
        r.raise_for_status()
        emb = r.json().get("embedding")
        if not emb:
            raise RuntimeError(
                f"Ollama returned no embedding for {self._embedding_model!r}"
            )
        return emb

    def find_central_match(self, text: str) -> dict | None:
        """
        Find the best matching node in central memory.
        The tracker's SC might map to any level (SC/Context/Anchor) in central memory.
        """
        embedding = self.embed_text(text)

        try:
            results = self.central.query(
                query_embeddings=[embedding],
                n_results=3,
            )
        except Exception as e:
            logger.error(f"Central memory query failed: {e}")
            return None

        if not results["ids"] or not results["ids"][0]:
            return None

        best_distance = results["distances"][0][0]
        # ChromaDB default is L2 distance; approximate cosine similarity
        similarity = max(0, 1 - (best_distance / 2))

        if similarity >= self.merge_threshold:
            return {
                "id": results["ids"][0][0],
                "document": results["documents"][0][0],
                "metadata": results["metadatas"][0][0] if results["metadatas"] else {},
                "similarity": similarity,
            }
        return None

    def merge_to_central(self, daily_entry: dict) -> str | None:
        """
        Merge a daily memory entry into central PMIS memory.

        Only called for SC and context levels (configured in settings.yaml).
        Anchors stay in tracker's own daily_memory table — too granular for central.
        """
        level = daily_entry.get("level", "anchor")
        if level not in self.merge_levels:
            return None

        text = self._build_entry_text(daily_entry)
        match = self.find_central_match(text)

        if match:
            existing_meta = match["metadata"]
            existing_meta["productivity_time_mins"] = (
                existing_meta.get("productivity_time_mins", 0) + daily_entry["time_mins"]
            )
            existing_meta["last_productivity_update"] = daily_entry["date"]
            existing_meta["human_mins"] = (
                existing_meta.get("human_mins", 0) + daily_entry["human_mins"]
            )
            existing_meta["agent_mins"] = (
                existing_meta.get("agent_mins", 0) + daily_entry["agent_mins"]
            )

            try:
                self.central.update(
                    ids=[match["id"]],
                    metadatas=[existing_meta],
                )
                logger.info(
                    f"Updated central node: {match['id']} "
                    f"(+{daily_entry['time_mins']:.0f} mins, similarity: {match['similarity']:.2f})"
                )
                return match["id"]
            except Exception as e:
                logger.error(f"Central memory update failed: {e}")
                return None
        else:
            # Create new node in central memory
            from src.pipeline.segmenter import sanitize_id
            entry_id = sanitize_id(
                f"prod-{daily_entry['date']}-{daily_entry.get('supercontext', 'unknown')}"
            )
            embedding = self.embed_text(text)

            try:
                self.central.upsert(
                    ids=[entry_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{
                        "level": level,
                        "supercontext": daily_entry.get("supercontext"),
                        "context": daily_entry.get("context"),
                        "source": "productivity_tracker",
                        "productivity_time_mins": daily_entry["time_mins"],
                        "human_mins": daily_entry["human_mins"],
                        "agent_mins": daily_entry["agent_mins"],
                        "date_added": daily_entry["date"],
                    }],
                )
                logger.info(f"Created new central node: {entry_id}")
                return entry_id
            except Exception as e:
                logger.error(f"Central memory insert failed: {e}")
                return None

    def mark_contribution(self, entry_id: str, deliverable_id: str, delivered: bool):
        """Mark a central memory node as contributed to a deliverable."""
        if not entry_id:
            return
        try:
            result = self.central.get(ids=[entry_id])
            if result["ids"]:
                meta = result["metadatas"][0]
                meta["contributed_to_delivery"] = delivered
                meta["deliverable_id"] = deliverable_id
                self.central.update(ids=[entry_id], metadatas=[meta])
        except Exception as e:
            logger.error(f"Failed to mark contribution for {entry_id}: {e}")

    def get_contribution_chain(self, deliverable_id: str) -> list[dict]:
        """Get all central memory nodes that contributed to a deliverable."""
        try:
            results = self.central.get(
                where={"deliverable_id": deliverable_id},
            )
            entries = []
            for i in range(len(results["ids"])):
                entries.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                })
            return entries
        except Exception as e:
            logger.error(f"Contribution chain query failed: {e}")
            return []

    @staticmethod
    def _build_entry_text(entry: dict) -> str:
        parts = [entry.get("supercontext", "")]
        if entry.get("context"):
            parts.append(entry["context"])
        if entry.get("anchor"):
            parts.append(entry["anchor"])
        return " > ".join(p for p in parts if p)
