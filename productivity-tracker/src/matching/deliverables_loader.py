"""
Deliverables loader — imports deliverables from YAML config and project tools.
"""

import json
import logging

import yaml

from src.storage.db import Database
from src.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger("tracker.deliverables_loader")


class DeliverablesLoader:
    """Loads deliverables from YAML and optional project tool APIs."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        self.chroma = ChromaDBStore(config)

    def load_from_yaml(self, path: str = "config/deliverables.yaml"):
        """Load deliverables from YAML config file."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Deliverables file not found: {path}")
            return

        for d in data.get("deliverables", []):
            contexts = d.get("expected_contexts", [])
            self.db.upsert_deliverable(
                id=d["id"],
                name=d["name"],
                supercontext=d.get("supercontext", ""),
                expected_contexts=json.dumps(contexts),
                owner=d.get("owner", ""),
                deadline=d.get("deadline", ""),
                status=d.get("status", "active"),
                source="yaml",
            )

            # Store embedding for semantic matching
            embed_text = f"{d['name']} — {d.get('supercontext', '')}. Contexts: {', '.join(contexts)}"
            self.chroma.store_deliverable(
                deliverable_id=d["id"],
                text=embed_text,
                metadata={
                    "name": d["name"],
                    "supercontext": d.get("supercontext", ""),
                    "owner": d.get("owner", ""),
                    "status": d.get("status", "active"),
                },
            )

        logger.info(f"Loaded {len(data.get('deliverables', []))} deliverables from YAML.")

    async def sync_from_asana(self):
        """Sync deliverables from Asana (if configured)."""
        if not self.config["matching"].get("asana_enabled"):
            return

        # Placeholder — implement with Asana API
        # Project → Deliverable (SC level)
        # Section → Context
        # Task → Anchor
        logger.info("Asana sync not yet implemented.")

    async def sync_from_notion(self):
        """Sync deliverables from Notion (if configured)."""
        if not self.config["matching"].get("notion_enabled"):
            return

        logger.info("Notion sync not yet implemented.")

    def add_deliverable(self, name: str, supercontext: str, contexts: list[str],
                        owner: str = "", deadline: str = "") -> str:
        """Add a deliverable programmatically (e.g., via MCP command)."""
        # Generate ID
        existing = self.db.get_active_deliverables()
        max_num = 0
        for d in existing:
            try:
                num = int(d["id"].split("-")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                pass
        new_id = f"D-{max_num + 1:03d}"

        self.db.upsert_deliverable(
            id=new_id,
            name=name,
            supercontext=supercontext,
            expected_contexts=json.dumps(contexts),
            owner=owner,
            deadline=deadline,
            status="active",
            source="manual",
        )

        embed_text = f"{name} — {supercontext}. Contexts: {', '.join(contexts)}"
        self.chroma.store_deliverable(
            deliverable_id=new_id,
            text=embed_text,
            metadata={"name": name, "supercontext": supercontext, "owner": owner, "status": "active"},
        )

        logger.info(f"Added deliverable {new_id}: {name}")
        return new_id
