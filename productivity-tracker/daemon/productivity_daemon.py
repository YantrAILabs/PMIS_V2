"""
Root-level always-active productivity daemon.

Manages the full pipeline from screenshot capture to memory update:
- Screenshot → Frame Analysis → Segment Classification
- 30-min Memory Sync (SQL → Vector → Hyperbolic → Matching → RSGD)
- Pipeline health monitoring

Install via: sudo daemon/install.sh
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("daemon")


class PipelineNodeStatus:
    """Tracks status and timing for a single pipeline node."""

    def __init__(self, name: str):
        self.name = name
        self.status = "idle"  # idle | running | ok | error
        self.last_run_at = None
        self.last_duration_secs = 0
        self.run_count = 0
        self.error_count = 0
        self.last_error = ""

    def start(self):
        self.status = "running"
        self._start_time = time.time()

    def finish_ok(self):
        self.status = "ok"
        self.last_run_at = datetime.now().isoformat()
        self.last_duration_secs = round(time.time() - self._start_time, 2)
        self.run_count += 1

    def finish_error(self, error: str):
        self.status = "error"
        self.last_run_at = datetime.now().isoformat()
        self.last_duration_secs = round(time.time() - self._start_time, 2)
        self.run_count += 1
        self.error_count += 1
        self.last_error = error[:200]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status,
            "last_run_at": self.last_run_at,
            "last_duration_secs": self.last_duration_secs,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


class DaemonAuth:
    """Simple admin password authentication."""

    def __init__(self, password_hash: str):
        self.password_hash = password_hash

    def verify(self, password: str) -> bool:
        h = hashlib.sha256(password.encode()).hexdigest()
        return h == self.password_hash

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()


class ProductivityDaemon:
    """Root-level always-active daemon managing the full pipeline."""

    def __init__(self, config_path: str = "config/settings.yaml",
                 admin_password_hash: str = ""):
        self.auth = DaemonAuth(admin_password_hash)
        self._running = False

        # Pipeline node tracking
        self.pipeline_nodes = {
            "screenshot_capture": PipelineNodeStatus("screenshot_capture"),
            "frame_analysis": PipelineNodeStatus("frame_analysis"),
            "segment_classification": PipelineNodeStatus("segment_classification"),
            "memory_sql_update": PipelineNodeStatus("memory_sql_update"),
            "vector_db_update": PipelineNodeStatus("vector_db_update"),
            "hyperbolic_update": PipelineNodeStatus("hyperbolic_update"),
            "project_matching": PipelineNodeStatus("project_matching"),
            "consolidation": PipelineNodeStatus("consolidation"),
        }

        self.config_path = config_path
        self._tracker = None
        self._monitor_app = None

    async def start(self):
        """Start all subsystems."""
        logger.info("Productivity Daemon starting...")
        self._running = True

        # Import and start the tracker
        from src.agent.tracker import ProductivityTracker
        self._tracker = ProductivityTracker(self.config_path)

        # Start the pipeline monitor API
        self._start_monitor_api()

        # Start the tracker (this runs the main loop)
        await self._tracker.start()

    async def stop(self):
        logger.info("Daemon shutting down...")
        self._running = False
        if self._tracker:
            await self._tracker.stop()
        logger.info("Daemon shutdown complete.")

    def _start_monitor_api(self):
        """Start the pipeline monitor API on port 8200."""
        import threading
        import uvicorn

        self._monitor_app = self._create_monitor_app()

        def run_monitor():
            uvicorn.run(self._monitor_app, host="0.0.0.0", port=8200, log_level="warning")

        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()
        logger.info("Pipeline Monitor API running on port 8200")

    def _create_monitor_app(self) -> FastAPI:
        """Create the FastAPI app for the pipeline monitor."""
        monitor = FastAPI(title="Pipeline Monitor", version="1.0")
        daemon = self

        @monitor.post("/api/pipeline/auth")
        async def pipeline_auth(password: str):
            if not daemon.auth.verify(password):
                raise HTTPException(401, "Invalid admin password")
            return {"authenticated": True}

        @monitor.get("/api/pipeline/status")
        async def pipeline_status():
            return {
                "running": daemon._running,
                "nodes": {k: v.to_dict() for k, v in daemon.pipeline_nodes.items()},
            }

        @monitor.get("/api/pipeline/node/{name}")
        async def pipeline_node(name: str):
            node = daemon.pipeline_nodes.get(name)
            if not node:
                raise HTTPException(404, f"Pipeline node '{name}' not found")
            return node.to_dict()

        @monitor.get("/api/pipeline/health")
        async def pipeline_health():
            nodes = daemon.pipeline_nodes
            total = len(nodes)
            errors = sum(1 for n in nodes.values() if n.status == "error")
            ok = sum(1 for n in nodes.values() if n.status == "ok")
            return {
                "health_score": round((ok / total) * 100, 1) if total > 0 else 0,
                "total_nodes": total,
                "ok_nodes": ok,
                "error_nodes": errors,
                "idle_nodes": total - ok - errors,
            }

        @monitor.get("/api/pipeline/errors")
        async def pipeline_errors():
            errors = []
            for node in daemon.pipeline_nodes.values():
                if node.error_count > 0:
                    errors.append({
                        "node": node.name,
                        "error_count": node.error_count,
                        "last_error": node.last_error,
                        "last_run_at": node.last_run_at,
                    })
            return {"errors": errors}

        @monitor.get("/", response_class=HTMLResponse)
        async def monitor_ui():
            ui_path = Path(__file__).parent / "monitor_ui" / "index.html"
            if ui_path.exists():
                return HTMLResponse(ui_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Pipeline Monitor</h1><p>UI not found.</p>")

        return monitor


def main():
    """Entry point for the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/var/log/productivity-daemon/daemon.log")
            if Path("/var/log/productivity-daemon").exists()
            else logging.FileHandler(
                Path.home() / ".productivity-tracker" / "daemon.log"
            ),
        ],
    )

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"

    # Load admin password hash
    cred_path = Path.home() / ".productivity-tracker" / "daemon_credentials.json"
    admin_hash = ""
    if cred_path.exists():
        creds = json.loads(cred_path.read_text(encoding="utf-8"))
        admin_hash = creds.get("admin_password_hash", "")

    daemon = ProductivityDaemon(config_path, admin_hash)

    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        loop.create_task(daemon.stop())

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(daemon.start())
    except KeyboardInterrupt:
        loop.run_until_complete(daemon.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
