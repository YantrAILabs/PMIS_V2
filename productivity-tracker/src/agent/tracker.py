"""
Main tracking daemon — orchestrates screenshot capture, window monitoring,
activity detection, agent detection, and feeds into the context pipeline.

Run via: python -m src.agent.tracker
Or via launchd (see scripts/install.sh)
"""

import asyncio
import logging
import signal
import time
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv

from src.agent.screenshot import ScreenshotCapture
from src.agent.window_monitor import WindowMonitor
from src.agent.activity_monitor import ActivityMonitor
from src.agent.agent_detector import AgentDetector
from src.agent.input_monitor import InputMonitor
from src.pipeline.segmenter import TargetFrameSegmenter
from src.pipeline.frame_analyzer import FrameAnalyzer
from src.pipeline.context_classifier import ContextClassifier
from src.storage.db import Database
from src.memory.hourly_aggregator import HourlyAggregator
from src.memory.daily_rollup import DailyRollup
from src.memory.central_merge import CentralMerge

load_dotenv()
logger = logging.getLogger("tracker")


class ProductivityTracker:
    """Main daemon that coordinates all tracking subsystems."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Subsystems
        self.screenshot = ScreenshotCapture(self.config)
        self.window_monitor = WindowMonitor()
        self.activity_monitor = ActivityMonitor(self.config)
        self.agent_detector = AgentDetector()
        self.input_monitor = InputMonitor(self.config)
        self.segmenter = TargetFrameSegmenter(self.config)
        self.frame_analyzer = FrameAnalyzer(self.config)
        self.context_classifier = ContextClassifier(self.config)
        self.db = Database()
        self.hourly_agg = HourlyAggregator(self.db, self.config)
        self.daily_rollup = DailyRollup(self.db, self.config)
        self.central_merge = CentralMerge(self.db, self.config)

        # State
        self._running = False
        self._paused = False
        self._current_segment_id = None
        self._frame_buffer = []

    async def start(self):
        """Start the tracking loop."""
        logger.info("Productivity Tracker starting...")
        self._running = True

        # Initialize DB
        self.db.initialize()

        # FIX M1: Load segment counter from DB to survive restarts
        today = date.today().strftime("%Y%m%d")
        max_counter = self.db.get_max_segment_number(today)
        self.segmenter.load_last_segment_counter(max_counter)

        # Start subsystems
        self.window_monitor.start()
        self.activity_monitor.start()
        self.input_monitor.start()

        # On startup: catch up ALL missed pipeline work across ALL dates
        logger.info("=== STARTUP CATCH-UP: Processing all missed pipeline work ===")

        # Step 1: Hourly — process all dates/hours that have segments but no hourly log
        hourly_count = await self.hourly_agg.run_all_pending()
        logger.info(f"Hourly catch-up: {hourly_count} hours processed.")

        # Step 2: Daily — rebuild daily entries for all dates with hourly data
        daily_count = await self.daily_rollup.run_all_pending()
        logger.info(f"Daily catch-up: {daily_count} dates rebuilt.")

        logger.info("=== STARTUP CATCH-UP COMPLETE ===")

        # Schedule periodic jobs
        asyncio.create_task(self._hourly_job())
        asyncio.create_task(self._six_hour_analysis())
        asyncio.create_task(self._daily_job())
        asyncio.create_task(self._thirty_min_sync())

        # Main tracking loop
        await self._tracking_loop()

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False

        # Finalize current segment if any
        if self._current_segment_id and self._frame_buffer:
            await self._finalize_segment()

        self.window_monitor.stop()
        self.activity_monitor.stop()
        self.input_monitor.stop()
        self.db.close()
        logger.info("Shutdown complete.")

    def pause(self):
        self._paused = True
        logger.info("Tracking paused.")

    def resume(self):
        self._paused = False
        logger.info("Tracking resumed.")

    @property
    def is_paused(self):
        return self._paused

    # ─── Main loop ──────────────────────────────────────────────────────

    async def _tracking_loop(self):
        """Core loop: capture → detect → segment → buffer → analyze."""
        interval = self.config["tracking"]["screenshot_interval_active"]
        idle_interval = self.config["tracking"]["screenshot_interval_idle"]

        while self._running:
            if self._paused:
                await asyncio.sleep(1)
                continue

            # 1. Check if idle
            is_idle = self.activity_monitor.is_idle()
            current_interval = idle_interval if is_idle else interval

            if is_idle:
                idle_duration = self.activity_monitor.idle_duration()
                gap_threshold = self.config["tracking"]["idle_gap_new_segment_secs"]
                if idle_duration > gap_threshold and self._current_segment_id:
                    await self._finalize_segment()
                await asyncio.sleep(current_interval)
                continue

            # 2. Get current window info
            window_info = self.window_monitor.get_active_window()
            if not window_info or self._is_excluded(window_info):
                await asyncio.sleep(current_interval)
                continue

            # 3. Check if agent is running
            agent_active = self.agent_detector.is_agent_active()

            # 4. Capture screenshot
            screenshot_path = self.screenshot.capture()
            if not screenshot_path:
                await asyncio.sleep(current_interval)
                continue

            # 5. Check if we need a new segment
            #    (segmenter now auto-updates its internal state — FIX C2)
            needs_new = self.segmenter.should_start_new_segment(
                window_info=window_info,
                screenshot_path=screenshot_path,
                agent_active=agent_active,
            )

            if needs_new and self._current_segment_id:
                await self._finalize_segment()

            if needs_new or self._current_segment_id is None:
                self._current_segment_id = self.segmenter.start_new_segment(
                    window_info=window_info,
                    agent_active=agent_active,
                )
                self._frame_buffer = []

            # 6. Detect keyboard/mouse activity for this frame
            input_state = self.input_monitor.get_activity_since_last_check()

            # 7. Add frame to buffer with input activity data
            frame = {
                "path": screenshot_path,
                "timestamp": datetime.now(),
                "frame_number": len(self._frame_buffer) + 1,
                "window_info": window_info,
                "agent_active": agent_active,
                "has_keyboard_activity": input_state["keyboard"],
                "has_mouse_activity": input_state["mouse"],
            }
            self._frame_buffer.append(frame)

            # 7. Batch analyze frames when buffer hits batch size
            batch_size = self.config["tracking"]["frame_batch_size"]
            if len(self._frame_buffer) % batch_size == 0:
                batch = self._frame_buffer[-batch_size:]
                await self._analyze_frame_batch(batch)

            await asyncio.sleep(current_interval)

    async def _analyze_frame_batch(self, frames: list):
        """Send a batch of frames to ChatGPT Vision for extraction."""
        try:
            results = await self.frame_analyzer.analyze_batch(frames)
            for frame, result in zip(frames, results):
                # Worker type: keyboard/mouse activity = human, no activity = ai
                has_kb = frame.get("has_keyboard_activity", False)
                has_mouse = frame.get("has_mouse_activity", False)
                worker = "human" if (has_kb or has_mouse) else "ai"

                self.db.insert_context2(
                    target_segment_id=self._current_segment_id,
                    target_frame_number=frame["frame_number"],
                    frame_timestamp=frame["timestamp"],
                    raw_text=result.get("text", ""),
                    detailed_summary=result.get("task", ""),
                    worker_type=worker,
                    has_keyboard_activity=has_kb,
                    has_mouse_activity=has_mouse,
                )
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")

    async def _finalize_segment(self):
        """Close current segment: flush remaining frames, run synthesis, store Context 1."""
        if not self._frame_buffer:
            return

        segment_id = self._current_segment_id
        success = False

        try:
            # FIX C3: Flush any unanalyzed tail frames before reading from DB
            batch_size = self.config["tracking"]["frame_batch_size"]
            analyzed_count = (len(self._frame_buffer) // batch_size) * batch_size
            remaining = self._frame_buffer[analyzed_count:]
            if remaining:
                await self._analyze_frame_batch(remaining)

            # Now all frames are in DB — read them back
            frame_results = self.db.get_frames_for_segment(segment_id)

            # Run ChatGPT synthesis for the full segment
            classification = await self.context_classifier.classify_segment(
                segment_id=segment_id,
                frame_results=frame_results,
                window_info=self._frame_buffer[0]["window_info"],
                agent_active=any(f["agent_active"] for f in self._frame_buffer),
            )

            # Calculate segment duration
            start = self._frame_buffer[0]["timestamp"]
            end = self._frame_buffer[-1]["timestamp"]
            duration_secs = max(int((end - start).total_seconds()), 1)

            # Compute human vs AI frame counts from input activity
            human_frames = sum(
                1 for f in self._frame_buffer
                if f.get("has_keyboard_activity") or f.get("has_mouse_activity")
            )
            ai_frames = len(self._frame_buffer) - human_frames
            # Worker type: any human input in the segment = human work
            segment_worker = "human" if human_frames > 0 else "ai"

            # Store in Context 1
            self.db.insert_context1(
                timestamp_start=start,
                timestamp_end=end,
                window_name=self._frame_buffer[0]["window_info"]["title"],
                platform=self._frame_buffer[0]["window_info"]["app_name"],
                medium=classification.get("medium", "other"),
                context=classification.get("context", "Unclassified"),
                supercontext=classification.get("supercontext", "Unclassified"),
                anchor=classification.get("anchor", "Unclassified"),
                target_segment_id=segment_id,
                target_segment_length_secs=duration_secs,
                worker=segment_worker,
                detailed_summary=classification.get("detailed_summary", ""),
                human_frame_count=human_frames,
                ai_frame_count=ai_frames,
            )

            logger.info(
                f"Segment {segment_id} finalized: "
                f"{classification.get('anchor', '?')} ({duration_secs}s, "
                f"{'agent' if classification.get('worker') == 'agent' else 'human'})"
            )
            success = True

        except Exception as e:
            logger.error(f"Segment finalization failed for {segment_id}: {e}")

        # FIX M3: Only clear state on success. On error, leave segment for retry.
        if success:
            for frame in self._frame_buffer:
                try:
                    Path(frame["path"]).unlink(missing_ok=True)
                except Exception:
                    pass
            self._frame_buffer = []
            self._current_segment_id = None
        else:
            logger.warning(f"Segment {segment_id} will be retried on next trigger.")

    def _is_excluded(self, window_info: dict) -> bool:
        """Check if current window is in the privacy exclusion list."""
        import yaml
        import re
        try:
            with open("config/privacy.yaml") as f:
                privacy = yaml.safe_load(f)
        except FileNotFoundError:
            return False

        bundle = window_info.get("bundle_id", "")
        title = window_info.get("title", "")

        for app in privacy.get("excluded_apps", []):
            if app.get("bundle_id") == bundle:
                url_patterns = app.get("url_patterns", [])
                if not url_patterns:
                    return True
                for pattern in url_patterns:
                    if re.match(pattern.replace("*", ".*"), title, re.IGNORECASE):
                        return True

        for pattern in privacy.get("excluded_windows", []):
            if re.match(pattern, title, re.IGNORECASE):
                return True

        return False

    # ─── Periodic jobs ──────────────────────────────────────────────────

    async def _hourly_job(self):
        """Run hourly aggregation aligned to clock boundaries."""
        while self._running:
            # FIX M2: Sleep until next hour boundary
            now = datetime.now()
            seconds_until_next_hour = 3600 - (now.minute * 60 + now.second)
            await asyncio.sleep(seconds_until_next_hour)

            if self._running:
                try:
                    await self.hourly_agg.run()
                    logger.info("Hourly aggregation complete.")
                except Exception as e:
                    logger.error(f"Hourly aggregation failed: {e}")

    async def _six_hour_analysis(self):
        """Run comprehensive analysis + central memory sync every 6 hours."""
        analysis_hours = self.config["memory"].get("analysis_hours", [0, 6, 12, 18])
        last_analysis_hour = -1

        while self._running:
            now = datetime.now()
            if now.hour in analysis_hours and now.hour != last_analysis_hour:
                last_analysis_hour = now.hour
                try:
                    logger.info(f"=== 6-HOUR ANALYSIS CHECKPOINT ({now.strftime('%H:%M')}) ===")

                    # Step 1: Catch up ALL missed hours for today (not just current hour)
                    await self.hourly_agg.run_all_pending(
                        target_date=date.today().strftime("%Y-%m-%d")
                    )

                    # Step 2: Build comprehensive analysis (daily rollup with current data)
                    await self.daily_rollup.run()
                    logger.info("6-hour comprehensive analysis complete.")

                    # Step 3: Sync with central PMIS v2 memory
                    today = date.today().strftime("%Y-%m-%d")
                    self.central_merge.run(today)
                    logger.info("Central memory sync complete.")

                    # Step 4: Log the analysis summary
                    hierarchy = self.daily_rollup.get_daily_hierarchy(today)
                    total_mins = sum(sc.get("time_mins", 0) for sc in hierarchy.values())
                    human_mins = sum(sc.get("human_mins", 0) for sc in hierarchy.values())
                    agent_mins = sum(sc.get("agent_mins", 0) for sc in hierarchy.values())
                    sc_names = list(hierarchy.keys())
                    logger.info(
                        f"Analysis: {total_mins:.0f}min tracked "
                        f"(H:{human_mins:.0f} A:{agent_mins:.0f}) "
                        f"across {len(sc_names)} supercontexts: {', '.join(sc_names)}"
                    )

                except Exception as e:
                    logger.error(f"6-hour analysis failed: {e}")
            await asyncio.sleep(60)  # Check every minute

    async def _daily_job(self):
        """Run daily rollup + central merge at configured hour."""
        rollup_hour = self.config["memory"]["daily_rollup_hour"]
        while self._running:
            now = datetime.now()
            if now.hour == rollup_hour and now.minute == 0:
                try:
                    # Step 1: Daily rollup (hourly → daily memory)
                    await self.daily_rollup.run()
                    logger.info("Daily rollup complete.")

                    # Step 2: Merge daily memory into central PMIS v2
                    today = date.today().strftime("%Y-%m-%d")
                    self.central_merge.run(today)
                    logger.info("Central memory merge complete.")

                except Exception as e:
                    logger.error(f"Daily job failed: {e}")
                await asyncio.sleep(3600)  # Don't re-trigger within same hour
            await asyncio.sleep(30)


    async def _thirty_min_sync(self):
        """Run full PMIS V2 memory pipeline sync every 30 minutes.

        Only triggers if new unique frame numbers were added in the last 30 minutes.
        Uses the same store/dedup logic as the nightly update — SQL → Vector → Hyperbolic → Matching → RSGD.
        """
        while self._running:
            await asyncio.sleep(1800)  # 30 minutes
            if not self._running:
                break

            try:
                # Guard: only sync if new frames exist
                new_frame_count = self.db.count_new_frames_since(minutes=30)
                if new_frame_count == 0:
                    logger.debug("30-min sync: no new frames, skipping.")
                    continue

                # Get unsynced segments
                unsynced = self.db.get_unsynced_segments()
                if not unsynced:
                    logger.debug("30-min sync: no unsynced segments, skipping.")
                    continue

                logger.info(
                    f"=== 30-MIN MEMORY SYNC === "
                    f"{len(unsynced)} unsynced segments, {new_frame_count} new frames"
                )

                # Import pipeline sync lazily to avoid circular imports
                from src.memory.pipeline_sync import ProductivityPipelineSync
                sync = ProductivityPipelineSync.from_config(self.config)
                result = sync.run(unsynced)

                logger.info(
                    f"30-min sync complete: {result.get('nodes_created', 0)} created, "
                    f"{result.get('nodes_updated', 0)} updated, "
                    f"{result.get('matches_found', 0)} matched "
                    f"(avg {result.get('avg_match_pct', 0):.1f}%)"
                )

            except Exception as e:
                logger.error(f"30-min memory sync failed: {e}")


def main():
    """Entry point for the tracker daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path.home() / ".productivity-tracker" / "tracker.log"
            ),
        ],
    )

    # Ensure data directory exists
    data_dir = Path.home() / ".productivity-tracker"
    data_dir.mkdir(exist_ok=True)

    tracker = ProductivityTracker()

    loop = asyncio.new_event_loop()

    def shutdown_handler(sig, frame):
        loop.create_task(tracker.stop())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        loop.run_until_complete(tracker.start())
    except KeyboardInterrupt:
        loop.run_until_complete(tracker.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
