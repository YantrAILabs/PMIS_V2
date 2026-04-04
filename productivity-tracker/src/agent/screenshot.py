"""
Screenshot capture using macOS native APIs.
Supports both screencapture CLI and CGWindowListCreateImage.
"""

import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image

logger = logging.getLogger("tracker.screenshot")


class ScreenshotCapture:
    def __init__(self, config: dict):
        self.max_width = config["tracking"]["screenshot_max_width"]
        self.screenshot_dir = Path(
            os.environ.get("SCREENSHOT_DIR", "/tmp/productivity-tracker/frames")
        )
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def capture(self) -> str | None:
        """
        Capture a screenshot of the current screen.
        Returns the file path of the captured (and resized) screenshot, or None on failure.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = self.screenshot_dir / f"raw_{timestamp}.png"
        final_path = self.screenshot_dir / f"frame_{timestamp}.jpg"

        try:
            # Use screencapture CLI (fastest, no accessibility permission needed for screen)
            # -x = no sound, -C = capture cursor, -t png
            result = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(raw_path)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0 or not raw_path.exists():
                logger.warning(f"screencapture failed: {result.stderr.decode()}")
                return self.capture_with_pyobjc()

            # Resize for cost optimization (ChatGPT Vision charges by token, smaller = cheaper)
            img = Image.open(raw_path)
            w, h = img.size
            if w > self.max_width:
                ratio = self.max_width / w
                new_size = (self.max_width, int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Convert RGBA to RGB for JPEG compatibility
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # Save as JPEG (smaller file, good enough for OCR)
            img.save(final_path, "JPEG", quality=85)

            # Clean up raw PNG
            raw_path.unlink(missing_ok=True)

            return str(final_path)

        except subprocess.TimeoutExpired:
            logger.error("Screenshot capture timed out")
            return None
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None

    def capture_with_pyobjc(self) -> str | None:
        """
        Alternative: capture using PyObjC / CGWindowListCreateImage.
        Use this if screencapture CLI has issues.
        """
        try:
            import Quartz

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            final_path = self.screenshot_dir / f"frame_{timestamp}.jpg"

            # Capture entire display
            image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectInfinite,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )

            if image is None:
                logger.error("CGWindowListCreateImage returned None")
                return None

            # Convert to JPEG via bitmap rep
            bitmap = Quartz.NSBitmapImageRep.alloc().initWithCGImage_(image)
            jpeg_data = bitmap.representationUsingType_properties_(
                Quartz.NSBitmapImageFileTypeJPEG,
                {Quartz.NSImageCompressionFactor: 0.85},
            )
            jpeg_data.writeToFile_atomically_(str(final_path), True)

            # Resize if needed
            img = Image.open(final_path)
            w, h = img.size
            if w > self.max_width:
                ratio = self.max_width / w
                new_size = (self.max_width, int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            # Convert RGBA to RGB for JPEG compatibility
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(final_path, "JPEG", quality=85)

            return str(final_path)

        except ImportError:
            logger.error("PyObjC not available, falling back to screencapture CLI")
            return self.capture()
        except Exception as e:
            logger.error(f"PyObjC screenshot failed: {e}")
            return None

    def cleanup_old(self, max_age_secs: int = 3600):
        """Remove screenshots older than max_age_secs."""
        now = datetime.now().timestamp()
        for f in self.screenshot_dir.glob("frame_*.jpg"):
            if now - f.stat().st_mtime > max_age_secs:
                f.unlink(missing_ok=True)
