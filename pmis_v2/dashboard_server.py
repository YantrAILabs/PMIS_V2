"""
PMIS V2 Dashboard Server (port 8200)

Lightweight server that serves the dashboard UI independently from the API server.
All API calls are routed to the main PMIS server at localhost:8100.

Usage:
    python3 pmis_v2/dashboard_server.py
"""

import sys
import re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PMIS_DIR = Path(__file__).parent
TEMPLATES_DIR = PMIS_DIR / "templates"
STATIC_DIR = PMIS_DIR / "static"
API_BASE = "http://localhost:8100"
DASHBOARD_PORT = 8200


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard HTML with API URLs rewritten to point at the PMIS server."""

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        # Route mapping
        if path in ("", "/dashboard"):
            self._serve_template("dashboard.html")
        elif path == "/integrations":
            self._serve_template("integrations.html")
        elif path.startswith("/static/"):
            # Serve static files
            file_path = STATIC_DIR / path[len("/static/"):]
            if file_path.exists():
                self._serve_file(file_path)
            else:
                self.send_error(404)
        else:
            self.send_error(404, f"Not found: {path}")

    def _serve_template(self, filename: str):
        template_path = TEMPLATES_DIR / filename
        if not template_path.exists():
            self.send_error(404, f"Template not found: {filename}")
            return

        content = template_path.read_text(encoding="utf-8")

        # Inject a fetch/EventSource proxy right after <head> that rewrites
        # all relative /api/* URLs to point at the PMIS API server.
        proxy_script = f"""
<script>
(function() {{
  const API_BASE = '{API_BASE}';
  const _fetch = window.fetch;
  window.fetch = function(url, opts) {{
    if (typeof url === 'string' && url.startsWith('/')) {{
      url = API_BASE + url;
    }}
    return _fetch.call(this, url, opts);
  }};
  const _ES = window.EventSource;
  window.EventSource = function(url, opts) {{
    if (typeof url === 'string' && url.startsWith('/')) {{
      url = API_BASE + url;
    }}
    return new _ES(url, opts);
  }};
  window.EventSource.prototype = _ES.prototype;
}})();
</script>"""
        content = content.replace("<head>", "<head>" + proxy_script, 1)

        # Rewrite navigation links
        content = content.replace('href="/"', 'href="/dashboard"')

        self._send_html(content)

    def _serve_file(self, file_path: Path):
        content = file_path.read_bytes()
        self.send_response(200)
        suffix = file_path.suffix.lower()
        content_types = {
            ".css": "text/css",
            ".js": "application/javascript",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".json": "application/json",
        }
        self.send_header("Content-Type", content_types.get(suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_html(self, content: str):
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        sys.stderr.write(f"[Dashboard] {args[0]}\n")


def main():
    server = HTTPServer(("0.0.0.0", DASHBOARD_PORT), DashboardHandler)
    print(f"[PMIS Dashboard] Serving on http://localhost:{DASHBOARD_PORT}")
    print(f"[PMIS Dashboard] API calls → {API_BASE}")
    print(f"[PMIS Dashboard] Pages: /dashboard, /integrations")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PMIS Dashboard] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
