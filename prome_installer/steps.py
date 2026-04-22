"""
Installer steps 1-8 — one function per step, each returning control to __main__.

Every step is idempotent. Re-running the installer should converge the system
to the intended state without destroying existing data or duplicating work.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import paths, service, ui


TOTAL_STEPS = 8


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════════

def step1_preflight() -> dict:
    """
    Return a context dict with: {python_exe, daemon_was_running, perm_issues}.
    Calls ui.fail() (sys.exit) on any non-recoverable problem.
    """
    ui.step(1, TOTAL_STEPS, "Pre-flight checks")

    # Platform check
    if sys.platform not in ("darwin", "win32"):
        ui.fail(f"Unsupported platform: {sys.platform}. ProMe supports macOS and Windows.")
    os_name = "macOS" if sys.platform == "darwin" else "Windows"
    os_version = platform.mac_ver()[0] if sys.platform == "darwin" else platform.win32_ver()[0]
    ui.ok(f"{os_name} detected ({os_version or 'unknown version'})")

    # Python >= 3.11 (we're already running on it, but double-check)
    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 11):
        hint = (
            "Install via: brew install python@3.12" if sys.platform == "darwin"
            else "Install via: winget install Python.Python.3.12"
        )
        ui.fail(f"Python >= 3.11 required (found {py_major}.{py_minor}). {hint}")
    ui.ok(f"Python {py_major}.{py_minor} ({sys.executable})")

    # pip availability
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
        )
        ui.ok("pip available")
    except subprocess.CalledProcessError:
        ui.fail(f"pip not available for {sys.executable}")

    # Stop existing daemon (idempotent)
    if service.stop_daemon():
        ui.ok("Existing daemon stopped")
    else:
        ui.ok("No existing daemon running")

    # Repo structure sanity
    if not paths.TRACKER_DIR.is_dir():
        ui.fail(f"productivity-tracker/ not found at {paths.TRACKER_DIR}")
    if not paths.PMIS_DIR.is_dir():
        ui.fail(f"pmis_v2/ not found at {paths.PMIS_DIR}")
    ui.ok("Repository structure valid")

    # Platform-specific permission pre-check (advisory)
    perm_issues = False
    if sys.platform == "darwin":
        test_png = Path("/tmp/.pmis_perm_test.png")
        try:
            result = subprocess.run(
                ["screencapture", "-x", str(test_png)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0 or not test_png.exists():
                perm_issues = True
        except Exception:
            perm_issues = True
        finally:
            test_png.unlink(missing_ok=True)
        if perm_issues:
            ui.warn("Screen Recording permission not granted — daemon won't capture screenshots")
            ui.warn("Grant it: System Settings > Privacy & Security > Screen Recording > Terminal")
        else:
            ui.ok("Screen Recording permission granted")

    return {
        "python_exe": sys.executable,
        "daemon_was_running": False,  # stopped above, caller doesn't need to know
        "perm_issues": perm_issues,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Python venv + dependencies
# ═══════════════════════════════════════════════════════════════════════════

def step2_venv(python_exe: str) -> None:
    ui.step(2, TOTAL_STEPS, "Python environment + dependencies")

    # Auto-fix broken venv (same logic as install.sh)
    if paths.VENV_DIR.is_dir():
        if paths.VENV_PYTHON.exists() and paths.VENV_PIP.exists():
            ui.info("Existing venv found, updating...")
        else:
            ui.info("Broken venv detected — rebuilding automatically...")
            shutil.rmtree(paths.VENV_DIR, ignore_errors=True)
            _create_venv(python_exe)
    else:
        ui.info("Creating virtual environment...")
        _create_venv(python_exe)
    ui.ok(f"Virtual environment at {paths.VENV_DIR}")

    # Upgrade pip + setuptools + wheel
    ui.info("Upgrading pip + setuptools...")
    _pip_install(["--upgrade", "pip", "setuptools", "wheel"], quiet=True)
    ui.ok("pip upgraded")

    # Stage A: core packages (must succeed)
    ui.info("Installing core dependencies...")
    core_packages = [
        "openai", "sqlalchemy", "numpy", "Pillow", "pyyaml", "python-dotenv",
        "fastapi", "uvicorn", "httpx", "psutil", "pyjwt", "bcrypt", "jinja2",
    ]
    for pkg in core_packages:
        if _pip_install([pkg], quiet=True, raise_on_error=False):
            ui.ok(f"  {pkg}")
        else:
            ui.info(f"Retrying {pkg} with verbose output...")
            if not _pip_install([pkg], quiet=False, raise_on_error=False):
                ui.fail(f"Core package {pkg} failed to install. Cannot continue.")
            ui.ok(f"  {pkg} (retry)")
    ui.ok("Core packages installed")

    # Stage B: heavy optional packages (track installed vs skipped)
    ui.info("Installing optional packages (may take a few minutes)...")
    opt_installed: list[str] = []
    opt_skipped: list[str] = []

    # Cross-platform optionals
    for pkg in ("chromadb", "scikit-image", "scikit-learn", "scipy"):
        if _pip_install([pkg], quiet=True, raise_on_error=False):
            ui.ok(f"  {pkg}")
            opt_installed.append(pkg)
        else:
            ui.warn(f"  {pkg} skipped")
            opt_skipped.append(pkg)

    # Platform-specific optionals
    if sys.platform == "darwin":
        pyobjc_pkgs = [
            "pyobjc-core",
            "pyobjc-framework-Cocoa",
            "pyobjc-framework-Quartz",
            "pyobjc-framework-ApplicationServices",
        ]
        for pkg in pyobjc_pkgs:
            if _pip_install([pkg], quiet=True, raise_on_error=False):
                ui.ok(f"  {pkg}")
                opt_installed.append(pkg)
            else:
                ui.warn(f"  {pkg} skipped")
                opt_skipped.append(pkg)
    elif sys.platform == "win32":
        # These are ALSO declared in productivity-tracker/pyproject.toml as
        # sys_platform-marked deps, but installing them explicitly here surfaces
        # any wheel-build failures with a readable error instead of the silent
        # skip we saw on at least one Python 3.12 Windows install.
        win_pkgs = ["mss>=9.0", "pywin32>=306", "pynput>=1.7"]
        for pkg in win_pkgs:
            if _pip_install([pkg], quiet=True, raise_on_error=False):
                ui.ok(f"  {pkg}")
                opt_installed.append(pkg)
            else:
                ui.warn(f"  {pkg} skipped — screenshots/window detection may fail")
                opt_skipped.append(pkg)

    if opt_skipped:
        print()
        ui.warn(f"Optional packages skipped: {' '.join(opt_skipped)}")
        _explain_skipped(opt_skipped)
        print()
    ui.ok(f"Optional packages done ({len(opt_installed)} installed, {len(opt_skipped)} skipped)")

    # Stage C: install tracker + ProMe + platform as editable packages
    ui.info("Installing productivity-tracker package...")
    if not _pip_install_editable(paths.TRACKER_DIR, quiet=True, raise_on_error=False):
        _pip_install_editable(paths.TRACKER_DIR, quiet=False, raise_on_error=False)
    ui.ok("Productivity tracker installed")

    ui.info("Installing ProMe requirements...")
    pmis_req = paths.PMIS_DIR / "requirements.txt"
    if pmis_req.is_file():
        _pip_install(["-r", str(pmis_req)], quiet=True, raise_on_error=False)
    ui.ok("ProMe done")

    ui.info("Installing platform requirements...")
    platform_req = paths.PLATFORM_DIR / "requirements.txt"
    if platform_req.is_file():
        _pip_install(["-r", str(platform_req)], quiet=True, raise_on_error=False)
    ui.ok("Platform done")

    # Verify critical imports
    ui.info("Verifying critical imports...")
    verify_script = """
ok, missing = [], []
for mod, name in [('openai','openai'), ('sqlalchemy','sqlalchemy'), ('fastapi','fastapi'),
                   ('numpy','numpy'), ('yaml','pyyaml'), ('PIL','Pillow'), ('dotenv','python-dotenv'),
                   ('uvicorn','uvicorn'), ('httpx','httpx'), ('psutil','psutil')]:
    try:
        __import__(mod)
        ok.append(name)
    except ImportError:
        missing.append(name)
print(f'{len(ok)} OK, {len(missing)} missing')
if missing:
    print('Missing: ' + ', '.join(missing))
"""
    result = subprocess.run(
        [str(paths.VENV_PYTHON), "-c", verify_script],
        capture_output=True,
        text=True,
    )
    print(f"  {result.stdout.strip()}")
    if "0 missing" in result.stdout:
        ui.ok("All critical imports verified")
    else:
        ui.warn("Some packages missing — check errors above")


def _create_venv(python_exe: str) -> None:
    try:
        subprocess.run(
            [python_exe, "-m", "venv", str(paths.VENV_DIR)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        ui.fail(f"Failed to create venv: {e.stderr.decode(errors='ignore') if e.stderr else e}")


def _pip_install(args: list[str], *, quiet: bool, raise_on_error: bool = True) -> bool:
    cmd = [str(paths.VENV_PYTHON), "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(args)
    try:
        subprocess.run(cmd, check=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError:
        if raise_on_error:
            raise
        return False


def _pip_install_editable(path: Path, *, quiet: bool, raise_on_error: bool = True) -> bool:
    cmd = [str(paths.VENV_PYTHON), "-m", "pip", "install", "-e", str(path), "--no-deps"]
    if quiet:
        cmd.append("-q")
    try:
        subprocess.run(cmd, check=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError:
        if raise_on_error:
            raise
        return False


def _explain_skipped(skipped: list[str]) -> None:
    notes = {
        "chromadb":      "No vector search (falls back to linear scan)",
        "scikit-image":  "No frame segmentation (screenshot dedup disabled)",
        "scikit-learn":  "Some ML consolidation features unavailable",
        "scipy":         "Matrix operations fall back to numpy",
    }
    pyobjc_prefix = "pyobjc"
    for pkg in skipped:
        if pkg in notes:
            ui.warn(f"  \u2192 {notes[pkg]}")
        elif pkg.startswith(pyobjc_prefix):
            ui.warn("  \u2192 Input monitor uses fallback (less accurate app detection)")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Directory structure
# ═══════════════════════════════════════════════════════════════════════════

def step3_directories() -> None:
    ui.step(3, TOTAL_STEPS, "Creating directory structure")
    for d in paths.all_dirs_to_create():
        d.mkdir(parents=True, exist_ok=True)
    ui.ok("All directories created")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Environment configuration (.env)
# ═══════════════════════════════════════════════════════════════════════════

# Placeholder markers in .env.example that indicate "not yet configured".
# Note: `yantrai.workers.dev` is the REAL default proxy URL for YantrAI employees,
# not a placeholder — do not add it here or step 4 will re-prompt on every run.
_TOKEN_PLACEHOLDER = "prome_your-personal-access-token-here"
_PROXY_PLACEHOLDER_OLD = "https://prome-openai-proxy.your-account.workers.dev/v1"
# Current default shipped in .env.example. Used as the prompt default so users
# can press Enter to accept.
_PROXY_DEFAULT = "https://prome-openai-proxy.yantrai.workers.dev/v1"


def step4_env() -> None:
    ui.step(4, TOTAL_STEPS, "Environment configuration")

    skip_prompt = False
    if paths.ENV_FILE.is_file():
        existing = paths.ENV_FILE.read_text(encoding="utf-8")
        has_placeholder = any(
            m in existing
            for m in (_TOKEN_PLACEHOLDER, _PROXY_PLACEHOLDER_OLD)
        )
        if has_placeholder:
            ui.warn("Existing .env has placeholders — will prompt for real values")
        else:
            ui.info("Existing .env found, preserving...")
            ui.ok(".env preserved")
            skip_prompt = True

    if skip_prompt:
        return

    print()
    ui.warn("This tracker talks to OpenAI through a company proxy.")
    ui.warn("Your admin should have given you:")
    ui.warn("  1. A proxy URL  (e.g. https://prome-openai-proxy.<account>.workers.dev)")
    ui.warn("  2. A personal access token  (starts with prome_)")
    print()

    # Prompt for proxy URL (default = YantrAI's production proxy)
    proxy_url = ""
    for attempt in range(2):
        proxy_url = ui.prompt("Proxy URL", default=_PROXY_DEFAULT).strip()
        if not proxy_url:
            ui.warn("No proxy URL — daemon screenshot analysis will fail until set.")
            proxy_url = _PROXY_PLACEHOLDER_OLD
            break
        if proxy_url.startswith("https://"):
            proxy_url = proxy_url.rstrip("/")
            if not proxy_url.endswith("/v1"):
                proxy_url += "/v1"
            ui.ok(f"Proxy URL accepted: {proxy_url}")
            break
        ui.warn("Proxy URL should start with https://")
        if attempt == 1:
            ui.warn("Using entered value as-is.")

    # Prompt for access token
    access_token = ""
    for attempt in range(2):
        access_token = ui.prompt("Personal access token (prome_...)").strip()
        if not access_token:
            ui.warn("No token entered — daemon will fail on screenshot analysis.")
            ui.warn(f"Add it later: edit {paths.ENV_FILE}")
            access_token = _TOKEN_PLACEHOLDER
            break
        if access_token.startswith("prome_") and len(access_token) >= 20:
            ui.ok("Access token accepted")
            break
        ui.warn("Token should start with 'prome_' and be at least 20 chars.")
        if attempt == 1:
            ui.warn("Using entered value as-is.")

    # Copy .env.example -> .env and substitute placeholders
    if not paths.ENV_EXAMPLE.is_file():
        ui.fail(f".env.example not found at {paths.ENV_EXAMPLE}")
    content = paths.ENV_EXAMPLE.read_text(encoding="utf-8")
    content = content.replace(_TOKEN_PLACEHOLDER, access_token)
    # Substitute whichever proxy URL is currently in .env.example (old placeholder
    # or the yantrai.workers.dev default) with the user's chosen URL.
    content = content.replace(_PROXY_PLACEHOLDER_OLD, proxy_url)
    content = content.replace(_PROXY_DEFAULT, proxy_url)
    content = content.replace("__HOME__", str(Path.home()))
    content = content.replace("__REPO__", str(paths.REPO_DIR))
    paths.ENV_FILE.write_text(content, encoding="utf-8")
    ui.ok(".env created")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Database initialization
# ═══════════════════════════════════════════════════════════════════════════

def step5_databases() -> None:
    ui.step(5, TOTAL_STEPS, "Initializing databases")

    # Tracker DB (SQLAlchemy)
    ui.info("Initializing tracker database...")
    tracker_db_path = paths.DATA_DIR / "tracker.db"
    tracker_script = f"""
import os, sys
os.environ['SQLITE_DB_PATH'] = r'{tracker_db_path}'
sys.path.insert(0, r'{paths.TRACKER_DIR}')
from src.storage.db import Database
db = Database(r'{tracker_db_path}')
db.initialize()
db.close()
print('  Tables: context_1, context_2, hourly_memory, daily_memory, deliverables')
"""
    if not _run_venv_script(tracker_script, cwd=paths.TRACKER_DIR):
        ui.fail("Tracker DB init failed")
    ui.ok(f"Tracker DB initialized at {tracker_db_path}")

    # ProMe DB
    ui.info("Initializing ProMe database...")
    pmis_db_path = paths.PMIS_DIR / "data" / "memory.db"
    pmis_script = f"""
import sys
sys.path.insert(0, r'{paths.PMIS_DIR}')
from db.manager import DBManager
db = DBManager(db_path=r'{pmis_db_path}')
n = db.count_nodes()
print(f'  Nodes: {{n}}, Tables: memory_nodes, embeddings, relations, projects, deliverables, ...')
db.close()
"""
    if not _run_venv_script(pmis_script, cwd=paths.PMIS_DIR):
        ui.fail("ProMe DB init failed")
    ui.ok(f"ProMe DB initialized at {pmis_db_path}")

    # Platform users DB (auto-init on import)
    ui.info("Initializing platform database...")
    platform_script = f"""
import sys
sys.path.insert(0, r'{paths.PLATFORM_DIR}')
from platform_db import get_users_db
conn = get_users_db()
conn.close()
print('  Tables: users, api_tokens, sharing_rules, access_requests, sync_log')
"""
    if not _run_venv_script(platform_script, cwd=paths.PLATFORM_DIR):
        ui.fail("Platform DB init failed")
    ui.ok(f"Platform DB initialized at {paths.PLATFORM_DIR / 'data' / 'users.db'}")


def _run_venv_script(script: str, *, cwd: Path) -> bool:
    try:
        subprocess.run(
            [str(paths.VENV_PYTHON), "-c", script],
            cwd=str(cwd),
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — End-to-end verification
# ═══════════════════════════════════════════════════════════════════════════

def step6_verify() -> None:
    ui.step(6, TOTAL_STEPS, "End-to-end verification")

    verify_script = paths.REPO_DIR / "verify_install.py"
    if not verify_script.is_file():
        ui.warn(f"verify_install.py not found at {verify_script} — skipping")
        return

    result = subprocess.run(
        [str(paths.VENV_PYTHON), str(verify_script)],
        cwd=str(paths.REPO_DIR),
    )
    if result.returncode != 0:
        ui.warn(f"E2E verification had issues (exit code {result.returncode}) — continuing anyway")
    else:
        ui.ok("E2E verification passed")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Install and start daemon
# ═══════════════════════════════════════════════════════════════════════════

def step7_daemon() -> None:
    label = "LaunchAgent" if sys.platform == "darwin" else "Startup folder shortcut"
    ui.step(7, TOTAL_STEPS, f"Installing tracker daemon ({label})")

    started, detail = service.install_daemon()
    if started:
        ui.ok(f"Tracker {detail}")
    else:
        ui.warn(f"Tracker daemon NOT installed — {detail}")
        ui.warn("Screenshots will not be captured until you fix this.")

    # Step 7b — write dev configs (cross-platform JSON, pathlib handles separators)
    _write_claude_launch_json()
    _write_mcp_config_json()


def _write_claude_launch_json() -> None:
    ui.info("Writing .claude/launch.json for dev servers...")
    paths.CLAUDE_LAUNCH_JSON.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "0.0.1",
        "configurations": [
            {
                "name": "pmis-v2-server",
                "runtimeExecutable": "python3",
                "runtimeArgs": [str(paths.PMIS_DIR / "server.py")],
                "port": 8100,
            }
        ],
    }
    paths.CLAUDE_LAUNCH_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    ui.ok(".claude/launch.json generated")


def _write_mcp_config_json() -> None:
    ui.info("Writing pmis_v2/claude_mcp_config.json...")
    config = {
        "mcpServers": {
            "pmis-memory": {
                "command": str(paths.VENV_PYTHON),
                "args": [str(paths.PMIS_DIR / "mcp_server.py"), "--transport", "stdio"],
                "env": {"PYTHONPATH": str(paths.PMIS_DIR)},
            }
        }
    }
    paths.MCP_CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    ui.ok("MCP config generated")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Launch web servers, verify, open browser, summarize
# ═══════════════════════════════════════════════════════════════════════════

_SERVER_PORTS = {
    "ProMe API + Wiki":   (8100, "pmis_v2",                  "server.py"),
    "Ops Dashboard":      (8200, "pmis_v2",                  "health_dashboard.py"),
    "Platform Portal":    (8000, "memory_system/platform",   "server.py"),
}


def step8_launch(perm_issues: bool) -> None:
    ui.step(8, TOTAL_STEPS, "Starting web servers and opening dashboard")

    # 1. Seed empty YAML files so /wiki/goals renders cleanly on fresh installs
    _seed_empty_yaml_files()

    # 2. Kill any stale listeners on our ports (idempotent re-install support)
    _kill_port_listeners([p for p, _, _ in _SERVER_PORTS.values()])

    # 3. Launch all three servers as detached background processes
    for label, (port, workdir_rel, script) in _SERVER_PORTS.items():
        workdir = paths.REPO_DIR / workdir_rel
        script_path = workdir / script
        if not script_path.exists():
            ui.warn(f"  {label}: {script_path} not found — skipping")
            continue
        try:
            _launch_detached(paths.VENV_PYTHON, script_path, workdir, port, label)
            ui.info(f"  Launching {label} on :{port}...")
        except Exception as e:
            ui.warn(f"  {label}: launch failed — {e}")

    # 4. Wait for servers to boot, then smoke-test each
    import urllib.request
    import urllib.error
    deadline_per_server = 15  # seconds
    time.sleep(2)

    all_ok = True
    for label, (port, _, _) in _SERVER_PORTS.items():
        url = f"http://127.0.0.1:{port}/"
        ok = False
        waited = 0
        last_err = ""
        while waited < deadline_per_server:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    # Any 2xx/3xx/4xx = server is up (some root paths 404, that's fine)
                    if resp.status < 500:
                        ok = True
                        break
                    last_err = f"HTTP {resp.status}"
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    ok = True
                    break
                last_err = f"HTTP {e.code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(2)
            waited += 2
        if ok:
            ui.ok(f"  {label} responding on :{port}")
        else:
            ui.warn(f"  {label} not responding on :{port} after {deadline_per_server}s ({last_err})")
            all_ok = False

    # 5. Open the main dashboard in the default browser
    main_url = "http://localhost:8100/wiki/goals"
    try:
        import webbrowser
        webbrowser.open(main_url)
        ui.ok(f"  Opening {main_url} in default browser")
    except Exception:
        ui.info(f"  Open manually: {main_url}")

    # 6. Final summary
    print()
    ui.banner("ProMe is running!")
    status_line = (
        f"  {ui.GREEN}Tracker:{ui.NC}      capturing screenshots in background"
        if _tracker_is_running()
        else f"  {ui.YELLOW}Tracker:{ui.NC}      NOT RUNNING — see step 7 warning"
    )
    print(status_line)
    print(f"  {ui.GREEN}Main UI:{ui.NC}      http://localhost:8100/wiki/goals")
    print(f"  {ui.GREEN}Ops:{ui.NC}          http://localhost:8200/")
    print(f"  {ui.GREEN}Portal:{ui.NC}       http://localhost:8000/")
    print(f"  {ui.GREEN}Data:{ui.NC}         {paths.DATA_DIR}")
    print(f"  {ui.GREEN}Logs:{ui.NC}         {paths.DATA_DIR / 'tracker-stderr.log'}")
    print()

    if perm_issues and sys.platform == "darwin":
        print(f"  {ui.RED}ACTION REQUIRED \u2014 Grant macOS permissions:{ui.NC}")
        print(f"  {ui.YELLOW}1.{ui.NC} System Settings > Privacy & Security > Screen Recording > Enable Terminal/Python")
        print(f"  {ui.YELLOW}2.{ui.NC} System Settings > Privacy & Security > Accessibility > Enable Terminal/Python")
        print()

    print(f"  {ui.BLUE}Useful commands:{ui.NC}")
    for label, cmd in service.summary_commands():
        print(f"    {label:15s} {cmd}")
    print()

    if not all_ok:
        ui.warn("Some servers are not responding. Check their terminal windows for errors,")
        ui.warn("or re-run the installer: {}".format(
            "install.bat" if sys.platform == "win32" else "./install.sh"
        ))


# ── Helpers for step 8 ─────────────────────────────────────────────────────

def _seed_empty_yaml_files() -> None:
    """Create empty goals.yaml / deliverables.yaml if they don't exist, so the
    wiki renders cleanly on a fresh install instead of choking on missing files."""
    config_dir = paths.TRACKER_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    goals = config_dir / "goals.yaml"
    deliv = config_dir / "deliverables.yaml"
    if not goals.exists():
        goals.write_text("goals: []\n", encoding="utf-8")
        ui.ok(f"  Seeded empty {goals.name}")
    if not deliv.exists():
        deliv.write_text("deliverables: []\n", encoding="utf-8")
        ui.ok(f"  Seeded empty {deliv.name}")


def _kill_port_listeners(ports: list[int]) -> None:
    """Kill whatever is listening on the given ports. Idempotent."""
    if sys.platform == "win32":
        port_list = ",".join(str(p) for p in ports)
        ps_cmd = (
            f"Get-NetTCPConnection -LocalPort {port_list} -ErrorAction SilentlyContinue | "
            f"Where-Object {{ $_.OwningProcess -gt 4 }} | "
            f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, timeout=10,
        )
    else:
        for port in ports:
            subprocess.run(
                f"lsof -ti :{port} 2>/dev/null | xargs -r kill -9 2>/dev/null",
                shell=True, capture_output=True, timeout=5,
            )


def _launch_detached(python_exe: Path, script: Path, workdir: Path,
                     port: int, label: str) -> None:
    """Start a Python server as a detached background process.
    Stdout/stderr go to a per-server log file in DATA_DIR for debugging."""
    log_path = paths.DATA_DIR / f"server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "cwd": str(workdir),
        "stdout": open(log_path, "ab"),
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        kwargs["creationflags"] = CREATE_NO_WINDOW | DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([str(python_exe), str(script)], **kwargs)


def _tracker_is_running() -> bool:
    """Best-effort check whether the tracker daemon is currently active."""
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["launchctl", "list", paths.LAUNCH_AGENT_LABEL],
                capture_output=True, text=True, timeout=3,
            )
            return r.returncode == 0 and '"PID"' in r.stdout
        except Exception:
            return False
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -like '*src.agent.tracker*' } | "
                 "Select-Object -First 1 -ExpandProperty ProcessId"],
                capture_output=True, text=True, timeout=5,
            )
            return (r.stdout or "").strip().isdigit()
        except Exception:
            return False
    return False
