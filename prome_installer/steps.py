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
    # On Windows, mss/pywin32/pynput come in via `pip install -e .` below
    # because they're sys_platform-marked deps in tracker's pyproject.toml.

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
        existing = paths.ENV_FILE.read_text()
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
    content = paths.ENV_EXAMPLE.read_text()
    content = content.replace(_TOKEN_PLACEHOLDER, access_token)
    # Substitute whichever proxy URL is currently in .env.example (old placeholder
    # or the yantrai.workers.dev default) with the user's chosen URL.
    content = content.replace(_PROXY_PLACEHOLDER_OLD, proxy_url)
    content = content.replace(_PROXY_DEFAULT, proxy_url)
    content = content.replace("__HOME__", str(Path.home()))
    content = content.replace("__REPO__", str(paths.REPO_DIR))
    paths.ENV_FILE.write_text(content)
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
    label = "LaunchAgent" if sys.platform == "darwin" else "Task Scheduler task"
    ui.step(7, TOTAL_STEPS, f"Installing daemon ({label})")

    started, detail = service.install_daemon()
    if started:
        ui.ok(f"Daemon running ({detail})")
    else:
        ui.warn(f"Daemon may not have started — {detail}")

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
    paths.CLAUDE_LAUNCH_JSON.write_text(json.dumps(config, indent=2))
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
    paths.MCP_CONFIG_JSON.write_text(json.dumps(config, indent=2))
    ui.ok("MCP config generated")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Summary
# ═══════════════════════════════════════════════════════════════════════════

def step8_summary(perm_issues: bool) -> None:
    ui.step(8, TOTAL_STEPS, "Installation complete")

    ui.banner("Installation Complete!")
    start_cmd = "./start.sh" if sys.platform == "darwin" else "start.bat"
    portal_path = paths.REPO_DIR / ("start.sh" if sys.platform == "darwin" else "start.bat")
    print(f"  {ui.GREEN}Daemon:{ui.NC}     Running (captures screenshots every 10s)")
    print(f"  {ui.GREEN}Portal:{ui.NC}     Start with: {start_cmd}")
    print(f"  {ui.GREEN}Portal URL:{ui.NC} http://localhost:8000")
    print(f"  {ui.GREEN}Data:{ui.NC}       {paths.DATA_DIR}")
    print(f"  {ui.GREEN}Logs:{ui.NC}       {paths.DATA_DIR / 'tracker-stderr.log'}")
    print()

    if perm_issues and sys.platform == "darwin":
        print(f"  {ui.RED}ACTION REQUIRED \u2014 Grant macOS permissions:{ui.NC}")
        print(f"  {ui.YELLOW}1.{ui.NC} System Settings > Privacy & Security > Screen Recording > Enable Terminal/Python")
        print(f"  {ui.YELLOW}2.{ui.NC} System Settings > Privacy & Security > Accessibility > Enable Terminal/Python")
        print()

    print(f"  {ui.BLUE}Useful commands:{ui.NC}")
    for label, cmd in service.summary_commands():
        print(f"    {label:15s} {cmd}")
    print(f"    {'Start portal':15s} {start_cmd}")
    print()
