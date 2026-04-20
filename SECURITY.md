# Reporting issues

If you find a defect that has privacy or safety implications, please report it privately rather than opening a public issue.

- **Email:** rohit.singh.nitr@gmail.com — expect acknowledgement within 48 hours.
- **Subject prefix:** `[PMIS security]` so it routes correctly.

Please include:

- Affected commit or version.
- Reproduction steps and any logs.
- Your suggested fix if you have one.

We'll keep you in the loop on the fix and credit you in the release notes unless you prefer otherwise.

## Scope

PMIS is alpha software. Assume that local data under `~/.pmis/` and `pmis_v2/data/` is as trustworthy as your OS user account — we don't isolate it from other processes on the same machine. Don't run PMIS with credentials you wouldn't already expose to your own shell.

## Out of scope

- Reports against third-party dependencies (Ollama, ChromaDB, FastAPI, SQLite). File those upstream.
- Feature-request framed issues — use a regular GitHub issue.
