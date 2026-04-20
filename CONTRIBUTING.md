# Contributing to PMIS

Thanks for considering a contribution.

## Development setup

```bash
git clone https://github.com/yourorg/pmis.git
cd pmis
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp .env.template .env  # add your OPENAI_API_KEY
```

Verify:

```bash
pmis --help
python3 -m pmis.cli stats
```

## Running tests

```bash
python3 pmis_v2/tests/test_e2e.py     # end-to-end pipeline (23 scenarios)
python3 pmis_v2/tests/test_p1_p2.py   # ChromaDB, batch embed, materialized stats, model consistency
```

Both must pass before sending a PR. CI runs them on Python 3.10 – 3.13.

## Linting

```bash
pip install ruff black
ruff check .
black --check .
```

Or just `black .` to auto-format.

## Pull request checklist

- [ ] One concern per PR — don't bundle unrelated changes.
- [ ] Tests added for new behavior; existing tests still pass.
- [ ] Public API changes documented in [README.md](README.md).
- [ ] If you touched hyperparameters, the default in [pmis_v2/hyperparameters.yaml](pmis_v2/hyperparameters.yaml) is commented.
- [ ] Commit messages are imperative and specific (`Fix surprise-gate off-by-one` not `Updates`).

## What's in scope

**Yes please:**
- Bug fixes with regression tests.
- New retrieval strategies, surprise metrics, or consolidation passes — behind a hyperparameter flag.
- Adapters for additional LLM providers in `pmis_v2/ingestion/embedder.py`.
- Dashboard panels under Work or Mind sections.
- Examples in `examples/`.

**Probably not without discussion first:**
- Breaking changes to the public 5-verb API (`ingest`, `attach`, `retrieve`, `consolidate`, `delete`).
- New top-level package directories.
- Replacing SQLite or ChromaDB.

Open an issue or Discussion before starting a large change.

## Filing issues

- **Bug** — use the bug report template. Include Python version, OS, and a minimal repro.
- **Feature request** — use the feature request template. Explain *why* first, implementation second.
- **Security** — see [SECURITY.md](SECURITY.md) (private disclosure).

## Code style

- Follow existing naming — snake_case for functions/vars, PascalCase for classes.
- Docstrings: one-line for small helpers; NumPy style for anything with non-obvious args.
- Prefer pure functions over classes where possible — most of `core/` and `ingestion/` is functional.
- Keep the public API (`pmis/api.py`) small and stable. New behavior goes into `pmis_v2/` first.

## Releases

Semver. `0.x.y` for alpha — anything may change. We'll promise API stability at `1.0`.

## Code of Conduct

Be decent. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
