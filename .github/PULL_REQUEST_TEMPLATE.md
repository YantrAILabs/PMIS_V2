## Summary
One or two sentences on what this changes.

## Motivation
Why this matters (link issue if there is one: `Closes #123`).

## Changes
- Bullet list of the meaningful edits.

## Test plan
- [ ] `python pmis_v2/tests/test_e2e.py` passes
- [ ] `python -m pmis.cli --help` still works
- [ ] Manually exercised: `<describe>`

## Notes for reviewers
- Anything non-obvious.
- Any hyperparameter defaults touched (and why).
- Schema migrations (use `CREATE TABLE IF NOT EXISTS` / column-add via `_migrate()`).
