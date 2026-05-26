# Changelog

## 0.2.0 — 2026-05-26

First release as a standalone repository. Previously lived as a subdirectory
of [`entity-id-app-v2`](https://github.com/andrewskatz/entity-id-app-v2);
extracted on 2026-05-25 via `git filter-repo` with full commit history
preserved. See
[`docs/planning/2026-05-25-qualitative-analysis-package-fork/`](docs/planning/2026-05-25-qualitative-analysis-package-fork/)
for the extraction plan and execution log.

### Repository changes
- Package code lifted from `qualitative-analysis/src/...` to `src/...` at the new repo root.
- 21 planning dirs, 11 progress notes, and 2 standalone docs (explainer +
  guide) carried over from the parent's `docs/`.
- Added top-level `LICENSE` (MIT) and `.gitignore`.

### No code changes in this release
This release is purely a re-homing. Package version bumped from 0.1.0 to 0.2.0
to signal the location change to any downstream consumer (notably the
publications-and-presentations scripts in `entity-id-app-v2`, which now
`pip install` this repo instead of importing from the sibling subdir).
