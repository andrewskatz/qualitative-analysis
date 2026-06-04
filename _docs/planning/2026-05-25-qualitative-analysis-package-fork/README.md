# Fork `qualitative-analysis` into a Standalone Repo

**Date:** 2026-05-25
**Status:** ✅ Complete (2026-05-26). All six phases finished. See [`02-execution-log.md`](02-execution-log.md) for the full record.

## Overview

Extract the `qualitative-analysis/` subdirectory from this repo (`entity-id-app-v2`) into a standalone Git repo + GitHub project at `andrewskatz/qualitative-analysis`. The package has outgrown its parent: it covers four analysis pathways (figurative, relationships, entity, decisions), totals ~30k LOC, has no code coupling to the rest of `entity-id-app-v2`, and is increasingly the artifact that downstream consumers (publications, collaborators, citations) want to point at.

## Motivation

- Parent repo name (`entity-id-app-v2`) no longer describes the package — entity scoring is only one of four pathways.
- Issue tracker, releases, and Git tags on the parent are not meaningful for the package.
- Citation / methods-paper identity needs a stable, dedicated repo URL.
- Parent dir mixes unrelated artifacts (web-app frontend/backend, pilot experiments, MLX prototype, vendored repos) that pollute the mental model and dependency surface.
- Package internals are already cleanly decoupled — no cross-pathway code coupling exists (see [Current State](#current-state-snapshot) below), so the lift is mostly mechanical.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Git history | Preserve via `git filter-repo` |
| New repo location | `/Users/akatz4/Documents/ak fac/research/projects/qualitative-analysis` (sibling of `entity-id-app-v2`) |
| GitHub repo name | `andrewskatz/qualitative-analysis` |
| Package-related docs in parent | Move (planning dirs + matching progress notes) |
| Parent's reference to new package | `pip install` from Git URL (no submodule) |

## Current state snapshot

- Package lives at [`qualitative-analysis/`](../../../qualitative-analysis/) (subdir of parent repo, not a submodule).
- 14 of 68 parent commits touch the package directory.
- Pathways (zero cross-pathway imports between them):
  - `figurative/` (~4,750 LOC)
  - `relationships/` (~4,770 LOC)
  - `entity/` (~9,630 LOC)
  - `decisions/` (~2,500 LOC)
  - `core/` (~1,470 LOC, shared by all)
- External code that imports the package: only two files under [`publications-and-presentations/llm-qualitative-scoring-methodology/`](../../../publications-and-presentations/llm-qualitative-scoring-methodology/) (`select_sample.py`, `generate_tables.py`).
- 37 uncommitted files in the package directory (26 modified + 11 untracked) — must be committed before extraction so they end up in filtered history.
- `qualitative-analysis/output/` and `.venv*` are already gitignored — no committed binary/data baggage.
- `git-filter-repo` is **not** installed locally — needs `brew install git-filter-repo`.

## Plan

### Phase 0 — Pre-flight (safe, reversible)

1. **Install git-filter-repo**: `brew install git-filter-repo`
2. **Commit the 37 uncommitted files** in `qualitative-analysis/`. Split into multiple commits if any are WIP / should be isolated.
3. **Push parent to `origin`** — safety net if anything goes sideways.
4. **Make a bundle backup**: `git bundle create ~/qa-extraction-safety.bundle --all` in the parent. Single-file full-history backup; can restore the entire repo from it.
5. **Resolve the unrelated dirty submodule** `external_repos/figurative-language-app-v1` so it doesn't get accidentally pulled into the staging commit.

### Phase 1 — Lock down the path inventory

Before running `filter-repo`, produce a definitive list of paths that move vs. stay.

**Definitely moves (code):**
- `qualitative-analysis/` (entire dir)

**Candidate docs to move (need review):**

Planning dirs:
- `docs/planning/2025-12-22-figurative-language-module/`
- `docs/planning/2026-01-06-package-updates-for-domains/`
- `docs/planning/2026-01-12-qualitative-analysis-package-refactor/`
- `docs/planning/2026-01-15-qa-package-entity-relationship-functionality/`
- `docs/planning/2026-01-23-qa-package-entity-scoring-enhancements/`
- `docs/planning/2026-01-29-bayesian-hierarchical-modeling/`
- `docs/planning/2026-02-02-entity-scoring-bayesian-etc-polishes/`
- `docs/planning/2026-02-05-entity-detect-pathway/`
- `docs/planning/2026-02-07-entity-pathway-audit/`
- `docs/planning/2026-02-08-relationship-analysis-pathway-audit/`
- `docs/planning/2026-02-18-qa-ent-score-dimensions-specification/`
- `docs/planning/2026-02-28-qa-package-decisions-factors-port/`
- `docs/planning/2026-03-09-entity-scoring-audit/`
- `docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/`
- `docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/`
- `docs/planning/2026-05-25-qualitative-analysis-package-fork/` (this dir — moves with the fork as historical record)

Plus individual progress notes mentioning the package (full list will be generated during Phase 1).

**Deliverable:** Tick-list of paths in this dir as `01-path-inventory.md`. Each entry annotated "primary about the package?" yes/no. User reviews + approves before any history rewrite.

### Phase 2 — Run `filter-repo` in a sibling staging dir

The staging dir is on the same filesystem as the final dir, so the Phase 3
rename is atomic. Same-filesystem also avoids cross-device copy-then-delete.
If filter-repo errors out partway, the final dir name (`qualitative-analysis`)
is never created — we just `rm -rf` the staging dir and retry.

1. Clone parent into a sibling staging dir:
   `git clone "/Users/akatz4/Documents/ak fac/research/projects/entity-id-app-v2" "/Users/akatz4/Documents/ak fac/research/projects/qualitative-analysis-staging"`
2. Remove `origin` in the clone (filter-repo refuses by default to operate on a clone with remotes — this protects us from rewriting the parent's remote history).
3. Run `git filter-repo` with two kinds of args:
   - `--path qualitative-analysis/ --path docs/planning/<each-moved-dir>/ ...` (keep only these paths — exact list in [`01-path-inventory.md`](01-path-inventory.md))
   - `--path-rename qualitative-analysis/:` (lift the package contents to the new repo root)
4. **Result:** `qualitative-analysis-staging/` is a repo containing only filtered history, package at root, moved docs still under `docs/planning/...`. Verify with `git log --stat`, `find . -maxdepth 2`, and a check that no STAY paths leaked in.

### Phase 3 — Polish the new repo locally

1. Atomic rename to final location:
   `mv "/Users/akatz4/Documents/ak fac/research/projects/qualitative-analysis-staging" "/Users/akatz4/Documents/ak fac/research/projects/qualitative-analysis"`

   This is the moment the final project directory comes into existence. The rename is a no-op on disk (same parent dir), so it's safe and instant.
2. Update `README.md` — drop "Run from repo root: `PYTHONPATH=qualitative-analysis/src ...`" (package is now the root). Add standalone install/usage section.
3. Add `LICENSE` file (MIT, matching `pyproject.toml`).
4. Bump version in `pyproject.toml` (e.g., `0.1.0` → `0.2.0`) and add a brief `CHANGELOG.md` noting the extraction.
5. Confirm `.gitignore` covers `.venv*`, `output/`, `.DS_Store`, `__pycache__/`.
6. **Smoke test**: fresh venv → `pip install -e ".[dev,viz]"` → `pytest` → `qa --help`.

### Phase 4 — Push to GitHub

1. `gh repo create andrewskatz/qualitative-analysis --public --source=. --remote=origin --description "LLM-powered qualitative data analysis toolkit"` (use `--private` if you'd rather keep it closed for now).
2. `git push -u origin main`
3. Verify on GitHub: README renders, history is intact, `git log --stat` shows commits with original dates/authors.

### Phase 5 — Clean up the parent repo

Only after Phase 4 succeeds:
1. **Sync the planning dir's live state into the new repo.** The Phase 0/1 planning + 02-execution-log.md were committed in the parent and frozen into the new repo at extraction time (Phase 2). Phase 3/4 entries were edited in the parent's copy only, so the new repo's copy is stale. Before deleting from the parent, copy the parent's current `docs/planning/2026-05-25-qualitative-analysis-package-fork/` contents over the new repo's copy and commit there. This ensures the new repo retains the full audit trail.
2. In parent: `git rm -r qualitative-analysis/` plus each moved docs path.
3. Update parent `requirements.txt` to add: `qualitative-analysis @ git+https://github.com/andrewskatz/qualitative-analysis@main` (pin a tag/sha once tags exist).
3. Verify the two parent scripts that import the package still work:
   - `publications-and-presentations/llm-qualitative-scoring-methodology/human-coding-study/analysis/select_sample.py`
   - `publications-and-presentations/llm-qualitative-scoring-methodology/scripts/generate_tables.py`
4. Update parent `README.md` to point at the new repo.
5. Commit + push to parent `origin`.

### Phase 6 — End-to-end verification

1. Fresh clone of new repo into `/tmp` → install → run `qa entity --help`.
2. In parent's venv: `pip install -U -r requirements.txt` → run one publications-and-presentations script end-to-end.
3. Update Claude Code auto-memory:
   - The "Virtual Environment" memory (path to `.venv-qa-pkg`) becomes stale — fix or remove.
   - The "Audit (2026-02-07)" memory's path `docs/planning/2026-02-07-entity-pathway-audit/` now lives in the new repo; update.

## Risks & open items

- **Working directory hygiene during extraction**: avoid editing files in `qualitative-analysis/` during Phases 2–5. Short focused session is easiest.
- **`qualitative-analysis/output/`** (gitignored, ~56 items, experiment results) won't be in new Git history but the *files* exist on disk in `/tmp/qa-extract`. Decide whether to keep locally in the new dir or scrub and regenerate.
- **`.venv-qa-pkg/`** in the current package — gitignored, but present on disk. Delete and recreate fresh after the move.
- **`external_repos/figurative-language-app-v1`** dirty submodule in current `git status` is unrelated to extraction but must be resolved in Phase 0.5 to prevent contamination.
- **GitHub repo visibility**: public vs. private — confirm at Phase 4.

## Rollback

If anything goes wrong:
1. Before Phase 3 rename: discard `qualitative-analysis-staging/`, no parent changes have happened.
2. After Phase 3 rename but before Phase 4: `rm -rf` the local sibling dir at `~/.../qualitative-analysis`. Parent is still untouched.
3. After Phase 4 but before Phase 5: delete the GitHub repo, `rm -rf` the local sibling dir. Parent is still untouched.
4. After Phase 5: restore parent from `~/qa-extraction-safety.bundle` via `git clone ~/qa-extraction-safety.bundle entity-id-app-v2-restored`.

## Files in this planning dir

- [`README.md`](README.md) — this document
- `01-path-inventory.md` (to be created during Phase 1) — tick-list of paths to move
- `02-execution-log.md` (to be created during execution) — actual commands run, output, deviations

---

**Done.** This repo is live at `git@github.com:andrewskatz/qualitative-analysis.git` (private, tagged `v0.2.0`). The parent (`entity-id-app-v2`) installs this package via pinned pip URL in its `requirements.txt`. See [`02-execution-log.md`](02-execution-log.md) for the complete six-phase record.
