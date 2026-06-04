# Execution Log

Running record of commands executed and decisions made during the package extraction.
SHAs, file lists, and any deviations from the plan are captured here for auditability.

---

## Phase 0 — Pre-flight

**Status:** Complete (2026-05-25)

### 0.1 Install git-filter-repo

```
$ brew install git-filter-repo
$ which git-filter-repo
/opt/homebrew/bin/git-filter-repo
```

### 0.2 Pre-commit verification: full test suite

Ran the entire test suite in `qualitative-analysis/.venv-qa-pkg/` to make sure
we wouldn't immortalize a broken baseline in the filtered history:

```
$ pytest -q --tb=short -p no:cacheprovider
... 468 passed, 1 skipped, 30 warnings in 279.22s (0:04:39)
```

Clean baseline. The 1 skip is pre-existing (not introduced by current dirty state).
Warnings are SwigPyObject deprecations (unrelated, from a transitive dep) and a
`fork()` warning in the bayesian tests (pre-existing, multiprocessing/PyMC).

### 0.3 Commit the 37 dirty package files in 4 pathway-based commits

Per the agreed strategy ("split into 4 pathway-based commits"). The cross-cutting
files (`cli.py`, `unified_cli.py`, `README.md`, `test_mock_pipeline.py`,
`test_unified_cli.py`) turned out to be primarily figurative-themed (document the
figurative output contract, exercise the figurative mock pipeline), so they
landed in commit 1 rather than a separate "misc" bucket.

| # | SHA | Subject | Files | +ins | -del |
|---|-----|---------|-------|------|------|
| 1 | `b6810fe` | qa figurative: scanner/domains/strategy improvements + new tests | 14 | 1499 | 190 |
| 2 | `bf086c5` | qa entity: pathway improvements (bayesian, clustering, comparison, viz) | 14 | 744 | 217 |
| 3 | `1c17988` | qa core: ollama think-block stripping + tokenizer init error handling | 3 | 125 | 2 |
| 4 | `53461f0` | qa scripts: quantization + temperature scoring experiment tooling | 6 | 2027 | 30 |

Total: **37 files, +4395 / -439**. Matches the dirty-state tally (1747+439 from
26 modified + ~2645 LOC from 11 new) within expected diff-counting noise.

After all four commits, `git status --short qualitative-analysis/` returned empty.

### 0.4 Push parent to origin

Local `main` was 6 commits ahead of `origin/main` (4 new + 2 prior unpushed),
zero commits behind. Clean fast-forward, no force needed:

```
$ git push origin main
   529cdd3..53461f0  main -> main
```

### 0.5 Safety bundle

Full-history bundle of the parent repo created at `~/qa-extraction-safety.bundle`
(33 MB, sha1, records `refs/heads/main`, `refs/remotes/origin/main`, `HEAD`):

```
$ git bundle create ~/qa-extraction-safety.bundle --all
$ git bundle verify ~/qa-extraction-safety.bundle
The bundle records a complete history.
```

Restore path if needed:
`git clone ~/qa-extraction-safety.bundle entity-id-app-v2-restored`

### 0.6 Deviation: skipped "resolve the dirty submodule" step

The plan listed `external_repos/figurative-language-app-v1` as a step to
resolve. On inspection it's a nested git repo (no `.gitmodules` entry —
undeclared), pinned at commit `04233a4` with uncommitted work inside *its*
working tree. The parent only sees `-dirty` and won't commit a new SHA unless
we explicitly stage `external_repos/...`. Since all commits above used explicit
file paths (no `git add -A` or `git add .`), the gitlink remained untouched.

**Decision:** leave the nested repo as-is. Not contaminating anything, and
resolving it is unrelated work that doesn't belong to this extraction effort.

---

## Phase 1 — Path inventory

**Status:** Complete (2026-05-26); awaiting reviewer approval before Phase 2.

### 1.1 Catalog `docs/` for package-related content

Surveyed all subdirs of `docs/` (planning, progress-notes, audits, explanations,
guides, bug-reports, testing, setup, team-protocols, api, onboarding). Most
audits/bug-reports/testing/setup/team-protocols/api/onboarding docs are web-app
concerns and stay in the parent.

### 1.2 Classify each candidate (Explore agent + spot checks)

Delegated the per-doc classification to an Explore agent, which read the lead
file of each candidate planning dir and grep-scanned the progress-notes
directory. Spot-checked four cases:

| Path | Agent → Reviewed | Reason for adjustment |
|---|---|---|
| `2025-12-25-fig-lang-source-target-labeling/` | SPLIT → **STAY** | Doc is about the web-app `text_items` JSONB schema and "post-extraction analysis" inside the web app, citing SETS as design inspiration only. Not a package design doc. |
| `2026-04-08-scoring-experiment-multi-model-analysis.md` | STAY → **MOVE** | Lead line: `**Package:** qualitative-analysis`. Multi-model factorial scoring experiment run through the package's entity pathway. The agent misclassified, likely because the file was untracked when it grep-scanned. |
| `2026-05-25-qualitative-analysis-package-fork/` | (not surveyed by agent) → **MOVE** | This planning dir documents the extraction itself; moves with the package as historical record. |
| Agent "summary" line said 30 progress notes MOVE | corrected to **10 progress notes** | Summary line had a counting error; reconciled against the 9 it actually listed + the one above. |

### 1.3 Verify no collisions after `--path-rename qualitative-analysis/:`

```
$ comm -12 \
    <(find qualitative-analysis/docs -type f | sed 's|qualitative-analysis/||' | sort) \
    <(find docs -type f | sort)
(empty)
```

The package's only internal docs file
`qualitative-analysis/docs/progress-notes/2026-01-26-entity-scoring-context-fix.md`
will land at `docs/progress-notes/...` in the new repo with no collision against
the 10 progress notes moving from the parent.

### 1.4 Commit untracked package-related docs (so filter-repo picks them up)

Three doc paths were untracked when the inventory was being built. They are
all MOVE candidates, so they had to be committed before filter-repo or they
would be silently excluded from the new repo's history:

```
$ git add \
    docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/ \
    docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/ \
    docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md
$ git commit -m "docs: commit untracked qa package audits + scoring experiment note"
[main 0eee5fa] ... 9 files changed, 1477 insertions(+)
$ git push origin main
   ae1f296..0eee5fa  main -> main
```

### 1.5 Final inventory

See [`01-path-inventory.md`](01-path-inventory.md). Tally:

- **Code:** 1 directory (`qualitative-analysis/`, rename to root)
- **Planning dirs:** 21 dirs under `docs/planning/`
- **Standalone docs:** 2 files (one in `docs/explanations/`, one in `docs/guides/`)
- **Progress notes:** 10 files under `docs/progress-notes/`
- **Total filter-repo path args:** 34, plus the path-rename.

The inventory file also pre-renders the exact `git filter-repo` command line.

### 1.6 Plan revision: staging dir instead of `/tmp/qa-extract`

After the inventory was complete, on review the original Phase 2/Phase 3 plan
had filter-repo running in `/tmp/qa-extract` and then `mv`-ing the result to
the final location. Two issues with that:

- macOS `/tmp` is wiped on reboot — risk of losing in-progress work.
- `/tmp` is on a different filesystem from the final location, so `mv` becomes
  a copy-then-delete rather than an atomic rename.

Plan was revised to use a sibling staging dir
(`qualitative-analysis-staging/`, on the same filesystem as the final
`qualitative-analysis/`). The final directory comes into existence in Phase 3
via an atomic rename. Same overall safety properties (no half-baked repo
at the final path if anything errors), but cleaner and faster.

README updated; the inventory's filter-repo command is unchanged.


---

## Phase 2 — filter-repo

**Status:** Complete (2026-05-26).

### 2.1 First two attempts (false starts)

filter-repo's safety checks are picky about the shape of the clone. Both
false starts were resolved by re-cloning fresh:

| Attempt | Issue | Fix |
|---|---|---|
| 1 | "Refusing to destructively overwrite repo history since this does not look like a fresh clone. (expected freshly packed repo)" — `git clone` from a local path hardlinks objects | Re-clone with `--no-local` so objects are copied, not hardlinked |
| 2 | "Refusing… (expected one remote, origin)" — I had pre-removed origin | filter-repo removes origin itself; leave it intact before running |

Reasoning for choosing the re-clone path over `--force`: the staging dir is
cheap to recreate, and trusting filter-repo's safety checks is preferable to
overriding them.

### 2.2 Successful run

```
$ cd /Users/akatz4/Documents/ak\ fac/research/projects/qualitative-analysis-staging
$ git filter-repo \
    --path qualitative-analysis/ \
    --path docs/explanations/figurative-language-detection-explained.md \
    --path docs/guides/extractor-selection-guide.md \
    --path docs/planning/2025-12-22-figurative-language-module/ \
    [... 30 more --path args, see 01-path-inventory.md ...] \
    --path-rename qualitative-analysis/:
NOTICE: Removing 'origin' remote; see 'Why is my origin removed?' …
Parsed 76 commits
New history written in 0.68 seconds; now repacking/cleaning...
Completely finished after 1.87 seconds.
```

New `HEAD` of staging repo: `c75efc4 docs(qa-fork): revise Phase 2/3 to use sibling staging dir`.

### 2.3 Verification

| Check | Result |
|---|---|
| Package contents at root (no nested `qualitative-analysis/`) | ✅ |
| 21 planning dirs in `docs/planning/` | ✅ |
| 11 progress notes in `docs/progress-notes/` (10 moved + 1 from package internal docs, no collision) | ✅ |
| 2 standalone files (explainer + extractor guide) | ✅ |
| 29 test files in `tests/` (incl. data CSVs + `run_*` dirs that were committed in the package) | ✅ |
| Original authorship / dates preserved (commits go back to `Initial commit: Entity ID App v2`) | ✅ |
| Commits retained: 25 of 76 parent commits — those that touched at least one moved path | ✅ |
| Zero web-app leakage: `backend/`, `frontend/`, `server/`, `experiments/`, `external_repos/`, `archive/`, `pilot_results/`, `publications-and-presentations/` all absent | ✅ |
| Working tree size: 6.7 MB; `.git` size: 1.6 MB | ✅ |
| `.gitignore` present | ⚠️ Missing — lived at parent root, not under `qualitative-analysis/`, so was not in the filter-repo `--path` list. **Resolution:** create one fresh in Phase 3. |


---

## Phase 3 — Local polish

**Status:** In progress (2026-05-26).

### 3.1 Atomic rename to final location

```
$ mv /Users/akatz4/Documents/ak\ fac/research/projects/qualitative-analysis-staging \
     /Users/akatz4/Documents/ak\ fac/research/projects/qualitative-analysis
```

Same-filesystem atomic rename, as planned.

### 3.2 Discovery: pre-existing sibling repos

After the rename, noticed two unrelated dirs in the same parent:

- `qualitative-analysis-v0/` → remote `andrewskatz/qualitative-analysis-v2` (name mismatch)
- `qualitative-analysis-v1/` → remote `andrewskatz/qualitative-analysis-v1`

Both are pre-existing prior extraction attempts. Paused and asked the user;
they confirmed both are abandoned. Continued with the planned new repo at
`qualitative-analysis/` (no suffix), GitHub `andrewskatz/qualitative-analysis`.
Saved this finding as a project memory so future sessions don't lose context.

### 3.3 Standalone repo bootstrap (commit `0e04862`)

Files created / modified to make the new repo publishable as a standalone
Python project:

| File | Change |
|---|---|
| `README.md` | Rewritten. Drops `PYTHONPATH=qualitative-analysis/src python -m ...` runner pattern. Documents the four pathways, install instructions, optional dep groups, CLI examples, and dev workflow. |
| `LICENSE` | New: MIT, matches `pyproject.toml` `license` field. |
| `.gitignore` | New: Python build outputs, venvs (`.venv*`, `venv/`, `env/`), test/coverage/mypy caches, Jupyter checkpoints, env files, logs, pathway run outputs (`output/`, `results/`), IDE files, macOS noise. |
| `CHANGELOG.md` | New: documents v0.2.0 re-homing release. |
| `pyproject.toml` | `version` 0.1.0 → 0.2.0; `description` expanded to name the four pathways; `authors` placeholder replaced with real identity. |
| `src/qualitative_analysis/__init__.py` | `__version__` 0.1.0 → 0.2.0. |

**Author identity note:** committed under the global gitconfig identity
`Andrew Katz <akatz4@vt.edu>`, but `pyproject.toml` author email is
`akatz@tabbiresearch.com` (from project auto-memory). Two different real
emails belonging to the same user. Flagged for the user; not blocking
extraction.

### 3.4 Smoke test

Fresh `.venv` at `qualitative-analysis/.venv/` (Python 3.13.0).
`pip install -e ".[dev,viz]"` succeeded; `qualitative-analysis-0.2.0`
registered. The smoke test caught **two real package bugs** that had been
masked in the parent's venv. Fixed in commit `87cc587` (after the bootstrap
commit `0e04862`).

**Bug 1 — missing `pandas` dependency.** `entity/agreement.py` and
`entity/bayesian.py` both `import pandas as pd` at module level. agreement.py
is loaded eagerly via `entity/__init__.py`, so `import qualitative_analysis.entity`
failed outright in a clean install. The parent's `.venv-qa-pkg` had pandas
present (transitive accident — pandas comes in via many ML/stats packages),
so this was never exercised. Fix: added `pandas>=2.0.0` to core deps in
`pyproject.toml`.

**Bug 2 — hardcoded "0.1.0" in five places.** `core/cli_utils.py:PACKAGE_VERSION`,
plus four output-metadata `package_version` literals in `cli.py` and
`figurative/domains_cli.py`. After the version bump to 0.2.0, `qa --version`
still printed "0.1.0". Fix: derive `__version__` in `__init__.py` from
`importlib.metadata.version("qualitative-analysis")` (with `0.0.0+unknown`
fallback when run uninstalled), then re-export it as `PACKAGE_VERSION`
from `core/cli_utils.py` and reference that constant elsewhere. Single
source of truth is now `pyproject.toml`'s `[project] version` field.

Also fixed `tests/test_unified_cli.py::test_qa_version`, which had
hardcoded `assertIn("0.1.0", ...)`; now reads `__version__` at test time.

**Final pytest result:** 448 passed, 21 skipped, in 124.98s (Python 3.13).
The 21 skips are the entity Bayesian tests, which require `pymc`/`arviz`
(in optional dep group `bayes`). Smoke env only installed `[dev,viz]`,
which is the lighter common path. Parent venv had 1 skip because pymc was
opportunistically present; in the new repo the contract is honest — users
who want Bayes install `[bayes]`.

### 3.5 Phase 3 commit summary

| SHA | Subject |
|---|---|
| `0e04862` | Standalone repo bootstrap: README, LICENSE, .gitignore, CHANGELOG, v0.2.0 |
| `87cc587` | fix: add pandas dep + single source-of-truth for package version |

**Status:** Phase 3 complete; awaiting Phase 4 go-ahead (push to GitHub).
Phase 4 requires:
- Decision on repo visibility (public vs private)
- Confirmation of repo name `andrewskatz/qualitative-analysis` (v0/v1 use different names; this one is the no-suffix canonical)
- `gh` CLI installation (currently missing on this machine — `brew install gh`)
- `gh auth login` (no auth currently)


---

## Phase 4 — Push to GitHub

**Status:** Complete (2026-05-26).

### 4.1 Tooling friction (resolved)

Installed `gh` CLI via `brew install gh` (v2.92.0). User attempted
`gh auth login` and got partway — SSH key uploaded to GitHub, auth flow
finished — but `gh` couldn't persist auth state because `~/.config` on
this machine is owned by `root` (not the user); some 2024 sudo install
of fish shell config left it that way.

Pivoted to manual repo creation rather than chase the sudo chown. User
created the empty `andrewskatz/qualitative-analysis` repo (private) on
github.com with no README/.gitignore/LICENSE (so the local repo's
contents push without conflict).

### 4.2 Add remote + push

```
$ cd /Users/akatz4/Documents/ak\ fac/research/projects/qualitative-analysis
$ git remote add origin git@github.com:andrewskatz/qualitative-analysis.git
$ git push -u origin main
To github.com:andrewskatz/qualitative-analysis.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

### 4.3 Verification

| Check | Result |
|---|---|
| Remote `HEAD` matches local `HEAD` (`87cc587`) | ✅ |
| 27 commits on `origin/main` | ✅ |
| SSH protocol (uses the key uploaded during the abandoned gh auth flow) | ✅ |
| Repo visibility: private | ✅ |

---

## Phase 5 — Clean up parent

**Status:** Complete (2026-05-26).

### 5.0 Pre-step: tag v0.2.0 on this repo

```
$ git tag -a v0.2.0 -m "v0.2.0 — first release as standalone repo"
$ git push origin v0.2.0
```

So `requirements.txt` in the parent can pin to `@v0.2.0` rather than tracking
`@main`.

### 5.1 Sync planning dir to this repo

Before the parent deleted its copy of this planning dir, rsync'd the
parent's current state over this repo's frozen-at-extraction copy.
Captured Phase 2/3/4 entries that had been written in the parent post-extraction.

Commit in this repo: `b44f7bb docs: sync fork plan from parent (adds Phase 2/3/4 entries)`.

### 5.2 Deprecation strategy: `mv` instead of `rm`

Per user preference, in the parent's working tree the to-be-removed paths
were `mv`'d into a gitignored safety folder
`entity-id-app-v2/_deprecated_qa_fork_20260526/` (atomic same-filesystem
renames) rather than `rm`'d. They are gone from git history but recoverable
on disk for as long as the user keeps the deprecated folder around.

Paths moved (34 total → 250 tracked files / 67 808 lines):
- `qualitative-analysis/` (the package directory itself)
- 21 planning dirs under `docs/planning/...`
- 10 progress notes under `docs/progress-notes/...`
- 2 standalone files (`docs/explanations/figurative-language-detection-explained.md`, `docs/guides/extractor-selection-guide.md`)

### 5.3 `requirements.txt` update

Added to parent:
```
qualitative-analysis @ git+ssh://git@github.com/andrewskatz/qualitative-analysis.git@v0.2.0
```
SSH (not HTTPS) because the new repo is private and the user already has
their SSH key on GitHub; HTTPS would have needed credential setup.

### 5.4 Install in parent venv + verify publications scripts

Two scripts in `publications-and-presentations/llm-qualitative-scoring-methodology/`
were known consumers. Two findings:

1. **Parent's `venv/bin/pip` is broken** — its shebang hardcodes a path to
   `/Users/akatz4/Documents/ak fac/research/projects/entity-id-app-v1/venv/bin/python`
   (the venv was copied from v1 to v2 long ago but never had its scripts
   rewritten). First install attempt silently installed into v1's venv.
   Workaround: use `venv/bin/python -m pip` (which is correct). Memory note
   added so this doesn't bite us again.

2. **Publications scripts don't actually import the package** — they only
   read CSV outputs from `qualitative-analysis/output/`. Since that dir was
   3.8 GB and 53 experiment subdirs, gitignored, and not really "package
   output" anymore (it's historical scoring-experiment data the manuscript
   cites), we relocated it: `_deprecated_.../qualitative-analysis/output/`
   → `entity-id-app-v2/scoring-experiment-output/` (added to parent
   .gitignore). Both scripts updated to point at the new path:
   - `human-coding-study/analysis/select_sample.py`
   - `scripts/generate_tables.py`
   (Both scripts are in an untracked dir in the parent — `git status`
   showed `publications-and-presentations/llm-qualitative-scoring-methodology/`
   as `??` — so the edits aren't part of the Phase 5 parent commit; they
   will be carried when the user eventually tracks/commits that dir.)

### 5.5 Parent README pointer

Added a brief note at the top of `entity-id-app-v2/README.md` directing
readers to the new repo and the planning dir.

### 5.6 Phase 5 commit + push (parent)

```
[main 13c8edc] qa package fork — Phase 5: parent cleanup
 253 files changed, 14 insertions(+), 67808 deletions(-)
```

Pushed to `andrewskatz/entity-id-app-v2`.

---

## Phase 6 — Verification

**Status:** Complete (2026-05-26).

### 6.1 Fresh-clone smoke test (this repo)

Simulated a brand-new contributor:

```
$ rm -rf /tmp/qa-fresh-clone
$ git clone git@github.com:andrewskatz/qualitative-analysis.git /tmp/qa-fresh-clone
$ cd /tmp/qa-fresh-clone
$ git log --oneline -1
b44f7bb docs: sync fork plan from parent (adds Phase 2/3/4 entries)
$ git tag -l
v0.2.0
$ python3.13 -m venv .venv
$ .venv/bin/pip install -e ".[dev]"
... (succeeded; pip notice only)
$ .venv/bin/qa --help
... shows figurative / relationships / entity / decisions pathways
$ .venv/bin/qa --version
qa 0.2.0
```

### 6.2 Publications script smoke test (parent)

```
$ cd /Users/akatz4/Documents/ak\ fac/research/projects/entity-id-app-v2
$ venv/bin/python publications-and-presentations/llm-qualitative-scoring-methodology/scripts/generate_tables.py --list
models-tested
cross-model-summary
scale-effects
prompt-effects
temperature-reliability
quantization-per-variant
quantization-pairwise
quantization-runs-sanity
```

Script resolved `QA_OUTPUT` to `scoring-experiment-output/`, found data,
listed available tables. End-to-end path resolution confirmed.

### 6.3 Auto-memory update

Stale memory entries updated in
`/Users/akatz4/.claude/projects/-Users-akatz4-Documents-ak-fac-research-projects-entity-id-app-v2/memory/MEMORY.md`:

- "Virtual Environment" entry: now points at the new repo's
  `.venv/` instead of the moved `.venv-qa-pkg/`; also notes the
  broken parent `venv/bin/pip` (use `python -m pip` instead).
- "Audit (2026-02-07)" entry: notes the audit docs are now in the
  new repo at the same relative path.

Also added: `project_qa_package_repos.md` documenting the canonical
`qualitative-analysis/` repo plus the abandoned `v0/`/`v1/` siblings.

---

## Final state

| Repo | Status |
|---|---|
| **`andrewskatz/qualitative-analysis`** (new, private) | Live at `v0.2.0`; 27 commits + this planning commit; tests green; CLI working |
| **`andrewskatz/entity-id-app-v2`** (parent) | 253 files / 67 k lines removed; depends on new package via pinned pip install; publications scripts updated to read from `scoring-experiment-output/`; safety bundle at `~/qa-extraction-safety.bundle` |
| **Local-only safety folders** | `entity-id-app-v2/_deprecated_qa_fork_20260526/` (gitignored, ~64 k items, recoverable until user deletes) |

Extraction complete. Both repos pushed; no outstanding work.
