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

**Status:** Pending.

---

## Phase 3 — Local polish

**Status:** Pending.

---

## Phase 4 — Push to GitHub

**Status:** Pending.

---

## Phase 5 — Clean up parent

**Status:** Pending.

---

## Phase 6 — Verification

**Status:** Pending.
