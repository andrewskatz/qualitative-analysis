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

**Status:** Pending — next step.

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
