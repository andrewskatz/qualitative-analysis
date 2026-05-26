# Path Inventory for Extraction

**Status:** Awaiting approval before Phase 2 (`git filter-repo`)

Definitive list of paths from `entity-id-app-v2` that should MOVE with the
package into the new `qualitative-analysis` repo, or STAY in the parent.

After approval, the paths in the MOVE list become `git filter-repo --path …`
arguments. Anything not on the MOVE list disappears from the new repo's
history.

---

## MOVE — Package code (1 path, gets path-renamed to root)

- [`qualitative-analysis/`](../../../qualitative-analysis/) — full package
  directory, including `src/`, `tests/`, `scripts/`, `docs/` (internal),
  `pyproject.toml`, `README.md`

After `--path-rename qualitative-analysis/:` this becomes the new repo's root.
The internal `qualitative-analysis/docs/progress-notes/2026-01-26-entity-scoring-context-fix.md`
lands at `docs/progress-notes/2026-01-26-entity-scoring-context-fix.md` — no
collision with parent docs (verified).

## MOVE — Planning dirs (21 dirs)

All have been verified as primarily about the package (design, audit,
implementation, or porting work for one of the four pathways), not about the
web app:

- `docs/planning/2025-12-22-figurative-language-module/` — package architecture / modularization
- `docs/planning/2026-01-05-fig-lang-domain-mapping-improvements/` — domain mapping workflow audit
- `docs/planning/2026-01-06-graph-semantic-layout/` — semantic graph layout feature
- `docs/planning/2026-01-06-package-updates-for-domains/` — domain mapping & normalization CLI
- `docs/planning/2026-01-07-fig-lang-domain-graph-fixes/` — raw domain graph abstraction selector
- `docs/planning/2026-01-12-qualitative-analysis-package-refactor/` — unified `qa` CLI refactor
- `docs/planning/2026-01-14-domain-graph-improvements/` — cluster-based region labeling
- `docs/planning/2026-01-15-qa-package-entity-relationship-functionality/` — entity/relationship port
- `docs/planning/2026-01-23-qa-package-entity-scoring-enhancements/` — multi-participant comparison
- `docs/planning/2026-01-29-bayesian-hierarchical-modeling/` — Bayesian modeling design
- `docs/planning/2026-02-02-entity-scoring-bayesian-etc-polishes/` — Bayesian polish/validation
- `docs/planning/2026-02-05-entity-detect-pathway/` — `qa entity detect` standalone command
- `docs/planning/2026-02-07-entity-pathway-audit/` — entity pathway audit (full)
- `docs/planning/2026-02-08-relationship-analysis-pathway-audit/` — relationship pathway audit
- `docs/planning/2026-02-17-bayesian-hierarchical-modeling-audit/` — Bayesian correctness audit
- `docs/planning/2026-02-18-qa-ent-score-dimensions-specification/` — configurable scoring scale
- `docs/planning/2026-02-28-qa-package-decisions-factors-port/` — decisions/factors port
- `docs/planning/2026-03-09-entity-scoring-audit/` — entity ID + SETS scoring audit
- `docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/` — figurative pathway audit
- `docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/` — final entity pathway audit
- `docs/planning/2026-05-25-qualitative-analysis-package-fork/` — this planning dir (moves itself)

## MOVE — Standalone docs (2 files)

- `docs/explanations/figurative-language-detection-explained.md` — explainer doc covering the package's detection strategies and use cases
- `docs/guides/extractor-selection-guide.md` — guide to picking entity extraction methods exposed by the package

## MOVE — Progress notes (10 files)

Progress notes whose primary topic is the package:

- `docs/progress-notes/2026-01-09-domain-mapping-cli-implementation.md`
- `docs/progress-notes/2026-01-15-qualitative-analysis-package-updates.md`
- `docs/progress-notes/2026-01-16-qa-package-entity-relationship-foundation.md`
- `docs/progress-notes/2026-01-19-qa-package-entity-pipeline-complete.md`
- `docs/progress-notes/2026-02-02-bayesian-hierarchical-modeling-implementation.md`
- `docs/progress-notes/2026-02-02-bayesian-polish-and-cli-improvements.md`
- `docs/progress-notes/2026-02-03-docstring-review-and-audit-preparation.md`
- `docs/progress-notes/2026-02-04-audit-option-c-tier1-tier2-fixes.md`
- `docs/progress-notes/2026-02-06-entity-detect-implementation.md`
- `docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md`

---

## STAY (representative — all unlisted paths stay)

For transparency, the docs paths explicitly classified as STAY:

**Planning dirs (web app / unrelated):**
- `docs/planning/2025-11-04-fig-lang-api-integration/` — web app API endpoint integration
- `docs/planning/2025-11-04-two-step-with-summaries-figurative-language/` — web app figurative UI + batch + CSV export
- `docs/planning/2025-11-06-fig-lang-export-update/` — web app CSV export robustness
- `docs/planning/2025-12-25-fig-lang-source-target-labeling/` — web app source/target labeling feature (cites SETS as design inspiration but is about the `text_items` JSONB schema, not the package). Originally flagged SPLIT by inventory agent; on review, the doc is web-app-centric.

All other `docs/planning/*` (entity-id-app-v2, web app, pilot, manuscript work) stay.

**Code that stays in parent:**
- `backend/`, `frontend/`, `server/` — web app
- `src/`, `tests/`, `scripts/` at parent root — pilot/experiment scripts
- `experiments/`, `pilot_results/`, `archive/` — pilot work
- `external_repos/` — vendored repos
- `publications-and-presentations/` — manuscript work (imports the package via pip after extraction)
- Top-level `test_*.py` files at parent root — web-app integration tests

**All progress notes not explicitly in the MOVE list above stay.**

---

## Filter-repo command this inventory derives

The actual command we'll run in Phase 2 (in a fresh clone with `origin` removed):

```bash
git filter-repo \
  --path qualitative-analysis/ \
  --path docs/explanations/figurative-language-detection-explained.md \
  --path docs/guides/extractor-selection-guide.md \
  --path docs/planning/2025-12-22-figurative-language-module/ \
  --path docs/planning/2026-01-05-fig-lang-domain-mapping-improvements/ \
  --path docs/planning/2026-01-06-graph-semantic-layout/ \
  --path docs/planning/2026-01-06-package-updates-for-domains/ \
  --path docs/planning/2026-01-07-fig-lang-domain-graph-fixes/ \
  --path docs/planning/2026-01-12-qualitative-analysis-package-refactor/ \
  --path docs/planning/2026-01-14-domain-graph-improvements/ \
  --path docs/planning/2026-01-15-qa-package-entity-relationship-functionality/ \
  --path docs/planning/2026-01-23-qa-package-entity-scoring-enhancements/ \
  --path docs/planning/2026-01-29-bayesian-hierarchical-modeling/ \
  --path docs/planning/2026-02-02-entity-scoring-bayesian-etc-polishes/ \
  --path docs/planning/2026-02-05-entity-detect-pathway/ \
  --path docs/planning/2026-02-07-entity-pathway-audit/ \
  --path docs/planning/2026-02-08-relationship-analysis-pathway-audit/ \
  --path docs/planning/2026-02-17-bayesian-hierarchical-modeling-audit/ \
  --path docs/planning/2026-02-18-qa-ent-score-dimensions-specification/ \
  --path docs/planning/2026-02-28-qa-package-decisions-factors-port/ \
  --path docs/planning/2026-03-09-entity-scoring-audit/ \
  --path docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/ \
  --path docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/ \
  --path docs/planning/2026-05-25-qualitative-analysis-package-fork/ \
  --path docs/progress-notes/2026-01-09-domain-mapping-cli-implementation.md \
  --path docs/progress-notes/2026-01-15-qualitative-analysis-package-updates.md \
  --path docs/progress-notes/2026-01-16-qa-package-entity-relationship-foundation.md \
  --path docs/progress-notes/2026-01-19-qa-package-entity-pipeline-complete.md \
  --path docs/progress-notes/2026-02-02-bayesian-hierarchical-modeling-implementation.md \
  --path docs/progress-notes/2026-02-02-bayesian-polish-and-cli-improvements.md \
  --path docs/progress-notes/2026-02-03-docstring-review-and-audit-preparation.md \
  --path docs/progress-notes/2026-02-04-audit-option-c-tier1-tier2-fixes.md \
  --path docs/progress-notes/2026-02-06-entity-detect-implementation.md \
  --path docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md \
  --path-rename qualitative-analysis/:
```

Total: 1 package path + 21 planning dirs + 2 explainer/guide files + 10 progress notes = **34 path arguments**, plus the rename that lifts the package to root.

---

## Things this inventory does not move (and why)

- **`backend/`, `frontend/`, `server/`** — entity-id-app-v2 web application. No package code.
- **`publications-and-presentations/llm-qualitative-scoring-methodology/`** — two scripts import the package, but the dir's primary purpose is manuscript and study materials, which belong with the parent. Scripts switch to `pip install` of the new package in Phase 5.
- **`external_repos/figurative-language-app-v1`** — vendored nested repo, unrelated to the Python package.
- **`pilot_results/`, `experiments/`, `archive/`** — pre-package pilot/experiment data.
- **`docs/audits/`, `docs/bug-reports/`, `docs/api/`, `docs/setup/`, `docs/team-protocols/`, `docs/testing/`, `docs/onboarding/`** — web app concerns.
- **Hidden files (`.github/`, `.vscode/`)** — parent repo's CI/IDE config; the new repo will get its own.

---

## Reviewer checklist

Before approving:

- [ ] All 21 planning dirs in MOVE are package-related (not web app)
- [ ] All 10 progress notes in MOVE are package-related
- [ ] The 2 standalone files (`figurative-language-detection-explained.md`, `extractor-selection-guide.md`) belong with the package
- [ ] The reclassification of `2025-12-25-fig-lang-source-target-labeling/` (SPLIT → STAY) is correct
- [ ] No other package-relevant doc was missed
- [ ] Filter-repo command above looks correct

If anything is wrong, edit this file and tell me what to change before Phase 2 runs.
