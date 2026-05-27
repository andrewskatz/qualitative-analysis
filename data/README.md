# Sample Data

Curated sample inputs for running each of the four analysis pathways. The aim
is to let a new contributor (or future-you) clone the repo, install, and
immediately try every CLI surface against representative data — without
needing to source their own files first.

These are **inputs and curated examples**, not test fixtures. Test fixtures
live in [`../tests/data/`](../tests/data/) and are wired into the pytest
suite via mocks; the files here are for hand-running the CLI and demoing
the pathways.

## Layout

```
data/
├── figurative/        # detection / domain mapping inputs
├── relationships/     # causal-relationship classification & extraction inputs
├── entity/            # entity scoring & detection inputs
└── decisions/         # decision/factor extraction inputs
```

## What's here

### `figurative/` (164 KB)

Carried over from `tests/` during the 2026-05-26 fork so figurative samples
sit alongside the other pathways' inputs.

| File | Size | Description |
|---|---|---|
| `master-fig-lang-dataset.csv` | 100 KB | Master dataset of figurative-language statements (text + labels). Use as a fuller benchmark input. |
| `master-fig-lang-dataset-sample-200.csv` | 33 KB | 200-row stratified sample of the master dataset; ideal for fast iteration. |
| `paired-figurative-statements-20.csv` | 8 KB | 20 statement pairs (literal/figurative twin sentences). The smallest demo dataset. |
| `paired-figurative-statements-60.csv` | 16 KB | 60 statement pairs (expanded). |
| `sample_instances.csv` | 1 KB | A few rows from a `qa figurative detect` instances output, useful as input to `qa fig map`. |

Example CLI runs:

```bash
qa figurative detect data/figurative/paired-figurative-statements-20.csv \
  --model gpt-oss:120b --output all
qa fig map data/figurative/sample_instances.csv
```

### `relationships/` (176 KB)

Synthetic curated statements and benchmark sets for the relationships
pathway. None of these are real participant text — all are LLM-generated
or hand-written examples designed to exercise specific causal patterns.

| File | Size | Description |
|---|---|---|
| `pilot_statements.csv` | 3 KB | Hand-curated statements with `ground_truth` labels (causal / non_causal) and category notes. Good for sanity-checking. |
| `expanded_statements.csv` | 10 KB | Expanded statement set in the same format. |
| `test_relationships_causal.csv` | 1 KB | Small fixture for causal extraction development. |
| `MCP_balanced_11x3_20251102.csv` | 13 KB | Balanced 11×3 MCP benchmark sample. |
| `MCP_400rows_causal-binary-classification_sys-think-prob-info-stakes-v_53.csv` | 140 KB | 400-row MCP causal binary classification set with system prompt variants (think/prob/info/stakes). |

Example CLI runs:

```bash
qa relationships detect data/relationships/pilot_statements.csv \
  --entities "Cause,Effect"
qa rel causal data/relationships/expanded_statements.csv
```

### `entity/` (256 KB)

Inputs for the entity-scoring and entity-detection workflows.

| File | Size | Description |
|---|---|---|
| `test_entities_scoring.csv` | 1 KB | Tiny scoring-input fixture; useful for fast iteration on `qa entity score`. |
| `sim-platform-15-participants_20260211.csv` | 110 KB | **Synthetic AI-generated personas** (n=15) with persona archetypes, simulated response types, and response text. Used as input for entity detection / scoring experiments. |
| `sim-platform-15-participants_20260211_combined.csv` | 140 KB | Same personas, with combined response rows for end-to-end pipeline testing. |

These sim-platform files contain **synthetic persona responses**, not real
participants — `Persona Name` and `Persona Archetype` columns reflect
LLM-generated personas (e.g., "college student — environmental and
ecological sciences emphasis").

Example CLI runs:

```bash
qa entity detect data/entity/sim-platform-15-participants_20260211_combined.csv
qa entity score data/entity/test_entities_scoring.csv \
  --dimensions "social,ecological,technological"
```

### `decisions/` (116 KB)

Two-domain synthetic decision/factor datasets. Each row is a hand-curated
or LLM-generated text with ground-truth `decisions_json` and `factors_json`
annotations.

| File | Size | Description |
|---|---|---|
| `ai_ml.csv` | 25 KB | AI/ML domain decision statements with ground-truth annotations. |
| `ai_ml_master.csv` | 34 KB | Larger AI/ML master set. |
| `climate_change.csv` | 21 KB | Climate-change domain decisions with annotations. |
| `climate_change_master.csv` | 28 KB | Larger climate-change master set. |

Example CLI runs:

```bash
qa decisions detect data/decisions/ai_ml.csv
qa dec aggregate data/decisions/climate_change.csv
```

## What was deliberately not included

The following data exists in the parent repo's
`_deprecated_qa_fork_20260526/qualitative-analysis/tests/data/` but was
**deliberately excluded** from this repo for IRB/privacy reasons:

| File / dir | Why excluded |
|---|---|
| `transcripts_merged_one-tx-per-row_2026-01-06_114238.csv` (5.7 MB) | Real interview transcripts — named participants and interviewers. IRB-protected. |
| `ent-rel-datasets/think-aloud/jt-transcripts-all.csv` | Real think-aloud protocols. IRB-protected. |

Any code that needs these files should expect them to come from a separate
location (e.g., a private data drive or IRB-approved storage), not from
this repo. Do **not** commit these to git, even in a private repo.

## Where this data came from

These files were copied during the 2026-05-26 package extraction from
`entity-id-app-v2/qualitative-analysis/tests/data/`. The original
location was a flat dump of test fixtures and experiment inputs mixed
together; here they are reorganized by pathway for clearer use.

See [`docs/planning/2026-05-25-qualitative-analysis-package-fork/`](../docs/planning/2026-05-25-qualitative-analysis-package-fork/)
for the extraction history.

## Adding new sample data

When adding a new file:

1. Put it in the matching pathway subdir (`figurative/`, `relationships/`,
   `entity/`, `decisions/`).
2. Add a row to the table above with size and one-line description.
3. If it might contain participant data, **don't** commit. Add a note in
   the "What was deliberately not included" table instead.
4. Keep individual files under a few MB. For larger benchmark sets, host
   them externally and reference the URL in this README.
