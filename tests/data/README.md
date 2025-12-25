# Test Data Notes

## Relationship CLI Outputs

Depending on `--output`, the relationships CLI can emit inside a `run_YYYYMMDD-HHMM/` subdirectory:

- `run_YYYYMMDD-HHMM/*_relationships_summary_YYYYMMDD-HHMM.csv`
  - One row per input text.
  - Columns: `text_id`, `text`, `entity_count`, `relationship_count`, `window_count`,
    `entities_json`, `strategy`, `error`.
- `run_YYYYMMDD-HHMM/*_relationships_edges_YYYYMMDD-HHMM.csv`
  - One row per relationship.
  - Columns: `text_id`, `window_index`, `source`, `target`, `type`, `description`.
- `run_YYYYMMDD-HHMM/*_relationships_windows_YYYYMMDD-HHMM.csv` (when `--output` includes `windows`)
  - One row per window.
  - Columns: `text_id`, `window_index`, `window_text`, `summary`, `relationship_count`,
    `entities_json`, `relationships_json`.

## Copied Dataset

`copied-master_dataset_complete.csv` uses:
- ID column: `text_id`
- Text column: `text`
