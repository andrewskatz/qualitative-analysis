# Test Data Notes

## Relationship CLI Outputs

Depending on `--output`, the relationships CLI can emit:

- `*_relationships_summary.csv`
  - One row per input text.
  - Columns: `text_id`, `text`, `entity_count`, `relationship_count`, `window_count`,
    `entities_json`, `strategy`, `error`.
- `*_relationships_edges.csv`
  - One row per relationship.
  - Columns: `text_id`, `window_index`, `source`, `target`, `type`, `description`.
- `*_relationships_windows.csv` (when `--output` includes `windows`)
  - One row per window.
  - Columns: `text_id`, `window_index`, `window_text`, `summary`, `relationship_count`,
    `entities_json`, `relationships_json`.

## Copied Dataset

`copied-master_dataset_complete.csv` uses:
- ID column: `text_id`
- Text column: `text`
