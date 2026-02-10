# CLI Orchestration & Pipeline Data Flow Audit

**Scope**: `relationships_cli.py`, `unified_cli.py` (relationships subcommands)

---

## H10: Verify Command Reads Normalized CSV as Raw Relationships

**Severity**: HIGH
**File**: `relationships_cli.py:1777`
**Impact**: Normalized relationship metadata lost during verification

```python
# Verify command uses basic CSV reader
relationships = _read_relationships_csv(
    input_path,
    source_col=args.source_col,
    ...
)
```

`_read_relationships_csv()` always returns `List[Relationship]` objects — it doesn't detect or handle the normalized relationship format. If a user runs:

```bash
qa relationships detect input.csv → edges.csv
qa relationships normalize edges.csv → normalized.csv
qa relationships verify normalized.csv source.csv  # ← loses normalization data
```

The normalized CSV has columns like `original_source`, `count`, `window_indices`, `text_ids` — all of which are ignored by `_read_relationships_csv()`. The verifier then receives bare `Relationship` objects with `count=1` and empty `text_ids`.

Compare with the causal command, which uses `_read_causal_input_csv()` that properly detects and handles normalized format.

**Fix**: Use a unified CSV reader that auto-detects format, or use `_read_causal_input_csv()` pattern for verify command.

---

## H11: Pipeline Stage Ordering Not Enforced

**Severity**: HIGH (design)
**File**: `relationships_cli.py` (all commands)
**Impact**: Users can run stages in wrong order with no warning

The CLI commands are independent and accept any CSV file. There's no validation that:
- `normalize` receives output from `detect`
- `verify` receives output from `detect` or `normalize`
- `causal` receives output from any prior stage
- `graph` receives the appropriate format

Running `qa relationships causal raw_text.csv` (where the CSV has text, not relationships) will fail with cryptic column-not-found errors. Running `qa relationships graph edges.csv` on raw detection output vs normalized vs causal output produces different results with no indication of what was expected.

**Fix**: Add format detection and warnings:
```python
# At start of each command
detected_format = _detect_csv_format(input_path)
if detected_format != expected_format:
    logger.warning(f"Input appears to be {detected_format} format, "
                   f"expected {expected_format}. Results may be incorrect.")
```

---

## M17: Three Duplicate CSV Reading Functions

**Severity**: MEDIUM
**File**: `relationships_cli.py` — `_read_relationships_csv`, `_read_causal_input_csv`, `_read_graph_input_csv`
**Impact**: DRY violation; bug fixes must be applied to 3 places

These three functions all:
1. Open a CSV file
2. Check required columns
3. Detect format (raw, normalized, causal)
4. Parse rows into model objects

Each handles the format detection slightly differently:

| Function | Detects Raw | Detects Normalized | Detects Causal |
|----------|------------|-------------------|----------------|
| `_read_relationships_csv` | ✓ | ✗ | ✗ |
| `_read_causal_input_csv` | ✓ | ✓ | ✗ |
| `_read_graph_input_csv` | ✓ | ✓ | ✓ |

**Fix**: Create a single `_read_relationships_from_csv()` that auto-detects format and returns the appropriate model type.

---

## M18: No Progress Reporting During Detection

**Severity**: MEDIUM
**File**: `relationships_cli.py:361-376`
**Impact**: No feedback for long-running detection on large CSVs

```python
for index, row in enumerate(reader, start=1):
    ...
    result = await detector.detect(text, current_entities=entities_input)
    # no progress output
```

The causal and verify commands both have progress callbacks:
```python
def on_progress(current, total):
    print(f"Progress: {current}/{total}")
```

But the detect command processes rows silently. For a 100-row CSV with multi-window processing, this could run for 30+ minutes with no output.

**Fix**: Add per-row progress reporting:
```python
print(f"Processing text {index}: {text_id}")
```

---

## M19: Output Directory Defaults to Package Directory

**Severity**: MEDIUM
**File**: `relationships_cli.py:206`
**Impact**: Output written to unexpected location; may fail on read-only installs

```python
def _resolve_output_dir(input_path, output_dir):
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path(__file__).parent.parent.parent / "output"  # ← package dir
```

The default output directory is relative to the installed package location, not the current working directory or the input file's directory. This is:
- Unexpected: Users would expect output near their input file
- Fragile: Package directory may be read-only (pip install, conda)
- Inconsistent: Other commands (normalize, causal, graph, verify) default to creating a subdirectory next to the input file

**Fix**: Default to a subdirectory next to the input file:
```python
path = input_path.parent / "output"
```

---

## M20: --include-causal Flag Is a No-Op

**Severity**: MEDIUM
**File**: `relationships_cli.py:1231-1234`
**Impact**: Confusing CLI interface; flag does nothing

```python
parser.add_argument(
    "--include-causal",
    action="store_true",
    default=True,  # ← already True
    help="Include causal attributes in graph if present (default: True).",
)
```

Since `default=True` and `action="store_true"`, the value is always `True`. The flag can never be `False` (you'd need `--no-include-causal` which doesn't exist). The `--no-causal` flag is the actual way to disable it.

**Fix**: Remove the `--include-causal` flag entirely since `--no-causal` is sufficient, or change to a `--causal/--no-causal` flag group.

---

## M21: No --timeout for Causal, Verify, and Graph Commands

**Severity**: MEDIUM
**File**: `relationships_cli.py`
**Impact**: Long LLM calls may hang indefinitely

The detect command has `--timeout` (line 113):
```python
parser.add_argument("--timeout", type=float, default=60.0)
```

But `add_relationships_causal_args`, `add_relationships_verify_args`, and `add_relationships_graph_args` don't expose a timeout. The default Ollama timeout is 60s, but users can't override it for these commands.

**Fix**: Add `--timeout` to all commands that make LLM calls.

---

## L8: Detector Instance Reuse Is Intentional but Undocumented

**Severity**: LOW
**File**: `relationships_cli.py:260-286`
**Impact**: Users don't know that detection state persists between CSV rows

A single `RelationshipDetector` instance processes all rows. While the entity context buffer is disabled by default (size=0), the system prompt, model configuration, and other state persist. This is probably intentional for efficiency, but the implications of reuse aren't documented.

---

## L9: Timestamp-Based Output Directories May Collide

**Severity**: LOW
**File**: `relationships_cli.py:249, 551, 855, 1303`
**Impact**: Rare — two runs in the same minute overwrite each other

```python
timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # minute precision
run_dir = output_dir / f"run_{timestamp}"
```

If two detect commands run within the same minute, they write to the same directory. Using seconds (`%S`) would reduce collision probability.
