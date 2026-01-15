# Unified CLI Refactoring Implementation Plan

## Goal
Unify three fragmented CLI entry points (`qualitative-analysis`, `qualitative-domains`, `qualitative-relationships`) under a single `qa` CLI with nested subcommands.

## Proposed Changes

### Core Infrastructure (`core/`)

#### [NEW] [cli_utils.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/core/cli_utils.py)

Shared CLI utilities to standardize common patterns across all commands:

```python
def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add --model, --base-url, --timeout, --log-llm."""

def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    """Add --output-dir, --output."""

def add_common_windowing_args(parser: argparse.ArgumentParser) -> None:
    """Add --window-size, --stride, --chunk-unit, --tokenizer, --no-windowing."""

def create_run_directory(output_dir: Path, prefix: str = "run") -> tuple[Path, str]:
    """Create timestamped run directory. Returns (run_dir, timestamp)."""

def write_run_metadata(run_dir: Path, metadata: dict) -> Path:
    """Write standardized run_metadata.json."""

def setup_progress_tracking(total: int) -> Callable:
    """Return a progress callback for consistent progress display."""
```

---

### Unified CLI Entry Point

#### [NEW] [unified_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/unified_cli.py)

Main unified CLI with hierarchical subcommands:

```python
qa
├── figurative (alias: fig)
│   ├── detect      # From cli.py
│   ├── map         # From domains_cli.py
│   ├── normalize   # From domains_cli.py
│   ├── graph       # From domains_cli.py
│   └── pipeline    # From domains_cli.py
└── relationships (alias: rel)
    └── detect      # From relationships_cli.py
```

**Key implementation details:**
- Use `argparse` with `add_subparsers()` for nested commands
- Implement aliases via `aliases=["fig"]` parameter
- Add `--version` flag showing package version
- Dispatch to handler functions in existing CLIs

---

### Refactor Figurative CLI

#### [cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/cli.py)

Extract reusable components:

```diff
+def add_figurative_detect_args(parser: argparse.ArgumentParser) -> None:
+    """Add all figurative detection arguments to parser."""
+    # Move argument definitions here

+async def run_figurative_detect(args: argparse.Namespace) -> int:
+    """Run figurative detection with parsed args."""
+    # Move _run() logic here, return exit code

 def main() -> None:
-    raise SystemExit(asyncio.run(_run()))
+    """Legacy entry point - delegates to unified CLI."""
+    parser = argparse.ArgumentParser(...)
+    add_figurative_detect_args(parser)
+    args = parser.parse_args()
+    raise SystemExit(asyncio.run(run_figurative_detect(args)))
```

---

### Refactor Domains CLI

#### [domains_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains_cli.py)

Extract reusable components:

```diff
+def add_domains_map_args(parser: argparse.ArgumentParser) -> None:
+def add_domains_normalize_args(parser: argparse.ArgumentParser) -> None:
+def add_domains_graph_args(parser: argparse.ArgumentParser) -> None:
+def add_domains_pipeline_args(parser: argparse.ArgumentParser) -> None:

 # Handlers already exist: _run_map, _run_normalize, _run_graph, _run_pipeline
 # Just need to make them callable from unified CLI
```

---

### Refactor Relationships CLI

#### [relationships_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/relationships_cli.py)

Extract reusable components:

```diff
+def add_relationships_detect_args(parser: argparse.ArgumentParser) -> None:
+    """Add all relationship detection arguments to parser."""

+async def run_relationships_detect(args: argparse.Namespace) -> int:
+    """Run relationship detection with parsed args."""

 def main() -> None:
     # Keep as legacy entry point
```

---

### Package Configuration

#### [pyproject.toml](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/pyproject.toml)

```diff
 [project.scripts]
+qa = "qualitative_analysis.unified_cli:main"
 qualitative-analysis = "qualitative_analysis.cli:main"
 qualitative-relationships = "qualitative_analysis.relationships_cli:main"
 qualitative-domains = "qualitative_analysis.figurative.domains_cli:main"
```

---

### Tests

#### [NEW] [test_unified_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/tests/test_unified_cli.py)

CLI integration tests using subprocess:

```python
class TestUnifiedCLIHelp(unittest.TestCase):
    """Test help output for all commands."""
    
    def test_qa_help(self):
        result = subprocess.run(["qa", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("figurative", result.stdout)
        self.assertIn("relationships", result.stdout)
    
    def test_qa_figurative_help(self):
        result = subprocess.run(["qa", "figurative", "--help"], ...)
        self.assertIn("detect", result.stdout)
        self.assertIn("map", result.stdout)
    
    def test_qa_fig_alias(self):
        result = subprocess.run(["qa", "fig", "--help"], ...)
        self.assertEqual(result.returncode, 0)

    def test_qa_version(self):
        result = subprocess.run(["qa", "--version"], ...)
        self.assertIn("0.1.0", result.stdout)

class TestUnifiedCLIExecution(unittest.TestCase):
    """Test actual command execution with sample data."""
    
    def test_figurative_detect_basic(self):
        # Uses tests/sample_instances.csv
        ...
```

---

## Verification Plan

### Automated Tests

**Existing tests (run first to ensure no regression):**
```bash
cd /Users/akatz4/Documents/ak\ fac/research/projects/entity-id-app-v2/qualitative-analysis
python -m pytest tests/test_domains.py -v
```

**New tests (will be added):**
```bash
# After implementation
python -m pytest tests/test_unified_cli.py -v
```

### Manual Verification

After implementation, verify these commands work as expected:

1. **Help output verification:**
   ```bash
   qa --help                    # Should show figurative, relationships
   qa figurative --help         # Should show detect, map, normalize, graph, pipeline
   qa fig --help                # Should be identical to above (alias)
   qa relationships --help      # Should show detect
   qa rel --help                # Should be identical to above (alias)
   qa --version                 # Should show 0.1.0
   ```

2. **Command equivalence check (using sample data):**
   ```bash
   # Old way
   qualitative-domains map tests/sample_instances.csv --model test --help
   
   # New way (should have identical arguments)
   qa figurative map tests/sample_instances.csv --model test --help
   ```

3. **Argument consistency check:**
   - Verify `--model`, `--base-url`, `--log-llm` are present on all detection commands
   - Verify all commands can be run with `--help` without errors

---

## User Review Required

> [!IMPORTANT]
> **Backward compatibility decision:** Per your earlier guidance, we are NOT maintaining backward compatibility with old entry points. However, I will keep them functional (without deprecation warnings) so existing scripts don't break immediately. Should I:
> - A) Keep old entry points working silently (current plan)
> - B) Add deprecation warnings to old entry points
> - C) Remove old entry points entirely
ANSWER: Add deprecation warnings to old entry points for now but make sure it's clear that they will be removed in a future release

---

## Implementation Order

1. Create `core/cli_utils.py` with shared utilities
2. Create `unified_cli.py` with command structure (help output works)
3. Extract arg functions from `cli.py` and integrate into unified CLI
4. Extract arg functions from `relationships_cli.py` and integrate
5. Wire up `domains_cli.py` handlers to unified CLI
6. Update `pyproject.toml` with `qa` entry point
7. Add CLI integration tests
8. Verify all acceptance criteria
