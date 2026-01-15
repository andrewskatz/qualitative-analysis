# Qualitative Analysis Package CLI Refactoring

## Problem Statement

The `qualitative-analysis` package currently has an inconsistent CLI structure:

1. **Figurative language detection**: `qualitative-analysis input.csv` (root command with positional arg)
2. **Domain processing**: `qualitative-domains map|normalize|graph` (subcommand pattern)

This is confusing because:
- Two separate entry points (`qualitative-analysis` vs `qualitative-domains`)
- Different command patterns (root command vs subcommands)
- Relationship between steps is unclear
- No unified help system

## Broader Vision

The package will eventually support **multiple analysis types**:

| Analysis Type | Description | Status | Current CLI |
|--------------|-------------|--------|-------------|
| **Figurative Language** | Detect metaphors, analogies, etc. + domain mapping | ✅ Implemented | `qualitative-analysis` + `qualitative-domains` |
| **Entity-Relationship Identification** | Extract relationships between entities | ✅ Implemented | `qualitative-relationships` |
| **Entity/Concept Identification** | Extract key concepts and entities | ⚠️ Backend Only | (via backend services) |
| **Causal Statement Identification** | Detect causal claims and reasoning | ⚠️ Backend Only | (via backend services) |
| **Decision/Factor Identification** | Extract decisions and influencing factors | 📋 Planned | - |

### Current Entry Points (pyproject.toml)
```toml
[project.scripts]
qualitative-analysis = "qualitative_analysis.cli:main"
qualitative-relationships = "qualitative_analysis.relationships_cli:main"
qualitative-domains = "qualitative_analysis.figurative.domains_cli:main"
```

Each analysis type may have its own multi-step workflow (detect → extract → normalize → visualize).

## Proposed CLI Structure

### Option 1: Analysis Type as Top-Level Subcommand

```bash
qualitative-analysis figurative detect input.csv --model gpt-oss:120b
qualitative-analysis figurative map instances.csv --model gpt-oss:120b
qualitative-analysis figurative normalize mappings.csv --merge-threshold 0.72
qualitative-analysis figurative graph mappings.csv --visualize
qualitative-analysis figurative pipeline input.csv --through graph

qualitative-analysis entities detect input.csv --model gpt-oss:120b
qualitative-analysis entities consolidate entities.csv
qualitative-analysis entities graph entities.csv

qualitative-analysis causal detect input.csv --model gpt-oss:120b
qualitative-analysis causal classify statements.csv
```

**Pros:**
- Clear hierarchical structure
- Scalable to new analysis types
- Discoverable (`--help` shows all analysis types)
- Consistent pattern across all modules

**Cons:**
- More verbose (two levels of subcommands)
- Slightly more typing for common operations

---

### Option 2: Flat Subcommands with Prefixes

```bash
qualitative-analysis fig-detect input.csv --model gpt-oss:120b
qualitative-analysis fig-map instances.csv --model gpt-oss:120b
qualitative-analysis fig-normalize mappings.csv
qualitative-analysis fig-graph mappings.csv --visualize

qualitative-analysis entity-detect input.csv
qualitative-analysis entity-consolidate entities.csv

qualitative-analysis causal-detect input.csv
qualitative-analysis causal-classify statements.csv
```

**Pros:**
- Shorter commands
- Simpler parser structure
- Tab-completion shows all commands

**Cons:**
- Harder to discover related commands
- Naming can get awkward
- Less organized as more analyses are added

---

### Option 3: Separate CLIs per Analysis Type (Current Evolution)

```bash
figurative detect input.csv --model gpt-oss:120b
figurative map instances.csv
figurative normalize mappings.csv
figurative graph mappings.csv

entities detect input.csv
entities consolidate entities.csv

causal detect input.csv
```

**Pros:**
- Short, intuitive commands
- Each analysis is its own "product"
- Easier to version independently

**Cons:**
- Multiple entry points to remember
- Package becomes a collection of CLIs
- Harder to share common functionality visibly

---

## Recommendation

**Option 1 (Analysis Type as Top-Level Subcommand)** is recommended because:

1. **Scalability**: Clean structure as new analysis types are added
2. **Discoverability**: `qualitative-analysis --help` shows all capabilities
3. **Consistency**: Every step follows the same `<analysis> <action>` pattern
4. **Documentation**: Easier to document and teach
5. **Namespace management**: No naming collisions between analysis types

### Recommended Shortcuts

For common workflows, provide shorter aliases:

```bash
# Full form
qualitative-analysis figurative detect input.csv

# Short form (alias)
qa fig detect input.csv

# Or even shorter via shell alias
alias qf="qualitative-analysis figurative"
qf detect input.csv
```

## Next Steps

1. **User Decision**: Choose between options (or suggest modifications)
2. **Create detailed implementation plan** with file changes
3. **Consider backward compatibility** for existing scripts
4. **Plan migration path** from current CLI

---

## Questions for User

1. Do you prefer Option 1, 2, or 3? Or a hybrid approach?
ANSWER: Option 1
2. Should we maintain backward compatibility with the current CLI?
ANSWER: No, but we should be sure that we document the transition path and everything functions after the migration.
3. What should the short name be? (`qualitative-analysis`, `qa`, `qualitative-analysis`?)
ANSWER: `qa`
4. Are there other analysis types beyond the ones listed that should inform the design?
ANSWER: Aside from the ones listed (figurative language, entity identification, entity relationship identification, causal statement identification, decision/factor identification), there are none others planned for now. With that said, we should be careful to plan for the future and not make the CLI structure too rigid.
