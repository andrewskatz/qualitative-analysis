# Entity Extractor Selection Guide

## Overview

This guide helps you choose the right entity extraction method based on your specific requirements. Our experiments with 28 test cases across different complexity levels provide clear performance benchmarks.

## Quick Decision Matrix

| Priority | Recommended Mode | Extractor | F1 Score | Speed | Use Case |
|----------|------------------|-----------|----------|-------|----------|
| **Speed** | `speed` | SpaCy | 0.922 | 0.50s | Real-time processing |
| **Quality** | `quality` | Hybrid Normalized | 0.931 | 3.24s | Research, analysis |
| **Precision** | `precision` | LLM | 0.924 | 2.62s | Critical applications |
| **Balanced** | `balanced` | LLM | 0.924 | 2.62s | General purpose |

## Detailed Comparison

### 🚀 Speed Mode (SpaCy Extractor)
**Performance**: F1: 0.922, Speed: 0.50s/text

**Strengths:**
- ⚡ **6x faster** than other methods
- 🎯 Excellent quality (92.2% F1 score)
- 🔧 No external API dependencies
- 💾 Low memory usage
- 🌐 Works offline

**Best for:**
- Real-time applications
- High-volume batch processing
- Resource-constrained environments
- Production systems requiring low latency

**Example Use Cases:**
- Live chat entity extraction
- Document processing pipelines
- Mobile applications
- Edge computing scenarios

```python
from src.config import get_performance_mode_config
from src.entity_extraction.spacy_extractor import SpacyEntityExtractor

config = get_performance_mode_config('speed')
extractor = SpacyEntityExtractor(config=config['spacy_entity_extraction'])
```

### 🏆 Quality Mode (Hybrid Normalized Extractor)
**Performance**: F1: 0.931, Speed: 3.24s/text

**Strengths:**
- 🥇 **Best overall quality** (93.1% F1 score)
- 🔄 Combines spaCy + LLM strengths
- ✅ Cross-validation between methods
- 🧹 Advanced normalization
- 📊 Balanced precision/recall

**Best for:**
- Research and analysis
- High-stakes applications
- Complex domain texts
- When quality is paramount

**Example Use Cases:**
- Academic research
- Legal document analysis
- Medical text processing
- Policy document analysis

```python
from src.config import get_performance_mode_config
from src.entity_extraction.hybrid_extractor import HybridEntityExtractor

config = get_performance_mode_config('quality')
extractor = HybridEntityExtractor(config=config['hybrid_entity_extraction'])
```

### 🎯 Precision Mode (LLM Extractor)
**Performance**: F1: 0.924, Precision: 0.980, Speed: 2.62s/text

**Strengths:**
- 🎯 **Highest precision** (98.0%)
- 🧠 Deep semantic understanding
- 🔍 Conservative extraction (fewer false positives)
- 📝 Context-aware entity recognition
- 🎛️ Configurable extraction criteria

**Best for:**
- Applications where false positives are costly
- Domain-specific entity extraction
- Compliance and regulatory analysis
- Quality over quantity scenarios

**Example Use Cases:**
- Financial document analysis
- Regulatory compliance checking
- Scientific literature review
- Contract analysis

```python
from src.config import get_performance_mode_config
from src.entity_extraction.extractor import EntityExtractor

config = get_performance_mode_config('precision')
extractor = EntityExtractor(config=config['entity_extraction'])
```

### ⚖️ Balanced Mode (LLM Extractor)
**Performance**: F1: 0.924, Speed: 2.62s/text

**Strengths:**
- ⚖️ Good balance of speed and quality
- 🎯 High precision and recall
- 🔧 Flexible configuration
- 📈 Consistent performance
- 💡 General-purpose solution

**Best for:**
- General-purpose applications
- Mixed content types
- Moderate performance requirements
- Development and prototyping

**Example Use Cases:**
- Content management systems
- Knowledge base construction
- General text analysis
- Proof-of-concept projects

## Performance by Text Complexity

### Simple Texts (1-3 entities)
- **All extractors**: Near-perfect performance (F1 > 0.95)
- **Recommendation**: Use `speed` mode for efficiency

### Moderate Texts (4-8 entities)
- **SpaCy**: F1: 0.89
- **LLM**: F1: 0.92
- **Hybrid**: F1: 0.94
- **Recommendation**: Use `balanced` or `quality` mode

### Complex Texts (9+ entities)
- **SpaCy**: F1: 0.85
- **LLM**: F1: 0.91
- **Hybrid**: F1: 0.93
- **Recommendation**: Use `quality` mode for best results

## Domain-Specific Considerations

### Technical/Scientific Texts
- **Best**: Hybrid Normalized (handles technical terms well)
- **Alternative**: LLM with domain-specific prompts

### Business/Policy Documents
- **Best**: LLM Precision mode (conservative extraction)
- **Alternative**: Hybrid Normalized for comprehensive analysis

### News/General Content
- **Best**: SpaCy Speed mode (efficient for standard entities)
- **Alternative**: Balanced mode for mixed content

### Social Media/Informal Text
- **Best**: LLM modes (better context understanding)
- **Alternative**: Hybrid for comprehensive coverage

## Resource Requirements

### Computational Resources

| Mode | CPU Usage | Memory | GPU | Network |
|------|-----------|--------|-----|---------|
| Speed | Low | 2-4 GB | Optional | None |
| Balanced | Medium | 4-8 GB | Recommended | Ollama |
| Quality | High | 6-12 GB | Recommended | Ollama |
| Precision | Medium | 4-8 GB | Recommended | Ollama |

### Scalability Considerations

**High Volume (>1000 texts/hour):**
- Use `speed` mode with parallel processing
- Consider batch processing optimizations

**Medium Volume (100-1000 texts/hour):**
- `balanced` mode works well
- Monitor resource usage

**Low Volume (<100 texts/hour):**
- `quality` mode for best results
- Resource usage not a concern

## Configuration Examples

### Production Environment
```python
# High-throughput production system
config = get_performance_mode_config('speed')
config['spacy_entity_extraction']['normalize_entities'] = True
```

### Research Environment
```python
# Research with maximum quality
config = get_performance_mode_config('quality')
config['hybrid_entity_extraction']['print_debug'] = True
```

### Development Environment
```python
# Development and testing
config = get_performance_mode_config('balanced')
config['entity_extraction']['temperature'] = 0.1  # Slight randomness for testing
```

## Migration Strategies

### From Manual Configuration
1. **Assess current performance** with existing setup
2. **Test equivalent performance mode** 
3. **Compare results** on sample data
4. **Gradually migrate** production systems

### From Other Tools
1. **Benchmark current system** on test dataset
2. **Try multiple modes** to find best match
3. **Consider hybrid approach** during transition
4. **Validate results** before full migration

## Troubleshooting

### Low Performance
- **Check text complexity**: Use higher-quality mode for complex texts
- **Verify normalization**: Ensure enhanced normalization is enabled
- **Review configuration**: Use performance mode presets

### High Resource Usage
- **Switch to speed mode**: For resource-constrained environments
- **Batch processing**: Process multiple texts together
- **Optimize hardware**: Consider GPU acceleration for LLM modes

### Inconsistent Results
- **Enable normalization**: Improves consistency across runs
- **Use semantic evaluation**: More robust than exact matching
- **Check configuration**: Ensure consistent settings

## Best Practices Summary

1. **Start with performance modes** instead of manual configuration
2. **Always enable enhanced normalization** for better results
3. **Choose based on primary constraint** (speed vs quality vs precision)
4. **Test on representative data** before production deployment
5. **Monitor performance metrics** in production
6. **Use configuration files** for reproducible setups
7. **Consider text complexity** when selecting modes
8. **Plan for scalability** based on expected volume

## Getting Help

- **API Documentation**: See `docs/api/enhanced-entity-extraction.md`
- **Configuration Reference**: Check `src/config.py` for all options
- **Example Configurations**: Review `examples/example_config.json`
- **Performance Testing**: Use experiment scripts in `experiments/08-hybrid-entity-identification/`
