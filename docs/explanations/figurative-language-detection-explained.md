# Figurative Language Detection: Strategies and Text Processing Explained

**Date:** 2025-11-04  
**Purpose:** Comprehensive explanation of figurative language detection strategies and text processing approaches

## Overview

The figurative language detection system uses a combination of **detection strategies** (one-pass vs two-pass) and **text processors** (baseline, chunking, sliding window) to identify figurative language in text. This document explains how each component works and when to use them.

---

## Detection Strategies

### 1. One-Pass Strategy

**How it works:**
- Makes a **single LLM call** per text chunk
- Directly asks: "Does this text contain figurative language?"
- Returns: detection result, confidence, and type in one step

**Prompt structure:**
```
Analyze the following text and determine if it contains figurative language.

Text: [your text here]

Figurative language includes:
- Metaphor, Simile, Personification, Hyperbole, Analogy

Respond in JSON format: {has_figurative, confidence, figurative_type}
```

**Performance:**
- **Speed:** Fast (~89-90% accuracy)
- **LLM calls:** 1 per chunk
- **Best for:** Quick analysis, batch processing

**Pros:**
- ✅ Faster processing (fewer LLM calls)
- ✅ Lower cost
- ✅ Simpler logic

**Cons:**
- ❌ Slightly lower accuracy (~89-90% vs 91-92%)
- ❌ May miss subtle figurative language
- ❌ Less semantic understanding

---

### 2. Two-Pass Strategy (Recommended)

**How it works:**
- Makes **two LLM calls** per text chunk
- **Pass 1:** Extracts the main topic/subject
- **Pass 2:** Asks if that topic is used literally or figuratively

**Prompt structure:**

**Pass 1 - Topic Extraction:**
```
What is the main topic or subject of this text?

Text: [your text here]

Respond in JSON format: {topic: "the main topic"}
```

**Pass 2 - Figurative Check:**
```
Is the concept "[topic]" used literally or figuratively in this text?

Text: [your text here]

Consider:
- Literal use: standard dictionary meaning
- Figurative use: metaphorical, symbolic, non-literal

Respond in JSON format: {is_figurative, confidence, figurative_type}
```

**Performance:**
- **Accuracy:** Higher (~91-92% accuracy)
- **LLM calls:** 2 per chunk
- **Best for:** Accurate analysis, research, quality over speed

**Pros:**
- ✅ Higher accuracy (~91-92%)
- ✅ Better semantic understanding
- ✅ Identifies the topic being discussed
- ✅ More nuanced detection

**Cons:**
- ❌ Slower (2x LLM calls)
- ❌ Higher cost
- ❌ More complex logic

**Why it's better:**
By first identifying the topic, the LLM can focus on whether that specific concept is used figuratively, rather than trying to detect all types of figurative language at once. This semantic approach is more aligned with how humans understand figurative language.

---

## Text Processors

Text processors determine how the input text is segmented before being sent to the LLM for analysis.

### 1. Baseline (Full Text)

**How it works:**
- Sends the **entire text as-is** to the LLM
- No chunking or segmentation
- Single chunk = entire text

**Example:**
```
Input text (100 sentences) → [Full text as 1 chunk] → LLM
```

**Performance:**
- **Accuracy:** 85.5%
- **Chunks per text:** 1
- **LLM calls:** 1 (one-pass) or 2 (two-pass)

**Best for:**
- ✅ Short texts (1-5 sentences)
- ✅ Texts that fit within LLM context window
- ✅ When context is critical

**Limitations:**
- ❌ Poor for long texts (6+ sentences)
- ❌ May exceed LLM context limits
- ❌ Lower accuracy on longer texts

---

### 2. Chunking (Fixed-Size) - **RECOMMENDED**

**How it works:**
- Splits text into **fixed-size chunks** of N sentences
- Default: 3 sentences per chunk
- No overlap between chunks
- Each chunk analyzed independently

**Example:**
```
Input text (9 sentences):
  Chunk 1: Sentences 1-3
  Chunk 2: Sentences 4-6
  Chunk 3: Sentences 7-9
  
Each chunk → LLM → Results aggregated
```

**Performance:**
- **Accuracy:** 90.5% (**BEST PERFORMER**)
- **Chunks per text:** Variable (depends on text length)
- **LLM calls:** N chunks × strategy (1 or 2)
- **Improvement:** +5.0pp over baseline overall, +8.3pp for long texts

**Best for:**
- ✅ Long texts (6+ sentences)
- ✅ Balanced accuracy and speed
- ✅ Most use cases (default choice)

**Configuration:**
- `chunk_size`: Number of sentences per chunk (default: 3)

**Why it's best:**
- Provides enough context for accurate detection
- Avoids overwhelming the LLM with too much text
- Experimentally validated as the best performer

---

### 3. Sliding Window (Overlapping)

**How it works:**
- Creates **overlapping windows** of N sentences
- Moves forward by S sentences (stride)
- Overlap = window_size - stride
- Each window analyzed independently

**Example:**
```
Input text (9 sentences), window_size=3, stride=2:
  Window 1: Sentences 1-3
  Window 2: Sentences 3-5 (overlaps with Window 1)
  Window 3: Sentences 5-7 (overlaps with Window 2)
  Window 4: Sentences 7-9 (overlaps with Window 3)
  
Each window → LLM → Results aggregated
```

**Performance:**
- **Accuracy:** 89.1%
- **Chunks per text:** Variable (more than chunking due to overlap)
- **LLM calls:** N windows × strategy (1 or 2)
- **Improvement:** +3.6pp over baseline

**Best for:**
- ✅ Maximum coverage
- ✅ Context preservation across boundaries
- ✅ When figurative language might span chunk boundaries

**Configuration:**
- `window_size`: Number of sentences per window (default: 3)
- `stride`: Number of sentences to move forward (default: 2)
- `overlap`: window_size - stride (default: 1 sentence)

**Trade-offs:**
- ✅ Better coverage (overlapping windows)
- ✅ Less likely to miss figurative language at boundaries
- ❌ More LLM calls (more windows)
- ❌ Slightly lower accuracy than chunking (89.1% vs 90.5%)
- ❌ Higher cost and slower

---

## Results Aggregation

When text is split into multiple chunks/windows, results are aggregated:

1. **Detection:** If ANY chunk contains figurative language → overall result is TRUE
2. **Confidence:** Average confidence across all chunks
3. **Type:** Most common figurative type across chunks
4. **Topic:** Most common topic (two-pass only)

---

## Recommendations

### For Most Use Cases:
- **Strategy:** Two-pass (better accuracy)
- **Processor:** Chunking (best performer)
- **Configuration:** Default settings (3-sentence chunks)

### For Speed/Cost Optimization:
- **Strategy:** One-pass (faster)
- **Processor:** Chunking (still best)
- **Configuration:** Larger chunks (4-5 sentences) to reduce LLM calls

### For Short Texts (1-5 sentences):
- **Strategy:** Either (minimal difference)
- **Processor:** Baseline (no need to chunk)

### For Maximum Accuracy:
- **Strategy:** Two-pass (semantic understanding)
- **Processor:** Sliding window (maximum coverage)
- **Configuration:** Small windows with high overlap (window=3, stride=1)

---

## Experimental Validation

All strategies and processors were validated in **Experiment 03a** of the figurative-language-app-v1 project:

| Processor | Accuracy | Improvement | Best For |
|-----------|----------|-------------|----------|
| Baseline | 85.5% | baseline | Short texts (1-5 sentences) |
| **Chunking** | **90.5%** | **+5.0pp** | **Long texts (6+ sentences)** |
| Sliding Window | 89.1% | +3.6pp | Maximum coverage |

**Key findings:**
- Chunking is the best overall performer
- Chunking shows +8.3pp improvement for long texts
- Two-pass strategy adds ~1-2pp accuracy over one-pass
- Baseline is only suitable for very short texts

---

## Example Configurations

### Default (Recommended):
```python
FigurativeDetector(
    strategy="two_pass",           # Better accuracy
    text_processor="chunking",     # Best performer
    model="mistral-small3.2",
    temperature=0.7
)
```

### Fast Mode:
```python
FigurativeDetector(
    strategy="one_pass",           # Faster
    text_processor="chunking",     # Still best
    model="mistral-small3.2",
    temperature=0.7
)
```

### Maximum Coverage:
```python
FigurativeDetector(
    strategy="two_pass",           # Better accuracy
    text_processor="sliding_window", # Overlapping windows
    model="mistral-small3.2",
    temperature=0.7
)
```

---

## Future Enhancements

Potential improvements identified:

1. **Prompt improvements with XML tags** (in progress)
   - Separate `<text_to_analyze>`, `<background_information>`, `<formatting_instructions>`
   - Clearer structure for LLM parsing

2. **Prompt versioning** (in progress)
   - Track prompt changes over time
   - A/B test different prompt formulations
   - Maintain backward compatibility

3. **Instance extraction**
   - Identify specific phrases that are figurative
   - Provide explanations for why they're figurative
   - Enable text highlighting in UI

4. **Adaptive processing**
   - Automatically choose processor based on text length
   - Dynamic chunk sizing based on content

5. **Confidence calibration**
   - Improve confidence score accuracy
   - Threshold tuning for different use cases

