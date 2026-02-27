---
name: handbook_multi_llm_agents
description: Guide for implementing Multi-LLM Agent Systems research using Mirascope orchestration, HuggingFace datasets/evaluation, and proven multi-agent patterns. Use when implementing agent orchestration, selecting evaluation datasets, computing metrics, or learning multi-agent design patterns.
tools: Read, Write, Bash
---

# Multi-LLM Agent Systems Research

## Overview

This skill guides you through implementing Multi-LLM Agent Systems research following a 3-phase pipeline:

```
DATA → METHOD → EVALUATION
data.py → data_out.json → method.py → method_out.json → eval.py → eval_out.json
```

Each phase has gold-standard requirements in separate reference documents.

## Quick Start

```python
# 1. DATA PHASE - Select dataset (text-only, one of 5 types)
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k")  # Math problems

# 2. METHOD PHASE - Implement multi-agent pattern with Mirascope
from mirascope import llm
from pydantic import BaseModel

class Solution(BaseModel):
    answer: str

@llm.call(provider="openai", model="gpt-5", response_model=Solution)
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-nano", response_model=bool)
def verify(task: str, solution: Solution) -> str:
    return f"Verify if '{solution.answer}' correctly answers '{task}'"

# ALWAYS implement baseline (single LLM)
baseline = reason(task)

# Multi-agent method (with verification)
solution = reason(task)
if not verify(task, solution):
    solution = reason(f"{task} (try again)")

# 3. EVALUATION PHASE - Compute metrics with HuggingFace Evaluate
import evaluate
exact_match = evaluate.load("exact_match")

# Use 3+ metrics, compare baseline vs method
baseline_score = exact_match.compute(predictions=baseline_preds, references=refs)
method_score = exact_match.compute(predictions=method_preds, references=refs)
improvement = method_score['exact_match'] - baseline_score['exact_match']
```

**Dependencies:**
```bash
pip install "mirascope[openai]" pydantic>=2.0
pip install evaluate bert-score rouge-score sacrebleu nltk scikit-learn torch transformers
```

---

## Reference Documentation

For complete requirements and implementations:

- **[datasets.md](datasets.md)** - Dataset selection (5 types, text-only constraint, examples)
- **[method.md](method.md)** - Multi-agent patterns (7 patterns with code, baseline requirement)
- **[evaluation.md](evaluation.md)** - Evaluation metrics (8 metrics with code, selection guide)

---

## The 3 Phases

### Phase 1: Dataset Selection

Select text-only datasets in one of 5 types:

1. **Math** - Problem → numeric answer (GSM8K)
2. **Coding** - Problem → code solution (HumanEval)
3. **Fact-based** - Question → 1-2 word answer (HotpotQA)
4. **Multiple choice** - Question → choice (GPQA)
5. **Free-text** - Question → free-text answer

Dataset type determines which pattern (Phase 2) and metrics (Phase 3) to use.

**→ See [datasets.md](datasets.md) for complete requirements**

### Phase 2: Method Implementation

Implement multi-agent system using Mirascope. **ALWAYS implement baseline** (single LLM) for comparison.

**The 7 patterns:**
1. Reasoning + Verification
2. Majority Voting
3. Debate with Judge
4. Hierarchical Decomposition
5. Reflection
6. Tool-Augmented
7. Prompt Templates

**Critical:** No cheating (baseline and method get same information), vary configurations (test multiple prompts/temperatures).

**→ See [method.md](method.md) for pattern implementations**

### Phase 3: Evaluation

Compute metrics using HuggingFace Evaluate. **Use 3+ metrics** for robust evaluation.

**Match metric to task type:**
- Math/Multiple choice → Exact Match
- Coding → Unit Tests (NEVER text metrics)
- Fact-based → Exact Match OR text metrics
- Free-text → ROUGE + BLEU + METEOR or BERTScore

**→ See [evaluation.md](evaluation.md) for metric implementations**

---

## Decision Tree

**Choose pattern by dataset type:**

| Dataset Type | Pattern | Metric |
|--------------|---------|--------|
| Math | Reasoning + Verification | Exact Match |
| Coding | Hierarchical Decomposition | Unit Tests |
| Multiple choice | Majority Voting, Debate | Exact Match |
| Fact-based | Majority Voting | Exact Match or text metrics |
| Free-text | Reflection, Hierarchical | ROUGE/BLEU/METEOR |

**Metric selection strategy:**
- Clear right/wrong answer → Exact Match
- Multiple valid phrasings → Text metrics (ROUGE, METEOR, BERTScore)
- Code evaluation → Unit Tests (NEVER text metrics)

---

## Quick Reference

### Dataset Types
1. Math: Problem → numeric answer → exact match
2. Coding: Problem → code → unit tests
3. Fact-based: Question → 1-2 words → exact match OR text metrics
4. Multiple choice: Question → choice → exact match
5. Free-text: Question → free-text → text metrics

### Multi-Agent Patterns
1. Reasoning + Verification - Generate → verify → retry
2. Majority Voting - Multiple agents → pick best
3. Debate with Judge - Two perspectives → judge
4. Hierarchical Decomposition - Decompose → solve → synthesize
5. Reflection - Generate → score → improve
6. Tool-Augmented - Plan tools → execute → reason
7. Prompt Templates - System messages for roles

### Evaluation Metrics
1. **Exact Match** - Math, multiple choice (0-1 scale)
2. **Unit Tests** - Coding (ALWAYS for code)
3. **ROUGE** - Free-text, recall-oriented (0-1 scale)
4. **BLEU** - Free-text, precision-oriented (0-1 scale)
5. **METEOR** - Free-text, paraphrase-aware (0-1 scale)
6. **BERTScore** - Subjective free-text, semantic (0-1 scale)
7. **CER** - Character error rate (lower is better)
8. **WER** - Word error rate (lower is better)

### OpenAI Models
- `gpt-5` - Default, high-quality reasoning
- `gpt-5-mini` - Faster, creative tasks
- `gpt-5-nano` - Fastest, simple tasks (verification)
- `gpt-4o-mini` - Precise, aggregation

---

## Critical Rules

1. ✅ **Text-only datasets** - No images, audio, PDFs
2. ✅ **Use Mirascope** - For all LLM orchestration
3. ✅ **Always implement baseline** - Required for comparison
4. ✅ **No cheating** - Baseline and method get identical information
5. ✅ **Use HuggingFace Evaluate** - For all metrics
6. ✅ **Use 3+ metrics** - Single metrics misleading
7. ✅ **Match metric to task** - Exact match for math, unit tests for code
8. ✅ **Complete implementations** - No half-finished work
9. ✅ **Vary configurations** - Test multiple settings separately
10. ✅ **Handle multiple answers** - Don't penalize valid phrasings
