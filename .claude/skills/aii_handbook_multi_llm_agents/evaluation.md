# Multi-LLM Agent Systems: Evaluation Requirements

## Critical Requirements

- You must use 'evaluate' from Hugingface for all your evaluations.
- The evaluation metrics you use the better. Don't stop at just 2.
- It's very important to consider if there are multiple correct answers to each example. If an evaluation metric evaluates most of the correct answers as incorrect it's not a good metric. Example: Question: 'Where is the Eiffel Tower?', Answer: "The Eiffel Tower is in Paris" is wrong but "Paris" is right.
- Be very careful when evaluating free-text answers as they can be very subjective.
- Your evaluations must be one of the following types:
    - **Exact match**: Use only when the answer is clear, unambiguous and there is only one correct answer. (e.g. multiple choice, math problems, fact-based)
    - **Unit test**: Always use to evaluate code solutions. (e.g. bug fixing, coding)
    - **Text generation metrics**: Use when you have free-text answers, but ONLY when the answer is objective and short (e.g. short free-text, fact-based)

---

# Dependencies

```bash
pip install evaluate bert-score rouge-score sacrebleu jiwer nltk scikit-learn torch transformers
```

---

# Metric Implementations

```python
import evaluate
```

## Metric 1: Exact Match

**Use when:** Math, multiple choice, fact-based with single phrasing

```python
exact_match = evaluate.load("exact_match")
score = exact_match.compute(predictions=preds, references=refs)
print(score['exact_match'])  # 0.0-1.0
```

---

## Metric 2: BLEU

**Use when:** Translation, code generation, free-text with preferred phrasing

```python
bleu = evaluate.load("bleu")
score = bleu.compute(
    predictions=preds,
    references=[[r] for r in refs]  # List of lists!
)
print(score['bleu'])  # 0.0-1.0
```

**Note:** References MUST be list of lists

---

## Metric 3: ROUGE

**Use when:** Summarization, free-text QA, document generation

```python
rouge = evaluate.load("rouge")
scores = rouge.compute(predictions=preds, references=refs)
print(scores['rouge1'])   # Unigram overlap
print(scores['rouge2'])   # Bigram overlap
print(scores['rougeL'])   # Longest common subsequence
```

**Range:** 0.0-1.0 for all scores

---

## Metric 4: BERTScore

**Use when:** Paraphrasing acceptable, semantic equivalence matters, subjective free-text

```python
bertscore = evaluate.load("bertscore")
scores = bertscore.compute(
    predictions=preds,
    references=refs,
    lang="en",
    model_type="distilbert-base-uncased",
    device="cpu"
)
avg_f1 = sum(scores['f1']) / len(scores['f1'])
print(avg_f1)  # 0.0-1.0
```

**Note:** Returns per-example scores, must average manually. Use `device="cuda"` if GPU available.

---

## Metric 5: METEOR

**Use when:** Balance between precision and recall, paraphrasing should count

```python
meteor = evaluate.load("meteor")
score = meteor.compute(predictions=preds, references=refs)
print(score['meteor'])  # 0.0-1.0
```

---

## Metric 6: SacreBLEU

**Use when:** Standardized BLEU needed for reproducibility (translation tasks)

```python
sacrebleu = evaluate.load("sacrebleu")
score = sacrebleu.compute(
    predictions=preds,
    references=[[r] for r in refs]  # List of lists!
)
print(score['score'])  # 0.0-100.0 (different scale!)
```

**Note:** Range is 0-100, not 0-1 like BLEU

---

## Metric 7: Character Error Rate (CER)

**Use when:** Speech recognition, OCR, edit distance matters

```python
cer = evaluate.load("cer")
score = cer.compute(predictions=preds, references=refs)
print(score)  # 0.0+ (lower is better)
```

**Note:** Lower is better. Can exceed 1.0.

---

## Metric 8: Word Error Rate (WER)

**Use when:** Speech recognition transcription, word-level accuracy

```python
wer = evaluate.load("wer")
score = wer.compute(predictions=preds, references=refs)
print(score)  # 0.0+ (lower is better)
```

**Note:** Lower is better. Can exceed 1.0.

---

# Metric Selection Guide

| Task Type | Recommended Metrics | Notes |
|-----------|-------------------|-------|
| Math | Exact Match | Clear right/wrong |
| Coding | Unit Tests | NEVER text metrics |
| Multiple Choice | Exact Match | Single correct option |
| Fact-based (short) | Exact Match OR ROUGE/METEOR | Use text metrics if multiple phrasings |
| Free-text (objective) | ROUGE + BLEU + METEOR | Use 3+ metrics |
| Free-text (subjective) | BERTScore | Semantic similarity |

**Key rules:**
- Use 3+ metrics for robust evaluation
- Match metric to task type
- Handle multiple correct answer phrasings (use text metrics instead of exact match)
- NEVER use text metrics for code (always unit tests)
- Exact Match penalizes valid alternative phrasings ("Paris" vs "The capital is Paris")
