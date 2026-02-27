# Multi-LLM Agent Systems: Dataset Requirements

This document contains the gold-standard requirements for dataset selection in Multi-LLM Agent Systems research.

---

## Critical Requirements

- We only want text-based datasets (no images, audio, video, pdf etc.), the text can include anything (text, numbers, tabular, etc.)

---

## 5 Supported Dataset Types

Your datasets must be one of the following types:

### Math
- **Structure:** Problem statement -> single numeric answer -> exact match eval
- **Note:** In some datasets it's always present at the end of the solution and must be extracted
- **Example:** https://huggingface.co/datasets/openai/gsm8k

### Coding
- **Structure:** Problem statement -> code solution -> unit test eval
- **Example:** https://huggingface.co/datasets/openai/openai_humaneval

### Fact-based
- **Structure:** Question -> 1-2 word answer -> exact match eval or text generation metrics eval
- **Example:** https://huggingface.co/datasets/hotpotqa/hotpot_qa

### Multiple choice
- **Structure:** Question -> multiple choice answer -> exact match eval
- **Example:** https://huggingface.co/datasets/Idavidrein/gpqa

### Free-text
- **Structure:** Question -> free-text answer -> text generation metrics eval
