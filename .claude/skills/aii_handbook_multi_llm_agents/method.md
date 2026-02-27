# Multi-LLM Agent Systems: Method Implementation Requirements

## Dependencies

```bash
pip install "mirascope[openai]" pydantic>=2.0
```

**Note:** For other providers, use `mirascope[anthropic]` or `mirascope[all]`.

---

## Critical Requirements

- You must use Mirascope for creating your multiagent system.
- DO NOT FORGET TO IMPLEMENT THE BASELINE (comparison) METHOD. This method is typically a single LLM call or a typical multiagent system without any of our proposed method's features.
- Avoid 'cheating' when comparing creating your method relative to a baseline: do not give only your method few shot examples or answers, and not to the baseline. This is cheating as your method has access to more information, and is not necessarily better.
- DO NOT LEAVE YOUR WORK 'HALF-FINISHED', IT MUST FULLY MATCH THE ORIGINAL immediately.
- Multiagent systems are very chaotic, a small change in initial prompt or temperature can drastically change the results. Make sure to vary these so each variation can be evaluated separately later.
- Use best practices in prompt engineering, look up how to do prompt engineering for each AI model you use with WebSearch then WebFetch.
- When creating your multiagent systems, try to draw paralells to how humans organize themselves to solve similar tasks or hard problems in general.

---

# Multi-Agent Pattern Implementations

## Prerequisites

```python
from typing import List
from pydantic import BaseModel
from mirascope import llm, prompt_template, Messages
```

---

## Pattern 1: Reasoning + Verification

Generate solution → Verify correctness → Retry if wrong

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

@llm.call(provider="openai", model="gpt-5", response_model=Solution)
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-nano", response_model=bool)
def verify(task: str, solution: Solution) -> str:
    return f"Verify if '{solution.answer}' correctly answers '{task}'"

def reason_with_verification():
    solution = reason(TASK)
    is_correct = verify(TASK, solution)
    if not is_correct:
        solution = reason(f"{TASK} (try again)")
    return solution
```

---

## Pattern 2: Majority Voting

Multiple agents solve independently → Aggregator picks best

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

@llm.call(provider="openai", model="gpt-5", response_model=Solution)
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Solution,
          call_params={"temperature": 0.3})
def reason_precise(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "low", "verbosity": "high"})
def reason_creative(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=str,
          call_params={"temperature": 0.1})
def aggregate(solutions: List[Solution]) -> str:
    answers = [s.answer for s in solutions]
    return f"Choose best answer from: {answers}"

def majority_voting():
    solutions = []
    reasoners = [reason, reason_precise, reason_creative]
    for reasoner in reasoners:
        sol = reasoner(TASK)
        solutions.append(sol)
    best = aggregate(solutions)
    return best
```

---

## Pattern 3: Debate with Judge

Two specialized agents argue → Judge picks winner

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

class Judgment(BaseModel):
    winner: str
    reasoning: str
    margin: float

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "low", "verbosity": "high"})
def reason_creative(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Solution,
          call_params={"temperature": 0.3})
def reason_precise(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-mini", response_model=Judgment,
          call_params={"reasoning_effort": "low", "verbosity": "medium"})
def judge(task: str, arg1: Solution, arg2: Solution) -> str:
    return f"Judge which answer is better for '{task}': 1) {arg1.answer} 2) {arg2.answer}"

def debate():
    cultural = reason_creative(f"{TASK} from cultural perspective")
    mathematical = reason_precise(f"{TASK} from mathematical perspective")
    decision = judge(TASK, cultural, mathematical)
    return cultural if "1" in decision.winner or "cultural" in decision.winner.lower() else mathematical
```

---

## Pattern 4: Hierarchical Decomposition

Decompose task → Solve subtasks → Synthesize results

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

class Analysis(BaseModel):
    findings: List[str]
    score: int

@llm.call(provider="openai", model="gpt-5-nano", response_model=List[str])
def decompose(task: str) -> str:
    return f"Break down '{task}' into 3 subtasks"

@llm.call(provider="openai", model="gpt-5", response_model=Solution)
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Solution,
          call_params={"temperature": 0.3})
def reason_precise(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "low", "verbosity": "high"})
def reason_creative(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-mini", response_model=Analysis,
          call_params={"reasoning_effort": "medium"})
def analyze_subtasks(results: List[Solution]) -> str:
    return f"Analyze findings from subtasks: {[s.answer[:30] for s in results]}"

def hierarchical():
    subtasks = decompose(TASK)
    solutions = []
    reasoners = [reason, reason_precise, reason_creative]
    for st, reasoner in zip(subtasks, reasoners):
        sol = reasoner(st)
        solutions.append(sol)
    analysis = analyze_subtasks(solutions)
    return analysis
```

---

## Pattern 5: Reflection

Generate solution → Score quality → Improve if score < threshold

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

@llm.call(provider="openai", model="gpt-5", response_model=Solution)
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=int,
          call_params={"temperature": 0.1})
def score_answer(answer: str) -> str:
    return f"Score this answer from 1-10: {answer}"

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "high", "verbosity": "high"})
def improve(solution: Solution, feedback: bool) -> str:
    return f"Improve '{solution.answer}' (was {'correct' if feedback else 'incorrect'})"

def reflection():
    initial = reason(TASK)
    score = score_answer(initial.answer)
    if score < 8:
        refined = improve(initial, False)
        return refined
    return initial
```

---

## Pattern 6: Tool-Augmented Reasoning

Plan tools → Execute tools → Reason with results

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

class ToolPlan(BaseModel):
    tools: List[str]
    priority: int
    description: str

def calculator(expr: str) -> float:
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except:
        return 0.0

def search(query: str) -> str:
    knowledge = {
        "42": "Answer to Life, Universe, Everything (Hitchhiker's Guide)",
        "math": "6 × 7 = 42, highly composite number",
    }
    return knowledge.get(query.split()[0], "No results")

@llm.call(provider="openai", model="gpt-5-nano", response_model=ToolPlan,
          call_params={"verbosity": "low"})
def plan_tool_use(task: str) -> str:
    return f"What tools (calculator/search) and queries needed for: {task}"

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "low", "verbosity": "high"})
def reason_creative(task: str) -> str:
    return task

def with_tools():
    plan = plan_tool_use(TASK)
    results = {}
    if "calculator" in plan.tools or "calculator" in plan.description.lower():
        results["calc"] = calculator("6 * 7")
    if "search" in plan.tools or "search" in plan.description.lower():
        results["info"] = search("42")
    enriched_task = f"{TASK} Context: {results}"
    solution = reason_creative(enriched_task)
    return solution
```

---

## Pattern 7: Prompt Templates with System Messages

Define agent roles via system messages for consistent behavior

```python
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

@llm.call(provider="openai", model="gpt-5", response_model=List[str],
          call_params={"reasoning_effort": "low", "verbosity": "medium"})
@prompt_template()
def analysis_prompt(task: str, context: str = "") -> Messages.Type:
    return [
        Messages.System("You are an expert analyst. Be concise and precise."),
        Messages.User(f"Analyze this: {task}. Context: {context}" if context else f"Analyze this: {task}")
    ]

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution,
          call_params={"reasoning_effort": "low", "verbosity": "high"})
def reason_creative(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Solution,
          call_params={"temperature": 0.3})
def reason_precise(task: str) -> str:
    return task

def templated_analysis():
    prompt_messages = analysis_prompt(TASK, "Focus on cultural significance")
    cultural = reason_creative(str(prompt_messages))
    themes = analysis_prompt(TASK, "Extract 3 key themes")
    synthesis_prompt = f"Synthesize insights about {TASK} considering themes: {themes}"
    final = reason_precise(synthesis_prompt)
    return final
```
