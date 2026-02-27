"""
Result aggregation - combine results from multiple prompts.
"""

from datetime import datetime
from pathlib import Path

from ..models import AgentOptions, AgentResponse, PromptResult


def aggregate_summaries(prompt_results: list[PromptResult]) -> dict:
    """
    Aggregate summary data from multiple prompts into one.

    Args:
        prompt_results: List of results with summary_data

    Returns:
        Aggregated summary dict
    """
    summaries = [r.summary_data for r in prompt_results if r.summary_data]
    if not summaries:
        return {}

    # Aggregate numeric fields
    total_cost = sum(s.get("total_cost", 0) for s in summaries)
    token_cost = sum(s.get("token_cost", 0) for s in summaries)
    tool_cost = sum(s.get("tool_cost", 0) for s in summaries)
    num_calls = sum(s.get("num_calls", 0) for s in summaries)
    runtime_seconds = sum(s.get("runtime_seconds", 0) for s in summaries)
    llm_time_seconds = sum(s.get("llm_time_seconds", 0) for s in summaries)
    input_tokens = sum(s.get("input_tokens", 0) for s in summaries)
    output_tokens = sum(s.get("output_tokens", 0) for s in summaries)
    reasoning_tokens = sum(s.get("reasoning_tokens", 0) for s in summaries)
    cache_write_tokens = sum(s.get("cache_write_tokens", 0) for s in summaries)
    cache_read_tokens = sum(s.get("cache_read_tokens", 0) for s in summaries)

    # Aggregate tool calls
    tool_calls = {}
    for s in summaries:
        for tool, count in s.get("tool_calls", {}).items():
            tool_calls[tool] = tool_calls.get(tool, 0) + count

    # Get model (use first one or "aggregated" if multiple)
    models = set(s.get("model", "") for s in summaries if s.get("model"))
    model = list(models)[0] if len(models) == 1 else "aggregated" if models else ""

    # Check if any had errors
    is_error = any(s.get("is_error", False) for s in summaries)
    status = "failed" if is_error else "aggregated"

    return {
        "type": "summary",
        "total_cost": total_cost,
        "token_cost": token_cost,
        "tool_cost": tool_cost,
        "model": model,
        "status": status,
        "is_aggregated": True,
        "num_calls": num_calls,
        "runtime_seconds": runtime_seconds,
        "llm_time_seconds": llm_time_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "tool_calls": tool_calls,
        "tool_costs": {},
        "message_text": f"Total cost: ${total_cost:.4f} ({len(summaries)} prompts)",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tool_name": "",
        "tool_id": "",
        "agent_context": "",
        "subagent_id": None,
        "parent_tool_use_id": None,
        "is_error": is_error,
        "message_metadata": {
            "num_prompts": len(summaries),
        },
    }


def aggregate_prompt_results(
    options: AgentOptions,
    prompt_results: list[PromptResult],
) -> AgentResponse:
    """
    Aggregate results from all executed prompts.

    This step:
    1. Calculates total cost across all prompts
    2. Collects all messages
    3. Extracts final response from last prompt
    4. Determines JSON log path
    5. If multiple prompts, creates and emits aggregated summary
    6. Builds final AgentResponse

    Args:
        options: Agent configuration options
        prompt_results: List of results from each prompt execution

    Returns:
        AgentResponse with aggregated data
    """
    # Handle empty prompt_results (all prompts failed)
    if not prompt_results:
        json_log_path = str(Path(options.json_log_path)) if options.json_log_path else str(
            (Path(options.cwd) if options.cwd else Path.cwd()) / "all_messages.jsonl"
        )
        return AgentResponse(
            final_response="",
            total_cost=0.0,
            prompt_results=[],
            final_session_id=None,
            all_messages=[],
            json_log_path=json_log_path,
            failed=True,
        )

    # Calculate totals
    total_cost = sum(result.cost for result in prompt_results)
    all_messages = []
    for result in prompt_results:
        all_messages.extend(result.messages)

    # Final response is from the last prompt
    final_result = prompt_results[-1]

    # For multi-prompt sequences, create aggregated summary for JSON log
    # Note: Summary emission is handled by caller (workflow) after verification passes/fails
    if len(prompt_results) > 1:
        aggregated_summary = aggregate_summaries(prompt_results)
        if aggregated_summary:
            # Add to messages for JSON log only (not emitted to console here)
            all_messages.append(aggregated_summary)

    # Determine JSON log path
    if options.json_log_path:
        json_log_path = str(Path(options.json_log_path))
    else:
        cwd = Path(options.cwd) if options.cwd else Path.cwd()
        json_log_path = str(cwd / "all_messages.jsonl")

    return AgentResponse(
        final_response=final_result.response,
        total_cost=total_cost,
        prompt_results=prompt_results,
        final_session_id=final_result.session_id,
        all_messages=all_messages,
        json_log_path=json_log_path,
    )
