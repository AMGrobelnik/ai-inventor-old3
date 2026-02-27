"""Knowledge graph generation workflow with verification and retry.

Generates knowledge graph triples from research papers with:
1. Initial prompt to extract triples
2. Wikipedia URL verification using aii_web_fetch_direct
3. Retry loop with conversation continuity for failed URLs

Uses aii_web_search_fast and aii_web_fetch_direct MCP tools for web access.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

from pydantic import BaseModel

from ...agent_backend import Agent, AgentOptions
from ...telemetry import AIITelemetry, MessageType
from ...utils import get_tooluniverse_mcp_config
from .verify import verify_wikipedia_urls


def _aggregate_summaries(summaries: list[dict], model: str) -> dict:
    """Aggregate multiple summary dicts into one (SummaryMetrics format).

    Must include all fields expected by console sink's _format_summary_message.
    """
    if not summaries:
        return {}

    # Costs
    total_cost = sum(s.get("total_cost", 0) or 0 for s in summaries)
    token_cost = sum(s.get("token_cost", s.get("total_cost", 0)) or 0 for s in summaries)
    tool_cost = sum(s.get("tool_cost", 0) or 0 for s in summaries)

    # Timing
    runtime_seconds = sum(s.get("runtime_seconds", 0) or 0 for s in summaries)
    llm_time_seconds = sum(s.get("llm_time_seconds", 0) or 0 for s in summaries)
    num_calls = sum(s.get("num_calls", 0) or 0 for s in summaries)

    # Tokens
    input_tokens = sum(s.get("input_tokens", 0) or 0 for s in summaries)
    output_tokens = sum(s.get("output_tokens", 0) or 0 for s in summaries)
    reasoning_tokens = sum(s.get("reasoning_tokens", 0) or 0 for s in summaries)
    cache_write_tokens = sum(s.get("cache_write_tokens", 0) or 0 for s in summaries)
    cache_read_tokens = sum(s.get("cache_read_tokens", 0) or 0 for s in summaries)

    # Tool calls aggregation
    tool_calls = {}
    for s in summaries:
        for tool, count in (s.get("tool_calls", {}) or {}).items():
            tool_calls[tool] = tool_calls.get(tool, 0) + count

    # Tool costs aggregation
    tool_costs = {}
    for s in summaries:
        for tool, cost_info in (s.get("tool_costs", {}) or {}).items():
            if tool not in tool_costs:
                tool_costs[tool] = {"count": 0, "unit": 0, "total": 0}
            if isinstance(cost_info, dict):
                tool_costs[tool]["count"] += cost_info.get("count", 0)
                tool_costs[tool]["total"] += cost_info.get("total", 0)
                # Keep first unit price seen
                if tool_costs[tool]["unit"] == 0:
                    tool_costs[tool]["unit"] = cost_info.get("unit", 0)

    # Error status
    is_error = any(s.get("is_error", False) for s in summaries)
    status = "failed" if is_error else summaries[-1].get("status", "success")

    return {
        "type": "summary",
        "model": model,
        "status": status,
        "is_aggregated": True,
        "is_error": is_error,
        # Costs
        "total_cost": total_cost,
        "token_cost": token_cost,
        "tool_cost": tool_cost,
        # Timing
        "runtime_seconds": runtime_seconds,
        "llm_time_seconds": llm_time_seconds,
        "num_calls": num_calls,
        # Tokens
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        # Tools
        "tool_calls": tool_calls,
        "tool_costs": tool_costs,
    }


@dataclass
class GenKGConfig:
    """Configuration for knowledge graph generation."""
    # Task identification
    paper_id: int
    paper_index: int
    title: str
    abstract: str

    # Prompt
    prompt: str
    system_prompt: str | None = None

    # Agent settings
    model: str = "claude-haiku-4-5"
    max_turns: int = 100
    agent_timeout: int | None = None  # Timeout for entire agent run (None = no timeout)
    agent_retries: int = 2  # Retries for entire agent on failure
    seq_prompt_timeout: int | None = 600
    seq_prompt_retries: int = 5
    cwd: str | Path = "./"
    json_log_path: str | None = None

    # MCP tools config - uses aii_web_search_fast and aii_web_fetch_direct by default
    mcp_servers: dict | None = None  # Custom MCP config (None = use default tooluniverse)
    allowed_tools: list[str] | None = None  # Tool restrictions

    # Structured output
    response_schema: type[BaseModel] | None = None

    # Verification settings
    verify_retries: int = 2  # Retries for URL verification failures
    min_valid_urls: int = 0  # Minimum valid URLs before restructure vs search again

    # Retry prompt builder
    build_retry_prompt_fn: Callable[[dict], str] | None = None


@dataclass
class GenKGResult:
    """Result from knowledge graph generation."""
    paper_id: int
    paper_index: int
    title: str
    triples: list[dict] | None = None
    paper_type: str | None = None
    verified: bool = False
    verification_result: dict | None = None
    retry_attempts: int = 0
    cost: float = 0.0
    run_dir: str | None = None
    error: str | None = None


def _fallback_retry_prompt(verification: dict) -> str:
    """Fallback retry prompt if none provided. Prefer passing build_retry_prompt_fn."""
    failed = verification.get('failed_triples', [])
    if not failed:
        return "Some Wikipedia URLs were invalid. Please fix them using WebSearch to find correct URLs."

    lines = ["The following Wikipedia URLs are invalid:\n"]
    for item in failed[:5]:
        triple = item.get('triple', {})
        name = triple.get('name', 'Unknown')
        url = triple.get('wikipedia_url', 'No URL')
        lines.append(f"- {name}: {url}")

    if len(failed) > 5:
        lines.append(f"... and {len(failed) - 5} more")

    lines.append("\nUse WebSearch with allowed_domains=[\"en.wikipedia.org\"] to find correct URLs and update triples_output.json.")
    return "\n".join(lines)


async def generate_kg_triples(
    config: GenKGConfig,
    telemetry: AIITelemetry | None = None,
) -> GenKGResult:
    """Generate knowledge graph triples with URL verification and retry.

    This workflow:
    1. Runs initial prompt to extract triples
    2. Verifies Wikipedia URLs exist using aii_web_fetch_direct
    3. Retries with conversation continuity if URLs are invalid

    Args:
        config: Generation configuration
        telemetry: Optional telemetry instance for logging

    Returns:
        GenKGResult with triples and verification status
    """
    # Build task identifiers
    task_id = f"triples_paper_idx{config.paper_index}"
    task_name = f"triple-paper_idx{config.paper_index}"

    result = GenKGResult(
        paper_id=config.paper_id,
        paper_index=config.paper_index,
        title=config.title,
    )

    # Emit helpers
    def emit(msg_type: MessageType, msg: str):
        if telemetry:
            telemetry.emit(msg_type, msg)

    def emit_msg(level: str, msg: str):
        if telemetry:
            telemetry.emit_message(level, msg, task_name, task_id)

    # Start task and create callback for summary aggregation
    callback = None
    if telemetry:
        telemetry.emit_task_start(task_id, task_name)
        callback = telemetry.create_callback(task_id, task_name, group="get_triples")

    cwd = Path(config.cwd).resolve()

    # Configure MCP servers for aii_web_search_fast and aii_web_fetch_direct
    mcp_servers = config.mcp_servers
    if mcp_servers is None:
        # Default: use tooluniverse for web tools
        mcp_servers = get_tooluniverse_mcp_config()

    # Disallow built-in WebSearch/WebFetch to force use of fast MCP tools
    disallowed = ["WebSearch", "WebFetch"]

    # Create Agent with SDK native structured output
    options = AgentOptions(
        model=config.model,
        cwd=cwd,
        max_turns=config.max_turns,
        agent_timeout=config.agent_timeout,
        agent_retries=config.agent_retries,
        seq_prompt_timeout=config.seq_prompt_timeout,
        seq_prompt_retries=config.seq_prompt_retries,
        permission_mode="bypassPermissions",
        system_prompt=config.system_prompt,
        continue_seq_item=True,  # Continue conversation between prompts
        json_log_path=config.json_log_path,
        mcp_servers=mcp_servers,
        allowed_tools=config.allowed_tools,
        disallowed_tools=disallowed,
        # Increase buffer size for large web search results (default 1MB too small)
        max_buffer_size=10 * 1024 * 1024,  # 10MB
        # Telemetry integration
        telemetry=telemetry,
        run_id=task_id,
        agent_context=task_name,
        # Structured JSON output (SDK native)
        output_format=config.response_schema.to_struct_output() if config.response_schema else None,
    )

    agent = Agent(options)
    all_responses: list = []
    total_cost = 0.0

    def emit_aggregated_summary():
        """Emit ONE aggregated summary from all responses."""
        if callback and all_responses:
            summaries = []
            for resp in all_responses:
                if resp.prompt_results:
                    for pr in resp.prompt_results:
                        if pr.summary_data:
                            summaries.append(pr.summary_data)
            if summaries:
                callback(_aggregate_summaries(summaries, config.model))

    try:
        # Initial prompt
        response = await agent.run(config.prompt)
        all_responses.append(response)
        total_cost += response.total_cost

        output = response.structured_output
        if not output:
            result.error = "No output generated"
            emit_msg("WARNING", "No output generated")
            emit_aggregated_summary()
            if telemetry:
                telemetry.emit_task_end(task_id, task_name, "No output")
            result.cost = total_cost
            return result

        # Extract triples from output
        if isinstance(output, dict):
            triples = output.get("triples", [])
            paper_type = output.get("paper_type")
        else:
            triples = getattr(output, "triples", [])
            paper_type = getattr(output, "paper_type", None)

        result.triples = triples
        result.paper_type = paper_type

        if not triples:
            result.error = "No triples extracted"
            emit_msg("WARNING", "No triples in output")
            emit_aggregated_summary()
            if telemetry:
                telemetry.emit_task_end(task_id, task_name, "No triples")
            result.cost = total_cost
            return result

        # URL verification retry loop
        def verify_callback(msg: str):
            emit_msg("VERIFY", msg)

        for attempt in range(config.verify_retries + 1):
            result.retry_attempts = attempt

            # Verify Wikipedia URLs
            verification = verify_wikipedia_urls(triples, callback=verify_callback)
            result.verification_result = verification
            result.verified = verification.get("valid", False)

            if verification.get("valid"):
                status = "VALID" + (f" (retry {attempt})" if attempt > 0 else "")
                emit_msg("SUCCESS", f"All URLs verified: {status}")
                emit_aggregated_summary()
                if telemetry:
                    telemetry.emit_task_end(task_id, task_name, status)
                result.cost = total_cost
                return result

            # Retry if attempts left
            if attempt < config.verify_retries:
                valid_count = verification.get('verified', 0)
                failed_count = verification.get('failed', 0)

                # Build retry prompt
                if config.build_retry_prompt_fn:
                    retry_prompt = config.build_retry_prompt_fn(verification)
                else:
                    retry_prompt = _fallback_retry_prompt(verification)

                emit_msg("RETRY", f"URLs failed ({valid_count} valid, {failed_count} failed), retrying...")

                # Retry with conversation continuity
                full_retry_prompt = (
                    f"Your previous response had invalid Wikipedia URLs.\n\n"
                    f"{retry_prompt}"
                )
                response = await agent.run(full_retry_prompt)
                all_responses.append(response)
                total_cost += response.total_cost

                output = response.structured_output
                if output:
                    if isinstance(output, dict):
                        triples = output.get("triples", [])
                        paper_type = output.get("paper_type")
                    else:
                        triples = getattr(output, "triples", [])
                        paper_type = getattr(output, "paper_type", None)
                    result.triples = triples
                    result.paper_type = paper_type
                else:
                    emit_msg("RETRY", "Retry produced no output, keeping previous")

        # All retries exhausted
        if verification:
            failed = verification.get('failed', 0)
            total = verification.get('total', 0)
            status = f"INVALID ({failed}/{total} URLs failed)"
            emit_msg("WARNING", f"URLs invalid after retries: {status}")
            emit_aggregated_summary()
            if telemetry:
                telemetry.emit_task_end(task_id, task_name, status)

        result.cost = total_cost
        return result

    except asyncio.TimeoutError:
        emit(MessageType.ERROR, "Timeout")
        emit_aggregated_summary()
        if telemetry:
            telemetry.emit_task_end(task_id, task_name, "Timeout")
        raise

    except Exception as e:
        emit(MessageType.ERROR, f"Error: {e}")
        emit_aggregated_summary()
        if telemetry:
            telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


__all__ = [
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
]
