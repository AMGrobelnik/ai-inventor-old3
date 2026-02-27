"""Cited argument generation workflow with verification and retry.

Reusable workflow for generating arguments with citations that are verified
against sources. Used by both novelty (web search) and feasibility (resources) workflows.

Pattern (default OpenRouter):
1. LLM call with optional tools (web search)
2. Force output when max tool iterations hit
3. Citation verification with configurable verify function
4. Retry loop with conversation continuation

Pattern (Claude agent):
1. Create Agent with SDK native output_format for structured output
2. Run initial prompt with agent.run()
3. Verify citations
4. Retry with agent.run(retry_prompt) - conversation continues
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Awaitable

from pydantic import BaseModel

from ...agent_backend import Agent, AgentOptions
from ...llm_backend import OpenRouterClient, chat
from ...llm_backend.tool_loop import _emit_summary
from ...telemetry import AIITelemetry, MessageType
from ...abilities.tools.utils import get_openrouter_tools
from ...utils import get_model_short


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude agent in cited args workflow."""
    model: str = "claude-haiku-4-5"  # claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-6
    max_turns: int = 50
    seq_prompt_timeout: int | None = None  # Timeout per prompt in seconds (None = no timeout)
    seq_prompt_retries: int = 10  # Max retry attempts per prompt on failure/timeout
    cwd: str | Path = "./"  # Working directory
    mcp_servers: dict | None = None  # MCP server config for tools
    allowed_tools: list[str] | None = None  # Restrict to these MCP tools (e.g., ["mcp__tooluniverse__aii_web_search_fast"])
    json_log_path: str | None = None  # Path for agent message JSON log


@dataclass
class CitedArgsConfig:
    """Configuration for cited argument generation."""
    # Task identification
    hypothesis_id: str
    hypothesis: dict
    stance: str  # "positive" or "negative"
    task_num: int
    dimension: str  # "novelty" or "feasibility"

    # Prompts
    prompt: str  # User prompt for argument generation
    system_prompt: str

    # LLM settings (only used if claude_agent is None)
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    reasoning_effort: str = ""
    suffix: str = ""

    # Tool settings (for novelty with web search)
    tools: list[str] | None = None  # Tool names like ["aii_web_search_fast", "aii_web_fetch_direct"]
    max_tool_iterations: int = 10
    web_search_backend: str = "tooluniverse"
    force_output_prompt: str | None = None  # Prompt when max iterations hit

    # Verification settings
    verify_retries: int = 1
    min_valid_citations: int = 5  # Minimum valid citations to restructure (vs search again)

    # AIITelemetry settings
    group: str = ""  # Group name for telemetry

    # Callbacks (set by caller)
    verify_fn: Callable[[str, Any], dict] | None = None  # (text, cache/resources) -> verification_result
    build_retry_prompt_fn: Callable[[dict, bool], str] | None = None  # (verification, no_citations) -> prompt

    # Optional context for verification (e.g., web cache or resources text)
    verify_context: Any = None

    # Claude agent config - if provided, uses Agent with conversation continuity
    claude_agent: ClaudeAgentConfig | None = None
    # Schema for structured output (required if claude_agent provided)
    response_schema: type[BaseModel] | None = None


@dataclass
class CitedArgsResult:
    """Result from cited argument generation."""
    hypothesis_id: str
    dimension: str
    stance: str
    model: str
    provider: str = "openrouter"
    argument: str | None = None
    verified: bool = False
    verification_result: dict | None = None
    retry_attempts: int = 0
    error: str | None = None


async def generate_cited_argument(
    telemetry: AIITelemetry,
    config: CitedArgsConfig,
) -> dict:
    """Generate a cited argument with verification and retry.

    This workflow handles:
    1. Initial LLM call with optional tools (for web search)
    2. Force output when max tool iterations hit
    3. Citation verification with configurable verify function
    4. Retry loop with configurable retry prompt builder
    5. Proper telemetry callbacks for status updates

    If config.claude_agent is provided, uses Claude Agent with conversation continuity.
    Otherwise uses OpenRouter with conversation continuation for retries.

    Returns:
        dict with keys: hypothesis_id, dimension, stance, model, provider,
                       argument, verified, verification_result, retry_attempts, error
    """
    # Build task identifiers
    stance_short = "pos" if config.stance == "positive" else "neg"
    dim_short = "nov" if config.dimension == "novelty" else "feas"
    model_short = get_model_short(config.model) if config.model else "agent"

    task_id = f"{config.hypothesis_id}_{dim_short}_{stance_short}_{config.task_num}"
    task_name = f"{dim_short}-{stance_short}-{config.task_num}__{model_short}"

    result = CitedArgsResult(
        hypothesis_id=config.hypothesis_id,
        dimension=config.dimension,
        stance=config.stance,
        model=config.model or "claude-agent",
        provider="claude_agent" if config.claude_agent else "openrouter",
    )

    # =========================================================================
    # CLAUDE AGENT PATH (conversation continuity for retries)
    # =========================================================================
    if config.claude_agent is not None:
        # Create callback with group tracking (same as OpenRouter path)
        callback = telemetry.create_callback(task_id, task_name, group=config.group)

        def verify_callback(msg: str):
            telemetry.emit_message("VERIFY", msg, task_name, task_id)

        return await _generate_cited_argument_claude_agent(
            telemetry, config, task_id, task_name, callback, verify_callback, result
        )

    # =========================================================================
    # OPENROUTER PATH (default)
    # =========================================================================
    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group=config.group)

    def verify_callback(msg: str):
        telemetry.emit_message("VERIFY", msg, task_name, task_id)

    return await _generate_cited_argument_openrouter(
        telemetry, config, task_id, task_name, callback, verify_callback, result
    )


async def _generate_cited_argument_claude_agent(
    telemetry: AIITelemetry,
    config: CitedArgsConfig,
    task_id: str,
    task_name: str,
    callback: Callable[[dict], None],
    verify_callback: Callable[[str], None],
    result: CitedArgsResult,
) -> dict:
    """Claude agent path with conversation continuity for retries.

    Creates ONE Agent that persists across citation verification retries.
    Uses SDK native output_format for structured JSON output.
    """
    # Register task with sequencer (must be before any emit_message calls)
    telemetry.emit_task_start(task_id, task_name)

    agent_cfg = config.claude_agent
    cwd = Path(agent_cfg.cwd).resolve()

    # Disallow built-in web tools:
    # - If MCP servers: force usage of aii_web_search_fast, aii_web_fetch_direct
    # - If no MCP servers (feasibility): prevent any web access, use resources only
    disallowed = ["WebSearch", "WebFetch"]

    # Create Agent with SDK native structured output
    options = AgentOptions(
        model=agent_cfg.model,
        cwd=cwd,
        max_turns=agent_cfg.max_turns,
        seq_prompt_timeout=agent_cfg.seq_prompt_timeout,
        seq_prompt_retries=agent_cfg.seq_prompt_retries,
        permission_mode="bypassPermissions",
        system_prompt=config.system_prompt,
        continue_seq_item=True,  # Continue conversation between prompts
        json_log_path=agent_cfg.json_log_path,
        mcp_servers=agent_cfg.mcp_servers or {},
        allowed_tools=agent_cfg.allowed_tools,  # Restrict to specific MCP tools
        disallowed_tools=disallowed,
        # Increase buffer size for large web search results (default 1MB too small)
        max_buffer_size=10 * 1024 * 1024,  # 10MB
        # Telemetry integration
        telemetry=telemetry,
        run_id=task_id,
        agent_context=task_name,
        # Structured JSON output (SDK native)
        output_format=config.response_schema.to_struct_output(),
    )

    agent = Agent(options)

    # Track all responses for aggregated summary at end
    all_responses: list = []

    def _emit_aggregated_summary():
        """Emit aggregated summary from all agent responses through callback for group aggregation."""
        if not all_responses:
            return
        # Aggregate summary data from all prompt results
        all_summaries = []
        for resp in all_responses:
            for pr in resp.prompt_results:
                if pr.summary_data:
                    all_summaries.append(pr.summary_data)
        if not all_summaries:
            return
        # Aggregate metrics
        is_error = any(s.get("is_error", False) for s in all_summaries)
        summary = {
            "type": "summary",
            "run_id": task_id,
            "agent_context": task_name,
            "total_cost": sum(s.get("total_cost", 0) for s in all_summaries),
            "token_cost": sum(s.get("token_cost", 0) for s in all_summaries),
            "tool_cost": sum(s.get("tool_cost", 0) for s in all_summaries),
            "model": all_summaries[0].get("model", "claude-agent") if all_summaries else "claude-agent",
            "status": "failed" if is_error else "aggregated",
            "is_aggregated": len(all_summaries) > 1,
            "is_error": is_error,
            "num_calls": sum(s.get("num_calls", 1) for s in all_summaries),
            "runtime_seconds": sum(s.get("runtime_seconds", 0) for s in all_summaries),
            "llm_time_seconds": sum(s.get("llm_time_seconds", 0) for s in all_summaries),
            "input_tokens": sum(s.get("input_tokens", 0) for s in all_summaries),
            "output_tokens": sum(s.get("output_tokens", 0) for s in all_summaries),
            "reasoning_tokens": sum(s.get("reasoning_tokens", 0) for s in all_summaries),
            "cache_write_tokens": sum(s.get("cache_write_tokens", 0) for s in all_summaries),
            "cache_read_tokens": sum(s.get("cache_read_tokens", 0) for s in all_summaries),
            "tool_calls": {},
            "tool_costs": {},
        }
        # Aggregate tool calls
        for s in all_summaries:
            for tool, count in s.get("tool_calls", {}).items():
                summary["tool_calls"][tool] = summary["tool_calls"].get(tool, 0) + count
        # Emit through callback for group aggregation
        callback(summary)

    try:
        current_text = None
        verification = None

        # Initial prompt
        response = await agent.run(config.prompt)
        all_responses.append(response)
        output = response.structured_output

        if not output:
            result.error = "No output generated"
            telemetry.emit_message("WARNING", "No output generated", task_name, task_id)
            _emit_aggregated_summary()
            telemetry.emit_task_end(task_id, task_name, "No output")
            return result.__dict__

        current_text = output.get("argument") if isinstance(output, dict) else output.argument

        if not current_text:
            result.error = "No argument in output"
            telemetry.emit_message("WARNING", "No argument in output", task_name, task_id)
            _emit_aggregated_summary()
            telemetry.emit_task_end(task_id, task_name, "No argument")
            return result.__dict__

        # Citation verification retry loop
        for attempt in range(config.verify_retries + 1):
            result.argument = current_text

            # Call verification function
            if config.verify_fn:
                try:
                    verification = config.verify_fn(
                        current_text,
                        config.verify_context,
                        verify_callback,
                    )
                except ValueError as e:
                    # No citations found in response
                    if attempt < config.verify_retries:
                        if config.build_retry_prompt_fn:
                            retry_instructions = config.build_retry_prompt_fn({}, True)
                        else:
                            retry_instructions = "Please include citations in your response."

                        telemetry.emit_message(
                            "RETRY",
                            "No citations found, requesting quotes...",
                            task_name, task_id
                        )
                        # Agent logs its own prompts, no need to emit here

                        # Retry with conversation continuity
                        retry_prompt = (
                            f"Your previous response had no citations.\n\n"
                            f"{retry_instructions}"
                        )
                        response = await agent.run(retry_prompt)
                        all_responses.append(response)
                        output = response.structured_output

                        if output:
                            current_text = output.get("argument") if isinstance(output, dict) else output.argument
                        continue
                    else:
                        result.error = f"Citation format error: {e}"
                        _emit_aggregated_summary()
                        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
                        return result.__dict__
            else:
                verification = {"valid": True, "results": [], "total": 0, "failed": 0}

            result.verified = verification.get("valid", False)
            result.verification_result = verification
            result.retry_attempts = attempt

            if verification.get("valid"):
                status = "VALID" + (f" (retry {attempt})" if attempt > 0 else "")
                telemetry.emit_message("SUCCESS", f"Citations verified: {status}", task_name, task_id)
                _emit_aggregated_summary()
                telemetry.emit_task_end(task_id, task_name, status)
                return result.__dict__

            # Retry if attempts left
            if attempt < config.verify_retries:
                valid_quotes = set(
                    r.get('quote', '')
                    for r in verification.get('results', [])
                    if r.get('status') == 'valid'
                )
                valid_count = len(valid_quotes)

                if config.build_retry_prompt_fn:
                    retry_instructions = config.build_retry_prompt_fn(verification, False)
                else:
                    retry_instructions = "Please try again with valid citations."

                retry_msg = f"Citations failed ({valid_count} valid), retrying..."
                telemetry.emit_message("RETRY", retry_msg, task_name, task_id)
                # Agent logs its own prompts, no need to emit here

                # Retry with conversation continuity (agent remembers previous response)
                retry_prompt = (
                    f"Your previous response had citation issues.\n\n"
                    f"{retry_instructions}"
                )
                response = await agent.run(retry_prompt)
                all_responses.append(response)
                output = response.structured_output

                if output:
                    current_text = output.get("argument") if isinstance(output, dict) else output.argument
                else:
                    telemetry.emit_message(
                        "RETRY",
                        "Retry produced no output, keeping previous",
                        task_name, task_id
                    )

        # All retries exhausted
        if verification:
            status = f"INVALID ({verification.get('failed', 0)}/{verification.get('total', 0)} failed)"
            telemetry.emit_message("WARNING", f"Citations invalid after retries: {status}", task_name, task_id)
            _emit_aggregated_summary()
            telemetry.emit_task_end(task_id, task_name, status)

    except asyncio.TimeoutError:
        telemetry.emit(MessageType.ERROR, "Timeout")
        _emit_aggregated_summary()
        telemetry.emit_task_end(task_id, task_name, "Timeout")
        raise
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Error: {e}")
        _emit_aggregated_summary()
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise

    return result.__dict__


async def _generate_cited_argument_openrouter(
    telemetry: AIITelemetry,
    config: CitedArgsConfig,
    task_id: str,
    task_name: str,
    callback: Callable[[dict], None],
    verify_callback: Callable[[str], None],
    result: CitedArgsResult,
) -> dict:
    """OpenRouter path with conversation continuation for retries."""
    try:
        effective_model = OpenRouterClient.resolve_model(config.model, config.suffix)
        tools = get_openrouter_tools(config.tools) if config.tools else None

        async with OpenRouterClient(
            api_key=config.api_key,
            model=effective_model,
            timeout=config.timeout,
        ) as client:
            # Initial generation (prompt logged by chat() via callback)
            loop_result = await chat(
                client=client,
                prompt=config.prompt,
                system=config.system_prompt,
                tools=tools,
                max_iterations=config.max_tool_iterations if tools else 1,
                message_callback=callback,
                reasoning_effort=config.reasoning_effort,
                web_search_backend=config.web_search_backend if tools else None,
                timeout=config.timeout,
                emit_summary=False,  # Emit once at end
            )

            messages = loop_result.messages
            conv_stats = loop_result.stats

            # Force output if max iterations hit with pending tool calls
            if (config.force_output_prompt and
                loop_result.hit_max_iterations and
                loop_result.last_response_has_tool_calls):
                telemetry.emit_message(
                    "INFO",
                    f"Tool limit ({config.max_tool_iterations}) reached, forcing output...",
                    task_name, task_id
                )
                messages.append({"role": "user", "content": config.force_output_prompt})
                loop_result = await chat(
                    client=client,
                    messages=messages,
                    tools=None,
                    message_callback=callback,
                    reasoning_effort=config.reasoning_effort,
                    timeout=config.timeout,
                    conversation_stats=conv_stats,
                    emit_summary=False,
                )
                messages = loop_result.messages

            output_text = client.extract_text_from_response(loop_result.response)
            if not output_text:
                result.error = "No output generated"
                telemetry.emit_task_end(task_id, task_name, "No output")
                return result.__dict__

            # Verification and retry loop
            current_text = output_text
            verification = None

            for attempt in range(config.verify_retries + 1):
                try:
                    result.argument = current_text

                    if config.verify_fn:
                        verification = config.verify_fn(
                            current_text,
                            config.verify_context,
                            verify_callback,
                        )
                    else:
                        verification = {"valid": True, "results": [], "total": 0, "failed": 0}

                    result.verified = verification.get("valid", False)
                    result.verification_result = verification
                    result.retry_attempts = attempt

                    if verification.get("valid"):
                        status = "VALID" + (f" (retry {attempt})" if attempt > 0 else "")
                        _emit_summary(callback, conv_stats, client)
                        telemetry.emit_task_end(task_id, task_name, status)
                        return result.__dict__

                    # Retry if attempts left
                    if attempt < config.verify_retries:
                        valid_quotes = set(
                            r.get('quote', '')
                            for r in verification.get('results', [])
                            if r.get('status') == 'valid'
                        )
                        valid_count = len(valid_quotes)
                        has_enough_valid = valid_count >= config.min_valid_citations

                        if config.build_retry_prompt_fn:
                            retry_prompt = config.build_retry_prompt_fn(verification, False)
                        else:
                            retry_prompt = "Please try again with valid citations."

                        retry_msg = (
                            f"Citations failed ({valid_count} valid), "
                            f"{'restructuring' if has_enough_valid else 'searching again'}..."
                        )
                        telemetry.emit_message("RETRY", retry_msg, task_name, task_id)
                        # Note: prompt is logged by chat() via message_callback, no need to emit here

                        messages.append({"role": "user", "content": retry_prompt})

                        retry_tools = None if has_enough_valid else tools
                        loop_result = await chat(
                            client=client,
                            messages=messages,
                            tools=retry_tools,
                            max_iterations=config.max_tool_iterations if retry_tools else 1,
                            message_callback=callback,
                            reasoning_effort=config.reasoning_effort,
                            web_search_backend=config.web_search_backend if retry_tools else None,
                            timeout=config.timeout,
                            conversation_stats=conv_stats,
                            emit_summary=False,
                        )
                        messages = loop_result.messages

                        new_text = client.extract_text_from_response(loop_result.response)
                        if new_text:
                            current_text = new_text
                        else:
                            telemetry.emit_message(
                                "RETRY",
                                "Retry produced no output, keeping previous",
                                task_name, task_id
                            )

                except ValueError as e:
                    if attempt < config.verify_retries:
                        if config.build_retry_prompt_fn:
                            retry_prompt = config.build_retry_prompt_fn({}, True)
                        else:
                            retry_prompt = "Please include citations in your response."

                        telemetry.emit_message(
                            "RETRY",
                            "No citations found, requesting quotes...",
                            task_name, task_id
                        )
                        # Note: prompt is logged by chat() via message_callback, no need to emit here

                        messages.append({"role": "user", "content": retry_prompt})
                        loop_result = await chat(
                            client=client,
                            messages=messages,
                            tools=tools,
                            max_iterations=config.max_tool_iterations if tools else 1,
                            message_callback=callback,
                            reasoning_effort=config.reasoning_effort,
                            web_search_backend=config.web_search_backend if tools else None,
                            timeout=config.timeout,
                            conversation_stats=conv_stats,
                            emit_summary=False,
                        )
                        messages = loop_result.messages

                        new_text = client.extract_text_from_response(loop_result.response)
                        if new_text:
                            current_text = new_text
                    else:
                        result.error = f"Citation format error: {e}"
                        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
                        return result.__dict__

            # All retries exhausted
            if verification:
                status = f"INVALID ({verification.get('failed', 0)}/{verification.get('total', 0)} failed)"
                _emit_summary(callback, conv_stats, client)
                telemetry.emit_task_end(task_id, task_name, status)

    except asyncio.TimeoutError:
        telemetry.emit_task_end(task_id, task_name, f"Timeout ({config.timeout}s)")
        raise
    except Exception as e:
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise

    return result.__dict__


def cap_results(
    positive: list[str],
    negative: list[str],
    mode: str = "none",
) -> tuple[list[str], list[str]]:
    """Balance positive/negative argument lists.

    Args:
        positive: List of positive arguments
        negative: List of negative arguments
        mode: Capping mode
            - "equal": Cap both to match smaller side
            - "none": No capping, keep all

    Returns:
        Tuple of (capped_positive, capped_negative)
    """
    if mode == "equal":
        min_count = min(len(positive), len(negative))
        return positive[:min_count], negative[:min_count]
    return positive, negative


def collect_verified_arguments(
    results: list[dict],
    cap_mode: str = "none",
) -> dict[str, list[str]]:
    """Collect and organize verified arguments from results.

    Args:
        results: List of CitedArgsResult dicts
        cap_mode: How to balance pos/neg ("equal" or "none")

    Returns:
        Dict with keys like "novelty_positive", "novelty_negative", etc.
        Each contains list of verified argument strings.
    """
    by_key: dict[str, list[str]] = {}

    for r in results:
        if r is None or not r.get("verified") or not r.get("argument"):
            continue
        key = f"{r['dimension']}_{r['stance']}"
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(r["argument"])

    # Apply cap mode per dimension
    dimensions = set(r["dimension"] for r in results if r)
    for dim in dimensions:
        pos_key = f"{dim}_positive"
        neg_key = f"{dim}_negative"
        pos = by_key.get(pos_key, [])
        neg = by_key.get(neg_key, [])
        by_key[pos_key], by_key[neg_key] = cap_results(pos, neg, cap_mode)

    return by_key


__all__ = [
    "CitedArgsConfig",
    "CitedArgsResult",
    "generate_cited_argument",
    "cap_results",
    "collect_verified_arguments",
]
