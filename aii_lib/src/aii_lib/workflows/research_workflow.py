"""
ResearchWorkflow - Research workflow with tool support.

Supports two backends:
- OpenRouter (default): Tool loop with aii_web_search_fast, aii_web_fetch_direct
- Claude Agent: MCP tools, structured output via file

Pattern (OpenRouter):
1. LLM calls with tools
2. Tool loop runs until model stops or hits max_iterations
3. Force output with custom prompt
4. Return structured output

Pattern (Claude Agent):
1. Agent runs with MCP tools
2. Writes structured output to file
3. Validates against schema
4. Retries with feedback if invalid

Usage:
    # OpenRouter
    async with OpenRouterClient(api_key=key, model=model) as client:
        result = await research_workflow(
            client=client,
            prompt="Research...",
            system="You are...",
            response_format=Hypothesis,
            config=ResearchWorkflowConfig(...),
        )

    # Claude Agent
    result = await research_workflow(
        prompt="Research...",
        system="You are...",
        response_format=Hypothesis,
        use_claude_agent=True,
        claude_model="claude-sonnet-4-5",
        cwd=Path("./workspace"),
    )
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from ..llm_backend.tool_loop import chat, ToolLoopResult, _emit_summary
from ..llm_backend.openrouter.or_to_json import extract_output
from ..abilities.tools.utils import get_openrouter_tools
from ..utils.agent_to_llm import ClaudeAgentToLLMStructOut, get_tooluniverse_mcp_config


# Default tools for research
RESEARCH_TOOLS = ["aii_web_search_fast", "aii_web_fetch_direct"]


@dataclass
class ResearchWorkflowConfig:
    """Configuration for research workflow."""
    max_tool_iterations: int = 10
    force_output_prompt: str = ""  # Required for OpenRouter - caller must provide
    tools: list[str] = field(default_factory=lambda: RESEARCH_TOOLS.copy())
    web_search_backend: str = "auto"
    timeout: float = 300
    # Claude agent specific
    max_retries: int = 2  # Retries if schema validation fails


@dataclass
class ResearchWorkflowResult:
    """Result from research workflow."""
    output: dict | None  # Parsed JSON output
    output_text: str | None  # Raw text output
    tool_result: ToolLoopResult | None  # Full tool loop result (None for Claude agent)
    forced_output: bool = False  # True if output was forced after max iterations
    provider: str = "openrouter"  # "openrouter" or "claude_agent"

    @property
    def success(self) -> bool:
        """True if we have any output (parsed dict OR raw text)."""
        return self.output is not None or bool(self.output_text)


async def research_workflow(
    prompt: str,
    system: str | None = None,
    response_format: type[BaseModel] | None = None,
    *,
    # Backend selection
    use_claude_agent: bool = False,
    # OpenRouter params (required if use_claude_agent=False)
    client=None,
    reasoning_effort: str | None = None,
    # Claude agent params (used if use_claude_agent=True)
    claude_model: str = "claude-sonnet-4-5",
    claude_max_turns: int = 100,
    cwd: Path | None = None,
    output_dir: Path | None = None,
    # Common params
    config: ResearchWorkflowConfig | None = None,
    message_callback: Callable[[dict], None] | None = None,
    telemetry=None,
    task_id: str | None = None,
) -> ResearchWorkflowResult:
    """Run research workflow with automatic tool loop and structured output.

    Args:
        prompt: User prompt for research task
        system: System prompt
        response_format: Pydantic model for structured output
        use_claude_agent: If True, use Claude agent instead of OpenRouter
        client: OpenRouterClient instance (required if use_claude_agent=False)
        reasoning_effort: Reasoning effort level for OpenRouter
        claude_model: Claude model name (sonnet, opus, etc.)
        claude_max_turns: Max turns for Claude agent
        cwd: Working directory for Claude agent
        output_dir: Output directory for logs
        config: Research configuration
        message_callback: Callback for logging messages
        telemetry: AIITelemetry instance
        task_id: Task ID for telemetry

    Returns:
        ResearchWorkflowResult with parsed output and metadata
    """
    cfg = config or ResearchWorkflowConfig()

    # =========================================================================
    # CLAUDE AGENT PATH
    # =========================================================================
    if use_claude_agent:
        return await _research_workflow_claude_agent(
            prompt=prompt,
            system=system or "",
            response_format=response_format,
            model=claude_model,
            max_turns=claude_max_turns,
            cwd=cwd or Path.cwd(),
            output_dir=output_dir,
            tools=cfg.tools,
            max_retries=cfg.max_retries,
            telemetry=telemetry,
            task_id=task_id,
            message_callback=message_callback,
        )

    # =========================================================================
    # OPENROUTER PATH
    # =========================================================================
    if client is None:
        raise ValueError("client is required when use_claude_agent=False")

    # Get tools
    tools = get_openrouter_tools(cfg.tools) if cfg.tools else None

    # Run tool loop WITHOUT structured output - let model research freely
    result = await chat(
        client=client,
        prompt=prompt,
        system=system,
        tools=tools,
        max_iterations=cfg.max_tool_iterations if tools else 1,
        response_format=None,  # No structured output during tool loop
        message_callback=message_callback,
        reasoning_effort=reasoning_effort,
        web_search_backend=cfg.web_search_backend,
        timeout=cfg.timeout,
        emit_summary=False,
    )

    messages = result.messages

    # Force output at the end to apply structured format
    force_reason = "Research complete"
    if result.hit_max_iterations and result.last_response_has_tool_calls:
        force_reason = f"Tool limit ({cfg.max_tool_iterations}) reached"

    if message_callback:
        message_callback({
            "type": "info",
            "message_text": f"{force_reason}, generating structured output...",
            "iso_timestamp": datetime.now().isoformat(),
        })

    # Add force prompt to messages
    force_messages = messages + [{"role": "user", "content": cfg.force_output_prompt}]

    # Final call with structured output
    result = await chat(
        client=client,
        messages=force_messages,
        tools=None,
        response_format=response_format,
        message_callback=message_callback,
        conversation_stats=result.stats,
        timeout=cfg.timeout,
        emit_summary=False,
    )

    # Extract output
    raw_text = extract_output(result.response)
    output_text = raw_text.strip() if raw_text else None

    # Try to parse as JSON
    output = None
    if output_text:
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError:
            from ..llm_backend.openrouter.or_to_json import extract_json_from_text
            json_text = extract_json_from_text(output_text)
            if json_text:
                try:
                    output = json.loads(json_text)
                except json.JSONDecodeError:
                    raise

    if message_callback:
        _emit_summary(message_callback, result.stats, client)

    return ResearchWorkflowResult(
        output=output,
        output_text=output_text,
        tool_result=result,
        forced_output=True,
        provider="openrouter",
    )


async def _research_workflow_claude_agent(
    prompt: str,
    system: str,
    response_format: type[BaseModel] | None,
    model: str,
    max_turns: int,
    cwd: Path,
    output_dir: Path | None,
    tools: list[str],
    max_retries: int,
    telemetry,
    task_id: str | None,
    message_callback: Callable[[dict], None] | None,
) -> ResearchWorkflowResult:
    """Claude agent path for research workflow."""
    if output_dir is None:
        output_dir = cwd

    task_id = task_id or "research"
    output_file = f"./{task_id}_output.json"
    json_log_path = str(output_dir / f"{task_id}_messages.jsonl")

    # Setup MCP servers for tools
    mcp_servers = get_tooluniverse_mcp_config(tools) if tools else None

    try:
        async with ClaudeAgentToLLMStructOut(
            schema=response_format,
            output_file=output_file,
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            max_retries=max_retries,
            system_prompt=system,
            mcp_servers=mcp_servers,
            telemetry=telemetry,
            task_id=task_id,
            task_name=task_id,
            json_log_path=json_log_path,
        ) as agent:
            result = await agent.run(prompt)

            if result.data:
                output = result.data if isinstance(result.data, dict) else result.data.model_dump()
                return ResearchWorkflowResult(
                    output=output,
                    output_text=json.dumps(output),
                    tool_result=None,
                    forced_output=False,
                    provider="claude_agent",
                )

            return ResearchWorkflowResult(
                output=None,
                output_text=None,
                tool_result=None,
                forced_output=False,
                provider="claude_agent",
            )

    except Exception as e:
        if message_callback:
            message_callback({
                "type": "error",
                "message_text": f"Claude agent failed: {e}",
                "iso_timestamp": datetime.now().isoformat(),
            })
        raise


__all__ = [
    "research_workflow",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "RESEARCH_TOOLS",
]
