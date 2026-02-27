"""
GenAIKit - Main API for GenAI operations.

Layer 1 (Base Operations):
- chat(): Simple LLM calls
- chat_tools(): LLM calls with tool loop
- agent(): Claude Agent SDK calls

For Layer 2 (Workflows), use directly from aii_lib.workflows:
- EloRanker: K-random opponents pairwise ranking with multiple LLMs
- CitedArgsConfig: Cited argument generation
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from aii_lib.telemetry import logger

from .types import GenAIRun, TokenUsage
from .telemetry import AIITelemetry
from .llm_backend import OpenRouterClient, ConversationStats
from .abilities.tools.utils import get_openrouter_tools, execute_tool_calls
from .agent_backend import Agent, AgentOptions, AgentResponse


T = TypeVar("T")


@dataclass
class GenAIKit:
    """
    Kit for GenAI operations.

    Provides base operations (chat, chat_tools, agent) and presets (elo_rank).
    All operations use telemetry for logging and return GenAIRun results.
    """

    # AIITelemetry for logging
    telemetry: AIITelemetry

    # Config dict (api_keys, model defaults, etc.)
    config: dict = field(default_factory=dict)

    # =========================================
    # Base Operations (Layer 1)
    # =========================================

    async def chat(
        self,
        prompts: str | list[str],
        *,
        model: str = "openai/gpt-4.1-mini",
        system: str | None = None,
        reasoning_effort: str = "medium",
        group: str | None = None,
        name_prefix: str = "chat",
        timeout: int = 300,
    ) -> list[GenAIRun]:
        """
        Simple LLM calls. Parallel if multiple prompts.

        Args:
            prompts: Single prompt or list of prompts
            model: Model identifier (e.g., "openai/gpt-4.1-mini")
            system: System prompt
            reasoning_effort: low/medium/high
            group: Group name for sequencer
            name_prefix: Prefix for task names
            timeout: Timeout in seconds

        Returns:
            List of GenAIRun results
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        api_key = self.config.get("api_keys", {}).get("openrouter")
        if not api_key:
            raise ValueError("Missing OpenRouter API key in config")

        async def run_single(prompt: str, idx: int) -> GenAIRun:
            task_id = f"{name_prefix}_{idx:03d}_{uuid.uuid4().hex[:6]}"
            task_name = f"{name_prefix}-{idx+1}"

            self.telemetry.emit_task_start(task_id, task_name)
            callback = self.telemetry.create_callback(task_id, task_name, group=group)

            start_time = datetime.now()
            run = GenAIRun(
                id=task_id,
                name=task_name,
                group=group,
                prompt=prompt,
                system=system,
                backend_type="llm",
                model=model,
            )

            try:
                async with OpenRouterClient(
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                ) as client:
                    conv_stats = ConversationStats()

                    response = await asyncio.wait_for(
                        client.call(
                            prompt=prompt,
                            system=system,
                            reasoning_effort=reasoning_effort,
                            message_callback=callback,
                            conversation_stats=conv_stats,
                        ),
                        timeout=timeout,
                    )

                    result_text = client.extract_text_from_response(response)

                    run.result = result_text
                    run.status = "completed"
                    run.cost = conv_stats.total_cost
                    run.tokens = TokenUsage(
                        input=conv_stats.input_tokens,
                        output=conv_stats.output_tokens,
                        reasoning=conv_stats.reasoning_tokens,
                    )
                    run.model = conv_stats.model or model
                    run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Emit for aggregation
                    self.telemetry.emit_run_end(task_id, task_name, {
                        "total_runs": 1,
                        "completed": 1,
                        "failed": 0,
                        "total_cost": run.cost,
                        "tokens": {"input": run.tokens.input, "output": run.tokens.output, "reasoning": run.tokens.reasoning},
                        "duration_ms": run.duration_ms,
                    })
                    self.telemetry.emit_task_end(task_id, task_name, f"OK ({len(result_text or '')} chars)")

            except asyncio.TimeoutError:
                run.status = "failed"
                run.error = f"Timeout after {timeout}s"
                run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.telemetry.emit_run_end(task_id, task_name, {
                    "total_runs": 1, "completed": 0, "failed": 1,
                    "total_cost": 0, "tokens": {}, "duration_ms": run.duration_ms,
                })
                self.telemetry.emit_task_end(task_id, task_name, f"Timeout ({timeout}s)")

            except Exception as e:
                run.status = "failed"
                run.error = str(e)
                run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.error("Chat failed", exc=e)
                self.telemetry.emit_run_end(task_id, task_name, {
                    "total_runs": 1, "completed": 0, "failed": 1,
                    "total_cost": 0, "tokens": {}, "duration_ms": run.duration_ms,
                })
                self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")

            return run

        tasks = [run_single(p, i) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

    async def chat_tools(
        self,
        prompts: str | list[str],
        tools: list[str],
        *,
        model: str = "openai/gpt-4.1-mini",
        system: str | None = None,
        reasoning_effort: str = "medium",
        max_iterations: int = 50,
        group: str | None = None,
        name_prefix: str = "chat_tools",
        timeout: int = 300,
    ) -> list[GenAIRun]:
        """
        LLM calls with tool loop. Loops until LLM stops calling tools.

        Args:
            prompts: Single prompt or list of prompts
            tools: List of tool names (e.g., ["aii_web_search_fast", "aii_web_fetch_direct"])
            model: Model identifier
            system: System prompt
            reasoning_effort: low/medium/high
            max_iterations: Max tool loop iterations
            group: Group name for sequencer
            name_prefix: Prefix for task names
            timeout: Timeout in seconds

        Returns:
            List of GenAIRun results
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        api_key = self.config.get("api_keys", {}).get("openrouter")
        if not api_key:
            raise ValueError("Missing OpenRouter API key in config")

        tool_defs = get_openrouter_tools(tools)

        async def run_single(prompt: str, idx: int) -> GenAIRun:
            task_id = f"{name_prefix}_{idx:03d}_{uuid.uuid4().hex[:6]}"
            task_name = f"{name_prefix}-{idx+1}"

            self.telemetry.emit_task_start(task_id, task_name)
            callback = self.telemetry.create_callback(task_id, task_name, group=group)

            start_time = datetime.now()
            run = GenAIRun(
                id=task_id,
                name=task_name,
                group=group,
                prompt=prompt,
                system=system,
                backend_type="llm",
                model=model,
            )

            try:
                async with OpenRouterClient(
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                ) as client:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})

                    conv_stats = ConversationStats()

                    # Emit prompt
                    callback({
                        "type": "prompt",
                        "message_text": prompt,
                        "iso_timestamp": datetime.now().isoformat(),
                    })

                    # Tool loop
                    iteration = 0
                    is_first_turn = True
                    last_response = None

                    while iteration < max_iterations:
                        iteration += 1

                        response = await asyncio.wait_for(
                            client.call(
                                messages=messages,
                                reasoning_effort=reasoning_effort,
                                tools=tool_defs,
                                message_callback=callback,
                                emit_system=is_first_turn,
                                emit_summary=False,
                                conversation_stats=conv_stats,
                            ),
                            timeout=timeout,
                        )
                        is_first_turn = False
                        last_response = response

                        if client.has_tool_calls(response):
                            tool_calls = client.extract_tool_calls(response)
                            tool_results = await execute_tool_calls(tool_calls)

                            # Update tool call counts
                            for tr in tool_results:
                                tool_name = tr.get("name", "unknown")
                                run.tool_calls[tool_name] = run.tool_calls.get(tool_name, 0) + 1

                            # Add to message history
                            assistant_msg = response.choices[0].message
                            messages.append({
                                "role": "assistant",
                                "content": getattr(assistant_msg, "content", None),
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                    }
                                    for tc in assistant_msg.tool_calls
                                ],
                            })

                            for tr in tool_results:
                                result_content = json.dumps(
                                    tr.get("result", tr.get("error", "No result")), default=str
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tr["tool_call_id"],
                                    "content": result_content,
                                })

                                # Emit tool result
                                callback({
                                    "type": "or_tool_out",
                                    "message_text": f"Tool: {tr['name']}",
                                    "iso_timestamp": datetime.now().isoformat(),
                                    "llm_provider": "openrouter",
                                    "message_metadata": {
                                        "tool_call_id": tr["tool_call_id"],
                                        "tool_name": tr["name"],
                                    },
                                })
                        else:
                            break

                    # Force final output if iterations exhausted
                    if last_response and client.has_tool_calls(last_response):
                        logger.warning(f"Tool iteration limit ({max_iterations}) reached")
                        messages.append({
                            "role": "user",
                            "content": "STOP. Write your final answer now using only the sources you have found.",
                        })
                        last_response = await asyncio.wait_for(
                            client.call(
                                messages=messages,
                                reasoning_effort=reasoning_effort,
                                tools=None,
                                message_callback=callback,
                                emit_system=False,
                                emit_summary=False,
                                conversation_stats=conv_stats,
                            ),
                            timeout=timeout,
                        )

                    result_text = client.extract_text_from_response(last_response)

                    run.result = result_text
                    run.status = "completed"
                    run.cost = conv_stats.total_cost
                    run.tokens = TokenUsage(
                        input=conv_stats.prompt_tokens,
                        output=conv_stats.completion_tokens,
                        reasoning=conv_stats.reasoning_tokens,
                    )
                    run.model = conv_stats.model or model
                    run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Calculate tool costs for standardized summary
                    tool_costs = {}
                    tool_cost_total = 0.0
                    for tool_name, count in run.tool_calls.items():
                        # Tool pricing: aii_web_search_fast = $0.001/call
                        if tool_name == "aii_web_search_fast":
                            unit_cost = 0.001
                        else:
                            unit_cost = 0.0  # Free tools
                        tool_total = count * unit_cost
                        tool_cost_total += tool_total
                        if unit_cost > 0:
                            tool_costs[tool_name] = {"count": count, "unit": unit_cost, "total": tool_total}

                    # Emit final standardized SummaryMetrics (since emit_summary=False in loop)
                    callback({
                        "type": "summary",
                        "total_cost": run.cost + tool_cost_total,
                        "token_cost": run.cost,
                        "tool_cost": tool_cost_total,
                        "model": run.model,
                        "status": conv_stats.finish_reason or "completed",
                        "is_aggregated": False,
                        "num_calls": conv_stats.num_turns,
                        "runtime_seconds": run.duration_ms / 1000,
                        "llm_time_seconds": conv_stats.get_runtime_minutes() * 60,
                        "input_tokens": run.tokens.input,
                        "output_tokens": run.tokens.output,
                        "reasoning_tokens": run.tokens.reasoning,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": conv_stats.cached_tokens or 0,
                        "tool_calls": run.tool_calls,
                        "tool_costs": tool_costs,
                        "iso_timestamp": datetime.now().isoformat(),
                    })

                    # Emit for aggregation
                    self.telemetry.emit_run_end(task_id, task_name, {
                        "total_runs": 1,
                        "completed": 1,
                        "failed": 0,
                        "total_cost": run.cost + tool_cost_total,
                        "tokens": {"input": run.tokens.input, "output": run.tokens.output, "reasoning": run.tokens.reasoning},
                        "tool_calls": run.tool_calls,
                        "duration_ms": run.duration_ms,
                    })
                    self.telemetry.emit_task_end(
                        task_id, task_name, f"OK ({len(result_text or '')} chars, {iteration} iters)"
                    )

            except asyncio.TimeoutError:
                run.status = "failed"
                run.error = f"Timeout after {timeout}s"
                self.telemetry.emit_task_end(task_id, task_name, f"Timeout ({timeout}s)")

            except Exception as e:
                run.status = "failed"
                run.error = str(e)
                logger.error("Chat tools failed", exc=e)
                self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")

            run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return run

        tasks = [run_single(p, i) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

    async def agent(
        self,
        prompts: str | list[str],
        *,
        workspace: Path | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        max_turns: int = 100,
        group: str | None = None,
        name_prefix: str = "agent",
        timeout: int = 600,
    ) -> list[GenAIRun]:
        """
        Claude Agent SDK calls. Agent manages its own tool loop.

        Args:
            prompts: Single prompt or list of prompts
            workspace: Working directory for agent
            allowed_tools: Tools to allow
            disallowed_tools: Tools to disallow
            max_turns: Max conversation turns
            group: Group name for sequencer
            name_prefix: Prefix for task names
            timeout: Timeout in seconds

        Returns:
            List of GenAIRun results
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        async def run_single(prompt: str, idx: int) -> GenAIRun:
            task_id = f"{name_prefix}_{idx:03d}_{uuid.uuid4().hex[:6]}"
            task_name = f"{name_prefix}-{idx+1}"

            self.telemetry.emit_task_start(task_id, task_name)

            start_time = datetime.now()
            run = GenAIRun(
                id=task_id,
                name=task_name,
                group=group,
                prompt=prompt,
                backend_type="agent",
            )

            try:
                options = AgentOptions(
                    cwd=str(workspace) if workspace else None,
                    allowed_tools=allowed_tools,
                    disallowed_tools=disallowed_tools,
                    max_turns=max_turns,
                    permission_mode="bypassPermissions",
                )

                agent = Agent(options=options)
                result: AgentResponse = await asyncio.wait_for(
                    asyncio.to_thread(agent.run, prompt),
                    timeout=timeout,
                )

                run.result = result.result
                run.status = "completed"
                run.cost = result.cost_usd
                run.model = "claude-agent"

                self.telemetry.emit_task_end(task_id, task_name, f"OK (${result.cost_usd:.4f})")

            except asyncio.TimeoutError:
                run.status = "failed"
                run.error = f"Timeout after {timeout}s"
                self.telemetry.emit_task_end(task_id, task_name, f"Timeout ({timeout}s)")

            except Exception as e:
                run.status = "failed"
                run.error = str(e)
                logger.error("Agent failed", exc=e)
                self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")

            run.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return run

        tasks = [run_single(p, i) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

