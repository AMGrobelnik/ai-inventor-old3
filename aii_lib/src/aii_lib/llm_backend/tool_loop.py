"""
Tool loop helper - chat() for any LLM client.

Works with any client that has:
- chat(messages, tools, ...) -> response
- has_tool_calls(response) -> bool
- extract_tool_calls(response) -> list[dict]
- extract_json_from_response(response) -> str (optional)

Usage:
    from aii_lib import chat, ToolLoopResult, OpenRouterClient

    async with OpenRouterClient(api_key=key) as client:
        # With tools - automatic tool loop
        result = await chat(
            client=client,
            prompt="Generate a hypothesis about...",
            system="You are a researcher...",
            tools=get_openrouter_tools(["aii_web_search_fast"]),
            response_format=Hypothesis,
            message_callback=callback,
        )

        # Without tools - single call (loop runs once, exits immediately)
        result = await chat(
            client=client,
            prompt="Summarize this text...",
            system="You are an assistant...",
        )

        # If iterations exhausted but still has tool calls, continue:
        if result.hit_max_iterations:
            result = await chat(
                client=client,
                messages=result.messages,  # Resume with full context
                tools=tools,
                max_iterations=50,  # More iterations
                conversation_stats=result.stats,  # Continue tracking
            )
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .openrouter.client import ConversationStats
from ..abilities.tools.utils import execute_tool_calls


def _get_tool_abbrev(tool_name: str, suffix: str) -> str:
    """Get abbreviated tool name for display (max 8 chars including suffix)."""
    # Common tool abbreviations
    abbrev_map = {
        "aii_web_search_fast": "SRCH",
        "aii_web_fetch_direct": "FTCH",
        "aii_web_fetch_grep": "GREP",
        "dblp_bib_search": "DBLP",
        "web_search": "SRCH",
        "web_fetch": "FTCH",
        "hf_dataset_search": "HF_S",
        "hf_dataset_info": "HF_I",
        "hf_dataset_preview": "HF_P",
        "hf_dataset_configs": "HF_C",
        "hf_dataset_download": "HF_D",
        "owid_search_datasets": "OWID",
        "owid_download_datasets": "OWID",
        "lean_run_code": "LEAN",
    }

    if tool_name in abbrev_map:
        return f"{abbrev_map[tool_name]}{suffix}"

    # Default: first 4 chars uppercase
    return f"{tool_name[:4].upper()}{suffix}"


def _validate_response_schema(response, schema: type, client) -> tuple[bool, str]:
    """Validate response JSON against Pydantic schema.

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check finish_reason first - if model was cut off, indicate that
        finish_reason = client.get_finish_reason(response) if hasattr(client, 'get_finish_reason') else "unknown"
        if finish_reason == "length":
            return False, "Response truncated (hit max_tokens) - increase max_tokens or simplify prompt"

        # Extract JSON text from response
        if hasattr(client, 'extract_json_from_response'):
            text = client.extract_json_from_response(response)
        else:
            text = client.extract_text_from_response(response)

        if not text or not text.strip():
            # Check if there was reasoning/thinking but no content
            has_reasoning = False
            if hasattr(response, 'choices') and response.choices:
                msg = response.choices[0].message if hasattr(response.choices[0], 'message') else None
                if msg:
                    reasoning = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_details', None)
                    has_reasoning = bool(reasoning)

            if has_reasoning:
                return False, "Model produced reasoning but no JSON output - may need clearer instructions"
            return False, "Empty response - no JSON content"

        # Validate against Pydantic schema
        schema.model_validate_json(text)
        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        # Pydantic ValidationError or other
        error_str = str(e)
        # Truncate long error messages
        if len(error_str) > 500:
            error_str = error_str[:500] + "..."
        return False, error_str


@dataclass
class ToolLoopResult:
    """Result from chat() - supports resuming conversations."""
    response: Any  # Final LLM response
    stats: ConversationStats  # Aggregated stats
    messages: list[dict]  # Full message history (for resuming)
    iterations_used: int = 0  # How many iterations were used
    max_iterations: int = 0  # What the limit was

    @property
    def hit_max_iterations(self) -> bool:
        """True if loop exited due to reaching max iterations limit."""
        return self.iterations_used >= self.max_iterations

    @property
    def last_response_has_tool_calls(self) -> bool:
        """True if the last assistant message has tool calls (model wants more)."""
        # Find the last assistant message in the conversation
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                return bool(tool_calls and len(tool_calls) > 0)
        return False

    @property
    def completed_naturally(self) -> bool:
        """True if model finished without hitting iteration limit."""
        return not self.hit_max_iterations


async def chat(
    client,
    prompt: str | list | None = None,
    system: str | None = None,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    max_iterations: int = 100,
    response_format: type | None = None,
    schema_retries: int = 2,
    message_callback: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    web_search_backend: str = "auto",
    timeout: float = 300,
    conversation_stats: ConversationStats | None = None,
    emit_summary: bool = True,
) -> ToolLoopResult:
    """Chat with automatic tool loop - keeps calling until model stops.

    Works with any LLM client that implements:
    - chat(messages, tools, message_callback, conversation_stats, ...)
    - has_tool_calls(response) -> bool
    - extract_tool_calls(response) -> list[dict]

    Args:
        client: Any LLM client (OpenRouterClient, AnthropicClient, etc.)
        prompt: User prompt - can be string or list of content blocks for multimodal
                (ignored if messages provided). List format:
                [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        system: System prompt (ignored if messages provided)
        messages: Existing message history (for resuming conversations)
        tools: List of tool definitions in OpenAI format
        max_iterations: Max tool loop iterations (safety limit)
        response_format: Pydantic model for structured output (applied on final call)
        schema_retries: Max retries if response_format validation fails (default 2)
        message_callback: Callback for logging messages
        reasoning_effort: Reasoning effort level (low/medium/high)
        web_search_backend: Backend for web_search tool (auto/google/bing/etc.)
        timeout: Timeout per LLM call in seconds
        conversation_stats: Existing stats to continue (for resuming)
        emit_summary: Whether to emit summary at end (default True)

    Returns:
        ToolLoopResult with response, stats, messages, and continuation flag
    """
    # Build or use existing messages
    if messages is not None:
        chat_messages = list(messages)  # Copy to avoid mutation
        # Log the last user message if this is a continuation (conversation_stats provided)
        # This handles cases like force output or summary prompts appended to existing conversation
        if message_callback and conversation_stats is not None and chat_messages:
            last_msg = chat_messages[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content", "")
                # Handle multimodal content
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    prompt_text = "\n".join(text_parts)
                else:
                    prompt_text = content
                message_callback({
                    "type": "prompt",
                    "message_text": prompt_text,
                    "iso_timestamp": datetime.now().isoformat(),
                })
    else:
        if prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        # Handle multimodal content - prompt can be string or list of content blocks
        chat_messages.append({"role": "user", "content": prompt})

        # Emit system prompt and user prompt via callback (only for new conversations)
        if message_callback:
            if system:
                message_callback({
                    "type": "s_prompt",
                    "message_text": system,
                    "iso_timestamp": datetime.now().isoformat(),
                })
            # For multimodal content, extract text parts for logging
            if isinstance(prompt, list):
                text_parts = [p.get("text", "") for p in prompt if p.get("type") == "text"]
                image_count = sum(1 for p in prompt if p.get("type") == "image_url")
                prompt_text = "\n".join(text_parts)
                if image_count:
                    prompt_text += f"\n[{image_count} image(s) attached]"
            else:
                prompt_text = prompt
            message_callback({
                "type": "prompt",
                "message_text": prompt_text,
                "iso_timestamp": datetime.now().isoformat(),
            })

    # Track stats across turns (use existing or create new)
    conv_stats = conversation_stats or ConversationStats()

    # Tool loop
    iteration = 0
    response = None
    is_first_turn = (conversation_stats is None)  # Only emit system on truly first turn

    while iteration < max_iterations:
        iteration += 1

        # Only emit system message on first turn of new conversation
        emit_system_msg = is_first_turn and (iteration == 1)

        # If no tools, pass response_format directly; otherwise defer to final call
        use_response_format = response_format if not tools else None

        response = await asyncio.wait_for(
            client.call(
                messages=chat_messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                response_format=use_response_format,
                message_callback=message_callback,
                emit_summary=False,  # Never emit until we're done
                emit_system=emit_system_msg,
                conversation_stats=conv_stats,
            ),
            timeout=timeout,
        )

        # Check if model wants to call tools
        if client.has_tool_calls(response):
            tool_calls = client.extract_tool_calls(response)
            tool_results = await execute_tool_calls(
                tool_calls,
                web_search_backend=web_search_backend,
            )

            # Track cache hits for aii_web_search_fast
            for result in tool_results:
                if result.get("original_name") == "aii_web_search_fast":
                    if result.get("cache_hit"):
                        conv_stats.tool_calls["aii_web_search_fast"] = max(
                            0, conv_stats.tool_calls.get("aii_web_search_fast", 0) - 1
                        )
                        conv_stats.tool_calls["aii_web_search_fast_cache_hit"] = (
                            conv_stats.tool_calls.get("aii_web_search_fast_cache_hit", 0) + 1
                        )

            # Add assistant message with tool calls to history
            if not response.choices:
                break  # No choices in response, exit loop
            assistant_msg = response.choices[0].message
            if not assistant_msg or not assistant_msg.tool_calls:
                break  # No tool calls to process, exit loop
            tool_call_dicts = []
            for tc in assistant_msg.tool_calls:
                try:
                    tool_call_dicts.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                except AttributeError as e:
                    raise AttributeError(f"Malformed tool call structure in response: {e}") from e
            chat_messages.append({
                "role": "assistant",
                "content": getattr(assistant_msg, 'content', None),
                "tool_calls": tool_call_dicts,
            })

            # Add tool results to history
            for result in tool_results:
                result_content = json.dumps(
                    result.get("result", result.get("error", "No result")),
                    default=str,
                )
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result_content,
                })

                # Log tool result via callback (sinks handle truncation)
                if message_callback:
                    result_display = json.dumps(
                        result.get("result", result.get("error", "No result")),
                        indent=2,
                        default=str,
                    )
                    tool_name_abbrev = _get_tool_abbrev(result["name"], "_OUT")
                    message_callback({
                        "type": "tool_output",
                        "message_text": f"Tool: {result['name']}\nResult:\n{result_display}",
                        "tool_name": tool_name_abbrev,
                        "tool_id": result["tool_call_id"],
                        "iso_timestamp": datetime.now().isoformat(),
                        "llm_provider": "openrouter",
                        "message_metadata": {
                            "tool_call_id": result["tool_call_id"],
                            "tool_name_full": result["name"],
                            "has_error": result.get("error") is not None,
                        },
                    })
        else:
            # No tool calls - model is done with tool phase
            break

    model_finished = response and not client.has_tool_calls(response)

    # If model finished and we want structured output, make a separate final call
    # This is the recommended approach: tools first, then response_format separately
    if response_format and model_finished and tools:
        response = await asyncio.wait_for(
            client.call(
                messages=chat_messages,
                tools=None,  # No tools on final structured output call
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                message_callback=message_callback,
                emit_summary=emit_summary,
                emit_system=False,
                conversation_stats=conv_stats,
            ),
            timeout=timeout,
        )
        # Add final response to messages
        final_content = getattr(response.choices[0].message, 'content', '') if response.choices else ''
        if final_content:
            chat_messages.append({"role": "assistant", "content": final_content})

    # Schema validation and retry (if response_format is set)
    if response_format and model_finished and hasattr(response_format, 'model_validate_json'):
        schema_name = getattr(response_format, '__name__', 'Schema')
        is_valid, validation_error = _validate_response_schema(response, response_format, client)

        retry_count = 0
        while not is_valid and retry_count < schema_retries:
            retry_count += 1
            if message_callback:
                message_callback({
                    "type": "retry",
                    "message_text": f"Schema validation failed ({retry_count}/{schema_retries}): {validation_error[:200]}",
                    "iso_timestamp": datetime.now().isoformat(),
                    "message_metadata": {
                        "retry_count": retry_count,
                        "max_retries": schema_retries,
                        "schema_name": schema_name,
                        "validation_error": validation_error[:500],
                    },
                })

            # Append feedback and retry
            feedback = f"Your JSON response has validation errors:\n\n{validation_error}\n\nFix the JSON to match the required schema exactly. Output only valid JSON."
            chat_messages.append({"role": "user", "content": feedback})

            if message_callback:
                message_callback({
                    "type": "prompt",
                    "message_text": feedback,
                    "iso_timestamp": datetime.now().isoformat(),
                })

            response = await asyncio.wait_for(
                client.call(
                    messages=chat_messages,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    message_callback=message_callback,
                    emit_summary=False,
                    emit_system=False,
                    conversation_stats=conv_stats,
                ),
                timeout=timeout,
            )

            # Add response to messages
            final_content = getattr(response.choices[0].message, 'content', '') if response.choices else ''
            if final_content:
                chat_messages.append({"role": "assistant", "content": final_content})

            is_valid, validation_error = _validate_response_schema(response, response_format, client)

        if not is_valid:
            if message_callback:
                message_callback({
                    "type": "schema_error",
                    "message_text": f"❌ JSON schema validation failed after {retry_count} retries\nSchema: {schema_name}\nErrors: {validation_error[:300]}",
                    "iso_timestamp": datetime.now().isoformat(),
                    "message_metadata": {
                        "retry_count": retry_count,
                        "max_retries": schema_retries,
                        "schema_name": schema_name,
                        "validation_error": validation_error[:500],
                    },
                })

    result = ToolLoopResult(
        response=response,
        stats=conv_stats,
        messages=chat_messages,
        iterations_used=iteration,
        max_iterations=max_iterations,
    )

    # Emit final summary if model finished naturally and emit_summary=True
    if message_callback and model_finished and emit_summary:
        _emit_summary(message_callback, conv_stats, client)

    return result


def _emit_summary(callback: Callable, stats: ConversationStats, client) -> None:
    """Emit final summary with aggregated stats."""
    # Calculate tool costs
    tool_costs = {}
    tool_cost_total = 0.0
    for tool_name, count in stats.tool_calls.items():
        if tool_name == "aii_web_search_fast":
            unit_cost = 0.001
        else:
            unit_cost = 0.0
        tool_total = count * unit_cost
        tool_cost_total += tool_total
        if unit_cost > 0:
            tool_costs[tool_name] = {"count": count, "unit": unit_cost, "total": tool_total}

    runtime_seconds = stats.get_runtime_minutes() * 60

    callback({
        "type": "summary",
        "total_cost": stats.total_cost + tool_cost_total,
        "token_cost": stats.total_cost,
        "tool_cost": tool_cost_total,
        "model": stats.model or getattr(client, 'model', 'unknown'),
        "status": stats.finish_reason,
        "is_aggregated": False,
        "is_error": stats.finish_reason in ("error", "failed"),
        "num_calls": stats.num_turns,
        "runtime_seconds": runtime_seconds,
        "llm_time_seconds": runtime_seconds,
        "input_tokens": stats.prompt_tokens,
        "output_tokens": stats.completion_tokens,
        "reasoning_tokens": stats.reasoning_tokens,
        "cache_write_tokens": 0,
        "cache_read_tokens": stats.cached_tokens or 0,
        "tool_calls": stats.tool_calls,
        "tool_costs": tool_costs,
        "iso_timestamp": datetime.now().isoformat(),
    })


__all__ = ["chat", "ToolLoopResult"]
