"""OpenRouter Client - Async.

Access 300+ models from multiple providers via OpenRouter.

Popular Models (December 2025):
    OpenAI:
        - openai/gpt-5.2         : GPT-5.2 flagship
        - openai/gpt-5-mini      : Fast GPT-5
        - openai/o4-mini         : Reasoning model
        - openai/o3              : Complex reasoning

    Anthropic:
        - anthropic/claude-opus-4.5   : Premium Claude
        - anthropic/claude-sonnet-4.5 : Balanced Claude
        - anthropic/claude-haiku-4.5  : Fast Claude

    Google:
        - google/gemini-3-flash-preview : Gemini 3 Flash (high-speed reasoning)
        - google/gemini-2.5-pro   : Gemini flagship
        - google/gemini-2.5-flash : Fast Gemini

    Meta Llama:
        - meta-llama/llama-3.3-70b-instruct  : Llama 3.3 70B
        - meta-llama/llama-3.1-405b-instruct : Llama 3.1 405B

    Mistral:
        - mistralai/mistral-large-latest : Mistral Large
        - mistralai/mixtral-8x22b-instruct

    DeepSeek:
        - deepseek/deepseek-r1   : R1 reasoning
        - deepseek/deepseek-chat : Chat model

    Recommended:
        - Flagship: openai/gpt-5.2
        - Balanced: anthropic/claude-sonnet-4.5
        - Fast: google/gemini-2.5-flash
        - Budget: deepseek/deepseek-chat
        - Open Source: meta-llama/llama-3.3-70b-instruct

Docs: https://openrouter.ai/models
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
from aii_lib.telemetry import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


# Retry configuration
MAX_RETRIES = 8  # Increased from 4 for better reliability


def _log_retry_error(retry_state) -> None:
    """Log retry error with full traceback via telemetry."""
    exc = retry_state.outcome.exception()

    # Extract detailed error message if available
    error_details = str(exc)
    if hasattr(exc, 'data') and hasattr(exc.data, 'error'):
        error_details = f"{exc.data.error.message} (code: {getattr(exc.data.error, 'code', 'N/A')})"

    logger.warning(
        f"OpenRouter request failed, retrying... (attempt {retry_state.attempt_number}/{MAX_RETRIES}) - {error_details}",
        exc=exc,
    )


from .or_to_json import serialize_response, extract_output, extract_usage, extract_json_from_text
from .json_to_log_format import create_summary


@dataclass
class ConversationStats:
    """Aggregated stats for multi-turn conversations."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_cost: float = 0.0
    num_turns: int = 0
    tool_calls: dict = field(default_factory=dict)  # {tool_name: count}
    start_time: datetime = field(default_factory=datetime.now)
    last_response: object = None  # Store last response for final summary
    model: str = ""
    finish_reason: str = "unknown"

    def add_turn(self, usage: dict, cost: float, tool_calls: list[dict] = None):
        """Add stats from a single turn."""
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.reasoning_tokens += usage.get("reasoning_tokens", 0)
        self.cached_tokens += usage.get("cached_tokens", 0)
        self.total_cost += cost
        self.num_turns += 1
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def get_runtime_minutes(self) -> float:
        """Get total runtime in minutes."""
        return (datetime.now() - self.start_time).total_seconds() / 60.0



class OpenRouterClient:
    """Async OpenRouter client for accessing 300+ models.

    Popular Models (Dec 2025):
        OpenAI: openai/gpt-5.2, openai/gpt-5-mini, openai/o4-mini
        Anthropic: anthropic/claude-sonnet-4.5, anthropic/claude-haiku-4.5
        Google: google/gemini-2.5-pro, google/gemini-2.5-flash
        Meta: meta-llama/llama-3.3-70b-instruct
        DeepSeek: deepseek/deepseek-r1, deepseek/deepseek-chat

    Docs: https://openrouter.ai/models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        config_path: str | None = None,
    ):
        # Load config defaults first
        from ..config import load_config, get_openrouter_config

        if config_path:
            load_config(config_path)

        config = get_openrouter_config()

        # Config defaults, overridden by explicit parameters
        self.model = model or config.get('default_model', 'anthropic/claude-sonnet-4')

        # Store API key and timeout for HTTP requests
        # Default timeout is 10 min (600s), but can be overridden
        self._api_key = api_key or config.get('api_key')
        if not self._api_key:
            logger.warning("No OpenRouter API key provided; requests will likely fail")
        self._timeout_ms = int((timeout if timeout is not None else 600.0) * 1000)  # Convert to ms

    @staticmethod
    def _dict_to_obj(data):
        """Convert nested dict to object with attribute access."""
        from types import SimpleNamespace

        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = OpenRouterClient._dict_to_obj(value)
            return SimpleNamespace(**data)
        elif isinstance(data, list):
            return [OpenRouterClient._dict_to_obj(item) for item in data]
        return data

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=2, max=120, jitter=5),  # 2s, 4s, 8s, 16s, 32s, 64s, 120s, 120s
        before_sleep=_log_retry_error,
    )
    async def _send_with_retry(self, payload: dict):
        """Send chat request via raw HTTP with usage accounting enabled.

        Uses aiohttp instead of OpenRouter SDK to support usage: {include: true}.
        See: https://openrouter.ai/docs/guides/usage-accounting
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Always include usage accounting
        payload["usage"] = {"include": True}

        timeout = aiohttp.ClientTimeout(total=self._timeout_ms / 1000)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenRouter API error {resp.status}: {error_text[:500]}")

                data = await resp.json()

                # Check for error in response body (OpenRouter sometimes returns 200 with error)
                if "error" in data:
                    error_info = data.get("error", {})
                    error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                    error_code = error_info.get("code", "unknown") if isinstance(error_info, dict) else "unknown"
                    raise Exception(f"OpenRouter API error (in body): {error_code}: {error_msg}")

                # Convert to object with attribute access (like SDK response)
                return self._dict_to_obj(data)

    async def call(
        self,
        prompt: str | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type | dict | None = None,
        provider: dict | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        message_callback=None,
        emit_summary: bool = True,
        emit_system: bool = True,
        conversation_stats: ConversationStats | None = None,
    ):
        """Send a chat message (async, no streaming).

        Args:
            prompt: The user prompt (ignored if messages is provided)
            system: System prompt (ignored if messages is provided)
            model: Override default model (e.g., "anthropic/claude-4.5-sonnet")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Pydantic model class or JSON schema dict for structured output
            provider: Provider preferences (e.g., {"sort": "price"})
            reasoning_effort: Reasoning effort level (low/medium/high) - passed through to supporting models
            tools: List of tool definitions in OpenAI format (tool_choice defaults to "auto")
            messages: Full message history (for multi-turn conversations like tool calling)
            message_callback: Callback for logging messages
            emit_summary: Whether to emit summary message (False for intermediate turns in multi-turn)
            emit_system: Whether to emit system/config message (False for subsequent turns in multi-turn)
            conversation_stats: Stats object to aggregate across turns (for multi-turn conversations)

        Returns:
            The API response object
        """
        resolved_model = model or self.model

        # Build messages - use provided messages or construct from prompt/system
        if messages is not None:
            chat_messages = messages
        else:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            chat_messages = []
            if system:
                chat_messages.append({"role": "system", "content": system})
            chat_messages.append({"role": "user", "content": prompt})

            # Log system prompt and user prompt (only when not using raw messages)
            if message_callback:
                if system:
                    message_callback({
                        "type": "s_prompt",
                        "message_text": system,
                        "iso_timestamp": datetime.now().isoformat(),
                    })
                message_callback({
                    "type": "prompt",
                    "message_text": prompt,
                    "iso_timestamp": datetime.now().isoformat(),
                })

        # Log system message (only on first turn for multi-turn conversations)
        if message_callback and emit_system:
            system_text = f"{resolved_model}"
            if reasoning_effort:
                system_text += f" | Reasoning: {reasoning_effort}"
            if tools:
                tool_names = [t.get("function", {}).get("name") for t in tools]
                system_text += f" | Tools: {', '.join(tool_names)}"
            if response_format:
                system_text += " | Structured output: enabled"

            message_callback({
                "type": "system",
                "message_text": system_text,
                "model": resolved_model,
                "iso_timestamp": datetime.now().isoformat(),
                "llm_provider": "openrouter",
                "message_metadata": {
                    "model": resolved_model,
                    "reasoning_effort": reasoning_effort,
                    "tools": tool_names if tools else None,
                    "structured_output": response_format is not None,
                },
                "raw_api_response": None,
            })

        # Build payload for raw HTTP request
        kwargs = {
            "model": resolved_model,
            "messages": chat_messages,
            "stream": False,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if provider:
            kwargs["provider"] = provider

        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # Add tools (tool_choice defaults to "auto")
        if tools:
            # Filter out internal fields like _original_name that might cause API errors
            clean_tools = []
            for tool in tools:
                clean_tool = {k: v for k, v in tool.items() if not k.startswith('_')}
                clean_tools.append(clean_tool)
            kwargs["tools"] = clean_tools

        # Add structured output format
        if response_format:
            if hasattr(response_format, 'model_json_schema'):
                # Pydantic model - convert to JSON schema
                schema = response_format.model_json_schema()
                # OpenAI strict mode requires: additionalProperties=false, all fields required
                schema = self._add_additional_properties_false(schema)
                schema = self._make_all_fields_required(schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                }
            elif isinstance(response_format, dict):
                # Already a dict schema
                kwargs["response_format"] = response_format

        # Track start time for runtime calculation
        start_time = datetime.now()

        # Make async request with connection retry (pass payload dict directly)
        response = await self._send_with_retry(kwargs)

        # Calculate runtime
        runtime_minutes = (datetime.now() - start_time).total_seconds() / 60.0

        # Process response
        if message_callback and hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and choice.message:
                    msg = choice.message

                    # Handle reasoning field FIRST (GPT-5 style - direct string)
                    reasoning = getattr(msg, 'reasoning', None)
                    if reasoning and isinstance(reasoning, str) and reasoning.strip():
                        message_callback({
                            "type": "or_reasoning",
                            "message_text": reasoning.strip(),
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "openrouter",
                            "message_metadata": {"reasoning_field": True},
                            "raw_api_response": None,
                        })

                    # Handle reasoning_details (if model exposes it as array)
                    reasoning_details = getattr(msg, 'reasoning_details', None)
                    if reasoning_details:
                        # Extract reasoning text from details array
                        reasoning_text = ""
                        for detail in reasoning_details:
                            if hasattr(detail, 'content'):
                                reasoning_text += detail.content + "\n"
                            elif isinstance(detail, dict) and 'content' in detail:
                                reasoning_text += detail['content'] + "\n"
                        if reasoning_text.strip():
                            message_callback({
                                "type": "or_reasoning",
                                "message_text": reasoning_text.strip(),
                                "iso_timestamp": datetime.now().isoformat(),
                                "llm_provider": "openrouter",
                                "message_metadata": {"reasoning_details": True},
                                "raw_api_response": None,
                            })

                    # Handle refusal (model refused to respond)
                    refusal = getattr(msg, 'refusal', None)
                    if refusal:
                        message_callback({
                            "type": "or_refusal",
                            "message_text": refusal,
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "openrouter",
                            "message_metadata": {},
                            "raw_api_response": None,
                        })

                    # Handle content (emit if non-empty)
                    content = getattr(msg, 'content', None)
                    tool_calls = getattr(msg, 'tool_calls', None)
                    has_other_output = bool(reasoning) or bool(tool_calls)

                    if content and content.strip():
                        # Pretty-print JSON content (from structured output)
                        display_content = content
                        if content.strip().startswith('{'):
                            try:
                                parsed = json.loads(content)
                                display_content = json.dumps(parsed, indent=2)
                            except json.JSONDecodeError:
                                pass  # Not valid JSON, keep as-is
                        message_callback({
                            "type": "or_msg",
                            "message_text": display_content,
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "openrouter",
                            "message_metadata": {},
                            "raw_api_response": choice.model_dump() if hasattr(choice, 'model_dump') else None,
                        })
                    elif not has_other_output:
                        # Only show "(empty response)" if there's truly nothing
                        message_callback({
                            "type": "or_msg",
                            "message_text": "(empty response)",
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "openrouter",
                            "message_metadata": {},
                            "raw_api_response": choice.model_dump() if hasattr(choice, 'model_dump') else None,
                        })

                    # Handle tool calls
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.function.name if hasattr(tc, 'function') else 'unknown'
                            tool_args = tc.function.arguments if hasattr(tc, 'function') else '{}'
                            # Parse args if string
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    pass
                            message_callback({
                                "type": "or_tool_in",
                                "message_text": f"Tool: {tool_name}\nArgs: {json.dumps(tool_args, indent=2) if isinstance(tool_args, dict) else tool_args}",
                                "iso_timestamp": datetime.now().isoformat(),
                                "llm_provider": "openrouter",
                                "message_metadata": {
                                    "tool_call_id": getattr(tc, 'id', None),
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                },
                                "raw_api_response": tc.model_dump() if hasattr(tc, 'model_dump') else None,
                            })

        # Extract usage and actual cost from OpenRouter response
        usage = extract_usage(response)
        total_cost = usage.get("cost", 0.0)

        # Get finish reason
        finish_reason = "unknown"
        if hasattr(response, 'choices') and response.choices:
            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown') or 'unknown'

        # Get actual model from response (may differ from requested)
        actual_model = getattr(response, 'model', resolved_model) or resolved_model

        # Extract tool calls for stats tracking
        turn_tool_calls = self.extract_tool_calls(response) if self.has_tool_calls(response) else []

        # Update conversation stats if provided (for multi-turn aggregation)
        if conversation_stats is not None:
            conversation_stats.add_turn(usage, total_cost, turn_tool_calls)
            conversation_stats.last_response = response
            conversation_stats.model = actual_model
            conversation_stats.finish_reason = finish_reason

        # Log summary (only if emit_summary=True)
        if message_callback and emit_summary:
            # Use conversation stats if available (multi-turn), otherwise single turn stats
            if conversation_stats is not None:
                input_tokens = conversation_stats.prompt_tokens
                output_tokens = conversation_stats.completion_tokens
                reasoning_tokens = conversation_stats.reasoning_tokens or 0
                cache_read_tokens = conversation_stats.cached_tokens or 0
                token_cost = conversation_stats.total_cost
                runtime_sec = conversation_stats.get_runtime_minutes() * 60
                num_turns = conversation_stats.num_turns
                tool_calls_dict = conversation_stats.tool_calls
            else:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                reasoning_tokens = usage.get("reasoning_tokens", 0)
                cache_read_tokens = usage.get("cached_tokens", 0)
                token_cost = total_cost
                runtime_sec = runtime_minutes * 60
                num_turns = 1
                # Build tool calls dict for single turn
                tool_calls_dict = {}
                for tc in turn_tool_calls:
                    name = tc.get("name", "unknown")
                    tool_calls_dict[name] = tool_calls_dict.get(name, 0) + 1

            # Calculate tool costs
            tool_costs = {}
            tool_cost_total = 0.0
            for tool_name, count in tool_calls_dict.items():
                # Tool pricing: aii_web_search_fast = $0.001/call
                if tool_name == "aii_web_search_fast":
                    unit_cost = 0.001
                else:
                    unit_cost = 0.0  # Free tools
                tool_total = count * unit_cost
                tool_cost_total += tool_total
                if unit_cost > 0:
                    tool_costs[tool_name] = {"count": count, "unit": unit_cost, "total": tool_total}

            # Emit standardized SummaryMetrics format
            message_callback({
                "type": "summary",
                "total_cost": token_cost + tool_cost_total,
                "token_cost": token_cost,
                "tool_cost": tool_cost_total,
                "model": actual_model,
                "status": finish_reason,
                "is_aggregated": False,
                "is_error": finish_reason in ("error", "failed"),
                "num_calls": num_turns,
                "runtime_seconds": runtime_sec,
                "llm_time_seconds": runtime_sec,  # Same for single client
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cache_write_tokens": 0,  # OpenRouter doesn't support
                "cache_read_tokens": cache_read_tokens,
                "tool_calls": tool_calls_dict,
                "tool_costs": tool_costs,
                "iso_timestamp": datetime.now().isoformat(),
            })

        return response

    def extract_text_from_response(self, response) -> str:
        """Extract output text from response."""
        return extract_output(response)

    def extract_tool_calls(self, response) -> list[dict]:
        """Extract tool calls from response.

        Returns:
            List of tool call dicts with keys: id, name, arguments
        """
        tool_calls = []
        if hasattr(response, 'choices') and response.choices:
            msg = response.choices[0].message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments if hasattr(tc, 'function') else '{}'
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tool_calls.append({
                        "id": getattr(tc, 'id', None),
                        "name": tc.function.name if hasattr(tc, 'function') else 'unknown',
                        "arguments": args,
                    })
        return tool_calls

    def has_tool_calls(self, response) -> bool:
        """Check if response contains tool calls."""
        if hasattr(response, 'choices') and response.choices:
            msg = response.choices[0].message
            return bool(getattr(msg, 'tool_calls', None))
        return False

    def get_finish_reason(self, response) -> str:
        """Get finish reason from response."""
        if hasattr(response, 'choices') and response.choices:
            return getattr(response.choices[0], 'finish_reason', 'unknown') or 'unknown'
        return 'unknown'

    def extract_json_from_response(self, response) -> str:
        """Extract JSON from response, handling markdown code blocks.

        Some models (e.g., haiku) wrap JSON in ```json ... ``` blocks.
        This method extracts the raw JSON for parsing.
        """
        text = extract_output(response)
        return extract_json_from_text(text)

    @staticmethod
    def _add_additional_properties_false(schema: dict) -> dict:
        """Add additionalProperties: false to all objects in schema."""
        if not isinstance(schema, dict):
            return schema

        result = schema.copy()

        if result.get("type") == "object":
            result["additionalProperties"] = False

        if "properties" in result:
            new_props = {}
            for key, value in result["properties"].items():
                new_props[key] = OpenRouterClient._add_additional_properties_false(value)
            result["properties"] = new_props

        if "items" in result:
            result["items"] = OpenRouterClient._add_additional_properties_false(result["items"])

        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = OpenRouterClient._add_additional_properties_false(value)
            result["$defs"] = new_defs

        return result

    @staticmethod
    def _make_all_fields_required(schema: dict) -> dict:
        """Make all properties required in schema (OpenAI strict mode requirement).

        OpenAI's structured output requires ALL fields in 'properties' to also
        be in 'required'. This recursively fixes Pydantic schemas where fields
        with defaults are marked as optional.
        """
        if not isinstance(schema, dict):
            return schema

        result = schema.copy()

        # If this object has properties, make them ALL required
        if result.get("type") == "object" and "properties" in result:
            result["required"] = list(result["properties"].keys())
            # Recursively process nested properties
            new_props = {}
            for key, value in result["properties"].items():
                new_props[key] = OpenRouterClient._make_all_fields_required(value)
            result["properties"] = new_props

        # Process array items
        if "items" in result:
            result["items"] = OpenRouterClient._make_all_fields_required(result["items"])

        # Process $defs (nested type definitions)
        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = OpenRouterClient._make_all_fields_required(value)
            result["$defs"] = new_defs

        return result

    @staticmethod
    def resolve_model(model: str, suffix: str = "") -> str:
        """Resolve model name with optional OpenRouter suffix.

        OpenRouter supports routing suffixes like :nitro (fast) and :floor (cheapest).
        See: https://openrouter.ai/docs/model-routing

        Args:
            model: Base model name (e.g., "openai/gpt-5-mini")
            suffix: Optional routing suffix (e.g., "nitro", "floor")

        Returns:
            Model string with suffix if provided (e.g., "openai/gpt-5-mini:nitro")
        """
        return f"{model}:{suffix}" if suffix else model

    @retry(
        stop=stop_after_attempt(4),  # 4 attempts = 1 initial + 3 retries
        wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
        before_sleep=_log_retry_error,
    )
    async def _send_image_with_retry(self, payload: dict, timeout_seconds: int = 180) -> dict:
        """Send image generation request with retry logic.

        Args:
            payload: Request payload with model, messages, modalities
            timeout_seconds: Timeout per request (default 180s = 3 min)

        Returns:
            Response JSON dict

        Raises:
            Exception on HTTP errors (triggers retry)
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error {response.status}: {error_text[:500]}")

                return await response.json()

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        message_callback=None,
        timeout_per_call: int = 180,  # 3 minutes per call
    ) -> bytes | None:
        """Generate an image using OpenRouter with image-capable models.

        Uses raw HTTP API to support modalities parameter.
        Includes retry logic with exponential backoff via tenacity.

        Args:
            prompt: The image generation prompt
            model: Override the model (should be an image-capable model)
            system: System prompt
            aspect_ratio: Aspect ratio (e.g., "1:1", "16:9", "9:16")
            image_size: Image size (e.g., "1K", "2K", "4K")
            message_callback: Callback for logging messages
            timeout_per_call: Timeout in seconds per attempt (default: 180 = 3 min)

        Returns:
            Image bytes (PNG) or None if generation failed
        """
        image_model = model or self.model

        # Log prompt
        if message_callback:
            message_callback({
                "type": "prompt",
                "message_text": f"[Image Generation] {prompt}",
                "iso_timestamp": datetime.now().isoformat(),
            })

        # Build request payload
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": image_model,
            "messages": messages,
            "modalities": ["image"],  # Only request image output, no text
        }

        # Add image config for Gemini models
        if aspect_ratio or image_size:
            payload["image_config"] = {}
            if aspect_ratio:
                payload["image_config"]["aspect_ratio"] = aspect_ratio
            if image_size:
                payload["image_config"]["image_size"] = image_size

        # Track start time for runtime calculation
        start_time = datetime.now()

        try:
            # Use tenacity-wrapped method for retries
            result = await self._send_image_with_retry(payload, timeout_per_call)

            # Calculate runtime
            runtime_seconds = (datetime.now() - start_time).total_seconds()

            # Extract usage info from response (OpenRouter format)
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # Cost may be in usage or top-level
            total_cost = usage.get("cost", 0.0) or result.get("cost", 0.0)
            actual_model = result.get("model", image_model)

            # Helper to emit summary before return
            def emit_summary(status: str = "success"):
                if message_callback:
                    message_callback({
                        "type": "summary",
                        "total_cost": total_cost,
                        "token_cost": total_cost,
                        "tool_cost": 0.0,
                        "model": actual_model,
                        "status": status,
                        "is_aggregated": False,
                        "is_error": status in ("error", "failed"),
                        "num_calls": 1,
                        "runtime_seconds": runtime_seconds,
                        "llm_time_seconds": runtime_seconds,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": 0,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": 0,
                        "tool_calls": {},
                        "tool_costs": {},
                        "iso_timestamp": datetime.now().isoformat(),
                    })

            # Extract image from response
            # Images are returned as base64-encoded data URLs in the format:
            # { "images": [{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }] }
            import base64 as b64_module

            if "choices" in result and result["choices"]:
                for choice in result["choices"]:
                    message = choice.get("message", {})

                    # Check for images field (OpenRouter image generation format)
                    images = message.get("images", [])
                    if images:
                        for img_block in images:
                            # Format: { "type": "image_url", "image_url": { "url": "data:..." } }
                            if isinstance(img_block, dict):
                                if img_block.get("type") == "image_url":
                                    url = img_block.get("image_url", {}).get("url", "")
                                    if url.startswith("data:image"):
                                        # Extract format: data:image/png;base64,... → png
                                        mime_part = url.split(",")[0]
                                        img_format = mime_part.split("/")[1].split(";")[0]
                                        _, b64_data = url.split(",", 1)
                                        if message_callback:
                                            message_callback({
                                                "type": "or_img",
                                                "message_text": f"Image generated: {img_format.upper()} ({len(b64_data)} base64 chars)",
                                                "iso_timestamp": datetime.now().isoformat(),
                                                "llm_provider": "openrouter",
                                                "message_metadata": {"format": img_format},
                                            })
                                        emit_summary("success")
                                        return b64_module.b64decode(b64_data)
                            # Also handle plain data URL strings (fallback)
                            elif isinstance(img_block, str) and img_block.startswith("data:image"):
                                mime_part = img_block.split(",")[0]
                                img_format = mime_part.split("/")[1].split(";")[0]
                                _, b64_data = img_block.split(",", 1)
                                if message_callback:
                                    message_callback({
                                        "type": "or_img",
                                        "message_text": f"Image generated: {img_format.upper()} ({len(b64_data)} base64 chars)",
                                        "iso_timestamp": datetime.now().isoformat(),
                                        "llm_provider": "openrouter",
                                        "message_metadata": {"format": img_format},
                                    })
                                emit_summary("success")
                                return b64_module.b64decode(b64_data)

                    # Check content for data URL (fallback)
                    content = message.get("content", "")
                    if isinstance(content, str) and "data:image" in content:
                        import re
                        match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
                        if match:
                            if message_callback:
                                message_callback({
                                    "type": "or_img",
                                    "message_text": f"Image extracted from content",
                                    "iso_timestamp": datetime.now().isoformat(),
                                    "llm_provider": "openrouter",
                                })
                            emit_summary("success")
                            return b64_module.b64decode(match.group(1))

                    # Check for content array with image blocks
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "image_url":
                                    url = block.get("image_url", {}).get("url", "")
                                    if url.startswith("data:image"):
                                        _, b64_data = url.split(",", 1)
                                        emit_summary("success")
                                        return b64_module.b64decode(b64_data)
                                elif block.get("type") == "image":
                                    data = block.get("data", "")
                                    if data:
                                        emit_summary("success")
                                        return b64_module.b64decode(data)

            # No image found in response
            if message_callback:
                message_callback({
                    "type": "warning",
                    "message_text": f"No image in response: {json.dumps(result)[:200]}",
                    "iso_timestamp": datetime.now().isoformat(),
                })
            emit_summary("no_image")
            return None

        except Exception as e:
            if message_callback:
                message_callback({
                    "type": "error",
                    "message_text": f"Image generation failed: {e}",
                    "iso_timestamp": datetime.now().isoformat(),
                })
            logger.exception(f"Image generation failed: {e}")
            raise

    async def close(self):
        """Close the client.

        OpenRouter uses per-request aiohttp sessions (created and closed within
        each request method), so there is no persistent connection to close.
        This method exists for API consistency with other provider clients.
        """
        try:
            # No persistent session to close; aiohttp sessions are context-managed per-request
            pass
        except Exception:
            pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
