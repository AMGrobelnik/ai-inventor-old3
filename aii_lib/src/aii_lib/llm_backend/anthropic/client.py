"""Anthropic Client - Async with aiohttp.

Supported Models (December 2025):
    Claude 4.5 Series (Latest):
        - claude-opus-4-5-20251101   : Premium, max intelligence
        - claude-sonnet-4-5-20250929 : Best balance speed/intelligence
        - claude-haiku-4-5-20251001  : Fastest, near-frontier

    Aliases (auto-update to latest):
        - claude-opus-4-5   -> claude-opus-4-5-20251101
        - claude-sonnet-4-5 -> claude-sonnet-4-5-20250929
        - claude-haiku-4-5  -> claude-haiku-4-5-20251001

    Claude 4 Series (Legacy):
        - claude-opus-4-1-20250805   : Previous Opus
        - claude-sonnet-4-20250514   : Claude 4 Sonnet
        - claude-opus-4-20250514     : Claude 4 Opus

    Claude 3.x Series (Legacy):
        - claude-3-7-sonnet-20250219 : Claude 3.7 Sonnet
        - claude-3-5-haiku-20241022  : Fast Claude 3.5
        - claude-3-haiku-20240307    : Fastest legacy

    Recommended:
        - Flagship: claude-opus-4-5-20251101
        - Balanced: claude-sonnet-4-5-20250929
        - Fast: claude-haiku-4-5-20251001

Docs: https://docs.anthropic.com/en/docs/about-claude/models
"""

from datetime import datetime
from anthropic import AsyncAnthropic, DefaultAioHttpClient
from aii_lib.telemetry import logger

from .pricing import calculate_cost
from .ant_to_json import (
    serialize_response,
    extract_thinking, extract_output, extract_usage
)
from .json_to_log_format import create_summary


class AnthropicClient:
    """Async Anthropic client using aiohttp for improved concurrency.

    Supported Models (Dec 2025):
        Claude 4.5: claude-opus-4-5-20251101, claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001
        Aliases: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5
        Legacy: claude-opus-4-1-20250805, claude-sonnet-4-20250514, claude-3-5-haiku-20241022

    Recommended: claude-sonnet-4-5 (balanced), claude-haiku-4-5 (fast)

    Docs: https://docs.anthropic.com/en/docs/about-claude/models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extended_thinking: bool | None = None,
        thinking_budget: int | None = None,
        timeout: float = 600,
        config_path: str | None = None,
    ):
        # Load config defaults first
        from ..config import load_config, get_anthropic_config

        if config_path:
            load_config(config_path)

        config = get_anthropic_config()

        # Config defaults, overridden by explicit parameters
        self.model = model or config.get('default_model', 'claude-sonnet-4-20250514')
        self.max_tokens = max_tokens or config.get('max_tokens', 20000)
        self.temperature = temperature if temperature is not None else config.get('temperature', 1.0)
        self.extended_thinking = extended_thinking if extended_thinking is not None else config.get('extended_thinking', True)
        self.thinking_budget = thinking_budget or config.get('thinking_budget', 16000)

        # Create async client with aiohttp backend
        resolved_api_key = api_key or config.get('api_key')
        if not resolved_api_key:
            logger.warning("No Anthropic API key provided; requests will likely fail")
        client_kwargs = {
            "http_client": DefaultAioHttpClient(),
            "timeout": timeout,
        }
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        self.client = AsyncAnthropic(**client_kwargs)

        logger.info(f"Anthropic async client initialized with model: {self.model}")

    async def create_message(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extended_thinking: bool | None = None,
        thinking_budget: int | None = None,
        tools: list | None = None,
        output_format: type | dict | None = None,
        message_callback=None,
    ):
        """Create a message (async, no streaming).

        Args:
            prompt: The user prompt
            system: System prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature
            extended_thinking: Enable extended thinking
            thinking_budget: Token budget for thinking
            tools: List of tools for function calling
            output_format: Pydantic model class or JSON schema dict for structured output
            message_callback: Callback for logging messages

        Returns:
            The API response object
        """
        # Log prompt
        if message_callback:
            message_callback({
                "type": "prompt",
                "message_text": prompt,
                "iso_timestamp": datetime.now().isoformat(),
            })

        # Build kwargs
        use_thinking = extended_thinking if extended_thinking is not None else self.extended_thinking
        budget = thinking_budget if thinking_budget is not None else self.thinking_budget
        resolved_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Only include max_tokens if specified
        if resolved_max_tokens is not None:
            kwargs["max_tokens"] = resolved_max_tokens

        if system:
            kwargs["system"] = system

        if use_thinking:
            thinking_config = {"type": "enabled"}
            if budget is not None:
                thinking_config["budget_tokens"] = budget
            kwargs["thinking"] = thinking_config
        else:
            kwargs["temperature"] = temperature if temperature is not None else self.temperature

        if tools:
            kwargs["tools"] = tools

        # Add structured output format (beta feature)
        use_beta = False
        if output_format:
            use_beta = True
            if hasattr(output_format, 'model_json_schema'):
                schema = output_format.model_json_schema()
            else:
                schema = output_format
            # Strict mode requires: additionalProperties=false, all fields required
            schema = self._add_additional_properties_false(schema)
            schema = self._make_all_fields_required(schema)
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": schema
            }

        # Log system message
        if message_callback:
            system_text = f"{self.model}"
            if resolved_max_tokens is not None:
                system_text += f" | Max tokens: {resolved_max_tokens}"
            if use_thinking:
                if budget is not None:
                    system_text += f" | Thinking: {budget} tokens"
                else:
                    system_text += " | Thinking: enabled"
            if use_beta:
                system_text += " | Structured output: enabled"

            message_callback({
                "type": "system",
                "message_text": system_text,
                "model": self.model,
                "iso_timestamp": datetime.now().isoformat(),
                "llm_provider": "anthropic",
                "message_metadata": {
                    "model": self.model,
                    "max_tokens": resolved_max_tokens,
                    "extended_thinking": use_thinking,
                    "thinking_budget": budget if use_thinking else None,
                    "structured_output": use_beta,
                },
                "raw_api_response": None,
            })

        # Make async request
        if use_beta:
            response = await self.client.beta.messages.create(
                betas=["structured-outputs-2025-11-13"],
                **kwargs
            )
            # Validate structured output response has expected content
            if not hasattr(response, 'content') or not response.content:
                logger.warning("Structured output response has no content blocks")
            elif not any(getattr(block, 'type', None) == 'text' for block in response.content):
                logger.warning("Structured output response has no text blocks")
        else:
            response = await self.client.messages.create(**kwargs)

        # Process response.content in order to preserve interleaving
        if message_callback and hasattr(response, 'content') and response.content:
            for block in response.content:
                block_type = getattr(block, 'type', None)
                block_raw = block.model_dump() if hasattr(block, 'model_dump') else None

                if block_type == 'thinking':
                    thinking_text = getattr(block, 'thinking', '')
                    if thinking_text:
                        message_callback({
                            "type": "thinking",
                            "message_text": thinking_text,
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "anthropic",
                            "message_metadata": {},
                            "raw_api_response": block_raw,
                        })

                elif block_type == 'text':
                    text = getattr(block, 'text', '')
                    if text:
                        message_callback({
                            "type": "ant_msg",
                            "message_text": text,
                            "iso_timestamp": datetime.now().isoformat(),
                            "llm_provider": "anthropic",
                            "message_metadata": {},
                            "raw_api_response": block_raw,
                        })

                elif block_type == 'tool_use':
                    tool_name = getattr(block, 'name', 'unknown')
                    message_callback({
                        "type": "ant_tool",
                        "message_text": f"Tool: {tool_name}",
                        "iso_timestamp": datetime.now().isoformat(),
                        "llm_provider": "anthropic",
                        "message_metadata": {},
                        "raw_api_response": block_raw,
                    })

        # Log summary with full response
        if message_callback:
            usage = extract_usage(response)
            total_cost = calculate_cost(usage, self.model)

            message_callback({
                "type": "summary",
                "message_text": create_summary(response, self.model),
                "llm_provider": "anthropic",
                "message_metadata": {
                    "total_cost": total_cost,
                    "usage": usage,
                },
                "raw_api_response": serialize_response(response),
                "iso_timestamp": datetime.now().isoformat(),
            })

        return response

    def extract_text_from_response(self, response) -> str:
        """Extract output text from response."""
        return extract_output(response)

    def extract_thinking_from_response(self, response) -> str:
        """Extract thinking from response."""
        return extract_thinking(response)

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
                new_props[key] = AnthropicClient._add_additional_properties_false(value)
            result["properties"] = new_props

        if "items" in result:
            result["items"] = AnthropicClient._add_additional_properties_false(result["items"])

        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = AnthropicClient._add_additional_properties_false(value)
            result["$defs"] = new_defs

        return result

    @staticmethod
    def _make_all_fields_required(schema: dict) -> dict:
        """Make all properties required in schema (strict mode requirement).

        Structured output requires ALL fields in 'properties' to also be in
        'required'. This recursively fixes Pydantic schemas where fields with
        defaults are marked as optional.
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
                new_props[key] = AnthropicClient._make_all_fields_required(value)
            result["properties"] = new_props

        # Process array items
        if "items" in result:
            result["items"] = AnthropicClient._make_all_fields_required(result["items"])

        # Process $defs (nested type definitions)
        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = AnthropicClient._make_all_fields_required(value)
            result["$defs"] = new_defs

        return result

    async def close(self):
        """Close the aiohttp client session."""
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
