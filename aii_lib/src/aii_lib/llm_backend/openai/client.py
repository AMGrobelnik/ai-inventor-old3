"""OpenAI Client - Async with aiohttp.

Supported Models (December 2025):
    GPT-5 Series:
        - gpt-5.2          : Latest flagship (Dec 2025)
        - gpt-5.1          : Previous flagship (Nov 2025)
        - gpt-5            : Base model (Aug 2025)
        - gpt-5-mini       : Fast, cost-efficient
        - gpt-5-nano       : Smallest variant
        - gpt-5-pro        : Extended reasoning
        - gpt-5.1-codex    : Code-optimized

    O-Series Reasoning:
        - o4-mini          : Fast reasoning (best AIME scores)
        - o3               : Complex reasoning
        - o3-mini          : Small reasoning model
        - o3-pro           : Extended compute reasoning
        - o1               : Previous reasoning model

    GPT-4 Series (Legacy):
        - gpt-4.1          : Improved GPT-4o (Apr 2025)
        - gpt-4.1-mini     : Fast variant
        - gpt-4o           : Legacy multimodal
        - gpt-4o-mini      : Legacy fast

    Recommended:
        - Flagship: gpt-5.2
        - Fast: gpt-5-mini
        - Reasoning: o4-mini
        - Coding: gpt-5.1-codex

Docs: https://platform.openai.com/docs/models
"""

import json
import re
from datetime import datetime
from openai import AsyncOpenAI, DefaultAioHttpClient
from aii_lib.telemetry import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .pricing import calculate_cost, calculate_token_cost, WEB_SEARCH_PRICING


def _log_retry_error(retry_state) -> None:
    """Log retry error with full traceback using exc= parameter."""
    exc = retry_state.outcome.exception()
    logger.warning(
        f"OpenAI request failed, retrying in {retry_state.next_action.sleep:.1f}s... "
        f"(attempt {retry_state.attempt_number}/3)",
        exc=exc,
    )


from .oai_to_json import (
    serialize_response,
    extract_reasoning, extract_output,
)
from .json_to_log_format import create_summary


class OpenAIClient:
    """Async OpenAI client using aiohttp for improved concurrency.

    Supported Models (Dec 2025):
        GPT-5: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro
        O-Series: o4-mini, o3, o3-mini, o3-pro, o1
        GPT-4 (legacy): gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini

    Recommended: gpt-5.2 (flagship), gpt-5-mini (fast), o4-mini (reasoning)

    Docs: https://platform.openai.com/docs/models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
        config_path: str | None = None,
    ):
        # Load config defaults first
        from ..config import load_config, get_openai_config

        if config_path:
            load_config(config_path)

        config = get_openai_config()

        # Config defaults, overridden by explicit parameters
        self.model = model or config.get('default_model', 'gpt-5')
        self.default_reasoning_effort = reasoning_effort or config.get('reasoning_effort', 'high')
        self.default_verbosity = verbosity or config.get('verbosity', 'medium')
        self.default_service_tier = service_tier or config.get('service_tier', 'default')

        # Create async client with aiohttp backend
        # Default timeout is 10 min (600s), but can be overridden
        resolved_api_key = api_key or config.get('api_key')
        if not resolved_api_key:
            logger.warning("No OpenAI API key provided; requests will likely fail")
        resolved_timeout = timeout if timeout is not None else 600.0
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            http_client=DefaultAioHttpClient(),
            timeout=resolved_timeout,
        )


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=_log_retry_error,
    )
    async def _send_with_retry(self, **kwargs):
        """Send request with automatic retry on connection failures."""
        return await self.client.responses.create(**kwargs)

    async def create_response(
        self,
        prompt: str,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        message_callback=None,
        # Web search options
        web_search: bool = False,
        allowed_domains: list[str] | None = None,
        user_location: dict | None = None,
        external_web_access: bool = True,
        # Structured output
        text_format: type | None = None,
        response_format: dict | None = None,
        # Multi-turn conversation
        previous_response_id: str | None = None,
        # Emission controls (for multi-turn conversations)
        emit_system: bool = True,
        emit_summary: bool = True,
    ):
        """Create a response (async, no streaming).

        Args:
            prompt: The user prompt
            instructions: System prompt / instructions for the model
            reasoning_effort: Override default reasoning effort
            verbosity: Override default verbosity
            message_callback: Callback for logging messages

            # Web search options
            web_search: Enable web search tool
            allowed_domains: List of domains to filter results (up to 100)
                             Example: ["openai.com", "github.com"]
            user_location: Approximate user location dict with keys:
                           - country: Two-letter ISO code (e.g., "US", "GB")
                           - city: City name (e.g., "London")
                           - region: Region/state (e.g., "California")
                           - timezone: IANA timezone (e.g., "America/Chicago")
            external_web_access: If False, use cached/indexed results only

            # Structured output
            text_format: Pydantic model class for structured output
            response_format: Raw dict schema for structured output

            # Multi-turn conversation
            previous_response_id: ID of previous response for multi-turn conversation.
                                  Pass response.id from a previous call to continue the conversation.
            emit_system: Whether to emit system/config message (False for subsequent turns in multi-turn)
            emit_summary: Whether to emit summary message (False for intermediate turns in multi-turn)

        Returns:
            The API response object (has .id attribute for multi-turn)
        """
        # Log prompt
        if message_callback:
            message_callback({
                "type": "prompt",
                "message_text": prompt,
                "iso_timestamp": datetime.now().isoformat(),
            })

        # Log system message (only if emit_system=True)
        reasoning = reasoning_effort or self.default_reasoning_effort
        verb = verbosity or self.default_verbosity

        if message_callback and emit_system:
            system_text = f"{self.model} | Reasoning: {reasoning}"
            if web_search:
                system_text += " | Web Search: enabled"
                if allowed_domains:
                    system_text += f" | Domains: {len(allowed_domains)}"
            if text_format:
                system_text += " | Structured output: enabled"

            message_callback({
                "type": "system",
                "message_text": system_text,
                "model": self.model,
                "iso_timestamp": datetime.now().isoformat(),
                "llm_provider": "openai",
                "message_metadata": {
                    "model": self.model,
                    "reasoning_effort": reasoning,
                    "verbosity": verb,
                    "web_search": web_search,
                },
                "raw_api_response": None,
            })

        # Build request
        kwargs = {
            "model": self.model,
            "input": prompt,
            "reasoning": {
                "effort": reasoning,
            }
        }

        # Add service tier if configured
        if self.default_service_tier and self.default_service_tier != "default":
            kwargs["service_tier"] = self.default_service_tier

        # Add system instructions if provided
        if instructions:
            kwargs["instructions"] = instructions

        # Add previous response ID for multi-turn conversation
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        if verb != "medium":
            kwargs["reasoning"]["verbosity"] = verb

        # Add web search tool with all options
        if web_search:
            web_search_tool: dict = {"type": "web_search"}

            # Domain filtering (up to 100 domains)
            if allowed_domains:
                web_search_tool["filters"] = {"allowed_domains": allowed_domains}

            # User location for geo-specific results
            if user_location:
                web_search_tool["user_location"] = {
                    "type": "approximate",
                    **user_location
                }

            # Control live vs cached access
            if not external_web_access:
                web_search_tool["external_web_access"] = False

            kwargs["tools"] = [web_search_tool]

            # Always include sources in response
            kwargs["include"] = ["web_search_call.action.sources"]

        # Add structured output format
        if text_format:
            if hasattr(text_format, 'model_json_schema'):
                schema = text_format.model_json_schema()
                # OpenAI strict mode requires: additionalProperties=false, all fields required
                schema = self._add_additional_properties_false(schema)
                schema = self._make_all_fields_required(schema)
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": text_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }
            else:
                kwargs["text"] = {"format": text_format}
        elif response_format:
            kwargs["text"] = {"format": response_format}

        # Track start time for runtime calculation
        start_time = datetime.now()

        # Make async request with connection retry
        response = await self._send_with_retry(**kwargs)

        # Calculate runtime
        runtime_minutes = (datetime.now() - start_time).total_seconds() / 60.0

        # Process response.output in order to preserve interleaving
        if message_callback and hasattr(response, 'output') and response.output:
            for item in response.output:
                item_type = getattr(item, 'type', None)

                if item_type == 'reasoning':
                    # Extract reasoning text from summary list
                    if hasattr(item, 'summary') and isinstance(item.summary, list):
                        texts = [
                            summary_obj.text
                            for summary_obj in item.summary
                            if hasattr(summary_obj, 'text')
                        ]
                        if texts:
                            reasoning_text = "\n\n".join(texts)
                            message_callback({
                                "type": "thinking",
                                "message_text": reasoning_text,
                                "iso_timestamp": datetime.now().isoformat(),
                                "llm_provider": "openai",
                                "message_metadata": {},
                                "raw_api_response": item.model_dump(warnings=False) if hasattr(item, 'model_dump') else None,
                            })

                elif item_type == 'web_search_call':
                    # Build display text based on action type
                    action_type = None
                    display_text = "Web action"
                    if hasattr(item, 'action') and item.action:
                        action = item.action
                        action_type = getattr(action, 'type', None)

                        if action_type == "search":
                            query = getattr(action, 'query', None)
                            display_text = f"Search: {query or '(no query)'}"
                            # Add sources if available
                            if hasattr(action, 'sources') and action.sources:
                                source_urls = []
                                for src in action.sources:
                                    if isinstance(src, dict):
                                        url = src.get('url', '')
                                    else:
                                        url = getattr(src, 'url', None) or ''
                                    if url:
                                        source_urls.append(url)
                                if source_urls:
                                    display_text += "\nSources:"
                                    for url in source_urls:
                                        display_text += f"\n  {url}"
                        elif action_type == "open_page":
                            url = getattr(action, 'url', None)
                            if url:
                                display_text = f"Open: {url}"
                            else:
                                display_text = "Open: (current page)"
                        elif action_type == "find_in_page":
                            pattern = getattr(action, 'pattern', None)
                            url = getattr(action, 'url', None)
                            display_text = f"Find: '{pattern or '?'}'"
                            if url:
                                display_text += f"\n  in: {url}"
                        else:
                            display_text = f"Web: {action_type}"

                    message_callback({
                        "type": "oai_srch",
                        "message_text": display_text,
                        "iso_timestamp": datetime.now().isoformat(),
                        "llm_provider": "openai",
                        "message_metadata": {},
                        "raw_api_response": item.model_dump(warnings=False) if hasattr(item, 'model_dump') else None,
                    })

                elif item_type == 'message':
                    # Extract message content and annotations
                    msg_raw = item.model_dump(warnings=False) if hasattr(item, 'model_dump') else None

                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_part in item.content:
                            text = getattr(content_part, 'text', '') or ''
                            if text:
                                # Clean citation tokens for display
                                display_text = re.sub(r'\ue200.*?\ue201', '', text)
                                display_text = re.sub(r'citeturn\d+(?:search|academia)\d+(?:turn\d+(?:search|academia)\d+)*', '', display_text)

                                # Pretty-print JSON for structured output
                                if display_text.strip().startswith('{'):
                                    try:
                                        parsed = json.loads(display_text)
                                        display_text = json.dumps(parsed, indent=2, ensure_ascii=False)
                                    except json.JSONDecodeError:
                                        pass  # Not valid JSON, keep as-is

                                message_callback({
                                    "type": "oai_msg",
                                    "message_text": display_text,
                                    "iso_timestamp": datetime.now().isoformat(),
                                    "llm_provider": "openai",
                                    "message_metadata": {},
                                    "raw_api_response": msg_raw,
                                })

                            # Log annotations for this content part
                            annotations = getattr(content_part, 'annotations', []) or []
                            if annotations:
                                # Get raw annotation data
                                annotations_raw = [
                                    annot.model_dump(warnings=False) if hasattr(annot, 'model_dump') else {}
                                    for annot in annotations
                                ]

                                formatted_parts = []
                                for annot in annotations:
                                    start = getattr(annot, 'start_index', None)
                                    end = getattr(annot, 'end_index', None)
                                    quote = text[start:end] if start is not None and end is not None else ''
                                    formatted_parts.append(
                                        f"Quote: {quote}\n"
                                        f"Title: {getattr(annot, 'title', '') or ''}\n"
                                        f"Type: {getattr(annot, 'type', '') or ''}\n"
                                        f"Url: {getattr(annot, 'url', '') or ''}"
                                    )

                                message_callback({
                                    "type": "oai_sour",
                                    "message_text": "\n\n".join(formatted_parts),
                                    "iso_timestamp": datetime.now().isoformat(),
                                    "llm_provider": "openai",
                                    "message_metadata": {},
                                    "raw_api_response": annotations_raw,
                                })

        # Log summary with full response (only if emit_summary=True)
        if message_callback and emit_summary:
            # Count web search actions FIRST (before cost calculation)
            web_search_count = 0
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if getattr(item, 'type', None) == 'web_search_call':
                        web_search_count += 1

            # Calculate costs (token_cost doesn't include web search, total_cost does)
            token_cost = calculate_token_cost(response, self.model, self.default_service_tier)
            web_search_cost = (web_search_count / 1000) * WEB_SEARCH_PRICING["per_1k_calls"]
            total_cost = token_cost + web_search_cost

            # Get status from response
            status = getattr(response, 'status', 'unknown') or 'unknown'

            # Get actual model from response
            actual_model = getattr(response, 'model', self.model) or self.model

            # Extract usage and normalize field names
            usage_raw = {}
            if hasattr(response, 'usage') and response.usage:
                usage_raw = response.usage.model_dump(warnings=False) if hasattr(response.usage, 'model_dump') else {}

            normalized_usage = {
                "input_tokens": usage_raw.get("input_tokens"),
                "output_tokens": usage_raw.get("output_tokens"),
                "reasoning_tokens": usage_raw.get("output_tokens_details", {}).get("reasoning_tokens") if usage_raw.get("output_tokens_details") else None,
                # OpenAI: cached_tokens = tokens served from cache (at reduced price)
                # Unlike Anthropic, OpenAI doesn't report cache creation separately
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": usage_raw.get("input_tokens_details", {}).get("cached_tokens") if usage_raw.get("input_tokens_details") else None,
                "service_tier": self.default_service_tier,
                "server_tool_use": {
                    "web_search_requests": web_search_count if web_search_count > 0 else None,
                    "web_fetch_requests": None,
                },
            }

            message_callback({
                "type": "summary",
                "message_text": "",  # Will be formatted by format_summary_message
                "llm_provider": "openai",
                "message_metadata": {
                    "total_cost": total_cost,
                    "token_cost": token_cost,
                    "model": actual_model,
                    "status": status,
                    "runtime_minutes": runtime_minutes,
                    "num_turns": 1,
                    "usage": normalized_usage,
                },
                "raw_api_response": serialize_response(response),
                "iso_timestamp": datetime.now().isoformat(),
            })

        return response

    def extract_text_from_response(self, response, resolve_citations: bool = True) -> str:
        """Extract output text from response.

        Args:
            response: The API response object
            resolve_citations: If True, replace citeturn... tokens with markdown citations

        Returns:
            The extracted text, optionally with resolved citations
        """
        text = extract_output(response)

        if resolve_citations and text:
            text = self._resolve_citations(response, text)

        return text

    def extract_reasoning_from_response(self, response) -> str:
        """Extract reasoning from response."""
        return extract_reasoning(response)

    def extract_usage_from_response(self, response) -> dict:
        """Extract usage stats from response for aggregation.

        Returns dict with keys: input_tokens, output_tokens, reasoning_tokens,
        cached_tokens, web_search_count, token_cost, total_cost
        """
        usage_raw = {}
        if hasattr(response, 'usage') and response.usage:
            usage_raw = response.usage.model_dump(warnings=False) if hasattr(response.usage, 'model_dump') else {}

        # Count web search actions
        web_search_count = 0
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if getattr(item, 'type', None) == 'web_search_call':
                    web_search_count += 1

        # Calculate costs
        token_cost = calculate_token_cost(response, self.model, self.default_service_tier)
        web_search_cost = (web_search_count / 1000) * WEB_SEARCH_PRICING["per_1k_calls"]
        total_cost = token_cost + web_search_cost

        return {
            "input_tokens": usage_raw.get("input_tokens", 0),
            "output_tokens": usage_raw.get("output_tokens", 0),
            "reasoning_tokens": (
                usage_raw.get("output_tokens_details", {}).get("reasoning_tokens", 0)
                if usage_raw.get("output_tokens_details") else 0
            ),
            "cached_tokens": (
                usage_raw.get("input_tokens_details", {}).get("cached_tokens", 0)
                if usage_raw.get("input_tokens_details") else 0
            ),
            "web_search_count": web_search_count,
            "token_cost": token_cost,
            "total_cost": total_cost,
        }

    def _resolve_citations(self, response, text: str) -> str:
        """Replace citeturn... tokens with markdown-style citations.

        OpenAI returns citations as tokens like 'citeturn0search0' which map to
        annotations in the response. This method:
        1. Collects all annotations (which have URL, title, start/end indices)
        2. Replaces citation tokens with numbered markdown references [1], [2], etc.
        3. Appends a references section at the end

        Args:
            response: The API response containing annotations
            text: The text with citation tokens

        Returns:
            Text with markdown citations and references section
        """
        # Collect all annotations from the response
        annotations = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if getattr(item, 'type', None) == 'message':
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_part in item.content:
                            part_annotations = getattr(content_part, 'annotations', []) or []
                            for annot in part_annotations:
                                url = getattr(annot, 'url', None) or ''
                                title = getattr(annot, 'title', None) or ''
                                start = getattr(annot, 'start_index', None)
                                end = getattr(annot, 'end_index', None)
                                if url and start is not None and end is not None:
                                    annotations.append({
                                        'url': url,
                                        'title': title,
                                        'start': start,
                                        'end': end,
                                    })

        if not annotations:
            # No annotations found, strip citation tokens entirely
            # Handles both plain format (citeturn0search0) and Unicode-wrapped format
            # Unicode format uses: \ue200 (start), \ue201 (end), \ue202 (separator)
            text = re.sub(r'\ue200.*?\ue201', '', text)  # Unicode wrapped
            text = re.sub(r'citeturn\d+(?:search|academia)\d+(?:turn\d+(?:search|academia)\d+)*', '', text)  # Plain
            return text

        # Sort by start index (reverse order for replacement)
        annotations.sort(key=lambda x: x['start'], reverse=True)

        # Build unique references list (dedupe by URL)
        seen_urls = {}
        references = []

        # First pass: collect unique URLs and assign numbers
        for annot in sorted(annotations, key=lambda x: x['start']):
            url = annot['url']
            if url not in seen_urls:
                ref_num = len(references) + 1
                seen_urls[url] = ref_num
                references.append({
                    'num': ref_num,
                    'url': url,
                    'title': annot['title'],
                })

        # Second pass: replace citation tokens with numbered references
        # Process in reverse order to preserve indices
        for annot in annotations:
            ref_num = seen_urls[annot['url']]
            start, end = annot['start'], annot['end']
            # Replace the citation token with [N]
            text = text[:start] + f"[{ref_num}]" + text[end:]

        # Clean up any remaining orphan citation tokens
        text = re.sub(r'\ue200.*?\ue201', '', text)  # Unicode wrapped
        text = re.sub(r'citeturn\d+(?:search|academia)\d+(?:turn\d+(?:search|academia)\d+)*', '', text)

        # Append references section if we have any
        if references:
            text += "\n\n---\n**References:**\n"
            for ref in references:
                if ref['title']:
                    text += f"[{ref['num']}] [{ref['title']}]({ref['url']})\n"
                else:
                    text += f"[{ref['num']}] {ref['url']}\n"

        return text

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
                new_props[key] = OpenAIClient._add_additional_properties_false(value)
            result["properties"] = new_props

        if "items" in result:
            result["items"] = OpenAIClient._add_additional_properties_false(result["items"])

        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = OpenAIClient._add_additional_properties_false(value)
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
                new_props[key] = OpenAIClient._make_all_fields_required(value)
            result["properties"] = new_props

        # Process array items
        if "items" in result:
            result["items"] = OpenAIClient._make_all_fields_required(result["items"])

        # Process $defs (nested type definitions)
        if "$defs" in result:
            new_defs = {}
            for key, value in result["$defs"].items():
                new_defs[key] = OpenAIClient._make_all_fields_required(value)
            result["$defs"] = new_defs

        return result

    async def close(self):
        """Close the aiohttp client session."""
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
