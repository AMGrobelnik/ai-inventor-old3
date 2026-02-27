"""Gemini Client - Async with aiohttp.

Supported Models (December 2025):
    Gemini 3 Series (Preview):
        - gemini-3-pro-preview       : Latest Gemini 3 Pro
        - gemini-3-flash-preview     : High-speed reasoning, 1M ctx, agentic
        - gemini-3-pro-image-preview : With image generation

    Gemini 2.5 Series (Stable):
        - gemini-2.5-pro        : State-of-the-art thinking
        - gemini-2.5-flash      : Best price-performance, 1M ctx
        - gemini-2.5-flash-lite : 1.5x faster, lower cost
        - gemini-2.5-flash-image: Image generation

    Gemini 2.5 Specialized:
        - gemini-2.5-flash-preview-tts           : TTS low latency
        - gemini-2.5-pro-preview-tts             : TTS high quality
        - gemini-2.5-flash-native-audio-preview  : Live audio

    Gemini 2.0 Series (Previous):
        - gemini-2.0-flash      : Multimodal, 1M context
        - gemini-2.0-flash-lite : Fast variant

    Deprecated (retired Apr 2025):
        - gemini-1.5-pro, gemini-1.5-flash

    Recommended:
        - Flagship: gemini-2.5-pro
        - Fast: gemini-2.5-flash
        - Budget: gemini-2.5-flash-lite

Docs: https://ai.google.dev/gemini-api/docs/models
"""

from datetime import datetime
from google import genai
from google.genai import types
from aii_lib.telemetry import logger

from .pricing import calculate_cost
from .gem_to_json import (
    serialize_response,
    extract_thinking, extract_output, extract_usage
)
from .json_to_log_format import create_summary


class GeminiClient:
    """Async Gemini client using google-genai SDK with aiohttp.

    Supported Models (Dec 2025):
        Gemini 3: gemini-3-pro-preview
        Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
        Gemini 2.0: gemini-2.0-flash, gemini-2.0-flash-lite

    Recommended: gemini-2.5-pro (flagship), gemini-2.5-flash (fast)

    Docs: https://ai.google.dev/gemini-api/docs/models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        timeout: float = 600,
        config_path: str | None = None,
    ):
        # Load config defaults first
        from ..config import load_config, get_gemini_config

        if config_path:
            load_config(config_path)

        config = get_gemini_config()

        # Config defaults, overridden by explicit parameters
        self.model = model or config.get('default_model', 'gemini-2.5-flash')
        self.temperature = temperature if temperature is not None else config.get('temperature', 1.0)
        self.thinking_budget = thinking_budget if thinking_budget is not None else config.get('thinking_budget')

        # Create async client with aiohttp backend
        resolved_api_key = api_key or config.get('api_key')
        if not resolved_api_key:
            logger.warning("No Gemini API key provided; requests will likely fail")
        http_options = types.HttpOptions(
            async_client_args={},  # Enable aiohttp
            timeout=int(timeout * 1000),  # Convert seconds to milliseconds
        )
        client_kwargs = {"http_options": http_options}
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        self.client = genai.Client(**client_kwargs)

        logger.info(f"Gemini async client initialized with model: {self.model}")

    async def generate_content(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        tools: list | None = None,
        response_schema: type | dict | None = None,
        message_callback=None,
    ):
        """Generate content (async, no streaming).

        Args:
            prompt: The user prompt
            system_instruction: System prompt
            temperature: Override default temperature
            thinking_budget: Token budget for thinking
            tools: List of tools for function calling
            response_schema: Pydantic model class or JSON schema dict for structured output
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

        # Resolve params
        budget = thinking_budget if thinking_budget is not None else self.thinking_budget
        temp = temperature if temperature is not None else self.temperature

        # Build config
        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Build ThinkingConfig if budget is set
        if budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=True,
            )
        else:
            # Only use temperature when thinking is disabled
            config_kwargs["temperature"] = temp

        if tools:
            config_kwargs["tools"] = tools

        # Add structured output schema
        use_structured = False
        if response_schema:
            use_structured = True
            config_kwargs["response_mime_type"] = "application/json"
            if hasattr(response_schema, 'model_json_schema'):
                # Pydantic model - use response_schema directly
                config_kwargs["response_schema"] = response_schema
            else:
                # Dict schema - use response_json_schema
                config_kwargs["response_json_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Log system message
        if message_callback:
            system_text = f"{self.model}"
            if budget is not None:
                system_text += f" | Thinking: {budget} tokens"
            if use_structured:
                system_text += " | Structured output: enabled"

            message_callback({
                "type": "system",
                "message_text": system_text,
                "model": self.model,
                "iso_timestamp": datetime.now().isoformat(),
                "llm_provider": "gemini",
                "message_metadata": {
                    "model": self.model,
                    "thinking_budget": budget,
                    "structured_output": use_structured,
                },
                "raw_api_response": None,
            })

        # Make async request using client.aio
        if config:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        else:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
            )

        # Get raw response for logging
        raw_response = serialize_response(response)

        # Process response.candidates in order to preserve interleaving
        if message_callback and hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        part_raw = part.to_dict() if hasattr(part, 'to_dict') else {}

                        # Check for thinking part
                        if hasattr(part, 'thought') and part.thought:
                            if hasattr(part, 'text') and part.text:
                                message_callback({
                                    "type": "thinking",
                                    "message_text": part.text,
                                    "iso_timestamp": datetime.now().isoformat(),
                                    "llm_provider": "gemini",
                                    "message_metadata": part_raw,
                                    "raw_api_response": None,
                                })

                        # Check for function call
                        elif hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            func_name = getattr(fc, 'name', 'unknown')
                            func_args = dict(fc.args) if hasattr(fc, 'args') else {}
                            message_callback({
                                "type": "gem_func",
                                "message_text": f"Function: {func_name}",
                                "iso_timestamp": datetime.now().isoformat(),
                                "llm_provider": "gemini",
                                "message_metadata": {
                                    "name": func_name,
                                    "args": func_args,
                                    **part_raw,
                                },
                                "raw_api_response": None,
                            })

                        # Regular text part
                        elif hasattr(part, 'text') and part.text:
                            message_callback({
                                "type": "gem_msg",
                                "message_text": part.text,
                                "iso_timestamp": datetime.now().isoformat(),
                                "llm_provider": "gemini",
                                "message_metadata": part_raw,
                                "raw_api_response": None,
                            })

        # Log summary with full response
        if message_callback:
            usage = extract_usage(response)
            total_cost = calculate_cost(usage, self.model)

            message_callback({
                "type": "summary",
                "message_text": create_summary(response, self.model, usage=usage),
                "llm_provider": "gemini",
                "message_metadata": {
                    "total_cost": total_cost,
                    "usage": usage,
                    "full_response": raw_response,
                },
                "raw_api_response": raw_response,
                "iso_timestamp": datetime.now().isoformat(),
            })

        return response

    def extract_text_from_response(self, response) -> str:
        """Extract output text from response."""
        return extract_output(response)

    def extract_thinking_from_response(self, response) -> str:
        """Extract thinking from response."""
        return extract_thinking(response)

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        message_callback=None,
    ) -> bytes | None:
        """Generate an image using Gemini image generation models.

        Args:
            prompt: The image generation prompt
            model: Override the model (should be an image-capable model like gemini-3-pro-image-preview)
            message_callback: Callback for logging messages

        Returns:
            Image bytes (PNG) or None if generation failed
        """
        from datetime import datetime

        image_model = model or self.model

        # Log prompt
        if message_callback:
            message_callback({
                "type": "prompt",
                "message_text": f"[Image Generation] {prompt}",
                "iso_timestamp": datetime.now().isoformat(),
            })

        # Configure for image output
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=image_model,
                contents=prompt,
                config=config,
            )

            # Extract image from response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            # Check for inline_data (image bytes)
                            if hasattr(part, 'inline_data') and part.inline_data:
                                if message_callback:
                                    message_callback({
                                        "type": "gem_img",
                                        "message_text": f"Image generated ({len(part.inline_data.data)} bytes)",
                                        "iso_timestamp": datetime.now().isoformat(),
                                        "llm_provider": "gemini",
                                    })
                                return part.inline_data.data

            if message_callback:
                message_callback({
                    "type": "warning",
                    "message_text": "No image in response",
                    "iso_timestamp": datetime.now().isoformat(),
                })

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
        """Close the client (no-op for genai client, kept for API consistency)."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
