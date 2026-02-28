#!/usr/bin/env python
"""
Image Generation (nano_banana) - Generate research figures via Gemini API.

Uses google-genai SDK to call gemini-3-pro-image-preview directly for
image generation with aspect ratio and resolution (1K/2K/4K) control.

Usage (CLI via ability server):
    python aii_image_gen_nano_banana.py --prompt "Bar chart..." --output ./fig.png
    python aii_image_gen_nano_banana.py --prompt "Flowchart..." --style neurips

Usage (direct core function):
    from aii_image_gen_nano_banana import init_image_gen_nano_banana, core_image_gen_nano_banana
    init_image_gen_nano_banana()
    result = core_image_gen_nano_banana(prompt="...", output_path="./fig.png")
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
SERVER_NAME = "aii_image_gen_nano_banana"
DEFAULT_TIMEOUT = 180.0
REQUEST_TIMEOUT = 120       # Per-attempt timeout (seconds)
MAX_RETRIES = 3             # Total attempts before giving up
RETRY_BACKOFF_BASE = 2.0    # Exponential backoff: 2^attempt seconds

log = logging.getLogger("aii_image_gen_nano_banana")

MODEL = "gemini-3-pro-image-preview"

NEURIPS_STYLE = (
    "Clean white background, no borders or decorative elements. "
    "Sans-serif font labels (Helvetica/Arial style), clearly readable at print size. "
    "Properly formatted axes with labeled tick marks. "
    "Minimal gridlines (light gray, dotted if needed). "
    "No 3D effects, no shadows, no gradients. "
    "Proportions suitable for a two-column NeurIPS paper layout."
)

VALID_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]

VALID_IMAGE_SIZES = ["1K", "2K", "4K"]


# =============================================================================
# Gemini client
# =============================================================================

_client = None


def init_image_gen_nano_banana():
    """Initialize Gemini API client."""
    global _client
    from google import genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in .env or environment")

    _client = genai.Client(api_key=GEMINI_API_KEY)
    log.info("Gemini client initialized for image generation")


def _resolution_fallback_order(requested: str) -> list[str]:
    """Build resolution fallback chain: requested → higher → lower.

    Prefers higher resolutions before falling back to lower ones.
    Always ends with '1K' as last resort.
    """
    all_sizes = ["4K", "2K", "1K"]
    requested = requested.upper() if requested else "1K"
    chain = [requested]
    # Add remaining sizes: prefer higher first, then lower
    for s in all_sizes:
        if s not in chain:
            chain.append(s)
    return chain


def _try_generate(client, contents, aspect_ratio, image_size):
    """Single generation attempt with retries at one resolution.

    Returns (result_dict, last_error) — result_dict is None on failure.
    """
    from google.genai import types as gx

    image_config_kwargs = {}
    if aspect_ratio and aspect_ratio in VALID_ASPECT_RATIOS:
        image_config_kwargs["aspect_ratio"] = aspect_ratio
    if image_size and image_size.upper() in VALID_IMAGE_SIZES:
        image_config_kwargs["image_size"] = image_size.upper()

    config_kwargs = {"response_modalities": ["TEXT", "IMAGE"]}
    if image_config_kwargs:
        config_kwargs["image_config"] = gx.ImageConfig(**image_config_kwargs)

    gen_config = gx.GenerateContentConfig(**config_kwargs)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=gen_config,
            )

            # Extract image bytes from response
            img_bytes = None
            text_content = ""

            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                parts = getattr(candidates[0].content, "parts", [])
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_content += part.text
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and hasattr(inline_data, "data") and inline_data.data:
                        img_bytes = inline_data.data
                        break

            if not img_bytes:
                last_error = "No image data in response"
                log.warning(f"[{image_size}] Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue

            # Success
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

            return {
                "img_bytes": img_bytes,
                "text_content": text_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "attempts": attempt,
                "image_size_used": image_size,
            }, None

        except Exception as e:
            last_error = str(e)
            log.warning(f"[{image_size}] Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

    return None, last_error


def core_image_gen_nano_banana(**kwargs) -> dict:
    """Generate an image via Gemini API (gemini-3-pro-image-preview).

    If generation fails at the requested resolution, falls back through
    other resolutions (preferring higher) before returning an error.

    Args:
        prompt: Image description with all details.
        output_path: Where to save the generated image.
        aspect_ratio: Canvas shape (e.g., '16:9', '4:3', '1:1').
        image_size: Resolution: '1K', '2K', '4K' (default: '1K').
        negative_prompt: Things to exclude from the image.
        style: Preset style ('neurips' appends academic style).
        system_instruction: System-level style guidance.

    Returns:
        Dict with success, output_path, model, dimensions, and metadata.
    """
    global _client

    prompt = kwargs.get("prompt", "")
    output_path = kwargs.get("output_path", "./generated_image.png")
    aspect_ratio = kwargs.get("aspect_ratio", "16:9")
    image_size = kwargs.get("image_size", "1K")
    negative_prompt = kwargs.get("negative_prompt")
    style = kwargs.get("style")
    system_instruction = kwargs.get("system_instruction")

    if not GEMINI_API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY not set"}

    if not prompt:
        return {"success": False, "error": "Prompt is required"}

    if _client is None:
        init_image_gen_nano_banana()

    # Build full prompt with style/negative
    full_prompt = prompt
    if style == "neurips":
        full_prompt = f"{prompt}\n\nStyle: {NEURIPS_STYLE}"
    if negative_prompt:
        full_prompt = f"{full_prompt}\n\nAvoid: {negative_prompt}"

    # Build contents list
    contents = []
    if system_instruction:
        contents.append(system_instruction)
    elif style == "neurips":
        contents.append("You are a scientific figure generator. Produce clean, publication-ready charts and diagrams.")
    contents.append(full_prompt)

    # Try requested resolution first, then fall back through others
    fallback_chain = _resolution_fallback_order(image_size)
    last_error = None

    for size in fallback_chain:
        if size != fallback_chain[0]:
            log.info(f"Falling back from {fallback_chain[0]} to {size}")

        result, err = _try_generate(_client, contents, aspect_ratio, size)

        if result is not None:
            # Success — save to disk
            img_bytes = result["img_bytes"]
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)

            dimensions = ""
            try:
                from PIL import Image
                with Image.open(out_path) as img:
                    dimensions = f"{img.width}x{img.height}"
            except Exception:
                pass

            return {
                "success": True,
                "output_path": str(out_path.resolve()),
                "model": MODEL,
                "dimensions": dimensions,
                "aspect_ratio": aspect_ratio,
                "image_size": result["image_size_used"],
                "image_size_requested": image_size,
                "prompt_length": len(full_prompt),
                "image_bytes": len(img_bytes),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "text_response": result["text_content"][:200] if result["text_content"] else "",
                "attempts": result["attempts"],
                "output": f"Image saved: {output_path} ({len(img_bytes)} bytes, {dimensions})",
            }

        last_error = err

    sizes_tried = ", ".join(fallback_chain)
    return {"success": False, "error": f"All resolutions failed ({sizes_tried}). Last error: {last_error}"}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate research figures via Gemini API (ability server)",
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Full image description (include all data values, labels, style)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./generated_image.png",
        help="Output file path (default: ./generated_image.png)",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=VALID_ASPECT_RATIOS,
        help="Canvas aspect ratio (default: 16:9)",
    )
    parser.add_argument(
        "--image-size",
        default="1K",
        choices=VALID_IMAGE_SIZES,
        help="Image resolution (default: 1K)",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Things to exclude from the image",
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=["neurips"],
        help="Preset style (neurips = academic paper style)",
    )
    parser.add_argument(
        "--system",
        default=None,
        dest="system_instruction",
        help="System instruction for style guidance",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "prompt": args.prompt,
        "output_path": args.output,
        "aspect_ratio": args.aspect_ratio,
        "image_size": args.image_size,
        "negative_prompt": args.negative_prompt,
        "style": args.style,
        "system_instruction": args.system_instruction,
    }, timeout=args.timeout)

    if result is None:
        print("Error: Ability service not available.", file=sys.stderr)
        sys.exit(1)

    if result.get("success"):
        print(result.get("output", ""))
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
