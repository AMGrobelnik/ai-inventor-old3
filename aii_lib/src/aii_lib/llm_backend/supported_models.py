"""
Supported Models Reference (December 2025)

This file documents available models for each LLM provider.
Use these model IDs when initializing clients.

Sources:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
- Google: https://ai.google.dev/gemini-api/docs/models
- OpenRouter: https://openrouter.ai/models
"""

# =============================================================================
# OPENAI MODELS
# =============================================================================
# Docs: https://platform.openai.com/docs/models

OPENAI_MODELS = {
    # GPT-5 Series (Latest - Dec 2025)
    "gpt-5.2": "Latest flagship model (Dec 2025)",
    "gpt-5.2-pro": "Pro version with extended reasoning",
    "gpt-5.1": "Previous flagship (Nov 2025)",
    "gpt-5.1-codex": "Code-optimized GPT-5.1",
    "gpt-5": "GPT-5 base (Aug 2025)",
    "gpt-5-mini": "Fast, cost-efficient GPT-5",
    "gpt-5-nano": "Smallest GPT-5 variant",
    "gpt-5-pro": "Pro reasoning mode",

    # O-Series Reasoning Models
    "o4-mini": "Fast reasoning model (best on AIME 2024/2025)",
    "o3": "Complex reasoning model",
    "o3-mini": "Small reasoning model",
    "o3-pro": "Extended compute reasoning",
    "o1": "Previous reasoning model",
    "o1-mini": "DEPRECATED - use o3-mini or o4-mini",

    # GPT-4 Series (Legacy)
    "gpt-4.1": "Improved GPT-4o (Apr 2025)",
    "gpt-4.1-mini": "Fast GPT-4.1 variant",
    "gpt-4.1-nano": "Smallest GPT-4.1",
    "gpt-4o": "Legacy multimodal model",
    "gpt-4o-mini": "Legacy fast model",
    "gpt-4o-audio-preview": "Audio capabilities",
    "gpt-4o-search-preview": "Web search enabled",
}

# Recommended defaults
OPENAI_RECOMMENDED = {
    "flagship": "gpt-5.2",
    "fast": "gpt-5-mini",
    "reasoning": "o4-mini",
    "coding": "gpt-5.1-codex",
    "budget": "gpt-5-nano",
}


# =============================================================================
# ANTHROPIC (CLAUDE) MODELS
# =============================================================================
# Docs: https://docs.anthropic.com/en/docs/about-claude/models

ANTHROPIC_MODELS = {
    # Claude 4.5 Series (Latest)
    "claude-opus-4-5-20251101": "Premium model, max intelligence (Aug 2025 training)",
    "claude-sonnet-4-5-20250929": "Best balance of speed/intelligence",
    "claude-haiku-4-5-20251001": "Fastest, near-frontier intelligence",

    # Aliases (auto-update to latest snapshot)
    "claude-opus-4-5": "Alias -> claude-opus-4-5-20251101",
    "claude-sonnet-4-5": "Alias -> claude-sonnet-4-5-20250929",
    "claude-haiku-4-5": "Alias -> claude-haiku-4-5-20251001",

    # Claude 4 Series (Legacy but available)
    "claude-opus-4-1-20250805": "Previous Opus with extended thinking",
    "claude-opus-4-20250514": "Claude 4 Opus",
    "claude-sonnet-4-20250514": "Claude 4 Sonnet",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",

    # Claude 3.5 Series (Legacy)
    "claude-3-5-haiku-20241022": "Fast Claude 3.5",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",

    # Claude 3 Series (Legacy)
    "claude-3-haiku-20240307": "Fastest legacy model",
    "claude-3-sonnet-20240229": "Legacy balanced model",
    "claude-3-opus-20240229": "Legacy premium model",
}

# Recommended defaults
ANTHROPIC_RECOMMENDED = {
    "flagship": "claude-opus-4-5-20251101",
    "balanced": "claude-sonnet-4-5-20250929",
    "fast": "claude-haiku-4-5-20251001",
    "budget": "claude-3-5-haiku-20241022",
}


# =============================================================================
# GOOGLE GEMINI MODELS
# =============================================================================
# Docs: https://ai.google.dev/gemini-api/docs/models

GEMINI_MODELS = {
    # Gemini 3 Series (Latest)
    "gemini-3-pro-preview": "Latest Gemini 3 Pro (preview)",
    "gemini-3-flash-preview": "High-speed reasoning, 1M context, agentic workflows",
    "gemini-3-pro-image-preview": "Gemini 3 Pro with image generation",

    # Gemini 2.5 Series (Stable)
    "gemini-2.5-pro": "State-of-the-art thinking model",
    "gemini-2.5-flash": "Best price-performance, 1M context",
    "gemini-2.5-flash-lite": "1.5x faster than 2.0 Flash, lower cost",
    "gemini-2.5-flash-image": "Image generation capabilities",

    # Gemini 2.5 Previews
    "gemini-2.5-flash-preview-09-2025": "Flash preview (Sep 2025)",
    "gemini-2.5-flash-lite-preview-09-2025": "Flash-Lite preview",
    "gemini-2.5-pro-preview-06-05": "Pro preview (Jun 2025)",

    # Gemini 2.5 Specialized
    "gemini-2.5-flash-preview-tts": "Text-to-speech (low latency)",
    "gemini-2.5-pro-preview-tts": "Text-to-speech (high quality)",
    "gemini-2.5-flash-native-audio-preview-12-2025": "Live audio/dialogue",

    # Gemini 2.0 Series (Previous gen)
    "gemini-2.0-flash": "Multimodal, 1M context",
    "gemini-2.0-flash-001": "Stable 2.0 Flash",
    "gemini-2.0-flash-lite": "Fast 2.0 variant",
    "gemini-2.0-flash-lite-001": "Stable 2.0 Flash-Lite",
    "gemini-2.0-flash-exp": "Experimental features",
    "gemini-2.0-flash-preview-image-generation": "Image generation",

    # Deprecated (retired Apr 2025)
    # "gemini-1.5-pro": "DEPRECATED",
    # "gemini-1.5-flash": "DEPRECATED",
}

# Recommended defaults
GEMINI_RECOMMENDED = {
    "flagship": "gemini-2.5-pro",
    "fast": "gemini-2.5-flash",
    "budget": "gemini-2.5-flash-lite",
    "preview": "gemini-3-pro-preview",
}


# =============================================================================
# OPENROUTER MODELS (300+ models from multiple providers)
# =============================================================================
# Docs: https://openrouter.ai/models

OPENROUTER_POPULAR = {
    # OpenAI via OpenRouter
    "openai/gpt-5.2": "OpenAI GPT-5.2",
    "openai/gpt-5-mini": "OpenAI GPT-5 Mini",
    "openai/o4-mini": "OpenAI o4-mini reasoning",
    "openai/o3": "OpenAI o3 reasoning",

    # Anthropic via OpenRouter
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",

    # Google via OpenRouter
    "google/gemini-3-flash-preview": "Gemini 3 Flash Preview (high-speed reasoning)",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",

    # Meta Llama
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "meta-llama/llama-3.1-405b-instruct": "Llama 3.1 405B",

    # Mistral
    "mistralai/mistral-large-latest": "Mistral Large",
    "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",

    # DeepSeek
    "deepseek/deepseek-r1": "DeepSeek R1 reasoning",
    "deepseek/deepseek-chat": "DeepSeek Chat",
}

OPENROUTER_RECOMMENDED = {
    "flagship": "openai/gpt-5.2",
    "balanced": "anthropic/claude-sonnet-4.5",
    "fast": "google/gemini-2.5-flash",
    "budget": "deepseek/deepseek-chat",
    "reasoning": "openai/o4-mini",
    "open_source": "meta-llama/llama-3.3-70b-instruct",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_models(provider: str) -> dict:
    """Get available models for a provider."""
    providers = {
        "openai": OPENAI_MODELS,
        "anthropic": ANTHROPIC_MODELS,
        "gemini": GEMINI_MODELS,
        "openrouter": OPENROUTER_POPULAR,
    }
    return providers.get(provider.lower(), {})


def get_recommended(provider: str) -> dict:
    """Get recommended models for a provider."""
    recommendations = {
        "openai": OPENAI_RECOMMENDED,
        "anthropic": ANTHROPIC_RECOMMENDED,
        "gemini": GEMINI_RECOMMENDED,
        "openrouter": OPENROUTER_RECOMMENDED,
    }
    return recommendations.get(provider.lower(), {})


def print_models(provider: str | None = None):
    """Print available models for one or all providers."""
    if provider:
        models = get_models(provider)
        print(f"\n{provider.upper()} Models:")
        for model_id, desc in models.items():
            print(f"  {model_id}: {desc}")
    else:
        for p in ["openai", "anthropic", "gemini", "openrouter"]:
            print_models(p)
