"""
Response models and result types.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """
    Token usage and cost information for a prompt execution.

    Attributes:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cache_creation_input_tokens: Tokens used for cache creation
        cache_read_input_tokens: Tokens read from cache
        total_cost: Total cost in USD for this usage
        raw_usage: Complete raw usage dictionary from Claude API
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_cost: float = 0.0
    raw_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptResult:
    """
    Result from executing a single prompt.

    Attributes:
        response: Claude's text response for this prompt
        session_id: Session ID for this execution
        cost: Cost in USD for this prompt
        prompt_index: Index of this prompt in the sequence (0-based)
        usage: Token usage and cost details
        messages: All logged messages for this prompt (same as JSON log)
        summary_data: Raw summary data for aggregation (when emit_summary=False)
        num_turns: Number of conversation turns used (for max_turns detection)
    """
    response: str
    session_id: str
    cost: float
    prompt_index: int
    usage: TokenUsage = field(default_factory=TokenUsage)
    messages: list[dict[str, Any]] = field(default_factory=list)  # All logged messages
    summary_data: dict[str, Any] = field(default_factory=dict)  # For aggregation
    num_turns: int = 0  # Number of turns used (for max_turns exceeded detection)
    structured_output: dict[str, Any] | None = None  # SDK native structured output (when output_format used)


@dataclass
class AgentResponse:
    """
    Complete response from agent execution.

    Attributes:
        final_response: Claude's final text response (from last prompt)
        total_cost: Total cost in USD across all prompts
        prompt_results: List of results for each prompt executed
        final_session_id: Session ID for resuming or forking
        all_messages: All logged messages across all prompts (same as JSON log)
        json_log_path: Path where messages were saved (if logging enabled)
        structured_output: Structured output data (when output_format used)
        expected_files_valid: True if all expected files exist (when expected_files_struct_out_field is set)
        failed: True if agent failed after all retries (response will be empty)
        error_message: Human-readable error description when failed=True
    """
    final_response: str
    total_cost: float
    prompt_results: list[PromptResult]
    final_session_id: str
    all_messages: list[dict[str, Any]] = field(default_factory=list)  # All messages from all prompts
    json_log_path: str | None = None  # Where the JSON log was saved
    structured_output: dict[str, Any] | None = None  # SDK native structured output (when output_format used)
    expected_files_valid: bool = True  # True if all expected files exist (or no expected_files_struct_out_field)
    failed: bool = False  # True if agent failed after all retries
    error_message: str | None = None  # Human-readable error when failed=True
