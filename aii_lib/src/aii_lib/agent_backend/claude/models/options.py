"""
Configuration options for Claude Agent SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
from pathlib import Path

from ..utils.constants import Defaults, PermissionModeValue
from .enums import SessionType, SystemPromptPreset


@dataclass
class ExpectedFile:
    """
    A single expected output file with its description.

    Attributes:
        path: File path relative to agent cwd (e.g., "data.py", "output/result.json")
        description: What the file should contain/purpose (shown in prompt)
    """
    path: str
    description: str = ""


@dataclass
class AgentOptions:
    """
    Configuration options for Claude Agent SDK.
    Exposes ALL available ClaudeAgentOptions parameters.
    """
    # Core options
    allowed_tools: list[str] | None = None  # None = use SDK defaults
    system_prompt: str | SystemPromptPreset | None = "claude_code"  # String, preset dict, or None
    permission_mode: PermissionModeValue = Defaults.PERMISSION_MODE  # Bypass all permissions by default
    max_turns: int = Defaults.MAX_TURNS  # Maximum conversation turns
    model: str = Defaults.MODEL  # Claude model
    effort: Literal["low", "medium", "high", "max"] | None = None  # Token effort level (None = SDK default "high"). Passed as --effort CLI flag.
    cwd: str | Path = "./"  # Working directory

    # Session management
    session_type: SessionType = SessionType.NEW  # NEW, RESUME, or FORK
    resume_session_id: str | None = None  # Session ID to resume/fork from
    continue_seq_item: bool = Defaults.CONTINUE_SEQ_ITEM  # Continue conversation for 2nd+ prompts in sequence

    # Tool restrictions (rarely used)
    disallowed_tools: list[str] | None = None

    # MCP servers
    mcp_servers: dict[str, Any] | str | Path = field(default_factory=dict)

    # Custom tools (file paths to Python files with @tool decorated functions)
    custom_tool_files: list[str | Path] = field(default_factory=list)

    # Custom agents (file paths to YAML files with agent definitions)
    custom_agent_files: list[str | Path] = field(default_factory=list)

    # Agent-level execution options (entire agent run)
    agent_timeout: int | None = None  # Timeout for entire agent run in seconds (None = no timeout)
    agent_retries: int = 2  # Max retry attempts for entire agent on failure/timeout

    # Per-prompt execution options (single prompt within agent)
    seq_prompt_timeout: int | None = None  # Timeout per prompt in seconds (None = no timeout)
    seq_prompt_retries: int = 5  # Max retry attempts per prompt on failure/timeout

    # Per-message execution options (individual SDK message within streaming loop)
    # When a single message hangs past message_timeout, raises MessageTimeoutError
    # which is retried up to message_retries times (separate budget from seq_prompt_retries).
    # When message_retries exhausted, escalates as asyncio.TimeoutError to seq_prompt retry.
    message_timeout: int | None = 300  # Timeout per SDK message in seconds (None = no timeout, default 5 min)
    message_retries: int = 3  # Max fork+resume attempts for message-level timeouts
    json_log_path: str | Path | None = None  # Path to JSONL log file (default: cwd/all_messages.jsonl)
    log_mode: Literal["full", "truncated", "none"] | None = None  # Console logging: None = use config, full = no truncation, truncated = use config, none = no output
    custom_metadata: dict[str, Any] = field(default_factory=dict)  # Custom fields to add to every message

    # SDK native structured output
    # Pass {"type": "json_schema", "schema": <json_schema_dict>} to enable
    # SDK handles validation and retry internally — no file I/O needed
    output_format: dict[str, Any] | None = None

    # Custom force output prompt (sent when max_turns is exceeded without structured output)
    # If None, uses the generic "STOP and output NOW" template
    force_output_prompt: str | None = None

    # Expected files validation via structured output (optional, off by default)
    # Set expected_files_struct_out_field to enable automatic file existence validation.
    # The agent reports created file paths in this structured output field.
    # SDK recursively extracts all string paths and validates they exist inside workspace.
    # Requires output_format to be set with a JSON schema that includes this field.
    expected_files_struct_out_field: str | None = None  # Field name in structured output (e.g., "expected_files"); None = disabled
    max_expected_files_retries: int = 2  # Max retries for missing files (only used when expected_files_struct_out_field is set)

    # AIITelemetry integration (for sequenced parallel execution)
    telemetry: Any = None  # Parent AIITelemetry instance (if None, creates new one)
    run_id: str | None = None  # Task/run ID for message sequencing (required when telemetry is provided)
    agent_context: str | None = None  # Display name for logs (e.g., "data-0", "exp-1", "comp-6")

    # Advanced options
    permission_prompt_tool_name: str | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=lambda: {"strict-mcp-config": None})  # --strict-mcp-config: only use MCP servers from mcp_servers param, ignore cloud/project MCPs
    max_buffer_size: int | None = None
    debug_stderr: Any = None
    can_use_tool: Any = None  # Callable for custom tool permissions
    hooks: dict[str, list[Any]] | None = None
    user: str | None = None
    include_partial_messages: bool = True
    setting_sources: list[Literal['user', 'project', 'local']] = field(default_factory=list)  # Empty by default to prevent .mcp.json auto-loading; set ["project"] explicitly when skills are needed

    # Resource selection (auto-prepared to workspace)
    selected_agents: list[Any] = field(default_factory=list)  # Agent names or AgentDefinition objects
    selected_mcps: list[Any] = field(default_factory=list)  # MCP names or McpDefinition objects

    # Internal: SDK agent definitions (populated from selected_agents during initialization)
    agents: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AgentOptions":
        """
        Load AgentOptions from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            AgentOptions instance with loaded configuration
        """
        import yaml
        from pathlib import Path as PathLib

        config_path = PathLib(config_path)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Convert session_type string to enum
        if "session_type" in config and isinstance(config["session_type"], str):
            config["session_type"] = SessionType(config["session_type"])

        # Convert YAML config to AgentOptions fields
        return cls(**config)

    def to_serializable_dict(self) -> dict[str, Any]:
        """
        Convert AgentOptions to a JSON-serializable dict.

        Excludes non-serializable fields (telemetry, callbacks, etc.).

        Returns:
            Dict suitable for JSON serialization and AgentOptions(**dict) reconstruction
        """
        # Fields that cannot be serialized (callables, complex objects, internal state)
        non_serializable_fields = {
            "telemetry",
            "debug_stderr",
            "can_use_tool",
            "hooks",
            "agents",  # Internal: populated during initialization
        }

        result = {}
        for field_name in self.__dataclass_fields__:
            if field_name in non_serializable_fields:
                continue

            value = getattr(self, field_name)

            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)
            # Convert lists of Paths
            elif isinstance(value, list):
                value = [str(v) if isinstance(v, Path) else v for v in value]
            # Convert Enum to string
            elif hasattr(value, "value") and hasattr(value, "name"):
                value = value.value

            result[field_name] = value

        return result
