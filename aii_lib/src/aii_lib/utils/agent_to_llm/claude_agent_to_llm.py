"""ClaudeAgentToLLMStructOut - Use Claude Agent as an LLM with structured JSON output.

Wraps the Claude Agent SDK to behave like an LLM client that outputs
structured JSON files matching Pydantic schemas.

Uses aii_lib for:
- Agent: Claude Code SDK wrapper
- AgentOptions: Agent configuration
- AgentInitializer/AgentFinalizer: Pre/post-agent utilities
- AIITelemetry: Task tracking

Usage:
    from pydantic import BaseModel
    from aii_lib.utils import ClaudeAgentToLLMStructOut

    class MySchema(BaseModel):
        title: str
        score: float

    async with ClaudeAgentToLLMStructOut(
        schema=MySchema,
        output_file="result.json",
        cwd="/my/project",
        system_prompt="You are a helpful assistant.",
        telemetry=telemetry,
        task_id="gen-1",
        task_name="gen-1",
    ) as agent:
        result = await agent.run("Analyze this code")
        print(result.data)  # Validated dict
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Type, ClassVar

from pydantic import BaseModel, ValidationError

from aii_lib.agent_backend import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AgentResponse

if TYPE_CHECKING:
    from aii_lib.telemetry import AIITelemetry


@dataclass
class ClaudeAgentToLLMStructOutResult:
    """Result from ClaudeAgentToLLMStructOut execution."""
    data: dict  # Validated JSON data
    raw_response: AgentResponse  # Full agent response
    output_path: Path  # Where the file was written
    attempts: int  # Number of attempts needed
    total_cost: float  # Total cost across all attempts


OUTPUT_INSTRUCTION_TEMPLATE = '''

---

Output the result as JSON to: `{output_file}`

JSON Schema:
```json
{schema_json}
```

IMPORTANT: This task is NOT complete until you use the Write tool to create `{output_file}`.
'''


FEEDBACK_TEMPLATE = '''
The output file has validation errors:

{errors}

Fix `{output_file}` to match the schema. All fields are required unless marked optional.
'''.strip()


class ClaudeAgentToLLMStructOut:
    """Use Claude Agent as an LLM with structured JSON output.

    This class wraps the Claude Agent SDK to:
    1. Build prompts that ensure file output
    2. Validate output against Pydantic schema
    3. Retry with feedback if validation fails

    Uses the same telemetry pattern as artifact executors (dataset, experiment, evaluation).

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from aii_lib.utils import ClaudeAgentToLLMStructOut
        >>>
        >>> class Analysis(BaseModel):
        ...     summary: str
        ...     score: float = Field(ge=0, le=1)
        >>>
        >>> async with ClaudeAgentToLLMStructOut(
        ...     schema=Analysis,
        ...     output_file="analysis.json",
        ...     cwd="/my/project",
        ...     system_prompt="You are a code reviewer.",
        ...     telemetry=telemetry,
        ...     task_id="analyze-1",
        ...     task_name="analyze-1",
        ... ) as agent:
        ...     result = await agent.run("Analyze the code quality")
        ...     print(result.data["summary"])
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        output_file: str = "./output.json",
        cwd: str | Path = "./",
        model: str = "claude-sonnet-4-5",
        max_turns: int = 50,
        max_retries: int = 2,
        timeout_seconds: int = 3600,
        system_prompt: str | None = None,
        mcp_servers: dict | None = None,
        # Telemetry integration (same as artifact executors)
        telemetry: "AIITelemetry | None" = None,
        task_id: str | None = None,
        task_name: str | None = None,
        task_sequence: int | None = None,
        json_log_path: str | None = None,
    ):
        """Initialize ClaudeAgentToLLMStructOut.

        Args:
            schema: Pydantic model class for output validation
            output_file: Relative path for JSON output (e.g., "./result.json")
            cwd: Working directory for agent execution
            model: Claude model (sonnet, opus, haiku)
            max_turns: Max turns per attempt
            max_retries: Max retry attempts on validation failure
            timeout_seconds: Timeout for agent execution (default: 3600)
            system_prompt: System prompt for the agent
            mcp_servers: Optional MCP server config for custom tools (e.g., ToolUniverse)
            telemetry: AIITelemetry instance for logging
            task_id: Task ID for telemetry sequencing
            task_name: Task name for display in logs
            task_sequence: Optional sequence number for ordering
            json_log_path: Optional path for agent message JSON log
        """
        self.schema = schema
        self.output_file = output_file
        self.cwd = Path(cwd).resolve()
        self.model = model
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        self.mcp_servers = mcp_servers

        # Telemetry integration
        self.telemetry = telemetry
        self.task_id = task_id
        self.task_name = task_name
        self.task_sequence = task_sequence
        self.json_log_path = json_log_path

        # Create initializer and finalizer (same pattern as artifact executors)
        self._initializer = AgentInitializer(
            telemetry=telemetry,
            task_id=task_id,
            task_name=task_name,
        )
        self._finalizer = AgentFinalizer(
            telemetry=telemetry,
            task_id=task_id,
            task_name=task_name,
        )

        self._agent: Agent | None = None

    def _build_prompt(self, prompt: str) -> str:
        """Append output instructions to the original prompt."""
        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)

        # Keep original prompt intact, just append output instructions
        output_instructions = OUTPUT_INSTRUCTION_TEMPLATE.format(
            output_file=self.output_file,
            schema_json=schema_json,
        )

        return prompt + output_instructions

    def _build_feedback(self, errors: str) -> str:
        """Build feedback prompt for retry."""
        return FEEDBACK_TEMPLATE.format(
            errors=errors,
            output_file=self.output_file,
        )

    def _validate_output(self) -> tuple[bool, str, dict | None]:
        """Validate the output file against schema.

        Returns:
            (is_valid, error_message, data)
        """
        output_path = self.cwd / self.output_file.lstrip("./")

        if not output_path.exists():
            return False, f"File `{self.output_file}` not found. Use Write tool to create it.", None

        try:
            content = output_path.read_text()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None

        try:
            self.schema(**data)
            return True, "", data
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"- {loc}: {err['msg']}")
            return False, "\n".join(errors), data

    def _emit(self, message: str, level: str = "INFO") -> None:
        """Emit message to telemetry if available."""
        if self.telemetry and self.task_id and self.task_name:
            self.telemetry.emit_message(level, message, self.task_name, self.task_id)

    def _get_agent_options(self) -> AgentOptions:
        """Build AgentOptions for execution."""
        options = AgentOptions(
            model=self.model,
            cwd=self.cwd,
            max_turns=self.max_turns,
            seq_prompt_timeout=self.timeout_seconds,
            permission_mode="bypassPermissions",
            system_prompt=self.system_prompt,
            continue_seq_item=True,  # Continue conversation between prompts
            json_log_path=self.json_log_path,
            # AIITelemetry integration for sequenced parallel execution
            telemetry=self.telemetry,
            run_id=self.task_id,
            agent_context=self.task_name,  # Display name for logs
        )
        if self.mcp_servers:
            options.mcp_servers = self.mcp_servers
        return options

    async def run(self, prompt: str) -> ClaudeAgentToLLMStructOutResult:
        """Execute task and return validated JSON output.

        Args:
            prompt: The user prompt (task description)

        Returns:
            ClaudeAgentToLLMStructOutResult with validated data and metadata

        Raises:
            ValueError: If validation fails after all retries
            asyncio.TimeoutError: If execution times out
        """
        # Start task (must be before any emit_message calls)
        self._initializer.start_task(sequence=self.task_sequence)

        # Log execution info
        self._emit(f"Executing with schema: {self.schema.__name__}")
        self._emit(f"Model: claude-{self.model}")
        self._emit(f"Workspace: {self.cwd}")

        # Build initial prompt
        full_prompt = self._build_prompt(prompt)

        # Create agent
        options = self._get_agent_options()
        self._agent = Agent(options)

        total_cost = 0.0
        attempts = 0

        try:
            for attempt in range(self.max_retries + 1):
                attempts += 1

                if attempt == 0:
                    # Initial attempt - new conversation
                    self._emit(f"Running initial prompt (attempt {attempts})")
                    response = await self._agent.run(full_prompt)
                else:
                    # Retry with feedback - continues same conversation
                    self._emit(f"Retry {attempt}/{self.max_retries} - validation failed", "WARN")
                    feedback = self._build_feedback(error_msg)
                    response = await self._agent.run(feedback)

                total_cost += response.total_cost

                # Validate
                is_valid, error_msg, data = self._validate_output()

                if is_valid:
                    self._emit(
                        f"Valid output on attempt {attempts}, ${total_cost:.4f}",
                        "SUCCESS"
                    )
                    self._finalizer.end_task_success(cost=total_cost)
                    return ClaudeAgentToLLMStructOutResult(
                        data=data,
                        raw_response=response,
                        output_path=self.cwd / self.output_file.lstrip("./"),
                        attempts=attempts,
                        total_cost=total_cost,
                    )

                if attempt < self.max_retries:
                    self._emit(f"Validation error: {error_msg[:100]}", "WARN")

            # All retries exhausted
            error = f"Failed after {attempts} attempts. Last error: {error_msg}"
            self._emit(error, "ERROR")
            self._finalizer.end_task_failure(error, cost=total_cost)
            raise ValueError(f"ClaudeAgentToLLMStructOut: {error}")

        except asyncio.TimeoutError:
            self._finalizer.end_task_timeout(self.timeout_seconds)
            raise

        except Exception as e:
            if not isinstance(e, ValueError):  # Don't double-report validation failures
                self._finalizer.end_task_error(str(e))
            raise

    async def close(self):
        """Cleanup resources."""
        self._agent = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
