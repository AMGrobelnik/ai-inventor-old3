"""
Low-level streaming executor wrapping ClaudeSDKClient
"""

from typing import AsyncGenerator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk._errors import ProcessError


class AgentProcessError(Exception):
    """Raised when the Claude agent subprocess terminates unexpectedly.

    This wraps ProcessError to provide a cleaner exception that can be caught
    and retried by the agent retry logic.
    """
    pass


class SubscriptionAccessError(Exception):
    """Raised when the Claude subscription/access is unavailable.

    Detected from SDK messages with error="authentication_failed" or
    text matching "does not have access to Claude". The agent should
    poll and wait (like threshold exceeded) rather than burn retries.
    """
    pass


class StreamingExecutor:
    """
    Low-level executor that wraps ClaudeSDKClient.
    Handles the streaming communication with Claude Agent SDK.
    """

    def __init__(self, sdk_options: ClaudeAgentOptions):
        """
        Initialize the executor.

        Args:
            sdk_options: ClaudeAgentOptions for the SDK
        """
        self.sdk_options = sdk_options

    async def _create_user_message_generator(self, prompt: str):
        """Create a generator that yields a single user message"""
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": prompt
            }
        }

    async def execute(self, prompt: str) -> AsyncGenerator:
        """
        Execute a single prompt and yield messages as they arrive.

        Args:
            prompt: The prompt to send to Claude

        Yields:
            Messages from the SDK (AssistantMessage, ResultMessage, etc.)

        Raises:
            AgentProcessError: If the subprocess terminates unexpectedly (can be retried)
        """
        try:
            async with ClaudeSDKClient(options=self.sdk_options) as client:
                # Send the message
                await client.query(self._create_user_message_generator(prompt))

                # Stream responses
                async for message in client.receive_response():
                    yield message
        except ProcessError as e:
            # Subprocess terminated (SIGTERM, SIGKILL, etc.)
            # Wrap in a cleaner exception that the retry logic can handle
            raise AgentProcessError(
                f"Agent subprocess terminated unexpectedly: {e}"
            ) from e
        except GeneratorExit:
            # Generator was closed (e.g., due to timeout or cancellation)
            # This is normal during cleanup, don't raise
            pass
