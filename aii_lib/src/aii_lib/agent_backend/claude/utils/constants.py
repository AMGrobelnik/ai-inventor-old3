"""
Constants for aii_lib agent backend.

All magic strings and numbers live here for easy discovery and modification.
"""
from enum import Enum
from typing import Literal


# ==================== TOOL NAMES ====================
class ToolName(Enum):
    """
    Tool name identifiers used for message sequencing and logging.

    These appear in the 'tool_name' field of tool-related messages.
    """
    # Task tool (subagent invocation)
    TASK_IN = "TASK_IN"
    TASK_OUT = "TASK_OUT"

    # Bash tool
    BASH_IN = "BASH_IN"
    BASH_OUT = "BASH_OUT"

    # File tools
    READ = "Read"
    WRITE = "Write"
    EDIT = "Edit"

    # Search tools
    GREP = "Grep"
    GLOB = "Glob"

    # Web tools
    WEB_SEARCH = "WebSearch"
    WEB_FETCH = "WebFetch"


# ==================== PERMISSION MODES ====================
class PermissionMode(Enum):
    """Permission modes for agent execution"""
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_ALL = "bypassPermissions"


PermissionModeValue = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


# ==================== DEFAULT VALUES ====================
class Defaults:
    """Default configuration values"""
    # Execution limits
    MAX_TURNS = 1000
    TIMEOUT_SECONDS = 2400  # 40 minutes

    # Model and permissions
    MODEL = "claude-sonnet-4-5"
    PERMISSION_MODE = PermissionMode.BYPASS_ALL.value

    # Logging
    TRUNCATE_LOGS = False

    # Session management
    CONTINUE_SEQ_ITEM = True


# ==================== TIMESTAMP FORMATS ====================
class TimestampFormat:
    """Timestamp format strings"""
    SHORT = "%m-%d %H:%M:%S"
    ISO = "%Y-%m-%dT%H:%M:%S"
