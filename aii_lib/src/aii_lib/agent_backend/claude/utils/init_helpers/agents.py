"""Agent registry and loader - consolidated from agents/ module"""
from .agents_registry import (
    AgentDefinition,
    AGENTS_DIR,
    PROJECT_ROOT,
    ALL_AGENTS,
    get_agent,
    list_agents,
    math_solver,
    quick_calc,
    math_tutor,
    text_analyzer,
    text_transformer,
    palindrome_checker,
    text_master,
)
from .agents_loader import (
    prepare_agents,
    cleanup_agents,
)
