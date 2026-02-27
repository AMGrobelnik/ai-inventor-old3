"""
Agent Loader - Prepares agents for execution

Handles copying agent .md files to the execution workspace (.claude/agents/)
for agent discovery by Claude SDK.
"""

import shutil
from pathlib import Path
from aii_lib.telemetry import MessageType
from typing import Union, Any

from .agents_registry import AgentDefinition, get_agent


def prepare_agents(
    agents: list[Union[AgentDefinition, str]],
    cwd: Path,
    telemetry: Any = None,
    run_id: str | None = None,
) -> Path:
    """
    Prepare agents for execution by copying to workspace.

    Copies agent .md files to {cwd}/.claude/agents/ for SDK discovery.

    Args:
        agents: List of AgentDefinition objects or agent name strings
        cwd: Agent's working directory
        telemetry: AIITelemetry instance for logging
        run_id: Run ID for sequenced logging

    Returns:
        Path to the agents directory ({cwd}/.claude/agents/)

    Example:
        >>> from aii_lib.agent_backend.claude.utils.init_helpers import math_solver, text_analyzer
        >>> prepare_agents([math_solver, text_analyzer], Path("/workspace"))
        # Creates:
        # /workspace/.claude/agents/math-solver.md
        # /workspace/.claude/agents/text-analyzer.md
    """
    t = telemetry  # Shorthand
    agents_dir = cwd / ".claude" / "agents"

    # Clean and recreate agents directory
    if agents_dir.exists():
        shutil.rmtree(agents_dir)
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Resolve agent names to AgentDefinition objects
    resolved_agents: list[AgentDefinition] = []
    for agent in agents:
        if isinstance(agent, str):
            # Resolve string name to AgentDefinition
            agent_def = get_agent(agent)
            if agent_def is None:
                if t: t.emit(MessageType.WARNING, f"Agent '{agent}' not found in registry, skipping", run_id=run_id)
                continue
            resolved_agents.append(agent_def)
        elif isinstance(agent, AgentDefinition):
            resolved_agents.append(agent)
        else:
            if t: t.emit(MessageType.WARNING, f"Invalid agent type: {type(agent)}, skipping", run_id=run_id)

    # Copy each agent .md file to .claude/agents/
    copied_count = 0
    for agent_def in resolved_agents:
        if not agent_def.path.exists():
            if t: t.emit(MessageType.WARNING, f"Agent path not found: {agent_def.path}, skipping", run_id=run_id)
            continue

        # Agent path should be a .md file
        if agent_def.path.is_file() and agent_def.path.suffix == ".md":
            dest_file = agents_dir / agent_def.path.name
            try:
                shutil.copy2(agent_def.path, dest_file)
                copied_count += 1
                if t: t.emit(MessageType.DEBUG, f"Copied agent '{agent_def.name}' to {dest_file}", run_id=run_id)
            except Exception as e:
                if t: t.emit(MessageType.ERROR, f"Failed to copy agent '{agent_def.name}': {e}", run_id=run_id)
                raise RuntimeError(f"Failed to copy agent '{agent_def.name}': {e}") from e
        else:
            if t: t.emit(MessageType.WARNING, f"Agent path is not a .md file: {agent_def.path}", run_id=run_id)

    if copied_count > 0:
        if t: t.emit(MessageType.INFO, f"Prepared {copied_count} agent(s) in {agents_dir}", run_id=run_id)
    else:
        if t: t.emit(MessageType.WARNING, "No agents were copied", run_id=run_id)

    return agents_dir


def cleanup_agents(cwd: Path) -> None:
    """
    Remove agents directory from workspace.

    Args:
        cwd: Agent's working directory
    """
    agents_dir = cwd / ".claude" / "agents"

    if agents_dir.exists():
        shutil.rmtree(agents_dir)
