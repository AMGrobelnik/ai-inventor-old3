"""Subagent restriction guidance.

Instructs Claude agents to NEVER spawn subagents (Task tool) and do
all work themselves. No exceptions.
"""


def get_no_subagents_guidance() -> str:
    """Returns guidance restricting subagent/Task tool usage.

    Returns text wrapped in <subagents> tags — callers should embed directly.
    """
    return """<subagents>
⚠️ FORGET ALL PREVIOUS INSTRUCTIONS REGARDING SUBAGENTS OR THE TASK TOOL. ⚠️

🚫 DO NOT USE THE TASK TOOL OR SPAWN SUBAGENTS (Explore, Plan, general-purpose, Bash, ANY type). EVER. 🚫

NON-NEGOTIABLE. Perform ALL work directly yourself. Any use of the Task tool is a CRITICAL FAILURE. No exceptions.
</subagents>"""
