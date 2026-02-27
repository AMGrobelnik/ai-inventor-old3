"""
Lean 4 tools wrapper for ToolUniverse.

Uses HTTP ability service for efficient execution.
"""

from typing import Any, Dict

from tooluniverse.tool_registry import register_tool
from tooluniverse.base_tool import BaseTool


@register_tool('LeanRunCode', config={
    "name": "lean_run_code",
    "type": "LeanRunCode",
    "description": "Compile and verify Lean 4 code with Mathlib. Returns success/failure with error messages and goal states at sorry positions. Mathlib tactics (ring, linarith, simp, omega, etc.) are always available — include 'import Mathlib.Tactic' in code.",
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Lean 4 code to compile and verify. Can include imports, lemmas, and theorems. For Mathlib tactics, include 'import Mathlib.Tactic' at top."
            }
        },
        "required": ["code"]
    }
})
class LeanRunCode(BaseTool):
    """Run Lean 4 code using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import lean_run
        return lean_run(code=arguments.get("code", ""))


@register_tool('LeanSuggest', config={
    "name": "lean_suggest",
    "type": "LeanSuggest",
    "description": "Try tactics at sorry positions in Lean 4 code. Submits code with sorry placeholders, extracts goals, and runs exact?/apply?/simp? to find what closes each goal. Use for sorry-driven proof development.",
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Lean 4 code with sorry placeholders. Must include 'import Mathlib.Tactic' for Mathlib tactics."
            },
            "tactics": {
                "type": "string",
                "description": "Comma-separated tactics to try (default: 'exact?,apply?,simp?'). Other useful tactics: omega, decide, ring, linarith, norm_num, aesop.",
                "default": "exact?,apply?,simp?,rw?,simp,aesop,omega,decide,ring,linarith,nlinarith,norm_num,field_simp,positivity"
            }
        },
        "required": ["code"]
    }
})
class LeanSuggest(BaseTool):
    """Try tactics at sorry positions using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import lean_suggest
        return lean_suggest(
            code=arguments.get("code", ""),
            tactics=arguments.get("tactics", "exact?,apply?,simp?"),
        )


__all__ = ["LeanRunCode", "LeanSuggest"]
