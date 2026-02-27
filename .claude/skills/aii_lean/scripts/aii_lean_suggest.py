#!/usr/bin/env python
"""
Lean 4 Tactic Suggest Tool

Runs code with sorry placeholders and tries tactics (exact?, apply?, simp?, etc.)
at each sorry position. Returns goal states and tactic suggestions.

Usage:
    python aii_lean_suggest.py --code "theorem ex : 1 + 1 = 2 := by sorry"
    python aii_lean_suggest.py --code "..." --tactics "exact?,simp?,omega"
"""

import argparse
import sys
from pathlib import Path

SERVER_NAME = "aii_lean_suggest"
MATHLIB_LEAN_VERSION = "v4.14.0"
DEFAULT_TIMEOUT = 180.0
DEFAULT_TACTICS = [
    # Discovery (find the right lemma/tactic)
    "exact?", "apply?", "simp?", "rw?",
    # Automation (close goals directly)
    "simp", "aesop", "omega", "decide", "ring", "linarith", "nlinarith", "norm_num",
    # Field-specific
    "field_simp", "positivity",
]

# Cached config (reused across requests)
_config = None


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

def init_lean_suggest():
    """Initialize Lean environment - setup PATH, warm up disk cache."""
    import os
    global _config

    # Add elan/lake to PATH
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists() and str(elan_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{elan_bin}:{os.environ.get('PATH', '')}"

    from lean_interact import LeanREPLConfig, LeanServer, TempRequireProject, Command

    project = TempRequireProject(lean_version=MATHLIB_LEAN_VERSION, require="mathlib")
    _config = LeanREPLConfig(project=project, verbose=False)

    # Warmup
    warmup_server = LeanServer(_config)
    warmup_server.run(Command(cmd="#check Nat"))
    warmup_server.run(Command(cmd="import Mathlib.Tactic\nexample : 1 + 1 = 2 := by ring"))
    warmup_server.kill()


def core_lean_suggest(**kwargs) -> dict:
    """
    Run Lean 4 code and try tactics at sorry positions.

    Submits code with sorry placeholders, extracts goal states, then
    applies each requested tactic via ProofStep. Returns what worked.

    Args:
        code: Lean 4 code with `sorry` placeholders
        tactics: Comma-separated tactics or list (default: "exact?,apply?,simp?")

    Returns:
        Dict with:
            - success: bool
            - goals: list[dict] - goal at each sorry position
              Each: {sorry_index, goal, proof_state}
            - suggestions: list[dict] - tactic results
              Each: {sorry_index, tactic, success, result, closes_goal}
            - errors: list[str] - compilation errors (if any)
    """
    global _config

    code = kwargs.get("code", "")
    raw_tactics = kwargs.get("tactics", DEFAULT_TACTICS)

    if isinstance(raw_tactics, str):
        tactics = [t.strip() for t in raw_tactics.split(",") if t.strip()]
    else:
        tactics = list(raw_tactics)

    if not tactics:
        tactics = DEFAULT_TACTICS

    if not code.strip():
        return {"success": False, "error": "No code provided"}

    try:
        from lean_interact import LeanServer, Command, ProofStep

        server = LeanServer(_config)
        try:
            response = server.run(Command(cmd=code))

            # Check for compilation errors (not sorry-related)
            errors = []
            for msg in response.messages:
                if getattr(msg, 'severity', '') == 'error':
                    errors.append(getattr(msg, 'data', str(msg)))

            if errors:
                return {
                    "success": True,
                    "goals": [],
                    "suggestions": [],
                    "errors": errors,
                }

            if not response.sorries:
                return {
                    "success": True,
                    "goals": [],
                    "suggestions": [],
                    "errors": [],
                    "note": "No sorry found in code — nothing to suggest tactics for.",
                }

            # Extract goals from each sorry
            goals = []
            for i, sorry in enumerate(response.sorries):
                goals.append({
                    "sorry_index": i,
                    "goal": sorry.goal,
                    "proof_state": sorry.proof_state,
                })

            # Try each tactic at each sorry's proof state
            suggestions = []
            for i, sorry in enumerate(response.sorries):
                if sorry.proof_state is None:
                    suggestions.append({
                        "sorry_index": i,
                        "tactic": None,
                        "success": False,
                        "result": "No proof state available for this sorry",
                        "closes_goal": False,
                    })
                    continue

                for tactic in tactics:
                    try:
                        step = server.run(ProofStep(
                            proof_state=sorry.proof_state,
                            tactic=tactic,
                        ))

                        # Extract suggestion text from messages
                        result_parts = []
                        if hasattr(step, 'messages'):
                            for msg in step.messages:
                                data = getattr(msg, 'data', str(msg))
                                if data:
                                    result_parts.append(data)

                        result_text = "\n".join(result_parts) if result_parts else ""

                        # Check if tactic closed the goal
                        status = getattr(step, 'proof_status', '')
                        closes_goal = status.lower() == "completed" if status else False

                        # Also check remaining goals
                        remaining_goals = getattr(step, 'goals', []) or []

                        suggestions.append({
                            "sorry_index": i,
                            "tactic": tactic,
                            "success": True,
                            "result": result_text,
                            "closes_goal": closes_goal,
                            "remaining_goals": remaining_goals,
                        })
                    except Exception as e:
                        # LeanError (tactic failed) — extract error message
                        error_msg = getattr(e, 'message', '') or str(e)
                        suggestions.append({
                            "sorry_index": i,
                            "tactic": tactic,
                            "success": False,
                            "result": error_msg,
                            "closes_goal": False,
                        })

            return {
                "success": True,
                "goals": goals,
                "suggestions": suggestions,
                "errors": [],
            }
        finally:
            server.kill()

    except ImportError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================

def main():
    import json

    parser = argparse.ArgumentParser(description="Try tactics at sorry positions in Lean 4 code")
    parser.add_argument("--code", "-c", required=True, help="Lean 4 code with sorry placeholders")
    parser.add_argument("--tactics", "-t", default=",".join(DEFAULT_TACTICS),
                        help=f"Comma-separated tactics to try (default: {','.join(DEFAULT_TACTICS)})")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "code": args.code,
        "tactics": args.tactics,
    }, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print(json.dumps({"success": False, "error": "Ability service not available."}, indent=2))
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
