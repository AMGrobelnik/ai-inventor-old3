"""User prompts for hypothesis ranking - pairwise comparison."""

from aii_pipeline.utils import to_prompt_yaml


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    hypothesis_a_formatted: str,
    hypothesis_b_formatted: str,
) -> str:
    return f"""<task_preview>
Compare two research hypotheses and select the better one.
</task_preview>

{hypothesis_a_formatted}

{hypothesis_b_formatted}

<comparison_criteria>
Evaluate based on:
1. **Hypothesis quality** - novelty, clarity, potential impact
2. **Audit evidence** - weigh the verified FOR and AGAINST arguments
3. **Novelty** - is it genuinely new based on the cited evidence?
4. **Feasibility** - can it be implemented with the cited resources?
5. **Research potential** - which has higher chance of meaningful contribution?

<CRITICAL>
- The audit arguments contain VERIFIED quotes - use them as evidence
- You MUST select either A or B - no ties allowed
- Reference specific audit quotes in your justification
</CRITICAL>
</comparison_criteria>

<YOUR_TASK>
Select A or B. Provide brief justification referencing audit evidence, then your preference.
</YOUR_TASK>"""


# =============================================================================
# HELPERS
# =============================================================================

def _format_audit_args(args: list[str]) -> str:
    """Format audit arguments as bullet list."""
    if not args:
        return "  None"
    return "\n".join(f"  - {arg}" for arg in args)


def _format_hypothesis_full(hypo: dict, audit: dict, label: str) -> str:
    """Format a hypothesis with all its info and audit arguments."""
    display = {k: v for k, v in hypo.items()
               if k not in ["hypothesis_id", "is_seeded", "model"]
               and v is not None}

    novelty = audit.get("novelty", {})
    feasibility = audit.get("feasibility", {})

    nov_pos = novelty.get("positive_args", [])
    nov_neg = novelty.get("negative_args", [])
    feas_pos = feasibility.get("positive_args", [])
    feas_neg = feasibility.get("negative_args", [])

    label_lower = label.lower()
    return f"""<hypothesis_{label_lower}>
<details>
{to_prompt_yaml(display)}
</details>

<audit_novelty label="VERIFIED EVIDENCE">
FOR novelty (this idea IS new):
{_format_audit_args(nov_pos)}

AGAINST novelty (similar work exists):
{_format_audit_args(nov_neg)}
</audit_novelty>

<audit_feasibility label="VERIFIED EVIDENCE">
FOR feasibility (CAN be built with provided resources):
{_format_audit_args(feas_pos)}

AGAINST feasibility (barriers exist with provided resources):
{_format_audit_args(feas_neg)}
</audit_feasibility>
</hypothesis_{label_lower}>"""


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    hypothesis_a: dict,
    hypothesis_b: dict,
    audit_a: dict,
    audit_b: dict,
) -> str:
    """Generate pairwise comparison prompt for two hypotheses with their audits."""
    return PROMPT(
        hypothesis_a_formatted=_format_hypothesis_full(hypothesis_a, audit_a, "A"),
        hypothesis_b_formatted=_format_hypothesis_full(hypothesis_b, audit_b, "B"),
    )
