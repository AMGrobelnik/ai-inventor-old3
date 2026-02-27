"""Optional pipeline steps (LLM-as-judge ranking modules).

These steps use LLM pairwise comparison tournaments for ranking.
They are not part of the default pipeline but can be re-enabled in config.yaml.

Modules:
- audit_hypo: Generate cited novelty/feasibility arguments per hypothesis
- rank_hypo: Swiss-BT tournament ranking of hypotheses
- rank_strat: Swiss-BT ranking of strategies in invention loop
- rank_plan: Swiss-BT ranking of plans per artifact_direction
- rank_narr: Swiss-BT narrative ranking with gap extraction
- rank_paper_text: Swiss-BT ranking of paper texts
"""
