#!/usr/bin/env python3
"""
Hypothesis Audit Module - Parallel Cited Argument Generation & Verification

For each hypothesis, generates arguments in parallel:
- Novelty (positive/negative): OpenRouter with web search to find existing work
- Feasibility (positive/negative): OpenRouter with multiple models for diverse perspectives

Each argument must contain citations with direct quotes: ["quote"](url)
Citations are verified against source pages - invalid arguments are filtered out.

Uses aii_lib workflows for cited argument generation with verification.
"""

import asyncio
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from aii_lib import (
    MessageType,
    CitedArgsConfig,
    ClaudeAgentConfig,
    generate_cited_argument,
    cap_results,
    AIITelemetry,
    JSONSink,
    create_telemetry,
    get_tooluniverse_mcp_config,
    get_model_short,
)

from aii_pipeline.prompts.steps.optional.audit_hypo.schema import (
    HypoAudit,
    AuditArgs,
)
from aii_pipeline.steps.optional._audit_hypo import (
    NoveltyCitedArg,
    FeasibilityCitedArg,
)
from aii_pipeline.prompts.steps.optional.audit_hypo.s_prompt import (
    get_novelty as get_novelty_sysprompt,
    get_feasibility as get_feasibility_sysprompt,
)
from aii_pipeline.prompts.steps.optional.audit_hypo.u_prompt import (
    get_feasibility_positive as get_feasibility_positive_prompt,
    get_feasibility_negative as get_feasibility_negative_prompt,
    get_novelty_positive as get_novelty_positive_prompt,
    get_novelty_negative as get_novelty_negative_prompt,
    build_novelty_retry_prompt,
    build_feasibility_retry_prompt,
    get_force_output_prompt,
)
from aii_pipeline.prompts.components.resources import get_resources_prompt
from aii_pipeline.utils import PipelineConfig, rel_path


def _create_verify_fn(dimension: str, resources_text: str = None):
    """Create verification function based on dimension."""
    if dimension == "novelty":
        def verify_fn(text: str, _context, callback):
            return NoveltyCitedArg(argument=text).verify_citations(callback=callback)
        return verify_fn
    else:
        def verify_fn(text: str, _context, callback):
            return FeasibilityCitedArg(argument=text).verify_against_resources(resources_text, callback=callback)
        return verify_fn


def _create_config(
    dimension: str,
    hypothesis_id: str,
    hypothesis: dict,
    stance: str,
    task_num: int,
    prompt: str,
    system_prompt: str,
    verify_retries: int,
    min_valid_citations: int,
    verify_fn,
    build_retry_prompt_fn,
    # OpenRouter-specific (only used if claude_agent is None)
    api_key: str = "",
    model: str = "",
    timeout: int = 120,
    reasoning_effort: str = "",
    suffix: str = "",
    max_tool_iterations: int = 10,
    # Claude agent params
    claude_agent: ClaudeAgentConfig | None = None,
    response_schema: type | None = None,
) -> CitedArgsConfig:
    """Create CitedArgsConfig for either novelty or feasibility."""
    is_novelty = dimension == "novelty"
    return CitedArgsConfig(
        hypothesis_id=hypothesis_id,
        hypothesis=hypothesis,
        stance=stance,
        task_num=task_num,
        dimension=dimension,
        prompt=prompt,
        system_prompt=system_prompt,
        api_key=api_key,
        model=model,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        suffix=suffix,
        tools=["aii_web_search_fast", "aii_web_fetch_direct", "aii_web_fetch_grep"] if is_novelty else None,
        max_tool_iterations=max_tool_iterations if is_novelty else 1,
        force_output_prompt=get_force_output_prompt() if is_novelty else None,
        verify_retries=verify_retries,
        min_valid_citations=min_valid_citations,
        group=dimension,
        verify_fn=verify_fn,
        build_retry_prompt_fn=build_retry_prompt_fn,
        claude_agent=claude_agent,
        response_schema=response_schema,
    )


def _organize_hypothesis_results(
    hypothesis_id: str,
    hypothesis: dict,
    task_results: list[dict],
    novelty_cap_mode: str,
    feasibility_cap_mode: str,
) -> dict:
    """Organize argument results for a single hypothesis and apply cap mode."""
    arguments = {
        "feasibility_positive": [],
        "feasibility_negative": [],
        "novelty_positive": [],
        "novelty_negative": [],
    }

    all_args = []
    for r in task_results:
        if r is None:
            continue
        key = f"{r['dimension']}_{r['stance']}"
        all_args.append(r)
        if r.get("verified") and r.get("argument"):
            arguments[key].append(r["argument"])

    # Apply cap mode using aii_lib
    feas_pos, feas_neg = cap_results(
        arguments["feasibility_positive"],
        arguments["feasibility_negative"],
        feasibility_cap_mode,
    )
    nov_pos, nov_neg = cap_results(
        arguments["novelty_positive"],
        arguments["novelty_negative"],
        novelty_cap_mode,
    )

    audit = HypoAudit(
        feasibility=AuditArgs(positive_args=feas_pos, negative_args=feas_neg),
        novelty=AuditArgs(positive_args=nov_pos, negative_args=nov_neg),
    )

    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "all_arguments": all_args,
        "audit": audit.model_dump(),
    }


async def run_audit_hypo_module(
    config: PipelineConfig,
    hypotheses=None,
    run_dir=None,
    workspace_dir=None,
    telemetry: AIITelemetry | None = None,
):
    """Run hypothesis audit with parallel cited argument generation and verification."""
    # Create output directory
    if run_dir:
        output_dir = run_dir / "3_audit_hypo"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{config.init.outputs_directory}/{timestamp}_audit_hypo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "audit_hypo")

    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    s1 = JSONSink(output_dir / "audit_hypo_pipeline_messages.jsonl")
    s2 = JSONSink(output_dir / "audit_hypo_pipeline_messages_sequenced.jsonl", sequenced=True)
    telemetry.add_sink(s1)
    telemetry.add_sink(s2)
    module_sinks.extend([s1, s2])

    try:

        telemetry.start_module("AUDIT_HYPO")

        if not hypotheses:
            telemetry.emit(MessageType.ERROR, "No hypotheses provided for audit")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return None

        # Get config
        api_keys = config.api_keys
        audit_cfg = config.audit_hypo
        novelty_cfg = audit_cfg.novelty
        feasibility_cfg = audit_cfg.feasibility
        verify_cfg = audit_cfg.verify_citations

        # Check which dimensions use Claude agent
        novelty_uses_agent = novelty_cfg.use_claude_agent
        feasibility_uses_agent = feasibility_cfg.use_claude_agent
        use_claude_agent = novelty_uses_agent or feasibility_uses_agent

        # =========================================================================
        # SETUP BACKEND (Claude agent or OpenRouter)
        # =========================================================================
        if use_claude_agent:
            # Claude agent path - create configs for each dimension
            max_concurrent = audit_cfg.max_concurrent_audits

            # Create workspace for agent
            agent_cwd = (output_dir / "claude_agent").resolve()
            agent_cwd.mkdir(parents=True, exist_ok=True)

            # Create novelty ClaudeAgentConfig
            if novelty_uses_agent:
                nov_agent_cfg = novelty_cfg.claude_agent
                nov_mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)
                nov_claude_agent = ClaudeAgentConfig(
                    model=nov_agent_cfg.model,
                    max_turns=nov_agent_cfg.max_turns,
                    seq_prompt_timeout=nov_agent_cfg.seq_prompt_timeout,
                    seq_prompt_retries=nov_agent_cfg.seq_prompt_retries,
                    cwd=agent_cwd,
                    mcp_servers=nov_mcp_servers,
                )
                # Single "model" for Claude agent
                nov_models = [{"model": nov_agent_cfg.model}]
            else:
                nov_claude_agent = None
                nov_llm = novelty_cfg.llm_client
                nov_models = [{
                    "model": m.model,
                    "reasoning_effort": m.reasoning_effort,
                    "suffix": m.suffix or nov_llm.suffix,
                    "max_tool_iterations": m.max_tool_iterations,
                } for m in nov_llm.models]

            # Create feasibility ClaudeAgentConfig
            if feasibility_uses_agent:
                feas_agent_cfg = feasibility_cfg.claude_agent
                feas_claude_agent = ClaudeAgentConfig(
                    model=feas_agent_cfg.model,
                    max_turns=feas_agent_cfg.max_turns,
                    seq_prompt_timeout=feas_agent_cfg.seq_prompt_timeout,
                    seq_prompt_retries=feas_agent_cfg.seq_prompt_retries,
                    cwd=agent_cwd,
                    mcp_servers=None,  # Feasibility doesn't use tools
                )
                # Single "model" for Claude agent
                feas_models = [{"model": feas_agent_cfg.model}]
            else:
                feas_claude_agent = None
                feas_llm = feasibility_cfg.llm_client
                feas_models = [{
                    "model": m.model,
                    "reasoning_effort": m.reasoning_effort,
                    "suffix": m.suffix or feas_llm.suffix,
                } for m in feas_llm.models]

            llm_provider = "claude_agent"
        else:
            # OpenRouter path
            nov_claude_agent = None
            feas_claude_agent = None
            max_concurrent = audit_cfg.max_concurrent_audits
            llm_provider = "openrouter"

            # Build models lists for both dimensions
            nov_llm = novelty_cfg.llm_client
            feas_llm = feasibility_cfg.llm_client
            nov_models = [{
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or nov_llm.suffix,
                "max_tool_iterations": m.max_tool_iterations,
            } for m in nov_llm.models]
            feas_models = [{
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or feas_llm.suffix,
            } for m in feas_llm.models]

        # Calculate totals (per_llm × num_models for each stance)
        nov_pos_total = novelty_cfg.num_positive_per_llm * len(nov_models)
        nov_neg_total = novelty_cfg.num_negative_per_llm * len(nov_models)
        feas_pos_total = feasibility_cfg.num_positive_per_llm * len(feas_models)
        feas_neg_total = feasibility_cfg.num_negative_per_llm * len(feas_models)
        novelty_calls = nov_pos_total + nov_neg_total
        feasibility_calls = feas_pos_total + feas_neg_total
        calls_per_hypo = novelty_calls + feasibility_calls
        total_calls = len(hypotheses) * calls_per_hypo

        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        # Load resources for feasibility verification
        resources_text = get_resources_prompt()

        telemetry.emit(MessageType.INFO, "AuditHypo - Generating cited arguments for hypotheses")
        telemetry.emit(MessageType.INFO, f"   Provider: {llm_provider}")
        telemetry.emit(MessageType.INFO, f"   Hypotheses: {len(hypotheses)} | Total calls: {total_calls}")
        telemetry.emit(MessageType.INFO, f"   Novelty: {len(nov_models)} models x {novelty_cfg.num_positive_per_llm}+/{novelty_cfg.num_negative_per_llm}- per LLM = {novelty_calls}/hypo")
        telemetry.emit(MessageType.INFO, f"   Feasibility: {len(feas_models)} models x {feasibility_cfg.num_positive_per_llm}+/{feasibility_cfg.num_negative_per_llm}- per LLM = {feasibility_calls}/hypo")
        telemetry.emit(MessageType.INFO, f"   Resources: {len(resources_text)} chars | Concurrent: {max_concurrent}")

        novelty_sysprompt = get_novelty_sysprompt()
        feasibility_sysprompt = get_feasibility_sysprompt()

        # Get LLM client configs for OpenRouter (used for cfg building)
        nov_llm = novelty_cfg.llm_client if not novelty_uses_agent else None
        feas_llm = feasibility_cfg.llm_client if not feasibility_uses_agent else None

        # Build all task configs - consolidated loop
        task_metadata = []
        task_configs = []
        task_counter = 0

        # Task specs: (dimension, stance, total_count, get_prompt_fn, sysprompt, retry_fn, claude_agent, models)
        task_specs = [
            ("novelty", "positive", nov_pos_total, get_novelty_positive_prompt, novelty_sysprompt, build_novelty_retry_prompt, nov_claude_agent, nov_models),
            ("novelty", "negative", nov_neg_total, get_novelty_negative_prompt, novelty_sysprompt, build_novelty_retry_prompt, nov_claude_agent, nov_models),
            ("feasibility", "positive", feas_pos_total, get_feasibility_positive_prompt, feasibility_sysprompt, build_feasibility_retry_prompt, feas_claude_agent, feas_models),
            ("feasibility", "negative", feas_neg_total, get_feasibility_negative_prompt, feasibility_sysprompt, build_feasibility_retry_prompt, feas_claude_agent, feas_models),
        ]

        for hypo_idx, hypo in enumerate(hypotheses):
            hypothesis_id = hypo.get('hypothesis_id', f"hypo_{hypo_idx+1}")

            for dimension, stance, count, get_prompt_fn, sysprompt, retry_fn, claude_agent, models in task_specs:
                for i in range(count):
                    task_counter += 1
                    model_short = get_model_short(model)
                    task_id = f"audit_{dimension[:3]}_{stance[:3]}_v{task_counter}__{model_short}"

                    # Rotate through models
                    model_cfg = models[task_counter % len(models)]
                    model = model_cfg["model"]
                    reasoning = model_cfg.get("reasoning_effort") or ""
                    suffix = model_cfg.get("suffix") or ""
                    max_tool_iters = model_cfg.get("max_tool_iterations", 20) if dimension == "novelty" else 1

                    task_metadata.append({
                        "task_id": task_id,
                        "hypo_id": hypothesis_id,
                        "dimension": dimension,
                        "stance": stance,
                    })

                    # Get llm_cfg for OpenRouter params (may be None for Claude agent)
                    llm_cfg = nov_llm if dimension == "novelty" else feas_llm

                    # Build config
                    cfg = _create_config(
                        dimension=dimension,
                        hypothesis_id=hypothesis_id,
                        hypothesis=hypo,
                        stance=stance,
                        task_num=task_counter,
                        prompt=get_prompt_fn(hypo),
                        system_prompt=sysprompt,
                        # OpenRouter params (only used if claude_agent is None)
                        api_key=api_keys.openrouter if llm_cfg else "",
                        model=model,
                        timeout=llm_cfg.llm_timeout if llm_cfg else 300,
                        reasoning_effort=reasoning,
                        suffix=suffix,
                        max_tool_iterations=max_tool_iters,
                        # Verification params
                        verify_retries=verify_cfg.retry,
                        min_valid_citations=verify_cfg.min_valid_citations,
                        verify_fn=_create_verify_fn(dimension, resources_text),
                        build_retry_prompt_fn=lambda v, nc, fn=retry_fn, mvc=verify_cfg.min_valid_citations: fn(v, nc, mvc),
                        # Claude agent params
                        claude_agent=claude_agent,
                        response_schema=NoveltyCitedArg if dimension == "novelty" else FeasibilityCitedArg,
                    )

                    task_configs.append({
                        "task_id": task_id,
                        "hypothesis_id": hypothesis_id,
                        "dimension": dimension,
                        "stance": stance,
                        "cfg": cfg,
                    })

        # Run tasks with semaphore
        async def run_with_limit(task_cfg: dict):
            async with sem if sem else nullcontext():
                cfg = task_cfg["cfg"]
                result = await generate_cited_argument(telemetry=telemetry, config=cfg)
                if result:
                    return task_cfg["task_id"], {
                        "hypothesis_id": task_cfg["hypothesis_id"],
                        "dimension": task_cfg["dimension"],
                        "stance": task_cfg["stance"],
                        **result,
                    }
                return task_cfg["task_id"], None

        start_time = time.perf_counter()
        task_results = await asyncio.gather(*[
            run_with_limit(task_cfg) for task_cfg in task_configs
        ], return_exceptions=True)
        gather_time = time.perf_counter() - start_time

        # Convert to dict (filter out exceptions)
        results = {}
        for tr in task_results:
            if isinstance(tr, Exception):
                telemetry.emit(MessageType.WARNING, f"AuditHypo task failed: {tr}")
                continue
            task_id, result = tr
            if result is not None:
                results[task_id] = result

        # Organize results by hypothesis
        hypo_results = {hypo.get('hypothesis_id', f"hypo_{i+1}"): [] for i, hypo in enumerate(hypotheses)}
        for meta in task_metadata:
            if (task_result := results.get(meta["task_id"])):
                hypo_results[meta["hypo_id"]].append(task_result)

        all_results = []
        for hypo_idx, hypo in enumerate(hypotheses):
            hypothesis_id = hypo.get('hypothesis_id', f"hypo_{hypo_idx+1}")
            organized = _organize_hypothesis_results(
                hypothesis_id, hypo, hypo_results.get(hypothesis_id, []),
                novelty_cfg.cap_mode, feasibility_cfg.cap_mode,
            )
            all_results.append(organized)

        # Log stats
        telemetry.emit(MessageType.INFO, "")
        telemetry.emit(MessageType.INFO, f"Tasks: {len(task_metadata)} | Concurrent: {max_concurrent} | Time: {gather_time:.1f}s")

        # Process results
        audited_hypotheses = []
        total_args = 0
        verified_args = 0

        for result in all_results:
            for arg in result["all_arguments"]:
                total_args += 1
                if arg["verified"]:
                    verified_args += 1
            audited_hypotheses.append(result)

            audit = result["audit"]
            f_pos, f_neg = len(audit["feasibility"]["positive_args"]), len(audit["feasibility"]["negative_args"])
            n_pos, n_neg = len(audit["novelty"]["positive_args"]), len(audit["novelty"]["negative_args"])
            telemetry.emit(MessageType.INFO, f"   {result['hypothesis_id']}: {f_pos+f_neg+n_pos+n_neg}/{calls_per_hypo} (F:{f_pos}+/{f_neg}-, N:{n_pos}+/{n_neg}-)")

        verification_rate = round(100 * verified_args / total_args) if total_args > 0 else 0
        telemetry.emit(MessageType.SUCCESS, f"AuditHypo completed - {len(audited_hypotheses)} hypotheses, {verified_args}/{total_args} verified ({verification_rate}%)")

        # Build and save output
        module_output = {
            'audited_hypotheses': audited_hypotheses,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'module': 'audit_hypo',
                'llm_provider': llm_provider,
                'num_hypotheses': len(hypotheses),
                'total_arguments': total_args,
                'verified_arguments': verified_args,
                'verification_rate': f"{verification_rate}%",
                'output_dir': str(output_dir),
            }
        }

        telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(module_output, indent=2, ensure_ascii=False))

        output_file = output_dir / "audit_hypo_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(module_output, f, indent=2, ensure_ascii=False)

        telemetry.emit(MessageType.INFO, f"Results saved to: {rel_path(output_file)}")

        # Emit module summary
        telemetry.emit_module_summary("AUDIT_HYPO")

        # Remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)

        return {
            'output_dir': str(output_dir),
            'output_file': str(output_file),
            'audited_hypotheses': audited_hypotheses,
            'metadata': module_output['metadata'],
        }

    finally:
        # Only flush/close if we created local telemetry
        if local_telemetry:
            telemetry.flush()


async def main():
    """Main function for standalone execution."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    sample_hypotheses = [{
        "title": "JSD Thermostat for Multi-LLM Deliberation",
        "hypothesis": "Controlling inter-agent Jensen-Shannon divergence to a mid-range band will reduce confident errors and API calls.",
        "motivation": "Multi-agent debate improves accuracy but costs scale poorly.",
        "assumptions": ["JSD can be computed efficiently", "Mid-range JSD correlates with better outcomes"],
        "methodology": "Use SciFact dataset, two agents produce label probabilities, compute JSD, spawn diversifier if too low or adjudicator if too high.",
        "validation": "Compare accuracy and cost vs baseline debate.",
        "inspiration": "Deliberative polling + JSD metrics",
        "terms": [{"term": "JSD", "definition": "Jensen-Shannon divergence"}],
    }]

    result = await run_audit_hypo_module(config, hypotheses=sample_hypotheses)
    return 0 if result else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
