"""B_GEN_ARTIFACT_DEMOS Step - Generate demo versions of artifacts.

Routes each artifact to the appropriate demo converter based on file type:
- Python scripts (.py) → Jupyter notebooks (gen_py_demo)
- Lean proofs (.lean) → Markdown with playground link (gen_lean_demo)
- Research JSON (.json) → Formatted markdown (gen_md_demo)

Output structure:
- {artifact_id}/demo/: Self-contained demos (notebooks, markdown)
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink
from aii_lib.agent_backend import ExpectedFile

from aii_pipeline.utils import PipelineConfig, rel_path

# Import artifact schemas to get demo_files and expected_files for each type
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.schema import ExperimentArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.schema import EvaluationArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.schema import DatasetArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.schema import ProofArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.schema import ResearchArtifact

from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import (
    BaseDemo, CodeDemo, DemoExpectedFiles, LeanDemo, MarkdownDemo,
)

# Per-type demo converters
from ._gen_artifact_demos.gen_py_demo import convert_to_notebook, github_to_colab_url
from ._gen_artifact_demos.gen_lean_demo import create_proof_markdown, lean_playground_url
from ._gen_artifact_demos.gen_md_demo import create_research_markdown


# Map artifact types to their schema classes
ARTIFACT_SCHEMAS = {
    "experiment": ExperimentArtifact,
    "evaluation": EvaluationArtifact,
    "dataset": DatasetArtifact,
    "proof": ProofArtifact,
    "research": ResearchArtifact,
}


def get_demo_files_for_type(artifact_type: str) -> list[str]:
    """Get demo file paths for this artifact type.

    Returns:
        List of filenames to include in demo (e.g., ["method.py"], ["research_out.json"])
    """
    schema = ARTIFACT_SCHEMAS.get(artifact_type)
    if schema:
        defaults = schema.model_fields["out_demo_files"].default or []
        return [f.path for f in defaults]
    return []


def get_expected_out_files_for_type(artifact_type: str) -> list[ExpectedFile]:
    """Get expected output files for this artifact type.

    Used for copying dependencies and verification.

    Returns:
        List of ExpectedFile objects from the artifact schema.
    """
    schema = ARTIFACT_SCHEMAS.get(artifact_type)
    if schema:
        return schema.get_expected_out_files()
    return []


async def run_gen_artifact_demos_module(
    config: PipelineConfig,
    artifacts: list,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    artifact_workspaces: dict[str, Path] | None = None,
    repo_url: str | None = None,
) -> list[BaseDemo]:
    """
    Run the B_GEN_ARTIFACT_DEMOS step (Level 0 - no dependencies).

    Collects files from artifact workspaces using get_github_deploy_outputs()
    and converts them to demo-ready formats.

    Output structure:
    - raw_code/{artifact_type}/: Original files from get_github_deploy_outputs()
    - demo_notebooks/{artifact_type}/: Self-contained Jupyter notebooks

    Args:
        config: Pipeline configuration
        artifacts: Original artifact dicts from invention_loop
        telemetry: AIITelemetry instance
        output_dir: Output directory
        artifact_workspaces: Dict mapping artifact_id to workspace Path
        repo_url: GitHub repo URL for constructing raw data URLs (for Colab compatibility)

    Returns:
        List of BaseDemo objects (CodeDemo, LeanDemo, MarkdownDemo)
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_artifact_demos_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_artifact_demos_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("B_GEN_ARTIFACT_DEMOS")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "B_GEN_ARTIFACT_DEMOS - Converting to demo formats")
        telemetry.emit(MessageType.INFO, "=" * 60)

        if not artifacts:
            telemetry.emit(MessageType.WARNING, "No artifacts to prepare")
            return []

        if not output_dir:
            output_dir = Path("./prepared_artifacts")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-artifact folder structure: {artifact_id}/src/ and {artifact_id}/demo/
        telemetry.emit(MessageType.INFO, f"   Artifacts: {len(artifacts)}")
        telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")
        telemetry.emit(MessageType.INFO, f"   Structure: {{artifact_id}}/src/ and {{artifact_id}}/demo/")

        prepared: list[BaseDemo] = []
        notebook_tasks = []

        for artifact in artifacts:
            aid = artifact.id
            artifact_type = artifact.type.value if hasattr(artifact.type, 'value') else str(artifact.type)

            # Get the demo file from artifact (first entry)
            demo_files = artifact.out_demo_files
            demo_file = demo_files[0].path if demo_files else None

            # Get workspace path for this artifact
            workspace_path = None
            if artifact_workspaces and aid in artifact_workspaces:
                workspace_path = artifact_workspaces[aid]
            elif artifact.workspace_path:
                workspace_path = Path(artifact.workspace_path)

            telemetry.emit(MessageType.INFO, f"   Processing {aid} ({artifact_type}): {demo_file or 'no demo'}")

            # Skip if no demo file
            if demo_file is None:
                telemetry.emit(MessageType.INFO, f"   Skipping {aid} - no demo file")
                continue

            # Create demo output directory
            demo_dir = output_dir / aid
            demo_dir.mkdir(exist_ok=True)

            # Get workspace path and validate it exists
            if not workspace_path or not workspace_path.exists():
                telemetry.emit(MessageType.WARNING, f"   No workspace for {aid}")
                continue

            # Route to appropriate converter based on file extension
            if demo_file.endswith(".lean"):
                # Lean proof -> Markdown with playground link
                script_path = workspace_path / demo_file
                if not script_path.exists():
                    telemetry.emit(MessageType.WARNING, f"   Demo file not found: {demo_file}")
                    continue

                lean_code = script_path.read_text()
                playground_url = lean_playground_url(lean_code)
                md_content, _ = create_proof_markdown(aid, lean_code, artifact)

                demo_path = demo_dir / f"{aid}.md"
                demo_path.write_text(md_content)

                prepared.append(LeanDemo(
                    id=aid,
                    title=artifact.title or f"Lean proof: {aid}",
                    summary=artifact.summary or "Formal proof with Lean playground link",
                    original_path=str(workspace_path),
                    demo_path=str(demo_path),
                    playground_url=playground_url,
                ))

            elif demo_file.endswith(".md"):
                # Research artifact -> copy pre-generated markdown from workspace
                source_md = workspace_path / demo_file
                demo_path = demo_dir / "research_demo.md"
                if source_md.exists():
                    shutil.copy(source_md, demo_path)
                else:
                    telemetry.emit(MessageType.WARNING, f"   Demo file not found: {demo_file}, generating from artifact")
                    md_content = create_research_markdown(artifact, workspace_path=workspace_path)
                    demo_path.write_text(md_content)

                prepared.append(MarkdownDemo(
                    id=aid,
                    title=artifact.title or f"Research: {aid}",
                    summary=artifact.summary or "Research findings",
                    original_path=str(workspace_path),
                    demo_path=str(demo_path),
                ))

            elif demo_file.endswith(".py"):
                # Python script -> Jupyter notebook (queue for parallel conversion)
                script_path = workspace_path / demo_file
                if not script_path.exists():
                    telemetry.emit(MessageType.WARNING, f"   Demo file not found: {demo_file}")
                    continue

                notebook_tasks.append({
                    "artifact_id": aid,
                    "artifact_type": artifact_type,
                    "artifact_name": demo_file,
                    "artifact": artifact,
                    "demo_dir": str(demo_dir),
                })

        # Run notebook conversions in parallel with semaphore control
        if notebook_tasks:
            max_concurrent = config.gen_paper_repo.gen_artifact_demos.claude_agent.max_concurrent_agents
            semaphore = asyncio.Semaphore(max_concurrent)
            telemetry.emit(MessageType.INFO, f"   Converting {len(notebook_tasks)} Python scripts to notebooks (max {max_concurrent} concurrent)...")

            async def convert_one(task_data, task_seq):
                aid = task_data["artifact_id"]
                atype = task_data["artifact_type"]
                name = task_data["artifact_name"]
                art = task_data["artifact"]
                demo_dir_str = task_data["demo_dir"]

                # Offset sequence by 100 so write_paper tasks (sequence=0..N)
                # always get console priority over demo tasks when running concurrently
                result = await convert_to_notebook(
                    config=config,
                    artifact_id=aid,
                    artifact_name=name,
                    artifact_type=atype,
                    artifact=art,
                    output_dir=output_dir,
                    telemetry=telemetry,
                    task_sequence=task_seq + 100,
                    repo_url=repo_url,
                )

                if result:
                    notebook_path, demo_title, demo_summary, demo_expected_files = result

                    # Copy to artifact's demo directory
                    artifact_demo_dir = Path(demo_dir_str)
                    workspace_dir = notebook_path.parent

                    # Use descriptive name based on Python file
                    nb_name = name.replace(".py", "_code_demo.ipynb")
                    demo_path = artifact_demo_dir / nb_name
                    shutil.copy(notebook_path, demo_path)

                    # Also copy demo data file if it exists
                    demo_data_file = workspace_dir / "mini_demo_data.json"
                    if demo_data_file.exists():
                        shutil.copy(demo_data_file, artifact_demo_dir / "mini_demo_data.json")

                    return CodeDemo(
                        id=aid,
                        title=art.title or demo_title,
                        summary=demo_summary,
                        original_path=str(art.workspace_path) if art.workspace_path else "",
                        demo_path=str(artifact_demo_dir),
                        notebook_path=str(demo_path),
                        out_expected_files=DemoExpectedFiles(**demo_expected_files),
                    )
                return None

            async def convert_with_semaphore(task_data, task_seq):
                """Convert notebook with semaphore control and staggered launch."""
                # Stagger agent launches by 5s each to avoid init contention
                if task_seq > 0:
                    await asyncio.sleep(task_seq * 5)
                async with semaphore:
                    return await convert_one(task_data, task_seq)

            results = await asyncio.gather(
                *[convert_with_semaphore(t, idx) for idx, t in enumerate(notebook_tasks)],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    telemetry.emit(MessageType.WARNING, f"   Notebook conversion failed: {r}")
                elif r is not None:
                    prepared.append(r)

        telemetry.emit(MessageType.SUCCESS, f"B_GEN_ARTIFACT_DEMOS complete: {len(prepared)} artifacts prepared")

        # Save output
        output_file = output_dir / "prepared_artifacts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "prepared": [p.model_dump() for p in prepared],
                "folder_structure": "per-artifact: {artifact_id}/src/ and {artifact_id}/demo/",
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "module": "gen_artifact_demos",
                    "llm_provider": "claude_agent",
                    "output_dir": str(output_dir) if output_dir else None,
                },
            }, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

        return prepared

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
