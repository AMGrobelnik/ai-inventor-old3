"""B_DEPLOY_TO_REPO Step - Deploy artifacts to GitHub repository.

Per-artifact structure:
- {artifact_id}/src/: Entire workspace (all files from invention loop execution)
- {artifact_id}/demo/: Self-contained Jupyter notebooks (code + data inlined)

README includes:
- Colab links for notebooks (auto-open in Google Colab)
- Lean playground links for proofs

Uses gh CLI for repo operations.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile

from aii_lib import AIITelemetry, MessageType, JSONSink

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._4_gen_paper_repo.schema import GistDeployment
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import LeanDemo
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import BaseArtifact
from ._1b_gen_artifact_demos import github_to_colab_url
from ._utils import get_readable_folder_name
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure


def generate_readme_with_colab_links(
    repo_url: str,
    prepared_artifacts: list,
    artifacts: list[BaseArtifact],
    has_paper_pdf: bool = False,
    has_paper_latex: bool = False,
    num_figures: int = 0,
    aid_to_folder: dict[str, str] | None = None,
) -> str:
    """Generate README.md with Colab links for notebooks.

    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/user/repo)
        prepared_artifacts: List of BaseDemo objects
        artifacts: List of BaseArtifact objects
        has_paper_pdf: Whether paper.pdf exists
        has_paper_latex: Whether paper_latex/ folder exists
        num_figures: Number of figures in figures/ folder

    Returns:
        README content as string
    """
    artifacts_by_id = {a.id: a for a in artifacts}

    readme = """# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

"""

    # Add paper section FIRST (before Quick Start)
    if has_paper_pdf or has_paper_latex:
        readme += """## Research Paper

"""
        if has_paper_pdf:
            pdf_url = f"{repo_url}/blob/main/paper/paper.pdf"
            pdf_badge = f"[![Download PDF](https://img.shields.io/badge/Download-PDF-red)]({pdf_url})"
            readme += f"{pdf_badge} "

        if has_paper_latex:
            latex_url = f"{repo_url}/tree/main/paper"
            latex_badge = f"[![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)]({latex_url})"
            readme += f"{latex_badge} "

        if num_figures > 0:
            figures_url = f"{repo_url}/tree/main/figures"
            figures_badge = f"[![Figures](https://img.shields.io/badge/Figures-{num_figures}-blue)]({figures_url})"
            readme += f"{figures_badge}"

        readme += "\n\n"

    readme += """## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

"""

    # Group artifacts by type
    notebooks = []
    markdown_demos = []
    proofs = []

    # Default to empty mapping if not provided
    folder_map = aid_to_folder or {}

    for prep in prepared_artifacts:
        aid = prep.id
        # Use sanitized folder name from mapping, fallback to artifact_id
        folder_name = folder_map.get(aid, aid)
        demo_type = prep.type.value
        artifact = artifacts_by_id.get(aid)
        short_desc = artifact.title if artifact else aid

        if demo_type == "code":
            # Build relative path for notebook: {folder_name}/demo/{name}.ipynb
            demo_path = Path(prep.demo_path)

            # demo_path may be a directory (contains notebook + demo data files) or a file
            if demo_path.is_dir():
                # Find the .ipynb file in the directory
                ipynb_files = list(demo_path.glob("*.ipynb"))
                notebook_name = ipynb_files[0].name if ipynb_files else "code_demo.ipynb"
            else:
                notebook_name = demo_path.name

            # Get relative path from repo root
            rel_path_str = f"{folder_name}/demo/{notebook_name}"

            # Build GitHub URL and Colab URL
            github_url = f"{repo_url}/blob/main/{rel_path_str}"
            colab_url = github_to_colab_url(github_url)

            notebooks.append({
                "id": aid,
                "folder": folder_name,
                "name": notebook_name,
                "desc": short_desc,
                "github_url": github_url,
                "colab_url": colab_url,
                "rel_path": rel_path_str,
            })

        elif isinstance(prep, LeanDemo) and prep.playground_url:
            proofs.append({
                "id": aid,
                "folder": folder_name,
                "desc": short_desc,
                "playground_url": prep.playground_url,
            })

        else:
            # Markdown demo (research)
            demo_path = Path(prep.demo_path)
            rel_path_str = f"{folder_name}/demo/{demo_path.name}"
            github_url = f"{repo_url}/blob/main/{rel_path_str}"

            markdown_demos.append({
                "id": aid,
                "folder": folder_name,
                "desc": short_desc,
                "github_url": github_url,
                "rel_path": rel_path_str,
            })

    # Add notebook section with Colab badges
    if notebooks:
        readme += """### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
"""
        for nb in notebooks:
            # Colab badge
            colab_badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({nb['colab_url']})"
            readme += f"| `{nb['folder']}` | {nb['desc'][:50]}... | {colab_badge} |\n"

    # Add proofs section with Lean badges
    if proofs:
        readme += """
### Formal Proofs

| Folder | Description | Verify in Lean |
|--------|-------------|----------------|
"""
        for proof in proofs:
            lean_badge = f"[![Verify in Lean](https://img.shields.io/badge/Lean_4-Verify_Proof-blue)]({proof['playground_url']})"
            readme += f"| `{proof['folder']}` | {proof['desc'][:50]}... | {lean_badge} |\n"

    # Add research section with view links
    if markdown_demos:
        readme += """
### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
"""
        for research in markdown_demos:
            view_badge = f"[![View Research](https://img.shields.io/badge/View-Research-green)]({research['github_url']})"
            readme += f"| `{research['folder']}` | {research['desc'][:50]}... | {view_badge} |\n"

    repo_name = repo_url.split("/")[-1] if "/" in repo_url else "repo"
    readme += f"""
## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone {repo_url}.git
cd {repo_name}

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
"""

    return readme


async def populate_repo(
    telemetry: AIITelemetry,
    repo_url: str,
    prepared_artifacts: list,  # list[BaseDemo] from prepare_artifacts step
    artifacts: list[BaseArtifact],
    paper_pdf_path: Path | None = None,
    paper_latex_dir: Path | None = None,
    figures: list[Figure] | None = None,
) -> tuple[bool, dict[str, str]]:
    """Populate the GitHub repo with prepared artifacts.

    Per-artifact repo structure:
    - {artifact_id}/src/: Entire workspace from invention loop execution
    - {artifact_id}/demo/: Self-contained Jupyter notebooks (code + data inlined)
    - figures/: Visualization figures
    - paper/: LaTeX and PDF

    README includes Colab links for notebooks.

    Args:
        telemetry: AIITelemetry instance
        repo_url: GitHub repo URL
        prepared_artifacts: BaseDemo objects from prepare_artifacts step
        artifacts: BaseArtifact objects for README
    """
    import time

    # Log what we're attempting to push
    telemetry.emit(MessageType.INFO, "")
    telemetry.emit(MessageType.SUCCESS, "=" * 50)
    telemetry.emit(MessageType.SUCCESS, ">>> PUSH ARTIFACTS TO GITHUB <<<")
    telemetry.emit(MessageType.INFO, f"   repo_url: {repo_url or 'NONE'}")
    telemetry.emit(MessageType.INFO, f"   prepared_artifacts: {len(prepared_artifacts)}")
    telemetry.emit(MessageType.INFO, f"   artifacts: {len(artifacts)}")
    telemetry.emit(MessageType.INFO, f"   paper_pdf: {paper_pdf_path} (exists={paper_pdf_path.exists() if paper_pdf_path else 'N/A'})")
    telemetry.emit(MessageType.INFO, f"   paper_latex_dir: {paper_latex_dir} (exists={paper_latex_dir.exists() if paper_latex_dir else 'N/A'})")

    if not repo_url:
        telemetry.emit(MessageType.ERROR, "   ABORTED: No repo_url provided")
        return False, {}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"

            # Clone the repo
            telemetry.emit(MessageType.INFO, "   Cloning repository...")
            clone_result = subprocess.run(
                ["gh", "repo", "clone", repo_url, str(repo_dir)],
                capture_output=True,
                text=True,
            )
            if clone_result.returncode != 0:
                telemetry.emit(MessageType.WARNING, f"Failed to clone repo: {clone_result.stderr}")
                return False, {}

            # Create common folders
            (repo_dir / "figures").mkdir(exist_ok=True)
            (repo_dir / "paper").mkdir(exist_ok=True)

            files_added = []

            # Map from artifact_id to sanitized folder name (for README links)
            aid_to_folder = {}

            # Directories/patterns to exclude from workspace copy
            _EXCLUDED_DIRS = {"temp", "tmp", "__pycache__", ".venv", "venv", "node_modules", ".git", ".cache", "dependencies"}
            _EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".log"}

            # Build lookup for original artifact titles (used for consistent folder naming)
            artifacts_by_id = {a.id: a for a in artifacts}

            # Copy per-artifact folders with proper /src/ and /demo/ structure
            # Both paths now use the same structure - prepared_dir is just an alternative source
            for prep in prepared_artifacts:
                aid = prep.id

                # IMPORTANT: Use the ORIGINAL artifact.title for folder name, NOT prep.title.
                # gen_py_demo builds GITHUB_DATA_URL using artifact.title, so deploy must match.
                original_artifact = artifacts_by_id.get(aid)
                title_for_folder = original_artifact.title if original_artifact else prep.title
                workspace_path = Path(prep.original_path)
                folder_name = get_readable_folder_name(aid, title_for_folder)
                aid_to_folder[aid] = folder_name

                # Create artifact directory structure: {folder_name}/src/ and {folder_name}/demo/
                artifact_dir = repo_dir / folder_name
                src_dir = artifact_dir / "src"
                demo_dir = artifact_dir / "demo"
                src_dir.mkdir(parents=True, exist_ok=True)
                demo_dir.mkdir(parents=True, exist_ok=True)

                telemetry.emit(MessageType.INFO, f"   Creating {folder_name}/src/ and {folder_name}/demo/")

                # Copy workspace to src/, excluding temp dirs and build artifacts
                if workspace_path.exists() and workspace_path.is_dir():
                    src_files_count = 0
                    skipped_count = 0
                    for item in workspace_path.iterdir():
                        # Skip excluded directories
                        if item.is_dir() and item.name in _EXCLUDED_DIRS:
                            skipped_count += 1
                            continue
                        # Skip excluded file extensions
                        if item.is_file() and item.suffix in _EXCLUDED_EXTENSIONS:
                            skipped_count += 1
                            continue
                        dst = src_dir / item.name
                        if item.is_file():
                            shutil.copy(item, dst)
                            files_added.append(f"{folder_name}/src/{item.name}")
                            src_files_count += 1
                        elif item.is_dir():
                            shutil.copytree(
                                item, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("temp", "tmp", "__pycache__", ".venv", "node_modules", ".git", "*.pyc"),
                            )
                            for f in dst.rglob("*"):
                                if f.is_file():
                                    rel = f.relative_to(dst)
                                    files_added.append(f"{folder_name}/src/{item.name}/{rel}")
                                    src_files_count += 1
                    if skipped_count:
                        telemetry.emit(MessageType.INFO, f"      Skipped {skipped_count} temp/build items")
                    telemetry.emit(MessageType.INFO, f"      Copied {src_files_count} files to {folder_name}/src/")
                else:
                    telemetry.emit(MessageType.WARNING, f"   Workspace not found for {aid}: {workspace_path}")

                # Copy demo files to demo/
                demo_src = Path(prep.demo_path)
                if demo_src.exists():
                    demo_files_count = 0
                    if demo_src.is_dir():
                        for f in demo_src.iterdir():
                            if f.is_file():
                                dst = demo_dir / f.name
                                shutil.copy(f, dst)
                                files_added.append(f"{folder_name}/demo/{f.name}")
                                demo_files_count += 1
                    else:
                        dst = demo_dir / demo_src.name
                        shutil.copy(demo_src, dst)
                        files_added.append(f"{folder_name}/demo/{demo_src.name}")
                        demo_files_count = 1
                    telemetry.emit(MessageType.INFO, f"      Copied {demo_files_count} demo files to {folder_name}/demo/")
                else:
                    telemetry.emit(MessageType.WARNING, f"   Demo file not found for {aid}: {demo_src}")

            # Copy paper files if provided
            has_paper_pdf = False
            has_paper_latex = False

            # Diagnostic: log what we received
            telemetry.emit(MessageType.INFO, f"   paper_pdf_path: {paper_pdf_path} (exists={paper_pdf_path.exists() if paper_pdf_path else 'N/A'})")
            telemetry.emit(MessageType.INFO, f"   paper_latex_dir: {paper_latex_dir} (exists={paper_latex_dir.exists() if paper_latex_dir else 'N/A'})")

            if paper_pdf_path and paper_pdf_path.exists():
                paper_dir = repo_dir / "paper"
                paper_dir.mkdir(exist_ok=True)
                dst = paper_dir / "paper.pdf"
                shutil.copy(paper_pdf_path, dst)
                files_added.append("paper/paper.pdf")
                has_paper_pdf = True
                telemetry.emit(MessageType.INFO, "   Added paper/paper.pdf")

            if paper_latex_dir and paper_latex_dir.exists():
                dst = repo_dir / "paper_latex"
                shutil.copytree(paper_latex_dir, dst, dirs_exist_ok=True)
                for f in paper_latex_dir.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(paper_latex_dir)
                        files_added.append(f"paper_latex/{rel}")
                has_paper_latex = True
                telemetry.emit(MessageType.INFO, f"   Added paper_latex/ ({len(list(paper_latex_dir.rglob('*')))} files)")

            # Fallback: detect paper files already in the repo clone (from push_paper_to_repo)
            # This handles the case where paper_pdf_path is None/invalid but the paper
            # was already pushed by a prior step.
            if not has_paper_pdf:
                clone_paper = repo_dir / "paper" / "paper.pdf"
                clone_paper_root = repo_dir / "paper.pdf"
                if clone_paper.exists() or clone_paper_root.exists():
                    has_paper_pdf = True
                    telemetry.emit(MessageType.WARNING, "   paper_pdf_path missing/invalid but paper.pdf found in repo clone — enabling paper section in README")
            if not has_paper_latex:
                clone_latex = repo_dir / "paper" / "paper.tex"
                if clone_latex.exists():
                    has_paper_latex = True
                    telemetry.emit(MessageType.WARNING, "   paper_latex_dir missing/invalid but paper.tex found in repo clone — enabling LaTeX section in README")

            # Copy figures to root figures/ (also done by push_paper_to_repo in step 4,
            # but we duplicate here so deploy_repo is self-contained for resume scenarios)
            figures_dir = repo_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            if figures:
                figs_copied = 0
                for fig in figures:
                    if fig.figure_path:
                        src = Path(fig.figure_path)
                        if src.exists():
                            shutil.copy(src, figures_dir / src.name)
                            figs_copied += 1
                if figs_copied:
                    telemetry.emit(MessageType.INFO, f"   Copied {figs_copied} figures to figures/")
                    files_added.extend([f"figures/{f.name}" for f in figures_dir.iterdir()])

            # Copy paper.pdf to repo root (also done by push_paper_to_repo,
            # duplicated here for self-contained deploy)
            if has_paper_pdf:
                root_pdf = repo_dir / "paper.pdf"
                if not root_pdf.exists():
                    # Try to copy from paper/ subfolder or from paper_pdf_path
                    paper_in_subfolder = repo_dir / "paper" / "paper.pdf"
                    if paper_in_subfolder.exists():
                        shutil.copy(paper_in_subfolder, root_pdf)
                        files_added.append("paper.pdf")
                        telemetry.emit(MessageType.INFO, "   Copied paper.pdf to repo root")
                    elif paper_pdf_path and paper_pdf_path.exists():
                        shutil.copy(paper_pdf_path, root_pdf)
                        files_added.append("paper.pdf")
                        telemetry.emit(MessageType.INFO, "   Copied paper.pdf to repo root")

            # Generate README with Colab links
            telemetry.emit(MessageType.INFO, "   Generating README with Colab links...")
            # Count figures in the figures/ folder
            num_figures = len([f for f in figures_dir.iterdir() if f.is_file()]) if figures_dir.exists() else 0

            readme_content = generate_readme_with_colab_links(
                repo_url=repo_url,
                prepared_artifacts=prepared_artifacts,
                artifacts=artifacts,
                has_paper_pdf=has_paper_pdf,
                has_paper_latex=has_paper_latex,
                num_figures=num_figures,
                aid_to_folder=aid_to_folder,
            )
            (repo_dir / "README.md").write_text(readme_content)
            files_added.append("README.md")

            if not files_added:
                telemetry.emit(MessageType.WARNING, "No files to add to repo")
                return False, {}

            # Git add, commit, push
            telemetry.emit(MessageType.INFO, f"   Running git add for {len(files_added)} files...")

            add_result = subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, text=True)
            if add_result.returncode != 0:
                telemetry.emit(MessageType.WARNING, f"   git add failed: {add_result.stderr}")

            telemetry.emit(MessageType.INFO, "   Running git commit...")
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Add artifacts from AI Inventor pipeline"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if commit_result.returncode != 0:
                if "nothing to commit" in commit_result.stdout.lower() or "nothing to commit" in commit_result.stderr.lower():
                    telemetry.emit(MessageType.WARNING, "   Nothing new to commit - files may already exist in repo")
                    return True, aid_to_folder  # Not an error, just nothing to push
                else:
                    telemetry.emit(MessageType.WARNING, f"   git commit failed: {commit_result.stderr}")

            # Push with retry logic for transient errors (HTTP 408, network issues)
            telemetry.emit(MessageType.INFO, "   Running git push...")
            max_retries = 3
            for attempt in range(max_retries):
                push_result = subprocess.run(
                    ["git", "push"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout per attempt
                )
                if push_result.returncode == 0:
                    break

                # Check for transient errors worth retrying
                stderr = push_result.stderr.lower()
                is_transient = any(err in stderr for err in [
                    "408", "timeout", "timed out", "connection reset",
                    "unexpected disconnect", "hung up", "network",
                ])

                if is_transient and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    telemetry.emit(MessageType.WARNING, f"   Push failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    telemetry.emit(MessageType.ERROR, f"   Push failed: {push_result.stderr}")
                    raise RuntimeError(f"Git push failed after {max_retries} retries: {push_result.stderr}")

            # Log detailed summary of what was pushed
            artifacts_pushed = len(aid_to_folder)
            telemetry.emit(MessageType.SUCCESS, f"   Pushed to repo: {artifacts_pushed} artifacts, {len(files_added)} total files")
            telemetry.emit(MessageType.SUCCESS, f"   GitHub: {repo_url}")
            for aid, folder in aid_to_folder.items():
                folder_files = [f for f in files_added if f.startswith(f"{folder}/")]
                telemetry.emit(MessageType.INFO, f"      {folder}: {len(folder_files)} files")
            telemetry.emit(MessageType.SUCCESS, "=" * 50)

            # Emit as SYSTEM so it appears in JSONL logs (SUCCESS/INFO filtered by JSONSink)
            telemetry.emit(MessageType.SYSTEM, f"Artifacts pushed to GitHub: {artifacts_pushed} artifacts, {len(files_added)} files → {repo_url}")

            return True, aid_to_folder

    except subprocess.TimeoutExpired as e:
        telemetry.emit(MessageType.ERROR, f"   FAILED: Git command timed out: {e}")
        raise
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"   FAILED: Exception during populate_repo: {e}")
        raise



async def run_deploy_to_repo_module(
    config: PipelineConfig,
    artifacts: list[BaseArtifact],
    prepared_artifacts: list,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    repo_url: str | None = None,
    paper_pdf_path: Path | None = None,
    paper_latex_dir: Path | None = None,
    figures: list[Figure] | None = None,
) -> list[GistDeployment]:
    """
    Run the B_DEPLOY_TO_REPO step.

    Populates the GitHub repo with:
    - {artifact_id}/src/: Entire workspace from invention loop
    - {artifact_id}/demo/: Self-contained Jupyter notebooks
    - figures/: Visualization figures
    - paper/: PDF of the research paper

    README includes Colab links for notebooks and Lean playground links for proofs.

    Args:
        config: Pipeline configuration
        artifacts: Original artifact dicts from discovery loop
        prepared_artifacts: Prepared artifacts from prepare_artifacts step
        artifacts: BaseArtifact objects from invention loop
        telemetry: AIITelemetry instance
        output_dir: Output directory
        repo_url: GitHub repo URL to populate
        paper_pdf_path: Path to paper.pdf if available
        paper_latex_dir: Path to LaTeX source directory if available

    Returns:
        List of GistDeployment objects (artifact URLs in repo)
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "deploy_to_repo_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "deploy_to_repo_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("B_DEPLOY_TO_REPO")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "B_DEPLOY_TO_REPO - Populating GitHub repository")
        telemetry.emit(MessageType.INFO, "=" * 60)

        if not artifacts:
            telemetry.emit(MessageType.WARNING, "No artifacts to deploy")
            return []

        # Check if gh CLI is available and authenticated
        try:
            check_gh = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            if check_gh.returncode != 0:
                telemetry.emit(MessageType.WARNING, "gh CLI not authenticated, skipping deployment")
                return []
        except FileNotFoundError:
            telemetry.emit(MessageType.WARNING, "gh CLI not installed, skipping deployment")
            return []

        telemetry.emit(MessageType.INFO, f"   Artifacts: {len(artifacts)}")
        telemetry.emit(MessageType.INFO, f"   Repo URL: {repo_url or 'N/A'}")

        deployments = []

        # Populate the GitHub repo
        if repo_url and prepared_artifacts:
            success, aid_to_folder = await populate_repo(
                telemetry=telemetry,
                repo_url=repo_url,
                prepared_artifacts=prepared_artifacts,
                artifacts=artifacts,
                paper_pdf_path=paper_pdf_path,
                paper_latex_dir=paper_latex_dir,
                figures=figures,
            )

            if not success:
                telemetry.emit(MessageType.ERROR, "   B_DEPLOY_TO_REPO: populate_repo FAILED - check logs above for details")

            if success:
                # Create deployment entries with Colab URLs for notebooks
                for prep in prepared_artifacts:
                    aid = prep.id
                    # Use sanitized folder name from mapping
                    folder_name = aid_to_folder.get(aid, aid)
                    demo_path = Path(prep.demo_path)

                    # demo_path may be a directory (for notebooks) or a file (for markdown)
                    if demo_path.is_dir():
                        # Find the main file in the directory
                        if prep.type.value == "code":
                            ipynb_files = list(demo_path.glob("*.ipynb"))
                            main_file = ipynb_files[0].name if ipynb_files else "code_demo.ipynb"
                        else:
                            main_file = f"{aid}.md"
                        all_files = [f.name for f in demo_path.iterdir() if f.is_file()]
                    else:
                        main_file = demo_path.name
                        all_files = [main_file]

                    # Build relative path in repo: {folder_name}/demo/{filename}
                    rel_path_str = f"{folder_name}/demo/{main_file}"

                    # Build URLs
                    github_url = f"{repo_url}/blob/main/{rel_path_str}"

                    # For notebooks, also provide Colab URL
                    colab_url = None
                    if prep.type.value == "code":
                        colab_url = github_to_colab_url(github_url)

                    deployments.append(GistDeployment(
                        artifact_id=aid,
                        gist_url=github_url,
                        gist_id=folder_name,  # Use folder name as gist_id
                        files=all_files,
                        colab_url=colab_url,
                    ))

        elif not repo_url:
            telemetry.emit(MessageType.WARNING, "No repo URL provided, skipping deployment")
        elif not prepared_artifacts:
            telemetry.emit(MessageType.WARNING, "No prepared artifacts provided, skipping deployment")

        telemetry.emit(MessageType.SUCCESS, f"B_DEPLOY_TO_REPO complete: {len(deployments)} artifacts in repo")

        # Save output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "gist_deployments.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "deployments": [
                        {
                            "artifact_id": d.artifact_id,
                            "gist_url": d.gist_url,
                            "gist_id": d.gist_id,
                            "files": d.files,
                            "colab_url": d.colab_url,
                        }
                        for d in deployments
                    ],
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "module": "deploy_to_repo",
                        "llm_provider": "gh_cli",
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                }, f, indent=2, ensure_ascii=False)
            telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

            # Save artifact URL mapping with Colab links
            artifact_urls = {}
            for d in deployments:
                artifact_urls[d.artifact_id] = {
                    "github": d.gist_url,
                    "colab": d.colab_url,
                }
            urls_file = output_dir / "gists.json"
            with open(urls_file, 'w', encoding='utf-8') as f:
                json.dump(artifact_urls, f, indent=2)

        telemetry.emit_module_summary("B_DEPLOY_TO_REPO")

        return deployments

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
