"""Push utilities for paper and figures to GitHub repo."""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from aii_lib import AIITelemetry, MessageType

from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure


def push_paper_to_repo(
    repo_url: str,
    paper_tex_path: Path | None,
    paper_pdf_path: Path | None,
    figures: list[Figure],
    telemetry: AIITelemetry,
    paper_bib_path: Path | None = None,
    workspace_dir: Path | None = None,
) -> bool:
    """Push paper (LaTeX + PDF + bibliography) and figures to GitHub repo."""
    # Log what we're attempting to push
    telemetry.emit(MessageType.INFO, "")
    telemetry.emit(MessageType.SUCCESS, "=" * 50)
    telemetry.emit(MessageType.SUCCESS, ">>> PUSH PAPER TO GITHUB <<<")
    telemetry.emit(MessageType.INFO, f"   repo_url: {repo_url or 'NONE'}")
    telemetry.emit(MessageType.INFO, f"   paper_tex: {paper_tex_path} (exists={paper_tex_path.exists() if paper_tex_path else 'N/A'})")
    telemetry.emit(MessageType.INFO, f"   paper_pdf: {paper_pdf_path} (exists={paper_pdf_path.exists() if paper_pdf_path else 'N/A'})")
    telemetry.emit(MessageType.INFO, f"   paper_bib: {paper_bib_path} (exists={paper_bib_path.exists() if paper_bib_path else 'N/A'})")
    telemetry.emit(MessageType.INFO, f"   figures: {len(figures)}")
    telemetry.emit(MessageType.INFO, f"   workspace_dir: {workspace_dir}")

    if not repo_url:
        telemetry.emit(MessageType.ERROR, "   ABORTED: No repo_url provided")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"

            # Clone the repo
            telemetry.emit(MessageType.INFO, "   Cloning repository to add paper...")
            clone_result = subprocess.run(
                ["gh", "repo", "clone", repo_url, str(repo_dir)],
                capture_output=True,
                text=True,
            )
            if clone_result.returncode != 0:
                telemetry.emit(MessageType.WARNING, f"Failed to clone repo: {clone_result.stderr}")
                return False

            # Create paper folder for LaTeX source files
            paper_folder = repo_dir / "paper"
            paper_folder.mkdir(exist_ok=True)

            # Copy paper.tex to paper/ folder
            if paper_tex_path and paper_tex_path.exists():
                shutil.copy(paper_tex_path, paper_folder / "paper.tex")

            # Copy paper.pdf to ROOT of repo (for easy access)
            if paper_pdf_path and paper_pdf_path.exists():
                shutil.copy(paper_pdf_path, repo_dir / "paper.pdf")

            # Copy references.bib to paper/ folder
            if paper_bib_path and paper_bib_path.exists():
                shutil.copy(paper_bib_path, paper_folder / "references.bib")

            # Create figures folder and copy figures
            figures_folder = repo_dir / "figures"
            figures_folder.mkdir(exist_ok=True)

            # Copy figures from VIZ_GEN
            figures_copied = 0
            figures_skipped = 0
            for fig in figures:
                if fig.figure_path:
                    src = Path(fig.figure_path)
                    if src.exists():
                        shutil.copy(src, figures_folder / src.name)
                        figures_copied += 1
                    else:
                        telemetry.emit(MessageType.WARNING, f"   Figure file not found: {fig.figure_path}")
                        figures_skipped += 1
                else:
                    telemetry.emit(MessageType.WARNING, f"   Figure {fig.id} has no figure_path")
                    figures_skipped += 1

            # If no VIZ_GEN figures, check workspace for agent-generated figures
            if figures_copied == 0 and workspace_dir:
                workspace_figures = workspace_dir / "figures"
                if workspace_figures.exists():
                    telemetry.emit(MessageType.INFO, f"   Checking workspace figures at {workspace_figures}")
                    for ext in ["*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg", "*.tex"]:
                        for fig_file in workspace_figures.glob(ext):
                            shutil.copy(fig_file, figures_folder / fig_file.name)
                            figures_copied += 1

            if figures_copied > 0:
                telemetry.emit(MessageType.INFO, f"   Copied {figures_copied} figures to repo")
            else:
                telemetry.emit(MessageType.WARNING, f"   No figures copied to repo (skipped: {figures_skipped})")

            # Update paper.tex to use relative figure paths (../figures/)
            tex_file = paper_folder / "paper.tex"
            if tex_file.exists():
                tex_content = tex_file.read_text()
                # Replace absolute paths with relative paths
                for fig in figures:
                    if fig.figure_path:
                        src = Path(fig.figure_path)
                        # Handle various path formats in LaTeX
                        tex_content = tex_content.replace(
                            f"figures/{src.name}",
                            f"../figures/{src.name}"
                        )
                        tex_content = tex_content.replace(
                            fig.figure_path,
                            f"../figures/{src.name}"
                        )
                tex_file.write_text(tex_content)

            # Update README with paper links
            readme_path = repo_dir / "README.md"
            if readme_path.exists():
                readme_content = readme_path.read_text()

                # Check if paper section already exists (h2 or h3)
                has_paper_section = (
                    "## Research Paper" in readme_content or
                    "### Research Paper" in readme_content
                )

                if not has_paper_section:
                    # Build paper section with h2 heading
                    paper_section = "## Research Paper\n\n"

                    if paper_pdf_path and paper_pdf_path.exists():
                        pdf_badge = f"[![Download PDF](https://img.shields.io/badge/Download-PDF-red)]({repo_url}/blob/main/paper.pdf)"
                        paper_section += f"{pdf_badge} "

                    if paper_tex_path and paper_tex_path.exists():
                        latex_badge = f"[![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)]({repo_url}/tree/main/paper)"
                        paper_section += f"{latex_badge}"

                    # Count figures being pushed
                    num_figures = len([f for f in figures if f.figure_path and Path(f.figure_path).exists()])
                    if num_figures > 0:
                        figures_badge = f"[![Figures](https://img.shields.io/badge/Figures-{num_figures}-blue)]({repo_url}/tree/main/figures)"
                        paper_section += f" {figures_badge}"

                    paper_section += "\n\n"

                    # Insert at TOP after the title and description
                    # Look for the first h2 section after the intro
                    lines = readme_content.split('\n')
                    insert_idx = 0

                    # Find the end of intro (after "# " title and description paragraph)
                    in_intro = True
                    for i, line in enumerate(lines):
                        if in_intro:
                            # Skip title line and empty lines
                            if line.startswith('# ') or line.strip() == '':
                                continue
                            # First non-empty, non-title line is part of description
                            # Continue until we hit the next h2 or blank line after description
                            if line.startswith('## '):
                                insert_idx = i
                                in_intro = False
                                break
                            # If we hit an empty line after description text, insert after it
                            if i > 0 and not lines[i-1].strip() == '' and line.strip() == '':
                                continue
                        else:
                            break

                    if insert_idx > 0:
                        # Insert paper section before the first h2
                        lines.insert(insert_idx, paper_section)
                        readme_content = '\n'.join(lines)
                    else:
                        # Fallback: append after intro
                        readme_content += "\n" + paper_section

                    readme_path.write_text(readme_content)

            # Git add, commit, push
            telemetry.emit(MessageType.INFO, "   Running git add...")
            add_result = subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, text=True)
            if add_result.returncode != 0:
                telemetry.emit(MessageType.WARNING, f"   git add failed: {add_result.stderr}")

            telemetry.emit(MessageType.INFO, "   Running git commit...")
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Add paper (LaTeX + PDF) and figures"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if commit_result.returncode != 0:
                if "nothing to commit" in commit_result.stdout.lower() or "nothing to commit" in commit_result.stderr.lower():
                    telemetry.emit(MessageType.WARNING, "   Nothing new to commit - files may already exist in repo")
                    return True  # Not an error, just nothing to push
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
                    telemetry.emit(MessageType.WARNING, f"Push failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    telemetry.emit(MessageType.ERROR, f"Push failed: {push_result.stderr}")
                    raise RuntimeError(f"Git push failed after {max_retries} retries: {push_result.stderr}")

            # Log what was actually pushed
            pushed_items = []
            if paper_pdf_path and paper_pdf_path.exists():
                pushed_items.append("paper.pdf")
            if paper_tex_path and paper_tex_path.exists():
                pushed_items.append("paper/paper.tex")
            if paper_bib_path and paper_bib_path.exists():
                pushed_items.append("paper/references.bib")
            if figures_copied > 0:
                pushed_items.append(f"{figures_copied} figures")

            telemetry.emit(MessageType.SUCCESS, f"   Pushed to repo: {', '.join(pushed_items) if pushed_items else 'README only'}")
            telemetry.emit(MessageType.SUCCESS, f"   GitHub: {repo_url}")
            telemetry.emit(MessageType.SUCCESS, "=" * 50)
            # Emit as SYSTEM so it appears in JSONL logs (SUCCESS filtered by JSONSink)
            telemetry.emit(MessageType.SYSTEM, f"Paper+figures pushed to GitHub: {', '.join(pushed_items)} → {repo_url}")
            return True

    except subprocess.TimeoutExpired as e:
        telemetry.emit(MessageType.ERROR, f"   FAILED: Git command timed out: {e}")
        raise
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"   FAILED: Exception during push: {e}")
        raise
