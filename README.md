# AI Inventor

Automated research pipeline for generating, executing, and evaluating novel research ideas. Uses Claude Code Agent SDK for autonomous code generation and multi-LLM orchestration (OpenRouter) for research planning.

**Tested on Ubuntu 22.04 LTS.**

## Prerequisites

- **Python 3.10.19** (3.12+ is NOT supported — numpy dependency constraints)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** — authenticate with `claude login`
- **[GitHub CLI](https://cli.github.com/)** — authenticate with `gh auth login` (for repo creation in gen_paper_repo)
- **[Lean 4 via elan](https://github.com/leanprover/elan)** — required for proof artifacts (scripts use Lean v4.14.0 with Mathlib)
- **API keys** — see [API Keys](#api-keys)

## Quick Start

### Option A: Automated setup

```bash
git clone https://github.com/AMGrobelnik/ai-inventor.git
cd ai-inventor
bash scripts/setup.sh
```

The setup script installs uv/Python/Claude CLI if missing, creates the venv with CPU-only PyTorch, installs packages, and creates `.env` from the template.

After setup:
```bash
# 1. Fill in API keys
nano .env

# 2. Authenticate CLI tools
claude login
gh auth login

# 3. Run the pipeline
source .venv/bin/activate
unset CLAUDECODE
aii_pipeline
```

### Option B: Manual setup

```bash
# 1. Clone
git clone https://github.com/AMGrobelnik/ai-inventor.git
cd ai-inventor

# 2. Create venv
uv venv .venv --python=3.10
source .venv/bin/activate

# 3. Install CPU-only PyTorch (avoids 2.5GB GPU download)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install packages (aii_lib MUST be installed first)
uv pip install -e aii_lib/
uv pip install -e aii_pipeline/

# 5. Link skill venvs to main .venv
for d in .claude/skills/*/scripts; do ln -sf "$PWD/.venv" "$d/.venv"; done

# 6. Set up API keys
cp .env.template .env
nano .env  # Fill in your keys

# 7. Authenticate CLI tools
claude login
gh auth login

# 8. Configure and run
# Edit aii_pipeline/config.yaml (research_direction, pipeline steps)
unset CLAUDECODE
aii_pipeline
```

**Important:** `unset CLAUDECODE` is required before running — it prevents a "nested session" error when the pipeline spawns Claude Code agents.

## API Keys

Copy `.env.template` to `.env` and fill in your keys:

| Key | Required | Used By |
|-----|----------|---------|
| `GEMINI_API_KEY` | Yes | Image generation (viz_gen) |
| `HF_TOKEN` | Yes | Dataset downloads |
| `SERPER_API_KEY` | Yes | Web search (get_triples) |
| `LEANEXPLORE_API_KEY` | If proofs | Lean 4 proof artifacts |
| `OPENAI_API_KEY` | No* | OpenRouter fallback |
| `OPENROUTER_API_KEY` | No* | OpenRouter fallback |

*Only needed if `use_claude_agent: false` in config.

## Pipeline Modules

The pipeline runs these stages sequentially:

```
seed_hypo → gen_hypo → invention_loop → gen_paper_repo
```

| Module | Description |
|--------|-------------|
| `seed_hypo` | Build knowledge graph, find blind spots |
| `gen_hypo` | Generate full hypotheses via LLMs |
| `invention_loop` | Iterative: strategy → plan → execute |
| `gen_paper_repo` | Write paper, generate figures, deploy |

### Running Specific Modules

Control which modules run via `config.yaml`:

```yaml
init:
  pipeline:
    first_step: "gen_hypo"       # Start here
    last_step: "gen_paper_repo"  # Stop here
```

### CLI Overrides

Override any config value with dot notation:

```bash
aii_pipeline --init.research_direction "Machine Learning"
aii_pipeline --gen_hypo.claude_agent.model "claude-sonnet-4-5"
aii_pipeline --init.run_name "my-experiment"
```

## Configuration

Main config: `aii_pipeline/config.yaml`

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `init.research_direction` | (set) | Your research topic |
| `init.outputs_directory` | `runs` | Output dir (relative to `aii_pipeline/`) |
| `init.pipeline.first_step` | `seed_hypo` | First module to run |
| `init.pipeline.last_step` | `gen_paper_repo` | Last module to run |

### Claude Agent Settings

Each module's `claude_agent` block controls:

```yaml
claude_agent:
  model: claude-opus-4-6        # LLM model
  max_turns: 200                 # Max agent turns
  agent_timeout: 7200            # Total timeout (seconds)
  message_timeout: 1200          # Per-message timeout
  max_concurrent_agents: 30      # Parallel agents
```

## Project Structure

```
ai-inventor/
├── aii_lib/              # Core library (agent backend, telemetry)
│   ├── src/aii_lib/
│   │   ├── agent_backend/  # Claude Code SDK integration
│   │   ├── abilities/      # Tool implementations (web search)
│   │   ├── telemetry/      # Logging and usage tracking
│   │   └── config.py       # Global config (API keys)
│   └── pyproject.toml
├── aii_pipeline/         # Research pipeline
│   ├── src/aii_pipeline/
│   │   ├── cli.py          # Entry point (aii_pipeline command)
│   │   ├── pipeline.py     # Pipeline orchestrator
│   │   ├── steps/          # Module implementations
│   │   ├── prompts/        # LLM prompt templates
│   │   └── utils/          # Config parsing, helpers
│   ├── config.yaml         # Pipeline configuration
│   └── pyproject.toml
├── scripts/              # Setup and utility scripts
├── .env.template         # API key template
└── CLAUDE.md             # Claude Code agent instructions
```

## Output Structure

Each run creates a timestamped directory:

```
aii_pipeline/runs/{run_name}/
├── 1_seed_hypo/          # Hypothesis seeds
├── 2_gen_hypo/           # Generated hypotheses
├── 3_invention_loop/     # Iterative invention artifacts
│   ├── pools/            # Strategy, plan, artifact pools
│   └── iter{N}_*/        # Per-iteration workspaces
├── 4_gen_paper_repo/     # Paper + repository output
└── .hf_cache/            # Run-specific HuggingFace cache
```

## Resuming Runs

Set checkpoint paths in `config.yaml` to resume from a previous run:

```yaml
init:
  pipeline:
    first_step: "invention_loop"
    gen_hypo_out_dir: "/path/to/runs/my_run/2_gen_hypo"
```

