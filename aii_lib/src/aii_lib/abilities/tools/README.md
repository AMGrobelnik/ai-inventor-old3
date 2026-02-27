# aii_pipeline ToolUniverse Tools

Custom tools for the AI Inventor pipeline, registered with ToolUniverse for MCP server access.

## Tool Categories

### HuggingFace Datasets (`hf_datasets.py`)
Tools for searching, previewing, and downloading datasets from HuggingFace Hub.

| Tool | Description |
|------|-------------|
| `hf_dataset_search` | Search datasets by query and task type |
| `hf_dataset_info` | Get detailed metadata for a dataset |
| `hf_dataset_preview` | Preview sample rows (streaming) |
| `hf_dataset_configs` | List available configurations |
| `hf_dataset_download` | Download and save to JSON files |

### Cited WebFetch (`cited_webfetch.py`)
Tools for fetching web pages and verifying citations.

| Tool | Description |
|------|-------------|
| `cited_webfetch_page` | Fetch web page/PDF as markdown |
| `cited_verify_quotes` | Verify quoted citations against sources |

### OpenRouter LLMs (`openrouter.py`)
Tools for searching and calling LLMs via OpenRouter API.

| Tool | Description |
|------|-------------|
| `openrouter_search_llms` | Search OpenRouter model catalog |
| `openrouter_call_llm` | Make API calls to LLM models |

### Our World in Data (`owid_datasets.py`)
Tools for searching and loading OWID datasets.

| Tool | Description |
|------|-------------|
| `owid_search_datasets` | Search OWID table catalog |
| `owid_download_datasets` | Load table data with preview |

## Usage

### Starting MCP Server

```bash
# From aii_pipeline directory
python tools/start_mcp_server.py --port 7005
```

### Programmatic Import

```python
# Import to register tools
import aii_pipeline.tools

# Use with ToolUniverse
from tooluniverse import ToolUniverse
tu = ToolUniverse()
tu.load_tools(include_tools=["hf_dataset_search", "openrouter_call_llm"])

# Execute tool
result = tu.execute("hf_dataset_search", {"query": "sentiment analysis"})
```

### With aii_lib agent

```python
from aii_lib.agent_backend.claude import AgentOptions

options = AgentOptions(
    mcp_servers={
        "aii_tools": {
            "type": "http",
            "url": "http://localhost:7005/mcp"
        }
    },
    allowed_tools=[
        "mcp__aii_tools__hf_dataset_search",
        "mcp__aii_tools__openrouter_call_llm"
    ]
)
```

## Dependencies

Required packages (add to pyproject.toml):
- `huggingface-hub>=0.20.0` - HuggingFace API
- `datasets>=2.0.0` - Dataset loading
- `crawl4ai[pdf]>=0.4.0` - Web fetching
- `requests>=2.31.0` - HTTP requests
- `owid-catalog>=0.3.0` - OWID data
- `tooluniverse` - Tool framework
