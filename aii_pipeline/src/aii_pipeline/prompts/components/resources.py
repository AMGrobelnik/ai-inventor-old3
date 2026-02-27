"""Resource definitions for prompts.

Split into reusable parts so different artifacts can include only what they need.
"""

from __future__ import annotations


def get_resources_software():
    """Software constraints."""
    return """<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- STRICTLY SYNCHRONOUS: no async/await, asyncio, multithreading, multiprocessing
- Each script run must complete in under 1 hour
- No distributed computing
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum 10,000 LLM API calls total across all scripts and tools
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>"""


def get_resources_tooluniverse():
    """ToolUniverse capabilities."""
    return """<tooluniverse>
ToolUniverse provides 650+ scientific tools organized into package libraries and specialized APIs.

PACKAGE LIBRARIES (Python library info, installation, usage examples):
- Packages Tools: PyPI package discovery and quality scoring with download stats and maintenance metrics
- Bioinformatics Core Tools: Biopython, scikit-bio, biotite, gget for molecular biology and sequence analysis
- Cheminformatics Tools: RDKit, OpenBabel, DeepChem for molecular modeling and drug-likeness prediction
- Earth Sciences Tools: GeoPandas, Cartopy, Rasterio, xESMF for geospatial data and climate modeling
- Genomics Tools: pysam, pyfaidx, PyRanges, pybedtools for SAM/BAM parsing and genomic intervals
- Image Processing Tools: Pillow, imageio, albumentations for image manipulation and augmentation
- Machine Learning Tools: scikit-learn, PyTorch, TensorFlow wrappers for model training and evaluation
- Neuroscience Tools: MNE, nilearn, brian2 for EEG/MEG analysis and neural network simulation
- Physics Astronomy Tools: Astropy, SunPy, galpy, QuTiP for astrophysics and quantum computing
- Scientific Computing Tools: NumPy, SciPy, pandas integrations for numerical computing
- Single Cell Tools: Scanpy, scRNA-seq analysis pipelines for single-cell genomics
- Structural Biology Tools: PDBFixer, protein structure preparation and validation
- Visualization Tools: Matplotlib, Seaborn, OpenCV, scikit-image for plotting and computer vision

SPECIALIZED TOOL APIS:

Literature & Knowledge:
- Europe PMC Tools: Literature search, clinical guidelines, full-text access to life sciences papers
- OpenAlex Tools: Academic literature search with citation metrics and institutional affiliations
- PubTator Tools: Biomedical entity recognition, gene/disease/chemical tagging in literature
- Semantic Scholar Tools: Academic paper search with AI-generated summaries and citations

Drug & Compound:
- ADMET AI Tools: Predict toxicity, BBB penetrance, bioavailability, CYP interactions from SMILES
- ChEMBL Tools: Drug similarity search, compound bioactivity data via OpenTargets integration
- DailyMed Tools: FDA drug label lookup by name/NDC/RxCUI, complete SPL document retrieval
- FDA Drug Labeling Tools: 158 tools for drug label sections (dosage, warnings, interactions)
- PubChem Tools: Compound properties, patents, substructure search, CID/SMILES conversions

Clinical & Safety:
- Adverse Event Tools: FDA FAERS queries, clinical trial adverse event extraction and aggregation
- ClinicalTrials.gov Tools: Search trials by condition/intervention, extract eligibility and outcomes
- FDA Drug Adverse Event Tools: FAERS database queries, adverse reaction counts by demographics
- MedlinePlus Tools: Health topics, genetics info, drug information for consumer health
- ODPHP Tools: Personalized preventive care recommendations by age/sex/health status

Genomics & Genetics:
- Enrichr Tools: Gene set enrichment analysis across biological pathways and processes
- Gene Ontology Tools: GO term queries, biological process/molecular function annotations
- GWAS Tools: Genome-wide association study search by trait/SNP/gene with effect sizes
- ID Mapping Tools: Convert between gene/protein identifiers (Ensembl, UniProt, HGNC, Entrez)

Protein & Structure:
- AlphaFold Tools: Retrieve protein 3D structure predictions, confidence scores, mutations
- RCSB PDB Tools: 39 tools for Protein Data Bank queries, structure metadata, ligands
- UniProt Tools: Protein sequences, functions, disease variants, subcellular localization

Pathways & Disease:
- Disease Target Score Tools: Aggregate evidence scores from literature, pathways, clinical sources
- EFO Tools: Experimental Factor Ontology lookups for disease/phenotype standardization
- HPA Tools: Human Protein Atlas expression data, tissue/cell line comparisons, antibodies
- HumanBase Tools: Tissue-specific protein-protein interaction networks from functional genomics
- Monarch Tools: Disease-phenotype-gene associations from integrated biomedical ontologies
- OpenTargets Tools: 54 tools for drug-target-disease associations, evidence aggregation
- Reactome Tools: Biological pathway reactions, disease-pathway associations

Agentic & Composition:
- Agentic Tools: HypothesisGenerator, LiteratureReviewer, ExperimentalDesignScorer, summarizers
- Compose Tools: Build multi-tool workflows, tool graph composition, output chaining
- Finder Tools: Tool discovery and search across ToolUniverse by keyword or capability
- Output Summarization Tools: AI-powered summarization of long tool outputs and research texts

Execution & Utilities:
- Dataset Tools: GEO dataset search, metadata retrieval for gene expression studies
- Embedding Tools: Create/search/add to vector databases with OpenAI/HuggingFace embeddings
- Execution Tools: Python code/script execution in sandboxed environment with resource limits
- URL Fetch Tools: Web page rendering with JavaScript support, text extraction from URLs
- Web Search Tools: General web search via multiple engines (Google, Bing, DuckDuckGo) no API keys
</tooluniverse>"""


def get_resources_custom_tools(use_aii_web_tools: bool = False):
    """AI-Inventor custom tools available via MCP.

    Args:
        use_aii_web_tools: If True, list aii_web_search_fast and aii_web_fetch_direct (MCP tools).
                          If False (default), list WebSearch and WebFetch (built-in tools).
    """
    if use_aii_web_tools:
        web_tools = """- aii_web_search_fast: Web search via Serper.dev (Google API). Returns titles, URLs, snippets.
- aii_web_fetch_direct: Fetch a web page or PDF as text. Supports pagination via char_offset."""
    else:
        web_tools = """- WebSearch: Search the web and get up-to-date information. Returns search results with titles, URLs, and snippets.
- WebFetch: Fetch content from a URL and process it. Retrieves web pages, converts HTML to readable text, and extracts information."""

    return f"""<ai_inventor_custom_tools>
{web_tools}
- aii_web_fetch_grep: Regex grep through a web page or PDF. Returns matches with context windows.
- dblp_bib_search: Search DBLP for academic papers by author, title, or year.
- openrouter_search_llms: Search OpenRouter's 300+ LLM catalog by name or capability.
- openrouter_call_llm: Call any LLM via OpenRouter (GPT, Claude, Gemini, Llama, DeepSeek, etc.).
- lean_run_code: Compile and verify Lean 4 code with Mathlib support.
</ai_inventor_custom_tools>"""


def get_resources_skills():
    """Skills available to agents."""
    return """<skills>
- aii_long_running_tasks: Gradual scaling pattern for experiments/evaluations (mini → 10 → 50 → 100 → 200 → max)
- aii_paper_writing: Academic paper writing (structure, figure placeholders, LaTeX, bibliography, citations)
- aii_nano_banana: AI image generation for research figures (Gemini 2.5 Flash, NeurIPS style, CLI script)
- aii_web_research_tools: Web research tool guidance (search, fetch, grep workflows)
- aii_handbook_multi_llm_agents: Multi-LLM Agent Systems implementation guide with patterns and evaluation
- aii_hf_datasets: Search, preview, and download HuggingFace datasets
- aii_json: JSON validation and formatting (schema compliance, mini/preview variants)
- aii_lean: Lean 4 formal verification with Mathlib search and tactic suggestions
- aii_get_hardware: Detect available hardware and OS (CPUs, RAM, GPU/VRAM, disk) before writing compute code
- aii_agent_conflicts: Isolation rules for parallel agents (PID-only process management, GPU/RAM sharing, file isolation)
- aii_parallel_computing: Maximize hardware utilization (GPU detection, ProcessPoolExecutor, asyncio, PyTorch GPU patterns)
- aii_openrouter_llms: Search and call LLMs from OpenRouter's catalog with reasoning control
- aii_owid_datasets: Our World in Data catalog search with BM25 (global statistics)
- amg_prompt_optim: LLM prompt optimization for conciseness
- anthropic_docx: Document creation and editing (.docx)
- anthropic_pdf: PDF manipulation (extract, create, merge, split, forms)
- anthropic_pptx: Presentation creation and editing (.pptx)
- dblp_bib: Build bibliographies from DBLP (search papers, fetch BibTeX)
</skills>"""


# =============================================================================
# REGISTRY & PUBLIC API
# =============================================================================

_RESOURCE_SECTIONS = {
    "software": get_resources_software,
    "tooluniverse": get_resources_tooluniverse,
}

DEFAULT_SECTIONS = ["software", "tooluniverse"]

# Per-artifact-type resource needs (used by gen_strat, gen_plan to select relevant sections)
ARTIFACT_RESOURCES: dict[str, set[str]] = {
    "research": {"software"},
    "proof": {"software"},
    "dataset": {"software", "tooluniverse"},
    "experiment": {"software", "tooluniverse"},
    "evaluation": {"software", "tooluniverse"},
}


def get_resources_prompt(
    include: list[str] | None = None,
    use_aii_web_tools: bool = False,
) -> str:
    """Get formatted resources block for prompts.

    Args:
        include: Which resource sections to include (e.g. ["software", "tooluniverse"]).
                 Options: "software", "tooluniverse".
                 If None, includes all DEFAULT_SECTIONS.
                 custom_tools and skills are ALWAYS included regardless.
        use_aii_web_tools: If True, list aii_web_search_fast/aii_web_fetch_direct (MCP).
                          If False (default), list WebSearch/WebFetch (built-in).

    Returns:
        Formatted string wrapped in <available_resources> tags.
    """
    if include is None:
        include = DEFAULT_SECTIONS

    sections = []
    for key in include:
        if key in _RESOURCE_SECTIONS:
            sections.append(_RESOURCE_SECTIONS[key]())

    # Always include custom tools and skills
    sections.append(get_resources_custom_tools(use_aii_web_tools=use_aii_web_tools))
    sections.append(get_resources_skills())

    content = "\n\n".join(sections)
    return f"""<available_resources>
{content}
</available_resources>"""
