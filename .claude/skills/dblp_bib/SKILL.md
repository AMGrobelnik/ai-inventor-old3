---
name: dblp_bib
description: Build bibliographies using DBLP. Search for papers by author/title/year, then fetch BibTeX entries for the ones you need. Use when writing papers, generating reference lists, or building .bib files.
---

## Contents

- Scripts (Search, Fetch)
- When to Use

**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for variables and **single-quoted** command templates so parallel's subshells can resolve them:
```
export PY=".claude/skills/dblp_bib/scripts/.venv/bin/python"
```

---

## Scripts

### DBLP Search (dblp_bib_search.py)

Search DBLP for papers. Returns metadata (title, authors, venue, year, `dblp_key`).

**Example input:**
```bash
SKILL_DIR=".claude/skills/dblp_bib" && \
$SKILL_DIR/scripts/.venv/bin/python $SKILL_DIR/scripts/dblp_bib_search.py --query "Vaswani attention 2017"
```

**Parallel execution (multiple queries):**

IMPORTANT: When searching for multiple papers, use GNU parallel instead of separate Bash tool calls:
```bash
export PY=".claude/skills/dblp_bib/scripts/.venv/bin/python" && \
export S=".claude/skills/dblp_bib/scripts/dblp_bib_search.py" && \
parallel -j 5 -k --group --will-cite '$PY $S -q {}' ::: 'Vaswani attention 2017' 'Wei chain of thought 2022' 'Yao tree of thoughts 2023'
```

**Example output:**
```
Search: Vaswani attention 2017
Found: 5 total, showing 5

1. Attention is All you Need.
   Authors: Ashish Vaswani, Noam Shazeer, ...
   Venue: NIPS  Year: 2017
   DBLP key: conf/nips/VaswaniSPUJGKP17
```

**Parameters:**

`--query, -q` (required)
- Search query string
- **Best results:** author last name + year (e.g. `"Vaswani 2017"`, `"Wei chain of thought 2022"`)
- Also works for topic searches (e.g. `"multi-agent debate LLM"`)

`--max-results, -n` (optional)
- Maximum papers to return (default: 5, max: 20)

`--year-from` (optional)
- Only include papers from this year onward

`--year-to` (optional)
- Only include papers up to this year

`--json, -j` (optional)
- Output raw JSON instead of formatted text

---

### DBLP BibTeX Fetch (dblp_bib_fetch.py)

Fetch BibTeX entries by `dblp_key`. Returns ready-to-use BibTeX strings.

**Example input:**
```bash
SKILL_DIR=".claude/skills/dblp_bib" && \
$SKILL_DIR/scripts/.venv/bin/python $SKILL_DIR/scripts/dblp_bib_fetch.py --keys "conf/nips/VaswaniSPUJGKP17" --years 2017
```

**Multiple keys in one call:**
```bash
SKILL_DIR=".claude/skills/dblp_bib" && \
$SKILL_DIR/scripts/.venv/bin/python $SKILL_DIR/scripts/dblp_bib_fetch.py \
  --keys "conf/nips/VaswaniSPUJGKP17" "conf/nips/YaoYZS00N23" \
  --years 2017 2023
```

**Example output:**
```
Fetched 2 BibTeX entries

@inproceedings{Vaswani2017,
  author    = {Ashish Vaswani and ...},
  title     = {Attention is All you Need},
  booktitle = {NIPS},
  year      = {2017},
}

@inproceedings{Yao2023,
  ...
}
```

**Parameters:**

`--keys, -k` (required)
- One or more DBLP keys from search results
- Example: `--keys "conf/nips/VaswaniSPUJGKP17" "conf/nips/YaoYZS00N23"`

`--years, -y` (optional)
- Publication year(s) for cleaner citation keys (e.g. `Vaswani2017` instead of raw slug)
- Must match order of `--keys`

`--json, -j` (optional)
- Output raw JSON instead of BibTeX text

---

## When to Use

Use this skill when you **already know which papers you want to cite** and need to build a bibliography. This is NOT for discovering new papers or exploring literature — use web search for that.

**Search tips:**
- Author last name + year is best: `"Vaswani 2017"`, `"Wei chain of thought 2022"`
- Topic searches work but are less precise: `"multi-agent debate LLM"`
- DBLP returns both conference and arxiv versions — prefer conference (has pages, booktitle)

**Using the citations:**
- **LaTeX:** Put BibTeX entries into your `.bib` file and cite with `\cite{Vaswani2017}`
- **Non-LaTeX:** Use the metadata (authors, title, venue, year) to construct your bibliography manually

Do NOT fabricate BibTeX — always fetch from DBLP. Only write entries manually as a fallback for papers DBLP doesn't have.
