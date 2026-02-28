---
name: aii_image_gen_nano_banana
description: AI image generation for research figures using Gemini API directly (google-genai SDK). Covers prompting best practices, NeurIPS figure style, and CLI script for batch generation.
---

## Contents

- Workflow (prompt design and generation)
- Scripts (Generate Image)
- Prompting Best Practices
- Figure Templates (bar charts, line plots, etc.)

**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for variables and **single-quoted** command templates so parallel's subshells can resolve them:
```
export PY="$(git rev-parse --show-toplevel)/.claude/skills/aii_image_gen_nano_banana/scripts/.venv/bin/python"
```

---

## Workflow: Image Generation

### Phase 1: Write a Detailed Prompt

The model generates from your text description ONLY. It cannot read data files, run code, or access external context. Every detail must be in the prompt.

**BAD:** "Show the performance results"
**GOOD:** "Grouped bar chart with white background. X-axis: 'Method A', 'Method B', 'Baseline'. Y-axis: Accuracy (0.0 to 1.0). Bar values: Method A = 0.847, Method B = 0.762, Baseline = 0.531. Error bars: 0.02, 0.03, 0.05. Sans-serif labels. No gridlines."

### Phase 2: Generate Image

```bash
SKILL_DIR="$(git rev-parse --show-toplevel)/.claude/skills/aii_image_gen_nano_banana" && \
$SKILL_DIR/scripts/.venv/bin/python $SKILL_DIR/scripts/aii_image_gen_nano_banana.py \
  --prompt "Grouped bar chart with white background..." \
  --output ./figures/fig_1.png \
  --aspect-ratio 16:9 \
  --image-size 1K \
  --style neurips
```

---

## Scripts

### Generate image (aii_image_gen_nano_banana.py)

Generate research figures via Gemini API (`gemini-3-pro-image-preview`).

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel)/.claude/skills/aii_image_gen_nano_banana" && \
$SKILL_DIR/scripts/.venv/bin/python $SKILL_DIR/scripts/aii_image_gen_nano_banana.py \
  --prompt "Bar chart with white background. X-axis: models. Y-axis: accuracy (0 to 1). Values: GPT-4=0.85, Claude=0.91, Baseline=0.45. Blue, orange, gray bars. Sans-serif labels, no gridlines." \
  --output ./figures/fig_1.png \
  --aspect-ratio 16:9 \
  --style neurips
```

**Parallel execution (multiple figures):**

IMPORTANT: When generating multiple figures, use GNU parallel instead of separate Bash tool calls:
```bash
export PY="$(git rev-parse --show-toplevel)/.claude/skills/aii_image_gen_nano_banana/scripts/.venv/bin/python" && \
export S="$(git rev-parse --show-toplevel)/.claude/skills/aii_image_gen_nano_banana/scripts/aii_image_gen_nano_banana.py" && \
parallel -j 3 -k --group --will-cite 'eval {}' ::: \
  '$PY $S -p "Bar chart..." -o ./figures/fig_1.png --style neurips' \
  '$PY $S -p "Line plot..." -o ./figures/fig_2.png --style neurips' \
  '$PY $S -p "Heatmap..." -o ./figures/fig_3.png --aspect-ratio 1:1 --style neurips'
```

**Example output:**
```
Image saved: ./figures/fig_1.png (363325 bytes, 1376x768)
{
  "success": true,
  "output_path": "/path/to/figures/fig_1.png",
  "model": "gemini-3-pro-image-preview",
  "dimensions": "1376x768",
  "aspect_ratio": "16:9",
  "image_size": "1K",
  "image_bytes": 363325,
  "input_tokens": 257,
  "output_tokens": 1624,
  "attempts": 1
}
```

**Parameters:**

`--prompt, -p` (required)
- Full image description with all data values, labels, and style
- Must include ALL numeric values explicitly

`--output, -o` (optional)
- Output file path (default: `./generated_image.png`)

`--aspect-ratio` (optional)
- Canvas shape (default: `16:9`)
- Valid: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

`--image-size` (optional)
- Resolution (default: `1K`)
- Valid: `1K`, `2K`, `4K`
- 2K works reliably at 16:9 and 1:1. 4K is inconsistent
- Auto-fallback: if generation fails, retries other sizes

`--negative-prompt` (optional)
- Things to exclude from the image

`--style` (optional)
- Preset style: `neurips` (appends NeurIPS camera-ready style)

`--system` (optional)
- System instruction for style guidance

`--timeout` (optional)
- Request timeout in seconds (default: 180)

**Tips:**
- Use `--style neurips` for all publication figures
- Include ALL data values in the prompt (axis ranges, bar values, labels, error bars)
- Specify colors, fonts, and what to exclude
- 1K resolution is most reliable; 2K works for 16:9 and 1:1
- Resolution auto-fallback: never fails due to resolution alone

---

## Prompting Best Practices

### Include All Numeric Values

For data-driven figures, list EVERY data point explicitly:
- Axis ranges and labels
- All bar/point values with labels
- Error bars or confidence intervals
- Legend entries

### Specify Layout and Style

- Canvas shape via `--aspect-ratio` (`16:9` for wide, `4:3` for standard, `1:1` for square)
- Color scheme (use distinct, accessible colors)
- Font style (sans-serif for publication)
- Background (white for papers)
- What to EXCLUDE (no shadows, no 3D effects, no decorative elements)

### NeurIPS Camera-Ready Style

Appended automatically with `--style neurips`:
```
Clean white background, no borders or decorative elements.
Sans-serif font labels (Helvetica/Arial style), clearly readable at print size.
Properly formatted axes with labeled tick marks.
Minimal gridlines (light gray, dotted if needed).
No 3D effects, no shadows, no gradients.
Proportions suitable for a two-column NeurIPS paper layout.
```

---

## Figure Templates

### 1. Bar Charts (most common)
```
Grouped bar chart with white background.
X-axis: [category labels]. Y-axis: [metric name] (range).
Values: [list all values].
Error bars: [list all values].
Sans-serif labels, no gridlines, distinct colors per group.
Legend in upper-right corner.
```

### 2. Line Plots
```
Line plot with white background.
X-axis: [variable] (range). Y-axis: [metric] (range).
Lines: [name1] with points at [...], [name2] with points at [...].
Solid lines, distinct colors, circle markers.
Sans-serif labels, light gray horizontal gridlines.
```

### 3. Architecture Diagrams
```
Left-to-right flowchart on white background.
Boxes: [component1] -> [component2] -> [component3].
Labeled arrows between boxes.
Rounded rectangle boxes with light fill colors.
Sans-serif labels, clean layout.
```

### 4. Confusion Matrices / Heatmaps
```
Heatmap on white background, square aspect ratio.
Rows: [labels]. Columns: [labels].
Cell values: [list all values in row-major order].
Color scale: white (0) to dark blue (1).
Numbers displayed in each cell. Sans-serif labels.
```
