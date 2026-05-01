# Blog figures — vLLM × Mooncake Store

Diagrams and charts for the blog post on the Mooncake Store integration in
vLLM. Each figure lives in its own self-contained subdirectory with the
rendering script, the SVG output, and (where applicable) a stand-alone
gpt-image-2 prompt.

| Subdirectory | Section in blog | What it shows |
|---|---|---|
| [`agentic-trace/`](./agentic-trace/) | Motivation | Per-turn anatomy of one agentic trace: token counts, cached prefix, new tool output, decode size, and timing. Numbers are corpus medians (codex / SWE-bench Pro, 610 trials). |
| [`overall-design/`](./overall-design/) | Distributed KV cache pool with Mooncake Store | Architecture of the integration: control plane (Mooncake Master ↔ vLLM Schedulers, lookup) and data plane (workers, Mooncake clients, GPU HBM ↔ CPU/DRAM/SSD over GPU Direct RDMA). |
| [`pd-multiconnector/`](./pd-multiconnector/) | Enabling PD + distributed KV cache pool with MultiConnector | How `MultiConnector` chains a PD Connector and a MooncakeStore Connector so a single KV block flows to both the cross-instance PD link and the distributed cache pool simultaneously. |

## Regenerate the SVGs

The Python scripts have no third-party dependencies — they emit raw SVG that
any modern browser will render. Each script writes its SVG next to itself, so
running them is order-independent:

```bash
python3 agentic-trace/agentic_trace.py
python3 overall-design/overall_design.py
python3 pd-multiconnector/pd_multiconnector.py
```

Quick local preview on macOS — open the SVGs directly:

```bash
open agentic-trace/agentic_trace.svg
open overall-design/overall_design.svg
open pd-multiconnector/pd_multiconnector.svg
```

## gpt-image-2 prompts

`overall-design/prompt.md` and `pd-multiconnector/prompt.md` are stand-alone
text prompts that describe each architecture diagram for a text-to-image
model. They are intentionally self-contained — **do not attach the SVG when
generating**, so the model is free to design its own visual treatment from
the architectural description alone.

The agentic-trace figure has no gpt-image-2 prompt; it is a quantitative chart
driven by corpus statistics, not a stylistic illustration.

## Data sources

`agentic-trace/agentic_trace.py` carries corpus medians as inline literals —
per-turn decode size, per-turn new tool output, the four explicit history
pairs in the final turn, and footer aggregate stats (cache-hit rate,
input:output ratio, context growth). They came from a one-off statistical
pass over the codex / SWE-bench Pro corpus and are baked in rather than
re-computed on every render. To update the numbers, edit the literals at the
top of the script (`SKIP`, `ROWS`, footer text near the end of `render_svg`).

The two architecture diagrams (`overall-design/`, `pd-multiconnector/`) are
hand-drawn schematics — geometry constants only, no data.
