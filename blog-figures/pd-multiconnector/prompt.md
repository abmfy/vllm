# Figure 2 — PD + distributed KV with MultiConnector (gpt-image-2 prompt)

Pure-text architectural description for `gpt-image-2`. Do not attach the
reference SVG when generating with this prompt — let the model design its own
visual treatment.

---

## Prompt

Generate a clean, modern technical architecture diagram for an
engineering-blog post about combining prefill/decode (PD) disaggregation with
a distributed KV cache pool in vLLM. The figure illustrates how vLLM's
`MultiConnector` composes multiple sub-connectors so that a single deployment
can use both direct PD transfer and a shared distributed KV cache at the same
time. Render as a flat schematic illustration on a white background, in the
polished style of an NVIDIA, Google Research, or Anthropic engineering blog:
pastel fills, dark-slate strokes, clean modern sans-serif typography
(Inter / SF Pro / Helvetica), no drop shadows, no gradients, no emojis. Use a
wide horizontal aspect ratio (about 16:10).

### What the figure must convey

There are two parallel vLLM serving roles in a PD-disaggregated deployment: a
**Prefill Instance** on the left and a **Decode Instance** on the right. Each
of them is its own outer container with a small amber header pill labeled
either "Prefill Instance" or "Decode Instance" placed near the top of the
container.

Inside each container is a single **MultiConnector** wrapper — a clearly
visible enclosing rectangle with a soft lavender / light-purple fill — that
contains two stacked child connectors. The top child is the **PD Connector**
(light blue), and the bottom child is the **MooncakeStore Connector** (mint
green). The visual nesting should make it obvious that MultiConnector is a
chain that owns both sub-connectors. To the side of the diagram, a small
muted side annotation reads "a chain of connectors", labeling this
composition pattern.

The two **PD Connectors** (one in each instance) are joined by a single bold
red horizontal double-headed arrow. Above the arrow place the label
"MooncakePD / NIXL Connector" in red, and below the arrow place a smaller
secondary label "Multi-Node NVLink / RDMA" in muted slate gray. Both labels
should remain readable on top of the red line — give them a white halo or
position them so the line does not cut through the glyphs.

Below both instances, draw a single wide rounded rectangle in soft orange,
spanning roughly the full width of the two instances combined, labeled in
bold "Mooncake Distributed KV Cache Pool". This is the shared distributed KV
cache layer.

### Arrows from the Store connectors to the pool

- From the **Prefill** instance's MooncakeStore Connector, a black
  double-headed vertical arrow runs down to the pool, with the label
  "put / get" placed beside the arrow segment between the instance and the
  pool. (The Prefill side both writes and reads the pool.)
- From the **Decode** instance's MooncakeStore Connector, a black single-
  headed vertical arrow points downward to the pool, with the label "put"
  beside it. (The Decode side currently only writes to the pool — it
  receives KV blocks for in-flight requests over the PD connector instead.)

The asymmetry between "put / get" and "put" is meaningful — please preserve
it.

### Color and typography palette

- Prefill / Decode header pills use the same amber as the Mooncake Master in
  the companion figure, so the two diagrams feel like a series.
- MultiConnector fill is a soft lavender / light purple.
- PD Connector is light blue (matching the worker-side blue of the companion
  figure).
- MooncakeStore Connector is mint green (matching the storage green of the
  companion figure).
- The Mooncake Distributed KV Cache Pool uses a soft orange that stands out
  from the pastels above it without overwhelming them.
- All component labels are bold (700 weight) and large enough to feel
  comfortable inside their containers; the pool's label should be the largest
  in the figure since it is the shared infrastructure.

The composition should feel calm, well-aligned, and visually parallel: the
Prefill and Decode columns mirror each other; the nested MultiConnector
hierarchy is unambiguous at a glance; the red PD-to-PD link clearly carries
the "direct prefill→decode transfer" story while the orange pool clearly
carries the "shared distributed cache" story.
