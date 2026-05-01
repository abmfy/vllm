# Figure 1 — Overall design (gpt-image-2 prompt)

Pure-text architectural description for `gpt-image-2`. Do not attach the
reference SVG when generating with this prompt — let the model design its own
visual treatment. The goal is for the model to produce a high-quality blog
illustration that conveys the architecture below.

---

## Prompt

Generate a clean, modern technical architecture diagram for an
engineering-blog post titled "Distributed KV cache pool with Mooncake Store."
The figure illustrates how multiple vLLM inference engines share a single
distributed KV cache through the Mooncake system, separated into a control
plane and a data plane. Render as a flat schematic illustration on a white
background, in the polished style of an NVIDIA, Google Research, or Anthropic
engineering blog: pastel fills, dark-slate strokes, clean modern sans-serif
typography (Inter / SF Pro / Helvetica), no drop shadows, no gradients, no
emojis, no mascots. Use a wide horizontal aspect ratio (16:10 or similar).

### What the figure must convey

There are two parallel vLLM inference engines, each labeled "vLLM Instance".
Both engines share a distributed KV cache pool managed by Mooncake. The
diagram has two horizontal lanes — a **Control plane** (top) and a **Data
plane** (bottom) — and these lane labels should be visible on the side.

**Control plane.** Each vLLM instance contains a vLLM Scheduler. A single
Mooncake Master sits between the two instances at the same vertical level as
the Schedulers, so the three boxes read as one row. Each Scheduler is
connected to the Mooncake Master with a short, straight, double-headed arrow
labeled "lookup". This represents the prefix-cache metadata lookup the
scheduler performs against the Master before scheduling a request.

**Data plane.** Below the Schedulers (still inside each vLLM Instance
container), each engine contains one or more vLLM Workers, and each Worker
embeds a Mooncake Client. Show this as a small 3D-stacked card to suggest a
GPU pool of multiple workers — visually it should be one card with two or
three subtle outline shadows behind it. Inside the front card, draw two
clearly stacked sub-pills labeled "vLLM Worker" (top) and "Mooncake Client"
(bottom).

Below each vLLM Instance, draw the storage hierarchy that the workers operate
on: a **GPU HBM** block per instance, and a single shared light-gray container
spanning both instances that holds two **CPU / DRAM / SSD** blocks (one per
instance).

### Arrows in the data plane

- Each vLLM Instance container connects downward to its own GPU HBM with a
  short double-headed vertical arrow centered horizontally on the block.
- Each GPU HBM connects downward to its same-side CPU / DRAM / SSD block with
  a short double-headed vertical arrow, also centered horizontally on both
  blocks.
- The two CPU / DRAM / SSD blocks are connected with a horizontal
  double-headed arrow that runs inside the shared container between them.
- Two diagonal arrows cross each other, drawn in clear red (#DC2626), forming
  an X between the two storage columns: one connects GPU HBM (left instance)
  with CPU / DRAM / SSD (right instance), and the mirror connects GPU HBM
  (right instance) with CPU / DRAM / SSD (left instance). Place a single bold
  red label that reads "GPU Direct RDMA" near these red arrows so it is
  clearly the label for them. The label must remain readable — give it a
  white halo or place it just above the X-crossing in clean space.

### Color and typography palette

- Schedulers and the Mooncake Master use a warm amber fill — the Master is a
  slightly stronger amber to read as the central authority.
- vLLM Workers and Mooncake Clients use a cool light blue.
- GPU HBM uses a soft pink. CPU / DRAM / SSD uses mint green. The shared
  container holding the two CPU / DRAM / SSD blocks is a neutral light gray.
- All component labels are bold (700 weight), large enough to feel
  comfortable inside their containers — labels should not look small relative
  to the block they sit in.
- The two side labels "Control plane" and "Data plane" appear in muted slate
  gray to the left of the diagram.
- Instance titles ("vLLM Instance 0", "vLLM Instance 1") sit just above each
  outer container in muted slate gray, smaller and lighter than the inner
  component labels.

The composition should feel calm, well-aligned, and information-dense without
being cluttered: generous whitespace, precise alignment, all blocks rounded
with consistent corner radius, all strokes consistent weight.
