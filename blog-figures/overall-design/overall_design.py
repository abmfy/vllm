"""Generate Figure 1: Overall design of vLLM distributed KV cache pool with Mooncake Store.

Run: python figure1_overall_design.py
Output: figure1_overall_design.svg
"""

from pathlib import Path

W, H = 1320, 850

COLORS = {
    "stroke": "#1f2937",
    "stroke_soft": "#475569",
    "instance_fill": "#f8fafc",
    "scheduler_fill": "#fde68a",   # amber-200
    "master_fill": "#fcd34d",       # amber-300
    "worker_fill": "#dbeafe",       # blue-100
    "client_fill": "#bfdbfe",       # blue-200
    "hbm_fill": "#fbcfe8",          # pink-200
    "dram_fill": "#bbf7d0",         # green-200
    "pool_bg": "#f1f5f9",
    "rdma": "#dc2626",
    "label": "#0f172a",
    "muted": "#64748b",
}

ARROW_DEFS = """
<defs>
  <!-- All arrow markers use refX=10 so the path tip (10,5) lands EXACTLY at
       the line endpoint and the body extends back into the line. This keeps
       arrowheads from poking past the line endpoint into adjacent shapes. -->
  <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#1f2937"/>
  </marker>
  <marker id="arrow-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#1f2937"/>
  </marker>
  <marker id="arrow-rdma" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#dc2626"/>
  </marker>
  <marker id="arrow-rdma-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#dc2626"/>
  </marker>
</defs>
""".strip()


def rect(x, y, w, h, fill="#fff", stroke=COLORS["stroke"], rx=12, sw=2):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, content, size=16, weight="600", anchor="middle", fill=COLORS["label"],
         halo=False, halo_color="#ffffff", halo_width=6):
    """Render text with optional white halo (paint-order stroke) so labels stay
    readable when they overlap arrows or other shapes."""
    if halo:
        style = (
            f' style="paint-order: stroke; stroke: {halo_color}; '
            f'stroke-width: {halo_width}px; stroke-linejoin: round;"'
        )
    else:
        style = ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}"{style}>{content}</text>'
    )


def arrow(x1, y1, x2, y2, color=None, sw=2, marker_end="arrow", marker_start=None):
    color = color or COLORS["stroke"]
    me = f' marker-end="url(#{marker_end})"' if marker_end else ""
    ms = f' marker-start="url(#{marker_start})"' if marker_start else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}"{ms}{me}/>'
    )


def dashed_line(x1, y1, x2, y2, color="#94a3b8", sw=1.6, dash="10 7"):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" stroke-dasharray="{dash}"/>'
    )


def build():
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(ARROW_DEFS)
    parts.append(rect(0, 0, W, H, fill="#ffffff", stroke="#ffffff", rx=0, sw=0))

    # ----- Geometry -----
    # Generous structural gaps so every arrow has visible body length:
    #   - Schedulers ↔ Master gap: ~70 px each side
    #   - Instance ↔ HBM gap: ~90 px
    #   - HBM ↔ DRAM gap: ~70 px
    # Inst_h is kept tight to minimise empty band below Mooncake Client.
    inst_y, inst_w, inst_h = 60, 340, 450
    inst0_x, inst1_x = 170, 830                # inst1 shifted right to widen master gap
    inst0_cx = inst0_x + inst_w / 2          # 340
    inst1_cx = inst1_x + inst_w / 2          # 1000

    # Top row: schedulers + master share the same y / height — they read as one
    # horizontal control-plane band.
    row_y, row_h = 90, 70
    sch_w = inst_w - 60                       # 280
    sch0_x = inst0_x + 30                     # 200
    sch1_x = inst1_x + 30                     # 860
    master_w = 240
    master_x = 550                            # symmetric: 40 px gap to each instance edge

    # Worker stack inside each instance (3D-stacked card)
    wbox_w, wbox_h = 260, 220
    wbox_y = 240
    wbox_x_0 = inst0_x + (inst_w - wbox_w) / 2 - 10   # shift left so shadow stays inside
    wbox_x_1 = inst1_x + (inst_w - wbox_w) / 2 - 10

    # GPU HBM row
    hbm_y, hbm_h, hbm_w = 600, 80, 280
    hbm0_x = inst0_x + (inst_w - hbm_w) / 2   # 200
    hbm1_x = inst1_x + (inst_w - hbm_w) / 2   # 860

    # CPU/DRAM/SSD pool container + inner blocks
    pool_x = inst0_x                          # 170
    pool_w = inst1_x + inst_w - inst0_x       # 1000
    pool_y, pool_h = 750, 90
    dram_y, dram_h = 768, 54
    dram_w = hbm_w
    dram0_x, dram1_x = hbm0_x, hbm1_x

    # Position the dashed separator midway between the Scheduler/Master row
    # bottom and the worker-stack top. Drawn last (top z-level) so it sits
    # above the Instance container fills.
    sep_y = (row_y + row_h + wbox_y) / 2

    # ----- Side labels (red, hugging the dashed separator) -----
    parts.append(text(70, sep_y - 10, "Control plane",
                      size=19, weight="700", fill=COLORS["rdma"]))
    parts.append(text(70, sep_y + 24, "Data plane",
                      size=19, weight="700", fill=COLORS["rdma"]))

    # ----- vLLM Instance 0 outer container -----
    parts.append(rect(inst0_x, inst_y, inst_w, inst_h,
                      fill=COLORS["instance_fill"], rx=20, sw=2.5))
    parts.append(text(inst0_x + inst_w / 2, inst_y - 18, "vLLM Instance 0",
                      size=19, weight="700", fill=COLORS["muted"]))

    # Scheduler 0 (same y/h as Master so the row reads as a single line)
    parts.append(rect(sch0_x, row_y, sch_w, row_h,
                      fill=COLORS["scheduler_fill"], rx=14))
    parts.append(text(sch0_x + sch_w / 2, row_y + row_h / 2 + 8, "vLLM Scheduler",
                      size=21, weight="700"))

    # Worker stack 0 (3-layer card)
    for i in (2, 1):
        parts.append(rect(wbox_x_0 + i * 10, wbox_y + i * 10, wbox_w, wbox_h,
                          fill="#ffffff", rx=14))
    parts.append(rect(wbox_x_0, wbox_y, wbox_w, wbox_h, fill="#ffffff", rx=14))
    parts.append(rect(wbox_x_0 + 18, wbox_y + 22, wbox_w - 36, 80,
                      fill=COLORS["worker_fill"], rx=10))
    parts.append(text(wbox_x_0 + wbox_w / 2, wbox_y + 22 + 50, "vLLM Worker",
                      size=20, weight="700"))
    parts.append(rect(wbox_x_0 + 18, wbox_y + 120, wbox_w - 36, 82,
                      fill=COLORS["client_fill"], rx=10))
    parts.append(text(wbox_x_0 + wbox_w / 2, wbox_y + 120 + 51, "Mooncake Client",
                      size=20, weight="700"))

    # ----- vLLM Instance 1 (mirror) -----
    parts.append(rect(inst1_x, inst_y, inst_w, inst_h,
                      fill=COLORS["instance_fill"], rx=20, sw=2.5))
    parts.append(text(inst1_x + inst_w / 2, inst_y - 18, "vLLM Instance 1",
                      size=19, weight="700", fill=COLORS["muted"]))

    parts.append(rect(sch1_x, row_y, sch_w, row_h,
                      fill=COLORS["scheduler_fill"], rx=14))
    parts.append(text(sch1_x + sch_w / 2, row_y + row_h / 2 + 8, "vLLM Scheduler",
                      size=21, weight="700"))

    for i in (2, 1):
        parts.append(rect(wbox_x_1 + i * 10, wbox_y + i * 10, wbox_w, wbox_h,
                          fill="#ffffff", rx=14))
    parts.append(rect(wbox_x_1, wbox_y, wbox_w, wbox_h, fill="#ffffff", rx=14))
    parts.append(rect(wbox_x_1 + 18, wbox_y + 22, wbox_w - 36, 80,
                      fill=COLORS["worker_fill"], rx=10))
    parts.append(text(wbox_x_1 + wbox_w / 2, wbox_y + 22 + 50, "vLLM Worker",
                      size=20, weight="700"))
    parts.append(rect(wbox_x_1 + 18, wbox_y + 120, wbox_w - 36, 82,
                      fill=COLORS["client_fill"], rx=10))
    parts.append(text(wbox_x_1 + wbox_w / 2, wbox_y + 120 + 51, "Mooncake Client",
                      size=20, weight="700"))

    # ----- Mooncake Master (same y/h as Schedulers) -----
    parts.append(rect(master_x, row_y, master_w, row_h,
                      fill=COLORS["master_fill"], rx=14))
    parts.append(text(master_x + master_w / 2, row_y + row_h / 2 + 8, "Mooncake Master",
                      size=21, weight="700"))

    # ----- Lookup arrows: straight, on the shared midline. Margin 4 px is
    # enough since refX=10 markers keep the arrowhead tip exactly at the line
    # endpoint (no overshoot into adjacent boxes). -----
    arrow_y = row_y + row_h / 2
    parts.append(arrow(sch0_x + sch_w + 4, arrow_y, master_x - 4, arrow_y,
                       marker_end="arrow", marker_start="arrow-start"))
    parts.append(arrow(master_x + master_w + 4, arrow_y, sch1_x - 4, arrow_y,
                       marker_end="arrow", marker_start="arrow-start"))
    parts.append(text((sch0_x + sch_w + master_x) / 2, arrow_y - 14, "lookup",
                      size=16, weight="600", fill=COLORS["muted"], halo=True))
    parts.append(text((master_x + master_w + sch1_x) / 2, arrow_y - 14, "lookup",
                      size=16, weight="600", fill=COLORS["muted"], halo=True))

    # ----- GPU HBM row -----
    parts.append(rect(hbm0_x, hbm_y, hbm_w, hbm_h, fill=COLORS["hbm_fill"], rx=12))
    parts.append(text(hbm0_x + hbm_w / 2, hbm_y + hbm_h / 2 + 8, "GPU HBM",
                      size=22, weight="700"))
    parts.append(rect(hbm1_x, hbm_y, hbm_w, hbm_h, fill=COLORS["hbm_fill"], rx=12))
    parts.append(text(hbm1_x + hbm_w / 2, hbm_y + hbm_h / 2 + 8, "GPU HBM",
                      size=22, weight="700"))

    # ----- CPU/DRAM/SSD pool container + inner blocks -----
    parts.append(rect(pool_x, pool_y, pool_w, pool_h,
                      fill=COLORS["pool_bg"], rx=16, sw=2))
    parts.append(rect(dram0_x, dram_y, dram_w, dram_h,
                      fill=COLORS["dram_fill"], rx=10))
    parts.append(text(dram0_x + dram_w / 2, dram_y + dram_h / 2 + 7, "CPU / DRAM / SSD",
                      size=20, weight="700"))
    parts.append(rect(dram1_x, dram_y, dram_w, dram_h,
                      fill=COLORS["dram_fill"], rx=10))
    parts.append(text(dram1_x + dram_w / 2, dram_y + dram_h / 2 + 7, "CPU / DRAM / SSD",
                      size=20, weight="700"))

    # ----- Vertical arrows: Instance → HBM, HBM ↔ DRAM (centered on each block).
    # 4 px margin is enough with refX=10 markers; the structural gap between
    # blocks (90 px Inst↔HBM, 70 px HBM↔DRAM) gives every line a visible body. -----
    parts.append(arrow(inst0_cx, inst_y + inst_h + 4, inst0_cx, hbm_y - 4,
                       marker_end="arrow", marker_start="arrow-start"))
    parts.append(arrow(inst1_cx, inst_y + inst_h + 4, inst1_cx, hbm_y - 4,
                       marker_end="arrow", marker_start="arrow-start"))
    parts.append(arrow(inst0_cx, hbm_y + hbm_h + 4, inst0_cx, dram_y - 4,
                       marker_end="arrow", marker_start="arrow-start"))
    parts.append(arrow(inst1_cx, hbm_y + hbm_h + 4, inst1_cx, dram_y - 4,
                       marker_end="arrow", marker_start="arrow-start"))

    # ----- Cross GPU Direct RDMA arrows (HBM 0 ↔ DRAM 1, HBM 1 ↔ DRAM 0) -----
    # Diagonals attach to HBM bottom edges and DRAM top edges (vertical access
    # is the natural orientation for these memory blocks). The x offset is
    # large enough (60 px from the inner corner) so the diagonal endpoints
    # never collide with the centered Instance↔HBM↔DRAM vertical arrows.
    rdma_x_offset = 60
    parts.append(arrow(hbm0_x + hbm_w - rdma_x_offset, hbm_y + hbm_h + 4,
                       dram1_x + rdma_x_offset, dram_y - 4,
                       color=COLORS["rdma"], sw=2.5,
                       marker_end="arrow-rdma", marker_start="arrow-rdma-start"))
    parts.append(arrow(hbm1_x + rdma_x_offset, hbm_y + hbm_h + 4,
                       dram0_x + dram_w - rdma_x_offset, dram_y - 4,
                       color=COLORS["rdma"], sw=2.5,
                       marker_end="arrow-rdma", marker_start="arrow-rdma-start"))
    # GPU Direct RDMA label sits in the band between HBM and DRAM, centered at
    # the X-crossing. Halo keeps it readable on top of the diagonals.
    cross_y = (hbm_y + hbm_h + dram_y) / 2
    parts.append(text(W / 2, cross_y + 6, "GPU Direct RDMA",
                      size=20, weight="700", fill=COLORS["rdma"],
                      halo=True, halo_width=10))

    # ----- Horizontal arrow between DRAM blocks (inside the pool container) -----
    parts.append(arrow(dram0_x + dram_w + 4, dram_y + dram_h / 2,
                       dram1_x - 4, dram_y + dram_h / 2,
                       marker_end="arrow", marker_start="arrow-start"))

    # ----- Dashed separator line (highest z-level: drawn last so it sits on top
    # of the Instance container fills and any underlying shapes). -----
    parts.append(dashed_line(40, sep_y, W - 40, sep_y))

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".svg")
    out.write_text(build())
    print(f"Wrote {out}")
