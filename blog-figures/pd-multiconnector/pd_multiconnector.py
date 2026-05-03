"""Generate Figure 2: PD + distributed KV cache pool with MultiConnector.

The figure tells the story of a single KV Block traversing the chain of
connectors inside MultiConnector: it is FAN-OUT to both the PD Connector
(which forwards it over the PD link to the other instance) and the
MooncakeStore Connector (which puts it into / gets it from the distributed
KV cache pool) at the same time.

Run: python figure2_pd_multiconnector.py
Output: figure2_pd_multiconnector.svg
"""

from pathlib import Path

W, H = 1380, 690

COLORS = {
    "stroke": "#1f2937",
    "muted": "#64748b",
    "label": "#0f172a",
    "instance_fill": "#f8fafc",
    # Each instance is colour-coded by the KV-block it produces. Prefill
    # produces orange blocks, Decode produces blue blocks. The header and the
    # KV-Block badge inside each instance share that producer colour, so the
    # animation's orange / blue pills tie back visually to their origin.
    "prefill_header_fill": "#fb923c",   # orange-400
    "decode_header_fill":  "#60a5fa",   # blue-400
    "prefill_kvb_fill":    "#fdba74",   # orange-300 (matches Round-1 prefill pill)
    "decode_kvb_fill":     "#93c5fd",   # blue-300   (matches Round-1 decode pill)
    "multi_fill": "#e9d5ff",        # MultiConnector wrapper (purple-200)
    "pd_fill": "#bfdbfe",           # PD Connector (blue-200)
    "store_fill": "#bbf7d0",        # MooncakeStore Connector (green-200)
    "pool_fill": "#fed7aa",         # Mooncake KV pool (orange-200)
    "pd_link": "#dc2626",
    "kvflow": "#7c3aed",            # KV-Block fan-out arrows: violet, distinct from PD red
}

ARROW_DEFS = """
<defs>
  <!-- All arrow markers use refX=10: the path tip (10,5) lands exactly at the
       line endpoint, the body extends back into the line. This keeps
       arrowheads from poking past the line endpoint into adjacent shapes. -->
  <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#1f2937"/>
  </marker>
  <marker id="arrow-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#1f2937"/>
  </marker>
  <!-- Soft slate markers for the Store↔Pool arrows: less visually loud than
       pure black/slate-800, sits closer to the muted text palette. -->
  <marker id="arrow-soft" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#64748b"/>
  </marker>
  <marker id="arrow-soft-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#64748b"/>
  </marker>
  <marker id="arrow-pd" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#dc2626"/>
  </marker>
  <marker id="arrow-pd-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#dc2626"/>
  </marker>
  <marker id="arrow-kv" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="#7c3aed"/>
  </marker>
  <marker id="arrow-kv-start" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#7c3aed"/>
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


def left_bracket(right_x, top_y, bot_y, arm_len=14, corner_r=10, color=None, sw=2.4):
    """Draw a `[` (rounded left square bracket) opening to the right.

    Top/bottom arm tips touch (right_x, top_y) and (right_x, bot_y); the
    vertical bar runs at x = right_x - arm_len. Corners use quadratic Beziers.
    """
    color = color or COLORS["stroke"]
    bar_x = right_x - arm_len
    return (
        f'<path d="'
        f'M {right_x} {top_y} '
        f'L {bar_x + corner_r} {top_y} '
        f'Q {bar_x} {top_y}, {bar_x} {top_y + corner_r} '
        f'L {bar_x} {bot_y - corner_r} '
        f'Q {bar_x} {bot_y}, {bar_x + corner_r} {bot_y} '
        f'L {right_x} {bot_y}" '
        f'fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def build(include_fanout: bool = True, include_kv_block: bool = True):
    """Render the static figure SVG.

    include_fanout:    when False, the four KV-Block fan-out arrows are
                       omitted. Useful for the animated version, where moving
                       particles depict the dispatch instead of static arrows.
    include_kv_block:  when False, the two KV Block pills (one per instance)
                       are hidden. The animated version uses moving KV pills
                       to show the block traversal, so the static badges are
                       redundant and would visually compete with them.
    """
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(ARROW_DEFS)
    parts.append(rect(0, 0, W, H, fill="#ffffff", stroke="#ffffff", rx=0, sw=0))

    # ----- Geometry -----
    # Layout story top-to-bottom inside each Instance container:
    #   Header pill ("Prefill Instance" / "Decode Instance")
    #   ↓ gap
    #   KV Block pill (the data unit — sits ABOVE MultiConnector to make the
    #   "data flows IN to MultiConnector" direction visually obvious)
    #   ↓ short gap
    #   MultiConnector wrapper, containing PD + Store side-by-side
    #   ↓ gap
    #   (instance bottom)
    # Below the instances, a long horizontal Pool spans both.
    #
    # p_x is shifted right (was 130 → 180) to leave a wider left margin for
    # the external [ bracket + "a chain of connectors" annotation that points
    # at the chain of sub-connectors inside MultiConnector.
    inst_w, inst_h = 440, 460
    p_x, p_y = 180, 50
    d_x, d_y = 800, p_y                        # gap of 180 between instances
    p_cx = p_x + inst_w / 2
    d_cx = d_x + inst_w / 2

    header_w, header_h = 220, 60
    header_y = p_y + 22                        # 72; header bottom = 132

    # KV Block pill — sits above MultiConnector, INSIDE the Prefill instance
    # container. Centered horizontally on the instance.
    kvb_w, kvb_h = 220, 50
    kvb_y = 144                                 # 12 px gap from header bottom
    kvb_x_p = p_cx - kvb_w / 2
    kvb_x_d = d_cx - kvb_w / 2
    kvb_cx_p = p_cx
    kvb_cx_d = d_cx

    # MultiConnector wrapper (now smaller — KV Block has moved out)
    mc_w, mc_h = inst_w - 50, 250
    mc_x_p = p_x + 25
    mc_x_d = d_x + 25
    mc_y = 224                                  # 30 px gap from KV Block bottom
    mc_inner_pad_x = 20
    mc_col_gap = 16
    col_w = (mc_w - 2 * mc_inner_pad_x - mc_col_gap) / 2     # 167

    # Connector vertical extent (PD + Store side-by-side, same y for both)
    conn_y = mc_y + 75                          # 299 — clear of MC header text
    conn_h = 140

    # Prefill: PD on the RIGHT (closer to Decode), Store on the LEFT.
    # Decode: mirror — PD on the LEFT, Store on the RIGHT.
    p_store_x = mc_x_p + mc_inner_pad_x
    p_pd_x    = mc_x_p + mc_inner_pad_x + col_w + mc_col_gap
    d_pd_x    = mc_x_d + mc_inner_pad_x
    d_store_x = mc_x_d + mc_inner_pad_x + col_w + mc_col_gap

    p_store_cx = p_store_x + col_w / 2
    p_pd_cx    = p_pd_x + col_w / 2
    d_pd_cx    = d_pd_x + col_w / 2
    d_store_cx = d_store_x + col_w / 2

    # Mooncake Distributed KV Cache Pool
    pool_x, pool_y, pool_h = p_x, 560, 100
    pool_w = d_x + inst_w - p_x

    # ===== Prefill Instance container & header =====
    parts.append(rect(p_x, p_y, inst_w, inst_h,
                      fill=COLORS["instance_fill"], rx=20, sw=2.5))
    parts.append(rect(p_x + (inst_w - header_w) / 2, header_y, header_w, header_h,
                      fill=COLORS["prefill_header_fill"], rx=14))
    parts.append(text(p_cx, header_y + header_h / 2 + 8, "Prefill Instance",
                      size=22, weight="700"))

    # MultiConnector wrapper (Prefill)
    parts.append(rect(mc_x_p, mc_y, mc_w, mc_h,
                      fill=COLORS["multi_fill"], rx=16, sw=2.5))
    parts.append(text(mc_x_p + mc_w / 2, mc_y + 28, "MultiConnector",
                      size=17, weight="700"))

    # KV Block pill (Prefill — produces ORANGE blocks)
    if include_kv_block:
        parts.append(rect(kvb_x_p, kvb_y, kvb_w, kvb_h,
                          fill=COLORS["prefill_kvb_fill"], rx=10, sw=2.2))
        parts.append(text(kvb_cx_p, kvb_y + kvb_h / 2 + 7, "KV Block",
                          size=19, weight="700"))

    # Prefill: Store on the left, PD on the right
    parts.append(rect(p_store_x, conn_y, col_w, conn_h, fill=COLORS["store_fill"], rx=12))
    parts.append(text(p_store_cx, conn_y + conn_h / 2 - 4, "MooncakeStore",
                      size=18, weight="700"))
    parts.append(text(p_store_cx, conn_y + conn_h / 2 + 22, "Connector",
                      size=18, weight="700"))

    parts.append(rect(p_pd_x, conn_y, col_w, conn_h, fill=COLORS["pd_fill"], rx=12))
    parts.append(text(p_pd_cx, conn_y + conn_h / 2 + 9, "PD Connector",
                      size=20, weight="700"))

    # ===== Decode Instance container & header =====
    parts.append(rect(d_x, d_y, inst_w, inst_h,
                      fill=COLORS["instance_fill"], rx=20, sw=2.5))
    parts.append(rect(d_x + (inst_w - header_w) / 2, header_y, header_w, header_h,
                      fill=COLORS["decode_header_fill"], rx=14))
    parts.append(text(d_cx, header_y + header_h / 2 + 8, "Decode Instance",
                      size=22, weight="700"))

    parts.append(rect(mc_x_d, mc_y, mc_w, mc_h,
                      fill=COLORS["multi_fill"], rx=16, sw=2.5))
    parts.append(text(mc_x_d + mc_w / 2, mc_y + 28, "MultiConnector",
                      size=17, weight="700"))

    # KV Block pill (Decode — produces BLUE blocks)
    if include_kv_block:
        parts.append(rect(kvb_x_d, kvb_y, kvb_w, kvb_h,
                          fill=COLORS["decode_kvb_fill"], rx=10, sw=2.2))
        parts.append(text(kvb_cx_d, kvb_y + kvb_h / 2 + 7, "KV Block",
                          size=19, weight="700"))

    # Decode: PD on the left, Store on the right
    parts.append(rect(d_pd_x, conn_y, col_w, conn_h, fill=COLORS["pd_fill"], rx=12))
    parts.append(text(d_pd_cx, conn_y + conn_h / 2 + 9, "PD Connector",
                      size=20, weight="700"))

    parts.append(rect(d_store_x, conn_y, col_w, conn_h, fill=COLORS["store_fill"], rx=12))
    parts.append(text(d_store_cx, conn_y + conn_h / 2 - 4, "MooncakeStore",
                      size=18, weight="700"))
    parts.append(text(d_store_cx, conn_y + conn_h / 2 + 22, "Connector",
                      size=18, weight="700"))

    # ===== KV Block fan-out arrows (the "chain of connectors" story) =====
    # From the KV Block bottom, two diagonals fan out to PD and Store. The
    # arrows are BIDIRECTIONAL so they capture both the "put" direction
    # (KV block flows out through the connector) and the "get" direction
    # (a hit pulls the block back into the worker via the same path).
    # 4 px margin is enough with refX=10 markers; the structural ~70 px gap
    # between KV Block bottom and connector top gives every arrow body length.
    if include_fanout:
        fan_top_y = kvb_y + kvb_h + 4
        fan_bot_y = conn_y - 4

        def fan_arrow(x1, y1, x2, y2):
            return arrow(x1, y1, x2, y2,
                         color=COLORS["kvflow"], sw=2.4,
                         marker_end="arrow-kv", marker_start="arrow-kv-start")

        # Prefill: bottom-left of KV Block ↔ Store (LEFT), bottom-right ↔ PD (RIGHT)
        parts.append(fan_arrow(kvb_x_p + kvb_w * 0.28, fan_top_y,
                               p_store_cx, fan_bot_y))
        parts.append(fan_arrow(kvb_x_p + kvb_w * 0.72, fan_top_y,
                               p_pd_cx, fan_bot_y))
        # Decode: bottom-left ↔ PD (LEFT), bottom-right ↔ Store (RIGHT)
        parts.append(fan_arrow(kvb_x_d + kvb_w * 0.28, fan_top_y,
                               d_pd_cx, fan_bot_y))
        parts.append(fan_arrow(kvb_x_d + kvb_w * 0.72, fan_top_y,
                               d_store_cx, fan_bot_y))

    # ===== PD ↔ PD link (red) =====
    pd_link_y = conn_y + conn_h / 2
    p_pd_right = p_pd_x + col_w
    parts.append(arrow(p_pd_right + 4, pd_link_y, d_pd_x - 4, pd_link_y,
                       color=COLORS["pd_link"], sw=2.5,
                       marker_end="arrow-pd", marker_start="arrow-pd-start"))
    pd_mid_x = (p_pd_right + d_pd_x) / 2
    parts.append(text(pd_mid_x, pd_link_y - 16, "MooncakePD / NIXL Connector",
                      size=16, weight="700", fill=COLORS["pd_link"],
                      halo=True, halo_width=8))
    parts.append(text(pd_mid_x, pd_link_y + 28, "Multi-Node NVLink / RDMA",
                      size=14, weight="600", fill=COLORS["muted"],
                      halo=True, halo_width=8))

    # ===== Mooncake Distributed KV Cache Pool =====
    parts.append(rect(pool_x, pool_y, pool_w, pool_h,
                      fill=COLORS["pool_fill"], rx=20, sw=2.5))
    parts.append(text(pool_x + pool_w / 2, pool_y + pool_h / 2 + 8,
                      "Mooncake Distributed KV Cache Pool",
                      size=24, weight="700"))

    # ===== Store ↔ Pool arrows. Soft slate colour (matches the muted text
    # palette) so the long arrows from the connector down to the pool don't
    # visually overpower the rest of the diagram. =====
    inst_bottom = p_y + inst_h
    label_y = (inst_bottom + pool_y) / 2 + 6
    # Prefill Store → pool: bidirectional put / get
    parts.append(arrow(p_store_cx, conn_y + conn_h + 4, p_store_cx, pool_y - 4,
                       color=COLORS["muted"], sw=2.5,
                       marker_end="arrow-soft", marker_start="arrow-soft-start"))
    parts.append(text(p_store_cx + 18, label_y, "put / get",
                      size=17, weight="700", anchor="start", halo=True, halo_width=6))
    # Decode Store → pool: one-way put
    parts.append(arrow(d_store_cx, conn_y + conn_h + 4, d_store_cx, pool_y - 4,
                       color=COLORS["muted"], sw=2.5,
                       marker_end="arrow-soft"))
    parts.append(text(d_store_cx + 18, label_y, "put",
                      size=17, weight="700", anchor="start", halo=True, halo_width=6))

    # ===== "a chain of connectors" annotation: a rounded `[` square bracket
    # sits OUTSIDE the Prefill instance container on the left, with the label
    # to its left. Placing it outside (rather than tucked inside the instance
    # padding) makes the annotation a clear, prominent side label that's hard
    # to miss. The bracket spans the two sub-connectors that form the chain. =====
    bracket_right = p_x - 14
    bracket_top = conn_y
    bracket_bot = conn_y + conn_h
    bracket_arm = 16
    bracket_bar_x = bracket_right - bracket_arm
    bracket_mid = (bracket_top + bracket_bot) / 2

    parts.append(left_bracket(bracket_right, bracket_top, bracket_bot,
                              arm_len=bracket_arm, corner_r=12,
                              color=COLORS["muted"], sw=2.6))
    label_x = bracket_bar_x - 18
    parts.append(text(label_x, bracket_mid - 12, "a chain of",
                      size=17, weight="700", fill=COLORS["muted"], anchor="end"))
    parts.append(text(label_x, bracket_mid + 14, "connectors",
                      size=17, weight="700", fill=COLORS["muted"], anchor="end"))

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".svg")
    out.write_text(build())
    print(f"Wrote {out}")
