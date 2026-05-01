"""
Render an agentic trace shape as a clean vector SVG.

Numbers driven by per-turn medians computed across the full codex_swebenchpro
corpus (610 trials).  Per turn k:
  - human[k] median: tool output / fresh content arriving at this call
  - gpt[k] median: total decode produced this call

System prompt + skills account for ~11.5K cross-trial cached prefix.
Decode is rendered as total output (output*); the thinking/tool-call breakdown
is NOT preserved in the ShareGPT export and is not shown.

Output: agentic_trace.svg
"""
import os
from pathlib import Path

OUTPUT = str(Path(__file__).with_suffix(".svg"))

# ---- Color palette --------------------------------------------------------
SEG_STYLE = {
    "sys":      ("#1d4ed8",  "white",   "system prompt"),
    "skills":   ("#60a5fa",  "white",   "skills / memory"),
    "user":     ("#a78bfa",  "white",   "user task"),
    "agent":    ("#86efac",  "#065f46", "past agent decode"),
    "tool":     ("#bae6fd",  "#075985", "past tool output"),
    "new_tool": ("#fb923c",  "white",   "new tool output"),
    "output":   ("#0891b2",  "white",   "output*"),
    "ellipsis": ("#e5e7eb",  "#9ca3af", "···"),
}
C_TEXT  = "#1f2937"
C_MUTED = "#6b7280"
C_GAP   = "#9ca3af"
C_DIV   = "#ffffff"

# ---- Per-turn data --------------------------------------------------------
# Tuple: (label, input_segs, decode_segs, in_note, llm_timing)
#
# Agent tokens in Turn N's history = prior turn's decode median (rounded).
# Tool tokens = the new_tool value from the turn that produced them.
# history_pairs(12) abbreviated: k=0,1 explicit + ellipsis(k=2–9) + k=10,11.
# Token totals preserved: history sum = 69,780.

SKIP = ("SKIP",)

ROWS = [
    ("Turn 0", [
        ("sys",    1_500),
        ("skills", 10_000),
        ("user",     500),
    ], [
        ("output", 250),
    ], "First call: prefix + user task", "~10–20s"),

    ("Turn 1", [
        ("sys",    1_500), ("skills", 10_000), ("user", 500),
        ("agent",    250),          # Turn 0 decode
        ("new_tool", 6_155),
    ], [
        ("output", 300),
    ], "+ Tool output: file contents (~6K)", "~3–10s"),

    ("Turn 2", [
        ("sys",    1_500), ("skills", 10_000), ("user", 500),
        ("agent",    250), ("tool", 6_155),    # Turn 0 decode + Turn 1 tool
        ("agent",    300),                      # Turn 1 decode
        ("new_tool", 5_821),
    ], [
        ("output", 300),
    ], "+ Tool output: grep results (~5.8K)", "~3–10s"),

    SKIP,

    ("Final Turn", [
        ("sys",    1_500), ("skills", 10_000), ("user", 500),
        # pairs k=0, k=1 (shown explicitly)
        ("agent",  250), ("tool", 6_500),
        ("agent",  280), ("tool", 6_300),
        # pairs k=2–9 (8 pairs, abbreviated)
        ("ellipsis", 46_520),
        # pairs k=10, k=11
        ("agent",  550), ("tool", 4_500),
        ("agent",  580), ("tool", 4_300),
        ("new_tool", 1_650),
    ], [
        ("output", 640),
    ], "+ Tool output: last command result (~1.6K)", "~20–40s"),
]


# ---- Helpers ---------------------------------------------------------------
ELLIPSIS_PX = 52   # fixed visual width for "···" — not token-proportional

def fmt_tok(n):
    if n >= 1000:
        return f"{n/1000:.0f}K" if n % 1000 == 0 else f"{n/1000:.1f}K"
    return str(n)

def sum_tok(segments, kinds):
    return sum(t for k, t in segments if k in kinds)

def vis_w(k, tok, px_per_tok):
    """Visual pixel width for a segment (ellipsis uses fixed width)."""
    return ELLIPSIS_PX if k == "ellipsis" else tok * px_per_tok


# ---- Render ----------------------------------------------------------------
def render_svg(rows, out_path):
    PAD_L    = 110
    PAD_R    = 30
    PAD_TOP  = 185
    PAD_BOT  = 90
    ROW_H    = 115
    BAR_H    = 28
    SKIP_H   = 40

    INPUT_W_MAX  = 920
    DECODE_W_MAX = 200   # expanded scale so decode bar is visible
    OUT_GAP      = 18
    LABEL_W      = 155

    real_rows = [r for r in rows if r is not SKIP]
    max_total_input  = max(sum(t for _, t in r[1]) for r in real_rows)
    max_total_decode = max(sum(t for _, t in r[2]) for r in real_rows)
    px_per_tok_in  = INPUT_W_MAX  / max_total_input
    px_per_tok_dec = DECODE_W_MAX / max_total_decode

    # Decode column after the widest *visual* bar (ellipsis uses fixed px)
    max_vis_input = max(
        sum(vis_w(k, t, px_per_tok_in) for k, t in r[1]) for r in real_rows
    )
    decode_x = PAD_L + max_vis_input + OUT_GAP
    label_x  = int(decode_x + DECODE_W_MAX + 12)
    width    = label_x + LABEL_W + PAD_R
    height   = PAD_TOP + sum(SKIP_H if r is SKIP else ROW_H for r in rows) + PAD_BOT

    p = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="Inter, -apple-system, system-ui, sans-serif" font-size="12" '
        f'style="background:#fafafa">'
    )
    p.append(
        '<defs>'
        '<pattern id="hatch" patternUnits="userSpaceOnUse" width="7" height="7" '
        'patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="7" stroke="rgba(255,255,255,0.38)" stroke-width="3"/>'
        '</pattern>'
        '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{C_GAP}"/>'
        '</marker>'
        '</defs>'
    )

    # ---- Title ----
    p.append(
        f'<text x="{PAD_L}" y="36" font-size="18" font-weight="700" fill="{C_TEXT}">'
        f'Anatomy of one agentic trace</text>'
    )
    p.append(
        f'<text x="{PAD_L}" y="57" font-size="12" fill="{C_MUTED}">'
        f'Each row = one LLM call. Per-turn sizes: medians from codex / SWE-bench Pro '
        f'corpus (610 trials). Decode block uses an expanded scale.'
        f'</text>'
    )

    # ---- Legend ----
    legend_y1 = 95
    legend_y2 = 135

    p.append(
        f'<text x="{PAD_L}" y="{legend_y1 - 14}" font-size="11" fill="{C_MUTED}" '
        f'font-weight="700" letter-spacing="0.5">CACHED PREFIX — KV cache reuse, no recompute</text>'
    )
    lx = PAD_L
    for k in ["sys", "skills", "user", "agent", "tool"]:
        fill, _, lab = SEG_STYLE[k]
        p.append(f'<rect x="{lx}" y="{legend_y1-10}" width="14" height="14" rx="2" fill="{fill}"/>')
        p.append(f'<rect x="{lx}" y="{legend_y1-10}" width="14" height="14" rx="2" fill="url(#hatch)"/>')
        p.append(f'<text x="{lx+20}" y="{legend_y1+1}" fill="{C_TEXT}" font-size="11">{lab}</text>')
        lx += 168

    p.append(
        f'<text x="{PAD_L}" y="{legend_y2 - 14}" font-size="11" fill="{C_MUTED}" '
        f'font-weight="700" letter-spacing="0.5">ACTIVE THIS TURN</text>'
    )
    lx = PAD_L
    for k, lab in [("new_tool", "new tool output (must prefill)"), ("output", "output*")]:
        fill, _, _ = SEG_STYLE[k]
        p.append(f'<rect x="{lx}" y="{legend_y2-10}" width="14" height="14" rx="2" fill="{fill}"/>')
        p.append(f'<text x="{lx+20}" y="{legend_y2+1}" fill="{C_TEXT}" font-size="11">{lab}</text>')
        lx += 230

    scale_ratio = px_per_tok_dec / px_per_tok_in
    p.append(
        f'<text x="{width - PAD_R}" y="{legend_y2 + 1}" text-anchor="end" '
        f'font-size="11" fill="{C_MUTED}" font-style="italic">'
        f'decode at ~{scale_ratio:.0f}× input scale</text>'
    )

    # ---- Column headers ----
    hdr_y = PAD_TOP - 16
    p.append(
        f'<text x="{PAD_L}" y="{hdr_y}" font-size="12" fill="{C_MUTED}" '
        f'font-weight="700" letter-spacing="0.5">INPUT (sent to model)</text>'
    )
    p.append(
        f'<text x="{decode_x}" y="{hdr_y}" font-size="12" '
        f'fill="{C_MUTED}" font-weight="700" letter-spacing="0.5">DECODE</text>'
    )

    # ---- Rows ----
    cur_y = PAD_TOP
    for ri, row in enumerate(rows):
        if row is SKIP:
            mid_x = PAD_L + max_vis_input / 2
            skip_text_y = cur_y + SKIP_H / 2 + 4
            p.append(
                f'<text x="{PAD_L - 12}" y="{skip_text_y}" text-anchor="end" '
                f'font-size="13" fill="{C_MUTED}" font-weight="600">⋮</text>'
            )
            p.append(
                f'<text x="{mid_x}" y="{skip_text_y}" text-anchor="middle" '
                f'font-size="12" fill="{C_MUTED}" font-style="italic">'
                f'…~26 more iterations of (decode → tool exec → new tool output)…'
                f'</text>'
            )
            after_idx = ri + 1
            if after_idx < len(rows) and rows[after_idx] is not SKIP:
                ax = PAD_L + 18
                p.append(
                    f'<line x1="{ax}" y1="{skip_text_y + 8}" '
                    f'x2="{ax}" y2="{cur_y + SKIP_H + 30}" '
                    f'stroke="{C_GAP}" stroke-width="1.4" stroke-dasharray="3,3" '
                    f'marker-end="url(#arr)"/>'
                )
            cur_y += SKIP_H
            continue

        label, in_segs, dec_segs, in_note, timing = row
        bar_y = cur_y + 36

        # Turn label
        p.append(
            f'<text x="{PAD_L - 12}" y="{bar_y + BAR_H/2 + 5}" text-anchor="end" '
            f'font-weight="700" fill="{C_TEXT}" font-size="13">{label}</text>'
        )

        # ---- Cached / new visual boundary ----
        new_tok = sum_tok(in_segs, ("new_tool",))
        cached_vis_w = sum(vis_w(k, t, px_per_tok_in) for k, t in in_segs if k != "new_tool")
        new_x_start  = PAD_L + cached_vis_w

        # ---- Input bar segments ----
        x = PAD_L
        for k, tok in in_segs:
            fill, txt_color, _ = SEG_STYLE[k]
            w = vis_w(k, tok, px_per_tok_in)
            if w <= 0:
                continue
            if k == "ellipsis":
                p.append(
                    f'<rect x="{x}" y="{bar_y}" width="{w}" height="{BAR_H}" '
                    f'fill="{fill}" stroke="{C_GAP}" stroke-width="0.6" '
                    f'stroke-dasharray="3,2"/>'
                )
                p.append(
                    f'<text x="{x + w/2}" y="{bar_y + BAR_H/2 + 5}" '
                    f'text-anchor="middle" font-size="15" fill="{txt_color}">···</text>'
                )
            else:
                p.append(
                    f'<rect x="{x}" y="{bar_y}" width="{w}" height="{BAR_H}" fill="{fill}"/>'
                )
                short = {"sys": "sys", "skills": "skills", "user": "user",
                         "new_tool": "new"}.get(k)
                if short and w > 35:
                    p.append(
                        f'<text x="{x + w/2}" y="{bar_y + BAR_H/2 + 4}" '
                        f'text-anchor="middle" font-size="11" fill="{txt_color}" '
                        f'font-weight="600">{short}</text>'
                    )
                p.append(
                    f'<line x1="{x+w}" y1="{bar_y}" x2="{x+w}" y2="{bar_y+BAR_H}" '
                    f'stroke="{C_DIV}" stroke-width="1.0" opacity="0.85"/>'
                )
            x += w
        input_end_x = x

        # ---- Hatch overlay on cached region ----
        if new_tok > 0:
            p.append(
                f'<rect x="{PAD_L}" y="{bar_y}" width="{cached_vis_w}" height="{BAR_H}" '
                f'fill="url(#hatch)" pointer-events="none"/>'
            )

        # ---- Above-bar token labels ----
        ss_tok = sum_tok(in_segs, ("sys", "skills"))
        ss_w   = (sum_tok(in_segs, ("sys",)) + sum_tok(in_segs, ("skills",))) * px_per_tok_in
        if ss_w > 0:
            p.append(
                f'<text x="{PAD_L + ss_w/2}" y="{bar_y - 6}" text-anchor="middle" '
                f'font-size="11" fill="{C_MUTED}">{fmt_tok(ss_tok)}</text>'
            )
        u_w    = sum_tok(in_segs, ("user",)) * px_per_tok_in
        hist_tok = sum_tok(in_segs, ("agent", "tool", "ellipsis"))
        hist_vis = sum(vis_w(k, t, px_per_tok_in) for k, t in in_segs
                       if k in ("agent", "tool", "ellipsis"))
        if hist_tok > 0 and hist_vis >= 50:
            p.append(
                f'<text x="{PAD_L + ss_w + u_w + hist_vis/2}" y="{bar_y - 6}" '
                f'text-anchor="middle" font-size="11" fill="{C_MUTED}">'
                f'history {fmt_tok(hist_tok)}</text>'
            )
        if new_tok > 0:
            new_vis = vis_w("new_tool", new_tok, px_per_tok_in)
            p.append(
                f'<text x="{new_x_start + new_vis/2}" y="{bar_y - 6}" '
                f'text-anchor="middle" font-size="11" fill="#9a3412" font-weight="700">'
                f'+{fmt_tok(new_tok)}</text>'
            )

        # ---- Decode block (expanded scale) ----
        total_dec = sum(t for _, t in dec_segs)
        dx = decode_x
        for k, tok in dec_segs:
            if tok <= 0:
                continue
            fill, txt_color, _ = SEG_STYLE[k]
            w = tok * px_per_tok_dec
            p.append(f'<rect x="{dx}" y="{bar_y}" width="{w}" height="{BAR_H}" fill="{fill}"/>')
            dx += w
        decode_end_x = dx
        dec_mid_x = (decode_x + decode_end_x) / 2

        # Above-decode: token count
        p.append(
            f'<text x="{dec_mid_x}" y="{bar_y - 6}" text-anchor="middle" '
            f'font-size="11" fill="#0e7490" font-weight="700">{fmt_tok(total_dec)}</text>'
        )

        # Dashed connector: input end → decode block
        p.append(
            f'<path d="M {input_end_x} {bar_y + BAR_H/2} L {decode_x} {bar_y + BAR_H/2}" '
            f'stroke="{C_GAP}" stroke-width="1" fill="none" stroke-dasharray="2,2"/>'
        )

        # ---- Right-side summary ----
        total_in = sum(t for _, t in in_segs)
        p.append(
            f'<text x="{label_x}" y="{bar_y + BAR_H/2 - 1}" '
            f'font-size="11" fill="{C_MUTED}">'
            f'<tspan font-weight="700" fill="{C_TEXT}">{fmt_tok(total_in)}</tspan> in'
            f' / <tspan font-weight="700" fill="#065f46">{fmt_tok(total_dec)}</tspan> out'
            f'<tspan x="{label_x}" dy="15" font-size="10.5" font-style="italic">{timing}</tspan>'
            f'</text>'
        )

        # ---- Below-bar narration ----
        narr_y = bar_y + BAR_H + 24

        if new_tok > 0:
            # Bracket under cached region — ticks point UP toward bar
            bk_y = bar_y + BAR_H + 9
            p.append(
                f'<path d="M {PAD_L} {bk_y-5} L {PAD_L} {bk_y} '
                f'L {new_x_start} {bk_y} L {new_x_start} {bk_y-5}" '
                f'stroke="{C_MUTED}" stroke-width="0.8" fill="none" opacity="0.55"/>'
            )
            p.append(
                f'<text x="{PAD_L + cached_vis_w/2}" y="{narr_y}" '
                f'text-anchor="middle" font-size="11" fill="{C_MUTED}" '
                f'font-style="italic">cached</text>'
            )
            new_vis = vis_w("new_tool", new_tok, px_per_tok_in)
            if new_x_start + 240 < decode_x:
                p.append(
                    f'<text x="{new_x_start}" y="{narr_y}" '
                    f'font-size="12" fill="#9a3412" font-weight="600">'
                    f'↑ {in_note}</text>'
                )
            else:
                p.append(
                    f'<text x="{new_x_start}" y="{bar_y - 20}" '
                    f'text-anchor="start" font-size="11" '
                    f'fill="#9a3412" font-weight="700">↓ {in_note}</text>'
                )
        else:
            p.append(
                f'<text x="{PAD_L}" y="{narr_y}" font-size="12" '
                f'fill="#9a3412" font-weight="600">↑ {in_note}</text>'
            )

        # Decode narration
        p.append(
            f'<text x="{decode_x}" y="{narr_y}" font-size="12" fill="{C_MUTED}">'
            f'<tspan fill="#065f46" font-weight="700">↓ output*</tspan></text>'
        )

        # ---- Tool-exec arrow ----
        if ri < len(rows) - 1:
            ax = PAD_L + 18
            ay1 = narr_y + 8
            next_row = rows[ri + 1]
            if next_row is SKIP:
                ay2 = cur_y + ROW_H + 8
                lbl = "tool exec   ~0.5–30s  (corpus: median 5s, P99 81s)"
            else:
                ay2 = cur_y + ROW_H + 30
                lbl = "tool exec   ~0.5–30s"
            p.append(
                f'<line x1="{ax}" y1="{ay1}" x2="{ax}" y2="{ay2}" '
                f'stroke="{C_GAP}" stroke-width="1.4" stroke-dasharray="3,3" '
                f'marker-end="url(#arr)"/>'
            )
            p.append(
                f'<text x="{ax + 8}" y="{(ay1 + ay2)/2 + 3}" '
                f'font-size="11" fill="{C_MUTED}" font-style="italic">{lbl}</text>'
            )

        cur_y += ROW_H

    # ---- Footer ----
    foot_y = height - PAD_BOT + 18
    p.append(
        f'<text x="{PAD_L}" y="{foot_y}" font-size="11" fill="{C_MUTED}">'
        f'Across the codex / SWE-bench Pro corpus (610 trials, ~33 calls/trial avg): '
        f'<tspan fill="#9a3412" font-weight="700">94.2% cache hit</tspan>, '
        f'<tspan fill="{C_TEXT}" font-weight="700">131:1 input:output ratio</tspan>, '
        f'median context grows from 12K → 80K tokens per trial.'
        f'</text>'
    )
    p.append(
        f'<text x="{PAD_L}" y="{foot_y + 17}" font-size="10.5" '
        f'fill="{C_MUTED}" font-style="italic">'
        f'* output token counts from corpus medians; '
        f'thinking / tool-call breakdown not preserved in ShareGPT export.'
        f'</text>'
    )

    p.append('</svg>')
    with open(out_path, "w") as f:
        f.write("\n".join(p))
    print(f"Wrote {out_path} ({os.path.getsize(out_path):,} bytes), canvas {width}×{height}")
    print(f"px/tok  input={px_per_tok_in:.4f}  decode={px_per_tok_dec:.3f}  "
          f"(ratio ×{px_per_tok_dec/px_per_tok_in:.1f})")
    for r in rows:
        if r is SKIP:
            continue
        ti = sum(t for _, t in r[1])
        td = sum(t for _, t in r[2])
        print(f"  {r[0]:>11s}: in={ti:>6,}  dec={td:>4,}  timing={r[4]}")


if __name__ == "__main__":
    render_svg(ROWS, OUTPUT)
