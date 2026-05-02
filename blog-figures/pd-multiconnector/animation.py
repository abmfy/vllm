"""Stepwise animated version of figure 2 (PD + MultiConnector).

Renders an MP4 + GIF that *narrates* a single KV-block traversal in clear
phases, rather than just showing balls flying along arrows:

  Phase 1 (emit + dispatch)   KV pill descends from the Prefill KV Block
                              into MultiConnector, briefly pauses at a fork
                              point, then SPLITS into two pills heading for
                              the two children.
  Phase 2 (children execute)  Each pill enters its connector — the connector
                              briefly glows to show it's "active" — and
                              traverses the connector body.
  Phase 3 (output operations) Store pill exits the bottom, drops to the
                              Pool (Pool flashes orange on receive). PD pill
                              exits the right edge, crosses the PD link
                              (the red link flashes), enters the Decode PD,
                              and finally lands at the Decode KV Block
                              (Decode KV pulses on receive).
  Phase 4 (settle)            Everything fades; loop seam is invisible.

The base figure is rendered without the static fan-out arrows
(`include_fanout=False`) — the moving pills depict the dispatch instead.

Particles are rendered as labeled "KV" pills (rounded rect + text), not
plain circles. Motion uses cubic ease-in-out for organic acceleration.
Connectors and the pool are highlighted by overlaying a glow stroke when
"active".

Pipeline:
  1. Generate N=180 SVG frames with particle + glow overlays.
  2. Inkscape `--shell` batch-converts to PNGs.
  3. ffmpeg composes:
       - animation.mp4  (H.264, ~few hundred KB)
       - animation.gif  (palettised, 900 px wide)

Run:
    python3 animation.py        # from this directory, or
    python3 pd-multiconnector/animation.py

Requires: inkscape, ffmpeg (both on PATH).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pd_multiconnector import build  # noqa: E402


# ---- Animation parameters ----
FPS = 30
DURATION_S = 6
N_FRAMES = FPS * DURATION_S            # 180
GIF_WIDTH_PX = 900
GIF_FPS = 20

# ---- Geometry constants (mirror pd_multiconnector.py) ----
INST_W = 440
P_X = 180
D_X = 800
P_CX = P_X + INST_W / 2                # 400
D_CX = D_X + INST_W / 2                # 1020

KVB_W, KVB_H = 220, 50
KVB_Y = 144
KVB_BOT = KVB_Y + KVB_H                # 194
KVB_X_P = P_CX - KVB_W / 2             # 290
KVB_X_D = D_CX - KVB_W / 2             # 910

MC_W = INST_W - 50
MC_X_P = P_X + 25                      # 205
MC_X_D = D_X + 25                      # 825
MC_Y = 224
PAD_X = 20
COL_GAP = 16
COL_W = (MC_W - 2 * PAD_X - COL_GAP) / 2     # 167
MC_CX_P = MC_X_P + MC_W / 2            # 400
MC_CX_D = MC_X_D + MC_W / 2            # 1020

CONN_Y = MC_Y + 75                     # 299
CONN_H = 140
CONN_BOT = CONN_Y + CONN_H             # 439

# Prefill: Store on the LEFT, PD on the RIGHT
P_STORE_X = MC_X_P + PAD_X             # 225
P_STORE_CX = P_STORE_X + COL_W / 2     # 308.5
P_PD_X = MC_X_P + PAD_X + COL_W + COL_GAP                   # 408
P_PD_CX = P_PD_X + COL_W / 2                                 # 491.5
P_PD_RIGHT = P_PD_X + COL_W                                  # 575

# Decode: PD on the LEFT, Store on the RIGHT
D_PD_X = MC_X_D + PAD_X                                       # 845
D_PD_CX = D_PD_X + COL_W / 2                                  # 928.5
D_STORE_X = MC_X_D + PAD_X + COL_W + COL_GAP                  # 1028
D_STORE_CX = D_STORE_X + COL_W / 2                            # 1111.5

PD_LINK_Y = CONN_Y + CONN_H / 2        # 369
POOL_X, POOL_W = 180, 1060
POOL_Y, POOL_H = 560, 100

# Fork point inside MultiConnector — where the single KV pill briefly pauses
# before splitting into two.
FORK_Y = CONN_Y - 30                   # 269

# ---- Style ----
KVB_FILL = "#fdba74"
KVB_STROKE = "#7c3aed"
GLOW_PD = "#ef4444"        # red glow for PD-related events
GLOW_STORE = "#10b981"     # emerald glow for Store-related events
GLOW_POOL = "#f97316"      # orange glow for Pool


# ---- Easing ----
def ease_io(t: float) -> float:
    """Cubic ease-in-out."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    if t < 0.5:
        return 4 * t * t * t
    return 1 - ((-2 * t + 2) ** 3) / 2


# ---- Triangular pulse (for glow events) ----
def pulse(t, t0, t1):
    """0 → 1 → 0 over [t0, t1]; 0 outside."""
    if t < t0 or t > t1:
        return 0.0
    mid = (t0 + t1) / 2
    return (t - t0) / (mid - t0) if t < mid else (t1 - t) / (t1 - mid)


# ---- Particle paths (keyframes) ----
# Each particle: dict with `alive` (start, end), `keyframes` [(t, (x,y)), ...],
# and optional fade_in / fade_out durations.

# Single pill emerges from KV Block, dives into MC, holds at fork while it
# "splits" (the two child pills appear at the same fork point a moment later).
SINGLE_PILL = {
    "alive": (0.00, 0.32),
    "keyframes": [
        (0.00, (P_CX, KVB_BOT)),
        (0.18, (MC_CX_P, FORK_Y)),
        (0.32, (MC_CX_P, FORK_Y)),
    ],
    "fade_in": 0.04,
    "fade_out": 0.06,        # fades out as the two children appear
}

# Store pill: spawns at fork, slides left, descends through Store, drops to Pool.
STORE_PILL = {
    "alive": (0.28, 0.92),
    "keyframes": [
        (0.28, (MC_CX_P, FORK_Y)),               # appears at fork
        (0.40, (P_STORE_CX, FORK_Y)),            # slides left
        (0.50, (P_STORE_CX, CONN_Y)),            # enters Store top
        (0.65, (P_STORE_CX, CONN_BOT)),          # exits Store bottom
        (0.80, (P_STORE_CX, POOL_Y)),            # arrives at Pool
        (0.92, (P_STORE_CX, POOL_Y)),            # holds at Pool while fading
    ],
    "fade_in": 0.04,
    "fade_out": 0.10,
}

# PD pill: spawns at fork, slides right, descends through PD, exits right,
# crosses PD link, enters Decode PD, exits top, arrives at Decode KV Block.
PD_PILL = {
    "alive": (0.28, 0.96),
    "keyframes": [
        (0.28, (MC_CX_P, FORK_Y)),
        (0.40, (P_PD_CX, FORK_Y)),
        (0.50, (P_PD_CX, CONN_Y)),
        (0.58, (P_PD_RIGHT, PD_LINK_Y)),         # exits Prefill PD right edge
        (0.66, (D_PD_X, PD_LINK_Y)),             # crosses PD link
        (0.74, (D_PD_CX, CONN_Y)),               # exits Decode PD top
        (0.86, (KVB_X_D + KVB_W * 0.28, KVB_BOT)),  # arrives at Decode KV Block
        (0.96, (KVB_X_D + KVB_W * 0.28, KVB_BOT)),
    ],
    "fade_in": 0.04,
    "fade_out": 0.10,
}

PARTICLES = [SINGLE_PILL, STORE_PILL, PD_PILL]

# ---- Glow / pulse events ----
# Each is a (kind, t_start, t_end) — a triangular pulse that overlays a
# bright stroke on the named element while it is "active".
GLOW_EVENTS = [
    # KV Block emits — pulses at the very start
    ("p_kv",     0.00, 0.18),
    # Both connectors light up while the children pills are inside them
    ("p_store",  0.50, 0.65),
    ("p_pd",     0.50, 0.62),
    # PD link flashes red while pill is crossing
    ("pd_link",  0.58, 0.68),
    # Decode PD lights up as pill enters
    ("d_pd",     0.66, 0.78),
    # Decode KV pulses as pill arrives
    ("d_kv",     0.84, 0.96),
    # Pool pulses as Store pill arrives
    ("pool",     0.78, 0.92),
]

# ---- Helpers ----
def particle_state(p, t):
    a0, a1 = p["alive"]
    if t < a0 or t > a1:
        return None
    kf = p["keyframes"]
    pos = kf[-1][1]
    for i in range(len(kf) - 1):
        t0, p0 = kf[i]
        t1, p1 = kf[i + 1]
        if t0 <= t <= t1:
            local = (t - t0) / (t1 - t0) if t1 > t0 else 0
            local = ease_io(local)
            pos = (p0[0] + local * (p1[0] - p0[0]),
                   p0[1] + local * (p1[1] - p0[1]))
            break
    fi = p.get("fade_in", 0)
    fo = p.get("fade_out", 0)
    op = 1.0
    if fi > 0 and t - a0 < fi:
        op *= (t - a0) / fi
    if fo > 0 and a1 - t < fo:
        op *= max(0.0, (a1 - t) / fo)
    return (pos[0], pos[1], max(0.0, min(1.0, op)))


def kv_pill_svg(x, y, opacity, label="KV"):
    return (
        f'<g transform="translate({x:.1f}, {y:.1f})" opacity="{opacity:.3f}">'
        f'<rect x="-22" y="-13" width="44" height="26" rx="7" '
        f'fill="{KVB_FILL}" stroke="{KVB_STROKE}" stroke-width="2.2"/>'
        f'<text x="0" y="5" text-anchor="middle" font-size="14" '
        f'font-weight="700" fill="#1f2937">{label}</text>'
        f'</g>'
    )


def glow_rect(x, y, w, h, color, opacity, rx=12, pad=5, sw=5):
    return (
        f'<rect x="{x - pad}" y="{y - pad}" width="{w + 2 * pad}" '
        f'height="{h + 2 * pad}" rx="{rx + pad}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}" opacity="{opacity:.3f}"/>'
    )


def glow_line(x1, y1, x2, y2, color, opacity, sw=8):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" opacity="{opacity:.3f}" '
        f'stroke-linecap="round"/>'
    )


# Map each glow kind to a renderer for that "active" state.
def render_glow(kind, opacity):
    if opacity <= 0.01:
        return ""
    if kind == "p_kv":
        return glow_rect(KVB_X_P, KVB_Y, KVB_W, KVB_H, GLOW_POOL, opacity, rx=10)
    if kind == "d_kv":
        return glow_rect(KVB_X_D, KVB_Y, KVB_W, KVB_H, GLOW_POOL, opacity, rx=10)
    if kind == "p_store":
        return glow_rect(P_STORE_X, CONN_Y, COL_W, CONN_H, GLOW_STORE, opacity)
    if kind == "p_pd":
        return glow_rect(P_PD_X, CONN_Y, COL_W, CONN_H, GLOW_PD, opacity)
    if kind == "d_pd":
        return glow_rect(D_PD_X, CONN_Y, COL_W, CONN_H, GLOW_PD, opacity)
    if kind == "pool":
        return glow_rect(POOL_X, POOL_Y, POOL_W, POOL_H, GLOW_POOL, opacity, rx=20, pad=4, sw=6)
    if kind == "pd_link":
        return glow_line(P_PD_RIGHT + 4, PD_LINK_Y, D_PD_X - 4, PD_LINK_Y,
                         GLOW_PD, opacity, sw=10)
    return ""


# ---- Frame rendering ----
def render_frame_svg(t):
    base = build(include_fanout=False)
    extras = []

    # Glow overlays go BENEATH the particles so the pills sit on top.
    for kind, t0, t1 in GLOW_EVENTS:
        opacity = pulse(t, t0, t1)
        if opacity > 0:
            extras.append(render_glow(kind, opacity))

    # Particles
    for p in PARTICLES:
        s = particle_state(p, t)
        if s is None:
            continue
        x, y, op = s
        if op > 0.01:
            extras.append(kv_pill_svg(x, y, op))

    return base.replace("</svg>", "\n".join(extras) + "\n</svg>")


# ---- Pipeline ----
def _check_tools():
    for tool in ("inkscape", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"error: `{tool}` is not on PATH")


def _generate_svg_frames(work: Path):
    print(f"[1/3] Generating {N_FRAMES} SVG frames in {work} …")
    for i in range(N_FRAMES):
        t = i / N_FRAMES
        (work / f"f{i:03d}.svg").write_text(render_frame_svg(t))


def _inkscape_to_png(work: Path):
    print(f"[2/3] Inkscape batch SVG → PNG ({N_FRAMES} frames)…")
    cmds = []
    for i in range(N_FRAMES):
        svg = work / f"f{i:03d}.svg"
        png = work / f"f{i:03d}.png"
        cmds.append(
            f"file-open:{svg}; export-filename:{png}; "
            f"export-dpi:96; export-do; file-close"
        )
    cmds.append("quit")
    subprocess.run(
        ["inkscape", "--shell"],
        input="\n".join(cmds) + "\n",
        text=True, check=True, capture_output=True,
    )


def _ffmpeg_compose(work: Path, out_dir: Path):
    mp4 = out_dir / "animation.mp4"
    gif = out_dir / "animation.gif"

    print(f"[3a/3] ffmpeg → {mp4.name} (H.264 / yuv420p, crf 20)…")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(work / "f%03d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            str(mp4),
        ], check=True,
    )

    print(f"[3b/3] ffmpeg → {gif.name} (palettised, {GIF_WIDTH_PX}px @ {GIF_FPS}fps)…")
    palette = work / "palette.png"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(work / "f%03d.png"),
            "-vf",
            f"fps={GIF_FPS},scale={GIF_WIDTH_PX}:-1:flags=lanczos,"
            "palettegen=stats_mode=diff",
            str(palette),
        ], check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(work / "f%03d.png"),
            "-i", str(palette),
            "-filter_complex",
            f"[0:v]fps={GIF_FPS},scale={GIF_WIDTH_PX}:-1:flags=lanczos[x];"
            "[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            "-loop", "0",
            str(gif),
        ], check=True,
    )

    return mp4, gif


def main():
    _check_tools()

    out_dir = Path(__file__).parent
    work = Path("/tmp/blog_anim_pd")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    _generate_svg_frames(work)
    _inkscape_to_png(work)
    mp4, gif = _ffmpeg_compose(work, out_dir)

    print()
    print(f"  {mp4}  ({mp4.stat().st_size:,} bytes)")
    print(f"  {gif}  ({gif.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
