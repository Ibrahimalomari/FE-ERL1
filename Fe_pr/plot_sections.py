"""
plot_sections.py
================
Visualise every fiber cross-section in the model using the
``openseespy.postprocessing.ops_vis.plot_fiber_section`` API.

Each section is encoded as a ``fib_sec_list`` -- the same format that
``ops_vis`` uses -- so no OpenSees domain needs to be open when
plotting.  Sections are saved as individual PNG files named by their
section tag:

    output/section_<tag>_<name>.png

Usage (standalone)
------------------
    python -m fe_model.plot_sections

Usage (from pipeline / run_model.py)
--------------------------------------
    from fe_model.plot_sections import plot_all_sections
    plot_all_sections(output_dir='./output')

Sections covered
----------------
Tag   Name          Type
---   ----          ----
1     W33x118       Composite steel I + concrete deck
2     W33x141       Composite steel I + concrete deck
3     W33x130       Composite steel I + concrete deck
4     C12x207       Steel channel
100   CB1           RC rectangular beam (full rebar)
101   CB2           RC rectangular beam (reduced rebar)
102   C1            RC circular column

If a ``Sections.csv`` is present in ``./data`` the composite sections
(tags 1-N) are built from it automatically.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')          # headless – works on any server / CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import openseespy.postprocessing.ops_vis as opsv
    _HAS_OPSV = True
except ImportError:
    _HAS_OPSV = False


# ── material colour map ────────────────────────────────────────────────────────
# mat_id -> display colour  (matches original script tag numbers)
# 1 = steel, 2 = unconfined concrete, 3 = confined concrete
MAT_COLORS: Dict[int, str] = {
    1: '#4a90d9',        # steel     – blue
    2: '#c8c8c8',        # unconfined concrete – light grey
    3: '#888888',        # confined concrete   – dark grey
    99: '#f0c040',       # rebar dots          – gold
}

# Human-readable material labels
MAT_LABELS: Dict[int, str] = {
    1:  'Steel',
    2:  'Unconfined Concrete',
    3:  'Confined Concrete',
    99: 'Rebar',
}

# ── shared geometry constants (mirror of Model_Defin.py) ──────────────────────
_STEEL  = 1
_UNC    = 2   # unconfined concrete
_CONF   = 3   # confined concrete

_Es  = 29000.0
_v   = 0.2
_Gs  = _Es / (1 + _v)
_Gc  = 3506 / (2 * 1.2)


# ══════════════════════════════════════════════════════════════════════════════
# fib_sec_list builders (one per section type)
# ══════════════════════════════════════════════════════════════════════════════

def _rc_rect_beam_cb1() -> Tuple[str, int, List]:
    """CB1 – 38x45 RC beam, full rebar cage (tag 100)."""
    b, d, cc = 38.0, 45.0, 2.25
    J  = b * d * (b*b + d*d) / 12.0
    GJ = _Gc * J
    nfy, nfz = 20, 30

    fib = [['section', 'Fiber', 100, '-GJ', GJ]]
    # confined core
    fib.append(['patch', 'rect', _CONF, nfy, nfz,
                -b/2+cc, -d/2+cc, b/2-cc,  d/2-cc])
    # unconfined cover (4 strips)
    fib.append(['patch', 'rect', _UNC,
                nfy, 1, -b/2,  d/2-cc, b/2,  d/2])
    fib.append(['patch', 'rect', _UNC, nfy, 1, -b/2, -d/2,   b/2, -d/2+cc])
    fib.append(['patch', 'rect', _UNC, 1, nfz, -b/2, -d/2+cc, -b/2+cc, d/2-cc])
    fib.append(['patch', 'rect', _UNC, 1, nfz, b/2-cc, -d/2+cc, b/2,  d/2-cc])
    # rebar layers
    fib.append(['layer', 'straight', _STEEL, 5, 0.79,
                -b/2+cc, d/2-cc, b/2-cc, d/2-cc])          # 5-#8  top
    fib.append(['layer', 'straight', _STEEL, 4, 1.00,
                -b/2+cc+4.1875, d/2-cc,
                 b/2-cc-4.1875, d/2-cc])                   # 4-#9  top (inset)
    fib.append(['layer', 'straight', _STEEL, 2, 0.44,
                -b/2+cc, 0.0, b/2-cc, 0.0])                # 2-#6  mid
    fib.append(['layer', 'straight', _STEEL, 9, 1.56,
                -b/2+cc, -(d/2-cc), b/2-cc, -(d/2-cc)])    # 9-#11 bottom
    return 'CB1', 100, fib


def _rc_rect_beam_cb2() -> Tuple[str, int, List]:
    """CB2 – 38x45 RC beam, reduced top rebar (tag 101)."""
    b, d, cc = 38.0, 45.0, 2.25
    J  = b * d * (b*b + d*d) / 12.0
    GJ = _Gc * J
    nfy, nfz = 20, 30

    fib = [['section', 'Fiber', 101, '-GJ', GJ]]
    fib.append(['patch', 'rect', _CONF, nfy, nfz,
                -b/2+cc, -d/2+cc, b/2-cc,  d/2-cc])
    fib.append(['patch', 'rect', _UNC, nfy, 1, -b/2,  d/2-cc, b/2,  d/2])
    fib.append(['patch', 'rect', _UNC, nfy, 1, -b/2, -d/2,   b/2, -d/2+cc])
    fib.append(['patch', 'rect', _UNC, 1, nfz, -b/2, -d/2+cc, -b/2+cc, d/2-cc])
    fib.append(['patch', 'rect', _UNC, 1, nfz, b/2-cc, -d/2+cc, b/2,  d/2-cc])
    fib.append(['layer', 'straight', _STEEL, 5, 0.79,
                -b/2+cc, d/2-cc, b/2-cc, d/2-cc])          # 5-#8 top
    fib.append(['layer', 'straight', _STEEL, 2, 0.44,
                -b/2+cc, 0.0, b/2-cc, 0.0])                # 2-#6 mid
    fib.append(['layer', 'straight', _STEEL, 9, 1.56,
                -b/2+cc, -(d/2-cc), b/2-cc, -(d/2-cc)])    # 9-#11 bottom
    return 'CB2', 101, fib


def _rc_circular_col_c1() -> Tuple[str, int, List]:
    """C1 – 36-inch diameter RC column (tag 102)."""
    r, cc = 18.0, 2.25
    Jc = math.pi / 2 * r**4
    GJ = _Gc * Jc

    fib = [['section', 'Fiber', 102, '-GJ', GJ]]
    fib.append(['patch', 'circ', _CONF, 10, 5, 0, 0, 0, r-cc, 0, 360])
    fib.append(['patch', 'circ', _UNC,  10, 1, 0, 0, r-cc, r, 0, 360])
    fib.append(['layer', 'circ', _STEEL, 8, 1.27, 0, 0, r-cc])
    return 'C1', 102, fib


def _steel_channel_c12x207() -> Tuple[str, int, List]:
    """C12x207 channel section (tag 4)."""
    d, bf, tf, tw = 12.0, 2.94, 0.501, 0.282
    GJ = _Gs * 0.37

    fib = [['section', 'Fiber', 4, '-GJ', GJ]]
    # top flange, bottom flange, web
    fib.append(['patch', 'rect', _STEEL, 10, 1, tw/2,  d/2-tf, bf-tw/2,  d/2])
    fib.append(['patch', 'rect', _STEEL, 10, 1, tw/2, -d/2,   bf-tw/2, -d/2+tf])
    fib.append(['patch', 'rect', _STEEL,  1, 20, -tw/2, -d/2, tw/2,  d/2])
    return 'C12x207', 4, fib


def _composite_section_from_row(
    row_index: int, row, GJ_scale: float = 1.0, hf: float = 7.5,
) -> Tuple[str, int, List]:
    """Build fib_sec_list for one composite steel+slab section from a
    Sections.csv row. Tag = row_index + 1 (1-based, mirrors original loop)."""
    tag = row_index + 1
    GJ  = GJ_scale * float(row['J'])

    fib = [['section', 'Fiber', tag, '-GJ', GJ]]
    fib.append(['patch', 'rect', _STEEL, 10, 1,
                row['yi1'], row['zi1'], row['yj1'], row['zj1']])
    fib.append(['patch', 'rect', _STEEL, 10, 1,
                row['yi2'], row['zi2'], row['yj2'], row['zj2']])
    fib.append(['patch', 'rect', _STEEL,  1, 20,
                row['yi3'], row['zi3'], row['yj3'], row['zj3']])
    # Concrete deck
    fib.append(['patch', 'rect', _UNC, 5, 5,
                -row['be']/2, row['d']/2,
                 row['be']/2, row['d']/2 + hf])
    return f'Composite_{tag}', tag, fib


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib-native fallback renderer
# (used when ops_vis is unavailable, or for circular/composite sections that
#  ops_vis doesn't handle well)
# ══════════════════════════════════════════════════════════════════════════════

def _color_for_mat(mat_id: int) -> str:
    return MAT_COLORS.get(mat_id, '#ffffff')


def _render_fib_sec_native(fib_sec_list: List, ax: plt.Axes, title: str) -> None:
    """Draw patches and rebar dots directly with matplotlib."""
    for cmd in fib_sec_list:
        if cmd[0] != 'patch' and cmd[0] != 'layer':
            continue

        if cmd[0] == 'patch':
            kind = cmd[1]
            mat  = cmd[2]
            color = _color_for_mat(mat)

            if kind == 'rect':
                # ['patch','rect', mat, nfy, nfz, y1, z1, y2, z2]
                _, _, _, nfy, nfz, y1, z1, y2, z2 = cmd
                rect = plt.Rectangle(
                    (min(y1, y2), min(z1, z2)),
                    abs(y2 - y1), abs(z2 - z1),
                    linewidth=0.3, edgecolor='#333333',
                    facecolor=color, zorder=1,
                )
                ax.add_patch(rect)

            elif kind == 'circ':
                # ['patch','circ', mat, ncirc, nrad, cy, cz, ri, ro, a0, a1]
                _, _, _, ncirc, nrad, cy, cz, ri, ro, a0, a1 = cmd
                # outer annulus (filled)
                outer = plt.Circle((cy, cz), ro,
                                   facecolor=color, edgecolor='#333333',
                                   linewidth=0.3, zorder=1)
                ax.add_patch(outer)
                # hollow inner if ri > 0
                if ri > 0:
                    inner_color = _color_for_mat(_CONF if mat == _UNC else _UNC)
                    inner = plt.Circle((cy, cz), ri,
                                       facecolor=inner_color, edgecolor='#333333',
                                       linewidth=0.3, zorder=2)
                    ax.add_patch(inner)

        elif cmd[0] == 'layer':
            kind = cmd[1]
            mat  = cmd[2]

            if kind == 'straight':
                # ['layer','straight', mat, n, area, y1,z1, y2,z2]
                _, _, _, n, area, y1, z1, y2, z2 = cmd
                r = math.sqrt(area / math.pi)
                if n == 1:
                    ys = [0.5 * (y1 + y2)]
                    zs = [0.5 * (z1 + z2)]
                else:
                    ys = [y1 + (y2 - y1) * i / (n - 1) for i in range(n)]
                    zs = [z1 + (z2 - z1) * i / (n - 1) for i in range(n)]
                for y, z in zip(ys, zs):
                    circ = plt.Circle((y, z), r,
                                      facecolor=MAT_COLORS[99], edgecolor='#222222',
                                      linewidth=0.5, zorder=5)
                    ax.add_patch(circ)

            elif kind == 'circ':
                # ['layer','circ', mat, n, area, cy, cz, radius]
                _, _, _, n, area, cy, cz, radius = cmd
                r = math.sqrt(area / math.pi)
                for i in range(n):
                    angle = 2 * math.pi * i / n
                    y = cy + radius * math.cos(angle)
                    z = cz + radius * math.sin(angle)
                    circ = plt.Circle((y, z), r,
                                      facecolor=MAT_COLORS[99], edgecolor='#222222',
                                      linewidth=0.5, zorder=5)
                    ax.add_patch(circ)


# ══════════════════════════════════════════════════════════════════════════════
# Figure assembly
# ══════════════════════════════════════════════════════════════════════════════

def _make_legend(ax: plt.Axes, fib_sec_list: List) -> None:
    """Add a compact legend showing only the materials present."""
    used_mats = set()
    for cmd in fib_sec_list:
        if cmd[0] == 'patch':
            used_mats.add(cmd[2])
        elif cmd[0] == 'layer':
            used_mats.add(99)   # rebar
    handles = []
    for mat in sorted(used_mats):
        handles.append(
            mpatches.Patch(
                facecolor=MAT_COLORS.get(mat, 'white'),
                edgecolor='#333333', linewidth=0.5,
                label=MAT_LABELS.get(mat, f'Mat {mat}'),
            )
        )
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.85, edgecolor='#aaaaaa')


def _save_section(
    name: str, tag: int, fib_sec_list: List, output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('#ffffff')

    _render_fib_sec_native(fib_sec_list, ax, name)
    _make_legend(ax, fib_sec_list)

    # Auto-scale with 10 % margin
    ax.autoscale_view()
    xl, xh = ax.get_xlim(); yl, yh = ax.get_ylim()
    mx = (xh - xl) * 0.12;  my = (yh - yl) * 0.12
    ax.set_xlim(xl - mx, xh + mx)
    ax.set_ylim(yl - my, yh + my)

    ax.set_xlabel('y  (in)', fontsize=9)
    ax.set_ylabel('z  (in)', fontsize=9)
    ax.set_title(f'Section {tag} – {name}', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=8)

    # Add dimension annotations (bounding box)
    xs = [l for l in ax.get_xlim()]
    ys = [l for l in ax.get_ylim()]
    width  = xh - xl
    height = yh - yl
    ax.annotate(
        f'W ≈ {width:.1f} in   H ≈ {height:.1f} in',
        xy=(0.03, 0.03), xycoords='axes fraction',
        fontsize=7, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8),
    )

    fname = output_dir / f'section_{tag:03d}_{name}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]   Saved  {fname.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_sections(
    data_dir:   str = './data',
    output_dir: str = './output',
) -> None:
    """Plot and save every fiber section.

    Parameters
    ----------
    data_dir : str
        Directory that may contain ``Sections.csv`` for composite girders.
    output_dir : str
        Where the PNG files are written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── fixed sections ────────────────────────────────────────────────────────
    fixed_sections = [
        _rc_rect_beam_cb1(),
        _rc_rect_beam_cb2(),
        _rc_circular_col_c1(),
        _steel_channel_c12x207(),
    ]
    for name, tag, fib in fixed_sections:
        _save_section(name, tag, fib, out)

    # ── composite sections from Sections.csv (if present) ────────────────────
    sections_csv = Path(data_dir) / 'Sections.csv'
    if sections_csv.exists():
        import pandas as pd
        s = pd.read_csv(sections_csv)
        print(f'[INFO] Found Sections.csv with {len(s)} composite sections')
        for i, row in s.iterrows():
            name, tag, fib = _composite_section_from_row(i, row, GJ_scale=_Gs)
            _save_section(name, tag, fib, out)
    else:
        print('[INFO] Sections.csv not found – skipping composite sections')

    print(f'\n✓  All section plots saved to  {out.resolve()}')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_all_sections()
