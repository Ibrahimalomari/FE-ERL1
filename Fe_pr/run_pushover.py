"""
run_pushover.py
===============
Gravity  →  Modal  →  Displacement-controlled Pushover

HOW TO USE
----------
1.  Edit the USER CONFIGURATION block below:
      pushover_dof         -- 1=X, 2=Y, 3=Z
      floors_to_load       -- which floor indices get lateral loads
      load_distribution    -- 'triangular' (∝ height) or 'uniform'
      control_floor        -- index of the floor that drives the analysis
      control_node_tag     -- exact node tag (None = auto, first on control_floor)
      max_disp             -- target roof displacement (inches)
      disp_incr            -- displacement step (inches)

2.  Run:
        python run_pushover.py

Outputs (written to ./output/)
---------------------------------
  pushover_curve.png            -- base-shear vs. roof-displacement plot
  pushover_curve_points.csv     -- all recorded (displacement, base-shear) points
  pushover_summary.csv          -- structural response parameters:
                                   first 5 periods, initial stiffness,
                                   max deflection, max force, area under curve
"""

import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openseespy.opensees import (
    loadConst, wipeAnalysis, timeSeries as ops_timeSeries,
    pattern, load,
    constraints, numberer, system, test,
    algorithm, integrator, analysis,
    analyze, nodeDisp, nodeReaction, reactions,
    eigen,
)

from fe_model import Config, build_full_model


# ================================================================
#  MODEL CONFIG  (mirror run_model.py — change only if needed)
# ================================================================
cfg = Config(
    data_dir            = './data',
    output_dir          = './output',
    gravity_load_steps  = 100,

    materials_csv       = 'materials.csv',
    transformations_csv = 'transformations.csv',
    bc_csv              = 'boundary_conditions.csv',
    nodes_csv           = 'nodes.csv',
    equal_dof_csv       = 'EqualDOF.csv',
    beams_csv           = 'Beam Elements.csv',
    columns_csv         = 'Column Elements.csv',
    loads_csv           = 'loads.csv',
    rc_rect_csv         = 'rc_rect_sections.csv',
    rc_circ_csv         = 'rc_circ_sections.csv',
    steel_sections_csv  = 'steel_sections.csv',
    sections_csv        = 'Sections.csv',
    slabs_csv           = 'slabs.csv',
    plot_model          = False,
    save_plots          = False,
)


# ================================================================
#  USER CONFIGURATION — edit these values
# ================================================================

# Direction of the lateral push: 1=X, 2=Y, 3=Z
pushover_dof = 1

# Floor indices to receive lateral loads (1 = base/support level, skip it).
# All nodes on each specified floor will be loaded.
# e.g. [2, 3, 4, 5]  →  floors 2 through roof
floors_to_load = [2, 3, 4, 5]

# Load distribution over height:
#   'triangular' — force ∝ (floor height above base) / (roof height above base)
#   'uniform'    — equal unit force on every loaded floor
load_distribution = 'triangular'

# Floor whose node drives displacement control (typically the roof = highest floor)
control_floor = 5

# Specific control-node tag.  Set to None to auto-select the lowest-numbered
# node on control_floor (deterministic across runs).
control_node_tag = 190   # e.g. 153

# Maximum roof displacement and step size (inches)
max_disp  = 10.0   # inches  — increase for ductile structures
disp_incr = 0.05   # inches  — decrease for finer resolution near yield

# ================================================================
#  END OF USER CONFIGURATION
# ================================================================


def _build_floor_map(builder):
    """Return {floor_index: [node_tags], ...} sorted by elevation.

    Floor 1 is the lowest elevation (usually the support level).
    """
    z_to_nodes: dict = defaultdict(list)
    for tag, (_, _, z) in builder.node_coords.items():
        z_to_nodes[round(z, 2)].append(tag)

    sorted_z = sorted(z_to_nodes.keys())
    floor_map = {
        i + 1: sorted(z_to_nodes[z])
        for i, z in enumerate(sorted_z)
    }
    z_map = {i + 1: z for i, z in enumerate(sorted_z)}
    return floor_map, z_map


def main():
    # --------------------------------------------------------
    # Phase 1: Build model, gravity, modal
    # --------------------------------------------------------
    print('=' * 60)
    print('Phase 1 — Gravity + Modal')
    print('=' * 60)
    results = build_full_model(
        cfg,
        run_gravity = True,
        run_modal   = True,
        n_modes     = 10,
    )
    builder = results['builder']

    # Extract first 5 vibration periods (try common keys, fall back to eigen())
    periods = _extract_periods(results, n=5)

    # --------------------------------------------------------
    # Auto-detect floor topology from model geometry
    # --------------------------------------------------------
    floor_map, z_map = _build_floor_map(builder)
    n_floors = len(floor_map)

    if n_floors < 2:
        raise RuntimeError(
            'Model has fewer than 2 distinct floor elevations. '
            'Check nodes.csv.'
        )

    base_z   = z_map[1]
    roof_z   = z_map[n_floors]
    total_h  = roof_z - base_z
    ctrl_z   = z_map[control_floor]

    # Validate user choices
    for fl in floors_to_load:
        if fl not in floor_map:
            raise ValueError(
                f'floors_to_load contains floor index {fl}, '
                f'but the model only has {n_floors} floors (1..{n_floors}).'
            )
    if control_floor not in floor_map:
        raise ValueError(
            f'control_floor={control_floor} not found. '
            f'Valid floor indices: 1..{n_floors}.'
        )

    # Control node
    ctrl_node = (
        control_node_tag
        if control_node_tag is not None
        else floor_map[control_floor][0]
    )

    # Reaction nodes = base (floor 1) nodes
    reaction_nodes = floor_map[1]

    # Print detected topology for reference
    print()
    print('--- Detected floor topology ---')
    for fl in range(1, n_floors + 1):
        tag_sample = floor_map[fl][:4]
        label = ' ← base/support' if fl == 1 else (
                ' ← roof'         if fl == n_floors else '')
        print(f'  Floor {fl}:  z = {z_map[fl]:.1f} in'
              f'  ({z_map[fl]/39.3701:.2f} m)'
              f'  {len(floor_map[fl])} nodes  {tag_sample}...{label}')
    print()
    print(f'Control node  : {ctrl_node}  (floor {control_floor}, DOF {pushover_dof})')
    print(f'Reaction nodes: {reaction_nodes[:5]}... ({len(reaction_nodes)} total, floor 1)')
    print()

    # --------------------------------------------------------
    # Phase 2: Lock gravity loads, reset pseudo-time
    # --------------------------------------------------------
    loadConst('-time', 0.0)
    wipeAnalysis()
    print('[OK]   Gravity loads locked constant. Time reset to 0.0.')

    # --------------------------------------------------------
    # Phase 3: Lateral load pattern (inverted-triangular or uniform)
    # --------------------------------------------------------
    ops_timeSeries('Linear', 10)        # tag 10: fresh series for lateral loads
    pattern('Plain', 2, 10)

    loaded_nodes: dict = {}
    for fl in floors_to_load:
        z = z_map[fl]
        h_ratio = (z - base_z) / total_h if total_h > 0 else 1.0
        factor  = h_ratio if load_distribution == 'triangular' else 1.0
        forces  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        forces[pushover_dof - 1] = factor                 # set the correct DOF
        for tag in floor_map[fl]:
            load(tag, *forces)
            loaded_nodes[tag] = forces

    print(f'[OK]   Applied {load_distribution} lateral loads '
          f'to {len(loaded_nodes)} nodes on floors {floors_to_load}.')

    # --------------------------------------------------------
    # Phase 4: Configure and run displacement-controlled pushover
    # --------------------------------------------------------
    constraints('Transformation')
    numberer('RCM')
    system('SuperLU')
    test('NormDispIncr', 1.0e-6, 2000, 0)
    algorithm('ModifiedNewton')
    integrator(
        'DisplacementControl',
        ctrl_node, pushover_dof,
        disp_incr,          # initial dU
        1,                  # Jd
        disp_incr ,  # min dU  (allows very small steps near limit)
        disp_incr*2,          # max dU
    )
    analysis('Static')

    print()
    print(f'=== Pushover: node {ctrl_node}, DOF {pushover_dof}, '
          f'target Δ = {max_disp:.2f} in ===')

    displacements = [0.0]
    base_shears   = [0.0]
    current_disp  = 0.0
    ok = 0
    step = 0

    while ok == 0 and abs(current_disp) < max_disp:
        ok = analyze(1)

        # Fallback: initial-stiffness iteration
        if ok != 0:
            algorithm('ModifiedNewton', '-initial')
            test('NormDispIncr', 1.0e-4, 1000, 0)
            ok = analyze(1)
            algorithm('ModifiedNewton')
            test('NormDispIncr', 1.0e-6, 1000, 0)

        if ok != 0:
            print(f'\n[FAIL] Convergence failure at step {step}, '
                  f'Δ = {current_disp:.4f} in. Stopping.')
            break

        current_disp = nodeDisp(ctrl_node, pushover_dof)
        reactions()
        shear = sum(nodeReaction(n, pushover_dof) for n in reaction_nodes)

        displacements.append(current_disp)
        base_shears.append(abs(shear))
        step += 1

        if step % 20 == 0 or abs(current_disp) >= max_disp:
            print(f'  step {step:5d}   Δ = {current_disp:8.4f} in'
                  f'   V_base = {abs(shear):.2f} kips')

    print()
    status = 'completed' if ok == 0 else 'stopped early (convergence)'
    print(f'Pushover {status}: {step} steps, '
          f'final Δ = {current_disp:.4f} in, '
          f'peak V = {max(base_shears):.2f} kips')

    # --------------------------------------------------------
    # Phase 5: Save CSVs and plot
    # --------------------------------------------------------
    _save_curve_csv(displacements, base_shears, ctrl_node)
    _save_summary_csv(displacements, base_shears, periods)
    _save_plot(displacements, base_shears, ctrl_node, reaction_nodes)

    print()
    print('=' * 60)
    print('Pushover complete. Output written to output/')
    print('  pushover_curve_points.csv')
    print('  pushover_summary.csv')
    print('  pushover_curve.png')
    print('=' * 60)


# ================================================================
#  I/O helpers
# ================================================================

def _extract_periods(results, n=5):
    """Pull first n vibration periods from modal-analysis output.

    Tries common keys in the `results` dict returned by build_full_model;
    if none are found, falls back to calling OpenSeesPy's `eigen()` directly
    and converting eigenvalues (rad^2/s^2) to periods (s).
    """
    # Try common keys returned by build_full_model
    for key in ('periods', 'T', 'modal_periods'):
        if key in results and results[key]:
            T = list(results[key])
            return (T + [float('nan')] * n)[:n]

    # Try nested under 'modal'
    modal = results.get('modal')
    if isinstance(modal, dict):
        for key in ('periods', 'T'):
            if key in modal and modal[key]:
                T = list(modal[key])
                return (T + [float('nan')] * n)[:n]
        # Derive from eigenvalues if exposed
        for key in ('eigenvalues', 'lambdas'):
            if key in modal and modal[key]:
                lam = list(modal[key])
                T = [2.0 * math.pi / math.sqrt(l) if l > 0 else float('nan')
                     for l in lam]
                return (T + [float('nan')] * n)[:n]

    # Fallback: ask OpenSees directly. Eigenvalues are in rad^2/s^2.
    try:
        lam = eigen(n)
        T = [2.0 * math.pi / math.sqrt(l) if l > 0 else float('nan')
             for l in lam]
        return (T + [float('nan')] * n)[:n]
    except Exception as exc:
        print(f'[WARN] Could not recover periods ({exc}); writing NaNs.')
        return [float('nan')] * n


def _compute_initial_stiffness(displacements, base_shears):
    """Estimate initial elastic stiffness K0 = dV / dΔ near the origin.

    Uses a least-squares fit through (0, 0) over the first few non-trivial
    points, which is more robust than the very first step (often zero or
    noise-dominated). Falls back gracefully on short data.
    """
    # Use up to the first ~5% of the analysis or 10 points, whichever is larger
    n_fit = max(2, min(len(displacements) - 1, max(10, len(displacements) // 20)))
    xs = displacements[1:1 + n_fit]
    ys = base_shears[1:1 + n_fit]
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    if den == 0:
        return float('nan')
    return num / den


def _area_under_curve(displacements, base_shears):
    """Trapezoidal integration of base shear vs displacement (kips·in)."""
    area = 0.0
    for i in range(1, len(displacements)):
        dx = displacements[i] - displacements[i - 1]
        area += 0.5 * (base_shears[i] + base_shears[i - 1]) * dx
    return area


def _save_curve_csv(displacements, base_shears, ctrl_node):
    """CSV #1 — all recorded (displacement, base shear) points."""
    path = cfg.in_output('pushover_curve_points.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            f'roof_disp_node{ctrl_node}_dof{pushover_dof}_in',
            'base_shear_kips',
        ])
        for d, v in zip(displacements, base_shears):
            w.writerow([round(d, 6), round(v, 4)])
    print(f'[OK]   Curve points → {path}  ({len(displacements)} rows)')


def _save_summary_csv(displacements, base_shears, periods):
    """CSV #2 — scalar response parameters."""
    k0       = _compute_initial_stiffness(displacements, base_shears)
    max_d    = max((abs(d) for d in displacements), default=0.0)
    max_v    = max(base_shears) if base_shears else 0.0
    area     = _area_under_curve(displacements, base_shears)

    # Pad / trim periods to exactly 5
    T = (list(periods) + [float('nan')] * 5)[:5]

    path = cfg.in_output('pushover_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value', 'unit'])
        for i, t in enumerate(T, start=1):
            w.writerow([f'T{i}_period_mode_{i}', _fmt(t), 's'])
        w.writerow(['initial_stiffness_K0', _fmt(k0),  'kips/in'])
        w.writerow(['max_deflection',       _fmt(max_d), 'in'])
        w.writerow(['max_base_shear',       _fmt(max_v), 'kips'])
        w.writerow(['area_under_curve',     _fmt(area),  'kips*in'])

    print(f'[OK]   Summary      → {path}')
    print(f'       Periods (s)     : '
          + ', '.join(f'T{i}={_fmt(t)}' for i, t in enumerate(T, 1)))
    print(f'       K0  (kips/in)   : {_fmt(k0)}')
    print(f'       max Δ (in)      : {_fmt(max_d)}')
    print(f'       max V (kips)    : {_fmt(max_v)}')
    print(f'       Area (kips*in)  : {_fmt(area)}')


def _fmt(x, ndp=6):
    """Compact float formatting that survives NaN."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 'nan'
        return round(float(x), ndp)
    except Exception:
        return 'nan'


def _save_plot(displacements, base_shears, ctrl_node, reaction_nodes):
    if len(displacements) < 2:
        print('[WARN] Not enough data to plot.')
        return

    # ---- capacity curve ----
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(displacements, base_shears, color='steelblue',
            linewidth=2.0, zorder=3)

    # light markers every ~50 steps so the curve isn't cluttered
    stride = max(1, len(displacements) // 50)
    ax.scatter(displacements[::stride], base_shears[::stride],
               s=18, color='steelblue', zorder=4, label='_nolegend_')

    # annotate the peak
    peak_v = max(base_shears)
    peak_d = displacements[base_shears.index(peak_v)]
    ax.annotate(
        f'Peak: {peak_v:.1f} kips\n@ Δ = {peak_d:.2f} in',
        xy=(peak_d, peak_v),
        xytext=(peak_d + max(displacements) * 0.05,
                peak_v * 0.90),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray'),
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8),
    )

    # axes
    ax.set_xlabel(
        f'Roof Displacement — node {ctrl_node}, DOF {pushover_dof} (in)',
        fontsize=11,
    )
    ax.set_ylabel(
        f'Base Shear — sum of reactions at {len(reaction_nodes)} support nodes (kips)',
        fontsize=11,
    )
    ax.set_title(
        f'Pushover Curve  '
        f'({"X" if pushover_dof==1 else "Y" if pushover_dof==2 else "Z"}-direction, '
        f'{load_distribution} loading)',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = cfg.in_output('pushover_curve.png')
    fig.savefig(str(path), dpi=600, bbox_inches='tight')
    path = cfg.in_output('pushover_curve.svg')
    fig.savefig(str(path),bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]   Plot     → {path}')


# ================================================================
if __name__ == '__main__':
    main()