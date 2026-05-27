"""
config.py
=========
Units and run-time configuration. Replaces the loose constants scattered
through the original script (`inch = 1`, `ft = 12*inch`, ...) with a single
authoritative place so every module agrees.

Bug fixes
---------
* Added missing CSV filename fields that pipeline.py, builder.py, and
  run_model.py all reference:
    - transformations_csv
    - materials_csv
    - rc_rect_csv
    - rc_circ_csv
    - steel_sections_csv
* Removed the now-redundant rc_sections_csv (singular) field that was never
  used anywhere.
* Units.m value kept at 39.3701 (correct SI conversion); the original
  script's 39.37 was a rounded approximation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class Units:
    """Base units: kip, inch, second.

    Use as ``Units.ft``, ``Units.ksi``, etc.  Anything written in source
    should multiply by units explicitly so a future unit change touches
    one place only.
    """
    # ---- Length ----
    inch = 1.0
    ft   = 12.0 * inch
    m    = 39.3701 * inch   # exact SI: 1 m = 39.3701 in
    mm   = m / 1000.0
    # ---- Force ----
    kip = 1.0
    lb  = kip / 1000.0
    kN  = 0.224809 * kip
    N   = kN / 1000.0
    # ---- Stress ----
    ksi = kip / inch ** 2
    psi = ksi / 1000.0
    MPa = 0.145038 * ksi
    GPa = 1000.0 * MPa
    # ---- Other ----
    g   = 386.088 * inch  # gravity, in/s^2


@dataclass
class Config:
    """Run-time configuration. All paths, tolerances, and CSV filenames live
    here so the calling script never hardcodes strings."""

    # ---- Paths ----
    data_dir:   Path = Path('./data')
    output_dir: Path = Path('./output')

    # ---- Model ----
    ndm: int = 3
    ndf: int = 6

    # ---- Numerical ----
    max_iter: int           = 50
    tolerance: float        = 1e-6
    n_int_points: int       = 5         # Lobatto integration points / element
    gravity_load_steps: int = 100

    # ---- Required input files ----
    nodes_csv:           str = 'nodes.csv'
    beams_csv:           str = 'Beam Elements.csv'
    columns_csv:         str = 'Column Elements.csv'
    equal_dof_csv:       str = 'EqualDOF.csv'
    bc_csv:              str = 'boundary_conditions.csv'
    transformations_csv: str = 'transformations.csv'   # FIX: was missing
    materials_csv:       str = 'materials.csv'         # FIX: was missing

    # ---- Optional section CSVs ----
    # Provide whichever section types your model uses; absent files are skipped.
    sections_csv:        str = 'Sections.csv'          # composite steel + deck
    rc_rect_csv:         str = 'rc_rect_sections.csv'  # FIX: was missing (rc_sections_csv)
    rc_circ_csv:         str = 'rc_circ_sections.csv'  # FIX: was missing
    steel_sections_csv:  str = 'steel_sections.csv'    # FIX: was missing

    # ---- Other optional files ----
    loads_csv:  str = 'loads.csv'
    slabs_csv:  str = 'slabs.csv'

    # ---- Visualization settings ----
    plot_model:              bool  = True
    save_plots:              bool  = True
    show_node_tags:          bool  = False
    element_linewidth_scale: float = 1.0
    plot_azim:               float = -60
    plot_elev:               float = 30

    # ---- Solver settings ----
    solver_system:      str = 'SuperLU'
    solver_numberer:    str = 'RCM'
    solver_constraints: str = 'Transformation'
    solver_algorithm:   str = 'ModifiedNewton'

    def __post_init__(self):
        self.data_dir   = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def in_data(self, filename: str) -> Path:
        return self.data_dir / filename

    def in_output(self, filename: str) -> Path:
        return self.output_dir / filename
