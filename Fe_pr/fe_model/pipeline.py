"""
pipeline.py
===========
High-level orchestration: wires every component in the correct order.

Bug fixes
---------
1. Replaced all calls to non-existent methods:
     materials.build_from_csv()           → now exists (added to materials.py)
     sections.build_rc_rect_from_csv()    → now exists (added to sections.py)
     sections.build_rc_circ_from_csv()    → now exists (added to sections.py)
     sections.build_steel_from_csv()      → now exists (added to sections.py)
     sections.build_composite_from_csv()  → now exists (added to sections.py)

2. Replaced references to non-existent Config attributes:
     config.materials_csv        → now declared in Config
     config.rc_rect_csv          → now declared in Config
     config.rc_circ_csv          → now declared in Config
     config.steel_sections_csv   → now declared in Config

3. Removed the import of SlabGenerator from .slabs (slabs.py does not exist
   in the package). The slab step is now guarded so it runs only when a slabs
   module is available; a clear ImportWarning is printed otherwise.
"""

from .config    import Config
from .data_io   import DataLoader
from .materials import MaterialLibrary
from .sections  import SectionLibrary
from .builder   import ModelBuilder
from .loads     import LoadBuilder
from .analysis  import AnalysisDriver


def build_full_model(
    config: Config,
    run_gravity: bool = True,
    run_modal:   bool = False,
    n_modes:     int  = 5,
) -> dict:
    """Build, load, and analyse the model end-to-end.

    Required CSVs
    -------------
    nodes.csv, EqualDOF.csv, boundary_conditions.csv,
    transformations.csv, materials.csv,
    Beam Elements.csv, Column Elements.csv, loads.csv

    Optional section CSVs (provide whichever types your model uses)
    ---------------------------------------------------------------
    rc_rect_sections.csv   -- RC rectangular fiber sections
    rc_circ_sections.csv   -- RC circular fiber sections
    steel_sections.csv     -- steel I-section / channel sections
    Sections.csv           -- composite steel-girder + concrete deck

    Optional
    --------
    slabs.csv              -- linear-elastic shell slab panels (requires slabs.py)
    """
    loader    = DataLoader(config)
    materials = MaterialLibrary()
    sections  = SectionLibrary(materials)
    builder   = ModelBuilder(config, loader, sections)

    # ---- 1. Domain ----
    builder.initialise()
    builder.build_nodes()
    builder.apply_equal_dofs()

    # ---- 2. Materials ----
    # FIX: method now exists in MaterialLibrary; falls back to build_defaults()
    # when materials.csv is absent.
    materials.build_from_csv(loader, filename=config.materials_csv)

    # ---- 3. Sections (all optional; absent files are skipped) ----
    # FIX: all four methods now exist in SectionLibrary.
    sections.build_rc_rect_from_csv(loader,  filename=config.rc_rect_csv)
    sections.build_rc_circ_from_csv(loader,  filename=config.rc_circ_csv)
    sections.build_steel_from_csv(loader,    filename=config.steel_sections_csv)
    sections.build_composite_from_csv(loader, filename=config.sections_csv)

    # ---- 3b. Pre-flight: every section tag used by elements must exist ----
    _check_section_tags(loader, sections, config)

    # ---- 4. Geometric transformations ----
    builder.build_transformations()

    # ---- 5. Beam integrations (covers all defined section tags) ----
    builder.build_beam_integrations(list(sections.tags.values()))

    # ---- 6. Frame elements ----
    builder.build_beam_elements()
    builder.build_column_elements()

    # ---- 7. Loads ----
    LoadBuilder(config, loader).apply_loads()

    # ---- 8. Boundary conditions ----
    builder.apply_boundary_conditions()

    # ---- 9. Slabs (optional — requires slabs.py in the package) ----
    # FIX: slabs.py does not exist in this package. The import is guarded so
    # the rest of the framework is not broken by the missing module. Add
    # slabs.py to the package to enable this step.
    try:
        from .slabs import SlabGenerator
        SlabGenerator(config, loader, builder, sections).build_all()
    except ImportError:
        if config.in_data(config.slabs_csv).exists():
            import warnings
            warnings.warn(
                'slabs.csv is present but slabs.py is not in the package. '
                'Slab panels will NOT be built. Add fe_model/slabs.py to enable this step.',
                stacklevel=2,
            )

    print('=' * 60)
    print('Model assembled successfully.')
    print('=' * 60)

    # ---- 10. Analysis ----
    driver = AnalysisDriver(config)
    if run_gravity:
        driver.run_gravity()
    if run_modal:
        driver.run_modal(n_modes)

    return {
        'loader': loader, 'materials': materials, 'sections': sections,
        'builder': builder, 'driver': driver,
    }


# ------------------------------------------------------------------ #
# Pre-flight helper
# ------------------------------------------------------------------ #
def _check_section_tags(
    loader: DataLoader,
    sections: SectionLibrary,
    config: Config,
) -> None:
    """Raise a descriptive error if any section tag used by elements is
    not yet defined in the section library."""
    defined = set(sections.tags.values())
    needed: dict = {}   # tag -> [source files]

    for csv_name in (config.beams_csv, config.columns_csv):
        path = config.in_data(csv_name)
        if not path.exists():
            continue
        df = loader.read(csv_name, schema_key=None, optional=True)
        if df is None:
            continue
        sec_col = next(
            (c for c in df.columns
             if c.strip().lower() in ('sectag', 'section', 'section_tag', 'sec')),
            None,
        )
        if sec_col is None:
            continue
        for tag in df[sec_col].dropna().astype(int).unique():
            if tag not in defined:
                needed.setdefault(tag, []).append(csv_name)

    if not needed:
        return

    lines = [
        '',
        '=' * 65,
        'MISSING SECTION DEFINITIONS',
        'The following section tags are used in your element CSVs but',
        'have no matching row in any section CSV:',
        '',
    ]
    for tag, sources in sorted(needed.items()):
        lines.append(f'  tag {tag:4d}  ← referenced in: {", ".join(sources)}')
    lines += [
        '',
        'Fix: add the missing tag(s) to the appropriate section CSV:',
        '  RC rectangular beam/column → rc_rect_sections.csv',
        '  RC circular column         → rc_circ_sections.csv',
        '  Steel I-section / channel  → steel_sections.csv',
        '  Composite steel+slab       → Sections.csv',
        '=' * 65,
    ]
    raise ValueError('\n'.join(lines))
