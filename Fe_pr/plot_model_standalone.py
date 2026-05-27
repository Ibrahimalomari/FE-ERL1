"""
plot_model_standalone.py
========================
Standalone script to visualize the 3D model without running the analysis.

Useful when you want to:
  * Check node/element connectivity before committing to a long analysis
  * Generate plots with different camera angles
  * Verify boundary conditions are correctly applied

Run with::

    python plot_model_standalone.py

The plot will be saved to output/model_3d.png (or shown interactively if
you set save_plots=False in the Config).
"""

from fe_model import Config, DataLoader, MaterialLibrary, SectionLibrary, ModelBuilder
from fe_model.visualize import plot_model_3d


def main():
    # ---- Configuration ----
    cfg = Config(
        data_dir   = './data',
        output_dir = './output',
        bc_csv     = 'boundary_conditions.csv',
    )

    # ---- Build geometry only (no loads, no analysis) ----
    print('Loading model geometry...')
    loader    = DataLoader(cfg)
    materials = MaterialLibrary()
    sections  = SectionLibrary(materials)
    builder   = ModelBuilder(cfg, loader, sections)

    builder.initialise()
    builder.build_nodes()
    builder.apply_equal_dofs()

    # Build materials and sections (so section tags exist for plotting)
    materials.build_defaults()
    sections.build_defaults()

    # Auto-build RC sections from catalogues
    col_cat = loader.read('column_sections.csv', optional=True)
    if col_cat is not None:
        sections.build_from_catalogue_csv(col_cat, kind='column')

    beam_cat = loader.read('beam_sections.csv', optional=True)
    if beam_cat is not None:
        sections.build_from_catalogue_csv(beam_cat, kind='beam')

    builder.build_default_transformations()
    builder.build_beam_integrations(list(sections.tags.values()))
    builder.build_beam_elements()
    builder.build_column_elements()
    builder.apply_boundary_conditions()

    print()
    print('=' * 60)
    print('Model geometry loaded. Generating plot...')
    print('=' * 60)

    # ---- Plot ----
    plot_model_3d(
        builder, cfg,
        save_path=cfg.in_output('model_3d.png'),
        show_nodes=True,
        show_node_tags=False,       # Set True to see node numbers (cluttered for large models)
        show_elements=True,
        show_bcs=True,
        show_axes=True,
        element_scale=1.5,
        azim=-60,                    # Camera azimuth angle
        elev=30,                     # Camera elevation angle
    )

    print()
    print('[OK]   Plot saved to output/model_3d.png')
    print('       Open it to verify:')
    print('         * Node positions')
    print('         * Beam (blue) and column (red) connectivity')
    print('         * Boundary conditions (green triangles = fixed, orange circles = pinned)')
    print('         * Coordinate system (red=X, green=Y, blue=Z)')


if __name__ == '__main__':
    main()
