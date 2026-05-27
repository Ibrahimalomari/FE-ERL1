"""
run_model.py
============
Top-level driver. Every piece of model data comes from CSV files in
``data_dir``. Nothing is hardcoded in this script or in the framework.

Required CSVs (must exist in data_dir)
---------------------------------------
nodes.csv                  -- node tags, coordinates, optional mass
EqualDOF.csv               -- multi-point constraints
boundary_conditions.csv    -- support fixities (1=fixed, 0=free)
transformations.csv        -- geometric transformation definitions
materials.csv              -- all uniaxial material definitions
Beam Elements.csv          -- beam element connectivity + section tag
Column Elements.csv        -- column element connectivity + section tag
loads.csv                  -- distributed element loads

Optional CSVs (used when present, silently skipped otherwise)
--------------------------------------------------------------
rc_rect_sections.csv       -- RC rectangular fiber sections
rc_circ_sections.csv       -- RC circular fiber sections
steel_sections.csv         -- steel I / channel fiber sections
Sections.csv               -- composite steel-girder + concrete deck
slabs.csv                  -- linear-elastic shell slab panels (requires slabs.py)

Run with:
    python run_model.py
"""

from fe_model import Config, build_full_model


def main():
    cfg = Config(
        data_dir           = './data',
        output_dir         = './output',
        gravity_load_steps = 100,

        # ---- CSV filenames (override defaults if your files differ) ----
        materials_csv        = 'materials.csv',
        transformations_csv  = 'transformations.csv',
        bc_csv               = 'boundary_conditions.csv',
        nodes_csv            = 'nodes.csv',
        equal_dof_csv        = 'EqualDOF.csv',
        beams_csv            = 'Beam Elements.csv',
        columns_csv          = 'Column Elements.csv',
        loads_csv            = 'loads.csv',
        rc_rect_csv          = 'rc_rect_sections.csv',
        rc_circ_csv          = 'rc_circ_sections.csv',
        steel_sections_csv   = 'steel_sections.csv',
        sections_csv         = 'Sections.csv',
        slabs_csv            = 'slabs.csv',

        # ---- Visualization ----
        plot_model              = True,
        save_plots              = True,
        show_node_tags          = False,
        element_linewidth_scale = 1.5,
        plot_azim               = -60,
        plot_elev               = 30,
    )

    results = build_full_model(
        cfg,
        run_gravity = False,
        run_modal   = True,
        n_modes     = 10,
    )

    # Save support-node displacements after gravity analysis!
    support_nodes = list([1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
     21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
     49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75])
    results['driver'].save_node_displacements(
        support_nodes, 'support_displacements.csv',
    )

    print()
    print('=' * 60)
    print('Analysis complete. Results written to output/')
    print('  support_displacements.csv')
    if cfg.save_plots:
        print('  model_3d.png')
    print('=' * 60)


if __name__ == '__main__':
    main()
