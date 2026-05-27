"""
builder.py
==========
Model assembly: nodes, geometric transformations, beam integrations,
beam/column elements, multi-point constraints, and boundary conditions.

Bug fixes
---------
1. ``_build_frame_elements`` — the ``nonlinearBeamColumn`` element call was
   passing ``self.config.n_int_points`` as the positional ``numIntgrPts``
   argument *and* ``'-integration', integ_tag`` simultaneously. When the named
   integration flag is present OpenSees obtains the point count from the
   Lobatto object, making the positional argument redundant and potentially
   conflicting. Removed the redundant positional ``n_int_points`` argument.

2. ``apply_equal_dofs`` — the filter ``int(r[f'D{i}']) > 0`` silently
   discarded any DOF coded as 0.  DOF indices in OpenSees are 1-6; a 0 in the
   CSV almost certainly indicates a data-entry error.  The code now raises a
   clear ValueError for any out-of-range value instead of silently ignoring it.
"""

from typing import Dict, Iterable, Optional, Tuple
import pandas as pd

from openseespy.opensees import (
    wipe, model,
    node, fix, equalDOF,
    geomTransf, beamIntegration, element,
)

from .config   import Config
from .data_io  import DataLoader
from .sections import SectionLibrary


class ModelBuilder:
    """Builds the OpenSees model from CSV inputs."""

    def __init__(self, config: Config, loader: DataLoader,
                 sections: SectionLibrary):
        self.config   = config
        self.loader   = loader
        self.sections = sections

        self.node_coords:  Dict[int, Tuple[float, float, float]] = {}
        self.element_tags: Dict[int, str]  = {}
        self.transf_tags:  Dict[str, int]  = {}   # name  -> tag
        self._integ_tags:  Dict[int, int]  = {}   # sec_tag -> integ_tag

        self._next_node_tag    = 1
        self._next_element_tag = 1

    # ================================================================== #
    # Domain initialisation
    # ================================================================== #
    def initialise(self) -> None:
        wipe()
        model('basic', '-ndm', self.config.ndm, '-ndf', self.config.ndf)

    # ================================================================== #
    # Nodes
    # ================================================================== #
    def build_nodes(self) -> None:
        df = self.loader.read(
            self.config.nodes_csv,
            required_cols=['nodetag', 'x', 'y', 'z'],
            schema_key='nodes',
        )
        scale    = self.loader.length_scale(self.config.nodes_csv)
        has_mass = 'mass' in df.columns and df['mass'].notna().any()

        for _, row in df.iterrows():
            tag = int(row['nodetag'])
            x   = float(row['x']) * scale
            y   = float(row['y']) * scale
            z   = float(row['z']) * scale
            if has_mass:
                m = float(row['mass'])
                node(tag, x, y, z, '-mass', m, m, m, 0.0, 0.0, 0.0)
            else:
                node(tag, x, y, z)
            self.node_coords[tag] = (x, y, z)
            if tag >= self._next_node_tag:
                self._next_node_tag = tag + 1

        zs = sorted({round(c[2], 4) for c in self.node_coords.values()})
        self.floor_elevations = zs
        msg = f'[OK]   Built {len(df)} nodes'
        if len(zs) > 1:
            msg += f'  ({len(zs)} distinct elevations)'
        print(msg)

    def add_node(
        self, x: float, y: float, z: float,
        mass_xyz: Optional[Tuple[float, float, float]] = None,
        tag: Optional[int] = None,
    ) -> int:
        """Create a new node programmatically (used by the slab generator)."""
        if tag is None:
            tag = self._next_node_tag
            self._next_node_tag += 1
        else:
            self._next_node_tag = max(self._next_node_tag, tag + 1)
        if mass_xyz is not None:
            mx, my, mz = mass_xyz
            node(tag, x, y, z, '-mass', mx, my, mz, 0.0, 0.0, 0.0)
        else:
            node(tag, x, y, z)
        self.node_coords[tag] = (x, y, z)
        return tag

    # ================================================================== #
    # Constraints
    # ================================================================== #
    def apply_equal_dofs(self) -> None:
        df = self.loader.read(
            self.config.equal_dof_csv,
            required_cols=['node1', 'node2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
            schema_key='equal_dof',
        )
        n = 0
        for row_idx, r in df.iterrows():
            dofs = []
            for i in range(1, 7):
                val = int(r[f'D{i}'])
                if val == 0:
                    continue           # 0 means "not constrained" — skip
                if val < 1 or val > 6:
                    # FIX: raise instead of silently swallowing bad data
                    raise ValueError(
                        f'EqualDOF row {row_idx}: D{i}={val} is out of range '
                        f'[1-6]. Use 0 to leave a DOF unconstrained.'
                    )
                dofs.append(val)
            if dofs:
                equalDOF(int(r['node1']), int(r['node2']), *dofs)
                n += 1
        print(f'[OK]   Applied {n} equalDOF constraints')

    def apply_boundary_conditions(self) -> None:
        """Apply fixities from ``boundary_conditions.csv``.

        Schema: node, ux, uy, uz, rx, ry, rz   (1=fixed, 0=free)
        """
        df = self.loader.read(
            self.config.bc_csv,
            required_cols=['node', 'ux', 'uy', 'uz', 'rx', 'ry', 'rz'],
            schema_key='bc',
            optional=False,
        )
        for _, r in df.iterrows():
            fix(int(r['node']),
                int(r['ux']), int(r['uy']), int(r['uz']),
                int(r['rx']), int(r['ry']), int(r['rz']))
        print(f'[OK]   Applied {len(df)} fixities from {self.config.bc_csv}')

    # ================================================================== #
    # Geometric transformations
    # ================================================================== #
    def build_transformations(self) -> None:
        """Read transformations.csv and register each geomTransf.

        Schema: tag, name, type, vecxz_x, vecxz_y, vecxz_z
        """
        df = self.loader.read(
            self.config.transformations_csv,
            required_cols=['tag', 'name', 'type', 'vecxz_x', 'vecxz_y', 'vecxz_z'],
            schema_key='transformations',
            optional=False,
        )
        for _, r in df.iterrows():
            tag  = int(r['tag'])
            name = str(r['name']).strip()
            kind = str(r['type']).strip()
            vx   = float(r['vecxz_x'])
            vy   = float(r['vecxz_y'])
            vz   = float(r['vecxz_z'])
            geomTransf(kind, tag, vx, vy, vz)
            self.transf_tags[name] = tag
        print(f'[OK]   Built {len(df)} geometric transformations')

    # ================================================================== #
    # Beam integrations
    # ================================================================== #
    def build_beam_integrations(self, section_tags: Iterable[int]) -> None:
        """Register one Lobatto integration rule per section tag.

        Convention: integ_tag == sec_tag. The mapping is stored in
        ``self._integ_tags`` and looked up by ``_build_frame_elements``.
        Must be called AFTER all sections are defined and BEFORE elements.
        """
        n_pts = self.config.n_int_points
        seen  = set()
        for tag in section_tags:
            if tag in seen:
                continue
            beamIntegration('Lobatto', tag, tag, n_pts)
            self._integ_tags[tag] = tag
            seen.add(tag)
        print(f'[OK]   Built {len(seen)} beam integration rules')

    # ================================================================== #
    # Frame elements
    # ================================================================== #
    def _build_frame_elements(
        self, df: pd.DataFrame, kind: str,
        default_transf_name: str,
    ) -> None:
        """Create nonlinearBeamColumn elements from a DataFrame.

        Transformation resolution (first match wins):
          1. ``transf_tag``  column (integer) in the CSV row
          2. ``transf_name`` column (string) looked up in self.transf_tags
          3. default_transf_name looked up in self.transf_tags
        """
        # Resolve column names case-insensitively so the code works
        # regardless of whether DataLoader already normalised them.
        col_lower = {c.lower(): c for c in df.columns}
        id_col     = col_lower.get('id',     col_lower.get('eletag', 'ID'))
        n1_col     = col_lower.get('node1',  col_lower.get('inode',  'node1'))
        n2_col     = col_lower.get('node2',  col_lower.get('jnode',  'node2'))
        sec_col    = col_lower.get('sectag', col_lower.get('section', 'Sectag'))

        for _, row in df.iterrows():
            tag    = int(row[id_col])
            n1, n2 = int(row[n1_col]), int(row[n2_col])
            sec    = int(row[sec_col])

            # --- transformation tag ---
            if 'transf_tag' in row.index and not pd.isna(row.get('transf_tag')):
                t_tag = int(row['transf_tag'])
            elif 'transf_name' in row.index and not pd.isna(row.get('transf_name')):
                name = str(row['transf_name']).strip()
                if name not in self.transf_tags:
                    raise KeyError(
                        f'Element {tag}: transf_name "{name}" not found. '
                        f'Available: {list(self.transf_tags)}'
                    )
                t_tag = self.transf_tags[name]
            else:
                t_tag = self.transf_tags.get(default_transf_name)
                if t_tag is None:
                    raise KeyError(
                        f'Default transformation "{default_transf_name}" not found. '
                        f'Available: {list(self.transf_tags)}'
                    )

            # --- integration tag ---
            integ_tag = self._integ_tags.get(sec)
            if integ_tag is None:
                raise KeyError(
                    f'No beam integration rule for section tag {sec} '
                    f'(element {tag}). Check that build_beam_integrations() '
                    f'was called with all section tags. '
                    f'Registered: {sorted(self._integ_tags)}'
                )

            # --- mass ---
            m_col = next(
                (c for c in row.index if c.lower().startswith('mass')), None,
            )
            mass = float(row[m_col]) if m_col and not pd.isna(row[m_col]) else 0.0

            # forceBeamColumn takes the beamIntegration tag positionally:
            #   element('forceBeamColumn', eleTag, iNode, jNode,
            #            transfTag, integTag, '-iter', maxIter, tol, '-mass', m)
            element(
                'forceBeamColumn',
                tag, n1, n2,
                t_tag, integ_tag,
                '-iter', self.config.max_iter, self.config.tolerance,
                '-mass', mass,
            )
            self.element_tags[tag] = kind
            if tag >= self._next_element_tag:
                self._next_element_tag = tag + 1

    def build_beam_elements(self) -> None:
        df = self.loader.read(
            self.config.beams_csv,
            required_cols=['ID', 'node1', 'node2', 'Sectag'],
            schema_key='beams',
        )
        self._build_frame_elements(df, kind='beam',
                                   default_transf_name='beam')
        print(f'[OK]   Built {len(df)} beam elements')

    def build_column_elements(self) -> None:
        df = self.loader.read(
            self.config.columns_csv,
            required_cols=['ID', 'node1', 'node2', 'Sectag'],
            schema_key='columns',
        )
        self._build_frame_elements(df, kind='col',
                                   default_transf_name='column')
        print(f'[OK]   Built {len(df)} column elements')

    # ================================================================== #
    # Helpers
    # ================================================================== #
    def reserve_element_tag(self) -> int:
        tag = self._next_element_tag
        self._next_element_tag += 1
        return tag