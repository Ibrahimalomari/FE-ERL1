"""
sections.py
===========
Cross-section library.  Each section type is a method that takes geometric
parameters and delegates to OpenSees primitives.

Bug fixes
---------
1. ``build_defaults`` — C1 column was passing ``Gc * J1`` (rectangular beam J)
   as its torsional stiffness instead of ``Gc * Jc`` (circular column polar
   moment).  Fixed to use ``Jc``.

2. ``add_composite_from_dataframe`` — tag calculation used the pandas row
   *index label* (which can be non-sequential after filtering) as an offset
   into section tags.  Replaced with an explicit counter so tags are always
   ``start_tag, start_tag+1, …`` regardless of the frame's index.  The
   composite name key uses the same counter.

3. ``build_from_catalogue_csv`` — when ``kind='column'`` the method called
   ``add_rc_rect_beam(name=f'column_{tag}', …)`` for what might be a circular
   column.  The name is now ``f'rc_col_{tag}'`` so section lookups are
   unambiguous, and a note explains that circular columns must be added via
   ``build_rc_circ_from_csv``.

4. Added the three CSV loader methods that ``pipeline.py`` calls:
   ``build_rc_rect_from_csv``, ``build_rc_circ_from_csv``,
   ``build_steel_from_csv``, and ``build_composite_from_csv``.
   Each is optional (skips gracefully when the file is absent).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from openseespy.opensees import section, patch, layer
from .materials import MaterialLibrary


# (n_bars, area_each, y_position_from_centroid)
RebarLayer = Tuple[int, float, float]


class SectionLibrary:
    """Named section registry."""

    def __init__(self, materials: MaterialLibrary, start_tag: int = 1):
        self.mat = materials
        self.tags: Dict[str, int] = {}
        self._next_tag = start_tag

    # -------------- internals --------------
    def _resolve_tag(self, requested: Optional[int]) -> int:
        if requested is None:
            tag = self._next_tag
            self._next_tag += 1
        else:
            tag = requested
            if tag >= self._next_tag:
                self._next_tag = tag + 1
        return tag

    def register(self, name: str, tag: int) -> int:
        self.tags[name] = tag
        if tag >= self._next_tag:
            self._next_tag = tag + 1
        return tag

    def get(self, name: str) -> int:
        if name not in self.tags:
            raise KeyError(
                f'Section "{name}" not defined. Available: {sorted(self.tags)}'
            )
        return self.tags[name]

    # ================================================================
    # Core section builders
    # ================================================================

    def add_rc_rect_beam(
        self, name: str,
        b: float, d: float, cover: float, GJ: float,
        rebar_layers: List[RebarLayer],
        confined_mat: str = 'confined',
        unconfined_mat: str = 'unconfined',
        steel_mat: str = 'steel',
        nfy: int = 20, nfz: int = 30,
        tag: Optional[int] = None,
    ) -> int:
        """RC rectangular fiber section."""
        tag = self._resolve_tag(tag)
        section('Fiber', tag, '-GJ', GJ)

        patch('rect', self.mat.get(confined_mat), nfy, nfz,
              -b/2 + cover, -d/2 + cover,
               b/2 - cover,  d/2 - cover)
        unc = self.mat.get(unconfined_mat)
        patch('rect', unc, nfy, 1, -b/2,  d/2 - cover,  b/2,          d/2)
        patch('rect', unc, nfy, 1, -b/2, -d/2,          b/2,         -d/2 + cover)
        patch('rect', unc, 1, nfz, -b/2, -d/2 + cover, -b/2 + cover,  d/2 - cover)
        patch('rect', unc, 1, nfz,  b/2 - cover, -d/2 + cover, b/2,   d/2 - cover)

        steel = self.mat.get(steel_mat)
        for n_bars, area, y_pos in rebar_layers:
            x_start = -b/2 + cover
            x_end   = -x_start
            layer('straight', steel, n_bars, area, x_start, y_pos, x_end, y_pos)

        self.tags[name] = tag
        return tag

    def add_rc_circular_col(
        self, name: str,
        radius: float, cover: float, GJ: float,
        n_bars: int, bar_area: float,
        n_circ: int = 10, n_rad_core: int = 5, n_rad_cover: int = 1,
        confined_mat: str = 'confined',
        unconfined_mat: str = 'unconfined',
        steel_mat: str = 'steel',
        tag: Optional[int] = None,
    ) -> int:
        tag = self._resolve_tag(tag)
        section('Fiber', tag, '-GJ', GJ)
        patch('circ', self.mat.get(confined_mat),   n_circ, n_rad_core,
              0, 0, 0, radius - cover, 0, 360)
        patch('circ', self.mat.get(unconfined_mat), n_circ, n_rad_cover,
              0, 0, radius - cover, radius, 0, 360)
        layer('circ', self.mat.get(steel_mat), n_bars, bar_area, 0, 0,
              radius - cover)
        self.tags[name] = tag
        return tag

    def add_steel_isection(
        self, name: str,
        d: float, bf: float, tf: float, tw: float, GJ: float,
        steel_mat: str = 'steel',
        n_flange: int = 10, n_web: int = 20,
        tag: Optional[int] = None,
    ) -> int:
        tag = self._resolve_tag(tag)
        section('Fiber', tag, '-GJ', GJ)
        steel = self.mat.get(steel_mat)
        patch('rect', steel, n_flange, 1, -bf/2,  d/2 - tf,  bf/2,  d/2)
        patch('rect', steel, n_flange, 1, -bf/2, -d/2,        bf/2, -d/2 + tf)
        patch('rect', steel, 1, n_web,  -tw/2, -d/2 + tf, tw/2,  d/2 - tf)
        self.tags[name] = tag
        return tag

    def add_steel_channel(
        self, name: str,
        d: float, bf: float, tf: float, tw: float, GJ: float,
        steel_mat: str = 'steel',
        tag: Optional[int] = None,
    ) -> int:
        tag = self._resolve_tag(tag)
        section('Fiber', tag, '-GJ', GJ)
        steel = self.mat.get(steel_mat)
        patch('rect', steel, 10, 1, tw/2,  d/2 - tf, bf - tw/2,  d/2)
        patch('rect', steel, 10, 1, tw/2, -d/2,      bf - tw/2, -d/2 + tf)
        patch('rect', steel, 1, 20, -tw/2, -d/2, tw/2, d/2)
        self.tags[name] = tag
        return tag

    # ================================================================
    # Composite steel + slab
    # ================================================================

    def add_composite_from_dataframe(
        self,
        sections_df: pd.DataFrame,
        slab_depth_above_steel: float = 7.5,
        GJ: float = 1.0,
        steel_mat: str = 'steel',
        slab_mat: str = 'unconfined',
        start_tag: int = 1,
    ) -> Dict[int, int]:
        """Build composite (steel girder + concrete deck) sections.

        FIX: previously used the pandas row index label as an offset, giving
        wrong tags when the frame was filtered or non-zero-indexed. Now uses
        an explicit counter (0, 1, 2, …) so tags are always
        start_tag, start_tag+1, … regardless of the DataFrame's index.

        Returns {1-based sequential counter: section_tag}.
        """
        gen = {}
        for counter, (_, row) in enumerate(sections_df.iterrows()):  # FIX
            tag = self._resolve_tag(start_tag + counter)             # FIX
            section('Fiber', tag, '-GJ', GJ * row['J'])
            steel = self.mat.get(steel_mat)
            slab  = self.mat.get(slab_mat)
            patch('rect', steel, 10, 1,  row['yi1'], row['zi1'], row['yj1'], row['zj1'])
            patch('rect', steel, 10, 1,  row['yi2'], row['zi2'], row['yj2'], row['zj2'])
            patch('rect', steel,  1, 20, row['yi3'], row['zi3'], row['yj3'], row['zj3'])
            patch('rect', slab,   5,  5,
                  -row['be']/2, row['d']/2,
                   row['be']/2, row['d']/2 + slab_depth_above_steel)
            name = f'composite_{counter + 1}'   # FIX: use counter, not row.name
            self.tags[name] = tag
            gen[counter + 1] = tag
        return gen

    # ================================================================
    # Default library (mirrors original script)
    # ================================================================

    def build_defaults(self) -> None:
        """Build the fixed sections CB1, CB2, C1, C12x207 from the original
        script. Geometric parameters are lifted directly from that code."""
        from .config import Units as U

        Gc = 3506 / (2 * 1.2)   # effective shear modulus for concrete torsion

        # ---- CB1 : RC beam (5-#8 + 4-#9 top, 2-#6 mid, 9-#11 bottom) ----
        b1, d1, cc1 = 38.0, 45.0, 2.25
        J1 = b1 * d1 * (b1*b1 + d1*d1) / 12.0
        self.add_rc_rect_beam(
            'CB1', b=b1, d=d1, cover=cc1, GJ=Gc * J1,
            rebar_layers=[
                (5, 0.79,  d1/2 - cc1),
                (4, 1.00,  d1/2 - cc1),
                (2, 0.44,  0.0),
                (9, 1.56, -(d1/2 - cc1)),
            ],
            tag=100,
        )
        # ---- CB2 : same as CB1 without the 4-#9 row ----
        self.add_rc_rect_beam(
            'CB2', b=b1, d=d1, cover=cc1, GJ=Gc * J1,
            rebar_layers=[
                (5, 0.79,  d1/2 - cc1),
                (2, 0.44,  0.0),
                (9, 1.56, -(d1/2 - cc1)),
            ],
            tag=101,
        )
        # ---- C1 : round column (r=18", 8-#11 bars) ----
        r  = 36.0 / 2.0
        Jc = 3.14159265358979 / 2.0 * r ** 4   # polar moment of inertia
        self.add_rc_circular_col(
            'C1', radius=r, cover=2.25,
            GJ=Gc * Jc,          # FIX: was Gc * J1 (beam J) — must be Gc * Jc
            n_bars=8, bar_area=1.27,
            tag=102,
        )
        # ---- C12x207 channel ----
        Es = 29000.0 * U.ksi
        Gs = Es / (1.0 + 0.2)
        self.add_steel_channel(
            'C12x207', d=12.0, bf=2.94, tf=0.501, tw=0.282,
            GJ=Gs * 0.37, tag=4,
        )

    # ================================================================
    # CSV loaders — called by pipeline.py
    # ================================================================

    def build_rc_rect_from_csv(
        self, loader, filename: str = 'rc_rect_sections.csv',
    ) -> int:
        """Load RC rectangular beam/column sections from CSV.

        Schema (all lengths in inches unless column header has a unit tag):
        Sectag, name, b, d, cover, GJ,
        rebar_<n>_nbars, rebar_<n>_area, rebar_<n>_ypos  (repeat for each layer)
        confined_mat, unconfined_mat, steel_mat           (optional, default names used if absent)

        Returns number of sections built.
        """
        df = loader.read(filename, optional=True)
        if df is None:
            return 0

        df.columns = [c.strip() for c in df.columns]
        n = 0
        for _, row in df.iterrows():
            tag  = int(row['Sectag'])
            name = str(row['name']).strip()
            b    = float(row['b'])
            d    = float(row['d'])
            cover = float(row['cover'])
            GJ   = float(row['GJ'])

            # Collect rebar layers. Support two naming conventions:
            #   rebar_{i}_nbars / rebar_{i}_area / rebar_{i}_ypos
            #   layer{i}_n     / layer{i}_area  / layer{i}_y
            # y-position may be the string "top" or "bottom" meaning
            # +(d/2-cover) and -(d/2-cover) respectively.
            rebar_layers: List[RebarLayer] = []
            i = 1
            while True:
                if f'rebar_{i}_nbars' in row.index:
                    nb_key, ar_key, yp_key = (
                        f'rebar_{i}_nbars', f'rebar_{i}_area', f'rebar_{i}_ypos')
                elif f'layer{i}_n' in row.index:
                    nb_key, ar_key, yp_key = (
                        f'layer{i}_n', f'layer{i}_area', f'layer{i}_y')
                else:
                    break
                if pd.isna(row[nb_key]):
                    break
                nb = int(row[nb_key])
                ar = float(row[ar_key])
                yp_raw = row[yp_key]
                if isinstance(yp_raw, str):
                    yp_raw = yp_raw.strip().lower()
                    if yp_raw == 'top':
                        yp = d / 2 - cover
                    elif yp_raw == 'bottom':
                        yp = -(d / 2 - cover)
                    else:
                        yp = float(yp_raw)
                else:
                    yp = float(yp_raw)
                rebar_layers.append((nb, ar, yp))
                i += 1

            c_mat  = str(row.get('confined_mat',   'confined')).strip()
            uc_mat = str(row.get('unconfined_mat', 'unconfined')).strip()
            s_mat  = str(row.get('steel_mat',      'steel')).strip()

            self.add_rc_rect_beam(
                name, b=b, d=d, cover=cover, GJ=GJ,
                rebar_layers=rebar_layers,
                confined_mat=c_mat, unconfined_mat=uc_mat, steel_mat=s_mat,
                tag=tag,
            )
            n += 1

        print(f'[OK]   Built {n} RC rectangular sections from {filename}')
        return n

    def build_rc_circ_from_csv(
        self, loader, filename: str = 'rc_circ_sections.csv',
    ) -> int:
        """Load RC circular column sections from CSV.

        Schema: Sectag, name, radius, cover, GJ, n_bars, bar_area,
                [n_circ, n_rad_core, n_rad_cover, confined_mat, unconfined_mat, steel_mat]

        Returns number of sections built.
        """
        df = loader.read(filename, optional=True)
        if df is None:
            return 0

        df.columns = [c.strip() for c in df.columns]
        n = 0
        for _, row in df.iterrows():
            tag      = int(row['Sectag'])
            name     = str(row['name']).strip()
            radius   = float(row['radius'])
            cover    = float(row['cover'])
            GJ       = float(row['GJ'])
            n_bars   = int(row['n_bars'])
            bar_area = float(row['bar_area'])

            def _opt_int(col, default):
                return int(row[col]) if col in row.index and row[col] == row[col] else default
            def _opt_str(col, default):
                return str(row[col]).strip() if col in row.index and row[col] == row[col] else default

            self.add_rc_circular_col(
                name, radius=radius, cover=cover, GJ=GJ,
                n_bars=n_bars, bar_area=bar_area,
                n_circ      = _opt_int('n_circ',       10),
                n_rad_core  = _opt_int('n_rad_core',    5),
                n_rad_cover = _opt_int('n_rad_cover',   1),
                confined_mat   = _opt_str('confined_mat',   'confined'),
                unconfined_mat = _opt_str('unconfined_mat', 'unconfined'),
                steel_mat      = _opt_str('steel_mat',      'steel'),
                tag=tag,
            )
            n += 1

        print(f'[OK]   Built {n} RC circular sections from {filename}')
        return n

    def build_steel_from_csv(
        self, loader, filename: str = 'steel_sections.csv',
    ) -> int:
        """Load steel fiber sections (I-section or channel) from CSV.

        Schema: Sectag, name, shape, d, bf, tf, tw, GJ, [steel_mat]
                shape: 'isection' | 'channel'

        Returns number of sections built.
        """
        df = loader.read(filename, optional=True)
        if df is None:
            return 0

        df.columns = [c.strip() for c in df.columns]
        n = 0
        for _, row in df.iterrows():
            tag   = int(row['Sectag'])
            name  = str(row['name']).strip()
            shape = str(row['shape']).strip().lower()
            d     = float(row['d'])
            bf    = float(row['bf'])
            tf    = float(row['tf'])
            tw    = float(row['tw'])
            GJ    = float(row['GJ'])
            s_mat = str(row.get('steel_mat', 'steel')).strip()

            if shape == 'isection':
                self.add_steel_isection(name, d=d, bf=bf, tf=tf, tw=tw,
                                        GJ=GJ, steel_mat=s_mat, tag=tag)
            elif shape == 'channel':
                self.add_steel_channel(name, d=d, bf=bf, tf=tf, tw=tw,
                                       GJ=GJ, steel_mat=s_mat, tag=tag)
            else:
                raise ValueError(
                    f'Unknown steel shape "{shape}" for section "{name}". '
                    f'Use "isection" or "channel".'
                )
            n += 1

        print(f'[OK]   Built {n} steel sections from {filename}')
        return n

    def build_composite_from_csv(
        self, loader, filename: str = 'Sections.csv',
        slab_depth_above_steel: float = 7.5,
        steel_mat: str = 'steel',
        slab_mat: str = 'unconfined',
        start_tag: int = 1,
    ) -> int:
        """Load composite (steel girder + concrete deck) sections from CSV.

        Delegates to ``add_composite_from_dataframe`` after reading the file.
        The CSV schema matches the original Sections.csv:
        J, yi1, zi1, yj1, zj1, yi2, zi2, yj2, zj2, yi3, zi3, yj3, zj3, be, d

        Returns number of sections built.
        """
        df = loader.read(filename, optional=True)
        if df is None:
            return 0

        gen = self.add_composite_from_dataframe(
            df,
            slab_depth_above_steel=slab_depth_above_steel,
            steel_mat=steel_mat,
            slab_mat=slab_mat,
            start_tag=start_tag,
        )
        print(f'[OK]   Built {len(gen)} composite sections from {filename}')
        return len(gen)

    # ================================================================
    # Catalogue builder (unchanged — only naming fix for columns)
    # ================================================================

    def build_from_catalogue_csv(
        self,
        catalogue_csv: pd.DataFrame,
        kind: str,
        cover: float = 2.25,
        confined_mat: str = 'confined',
        unconfined_mat: str = 'unconfined',
        steel_mat: str = 'steel',
    ) -> int:
        """Build RC rectangular sections from a catalogue DataFrame.

        Note: this method always creates rectangular fiber sections. For
        circular columns use ``build_rc_circ_from_csv`` instead.

        FIX: column sections now registered as 'rc_col_{tag}' instead of
        'column_{tag}' to avoid name collisions with the section type label.
        """
        from .config import Units as U

        if kind not in ('beam', 'column'):
            raise ValueError(f'kind must be "beam" or "column", got {kind!r}')

        n = 0
        for _, row in catalogue_csv.iterrows():
            tag = int(row['Sectag'])

            if kind == 'beam':
                b       = float(row['b_in'])
                d       = float(row['d_in'])
                n_bot   = int(row['n_bot'])
                bot_dia = float(row['bot_dia_in'])
                n_top   = int(row['n_top'])
                top_dia = float(row['top_dia_in'])
                bot_area = 3.14159265 / 4 * bot_dia ** 2
                top_area = 3.14159265 / 4 * top_dia ** 2
                rebar_layers = [
                    (n_top, top_area,  d / 2 - cover),
                    (n_bot, bot_area, -(d / 2 - cover)),
                ]
                GJ = 0.3 * b * d ** 3
                self.add_rc_rect_beam(
                    name=f'beam_{tag}',
                    b=b, d=d, cover=cover, GJ=GJ,
                    rebar_layers=rebar_layers,
                    confined_mat=confined_mat,
                    unconfined_mat=unconfined_mat,
                    steel_mat=steel_mat,
                    tag=tag,
                )
                n += 1

            elif kind == 'column':
                b = float(row['dx (m)']) * U.m
                d = float(row['dy (m)']) * U.m
                n_long      = int(row['long reinf #'])
                bar_dia_mm  = float(row['long reinf size (mm)'])
                bar_dia_in  = bar_dia_mm * U.mm
                bar_area    = 3.14159265 / 4 * bar_dia_in ** 2
                n_top = max(2, n_long // 2)
                n_bot = n_long - n_top
                rebar_layers = [
                    (n_top, bar_area,  d / 2 - cover),
                    (n_bot, bar_area, -(d / 2 - cover)),
                ]
                GJ = 0.3 * b * d ** 3
                self.add_rc_rect_beam(
                    name=f'rc_col_{tag}',   # FIX: was 'column_{tag}'
                    b=b, d=d, cover=cover, GJ=GJ,
                    rebar_layers=rebar_layers,
                    confined_mat=confined_mat,
                    unconfined_mat=unconfined_mat,
                    steel_mat=steel_mat,
                    tag=tag,
                )
                n += 1

        return n

    # ================================================================
    # Plate / slab sections
    # ================================================================

    def register_plate(self, name: str, tag: int) -> int:
        """Register a slab section tag (defined in slabs.py) for lookup."""
        self.tags[name] = tag
        if tag >= self._next_tag:
            self._next_tag = tag + 1
        return tag
