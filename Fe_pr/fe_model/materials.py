"""
materials.py
============
Material definitions. In the original script, material tags (1, 2, 3) were
hardcoded integers used directly in section definitions, so adding a new
material risked silent collisions. Here, materials are looked up by
**name** ('steel', 'confined_concrete', etc.) and tags are auto-managed.

Bug fixes
---------
* Added ``build_from_csv(loader, filename)`` — the method pipeline.py has
  always called but that never existed. Reads a materials.csv and dispatches
  to add_steel02 / add_concrete02 / add_elastic based on a 'type' column.
  Falls back to ``build_defaults()`` when the file is absent so the framework
  still runs out-of-the-box.

materials.csv schema
--------------------
name, type, [columns specific to type]

Steel02 columns  : Fy, E, b, R0, cR1, cR2, a1, a2, a3, a4, sig_init
Concrete02 columns: fc, eps_c, [fcu, eps_u, Lambda, ft, Ets]  — all optional
Elastic columns  : E
"""

from typing import Dict, Optional
from openseespy.opensees import uniaxialMaterial


class MaterialLibrary:
    """Named uniaxial material registry."""

    def __init__(self, start_tag: int = 1):
        self.tags: Dict[str, int] = {}
        self._next_tag = start_tag

    # -------------------- internals --------------------
    def _new_tag(self) -> int:
        tag = self._next_tag
        self._next_tag += 1
        return tag

    # -------------------- queries ----------------------
    def get(self, name: str) -> int:
        if name not in self.tags:
            raise KeyError(
                f'Material "{name}" not defined. '
                f'Available: {sorted(self.tags)}'
            )
        return self.tags[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tags

    # -------------------- builders ---------------------
    def add_steel02(
        self, name: str,
        Fy: float, E: float, b: float = 0.01,
        R0: float = 18.0, cR1: float = 0.925, cR2: float = 0.15,
        a1: float = 0.05, a2: float = 1.0,
        a3: float = 0.05, a4: float = 1.0,
        sig_init: float = 0.0,
    ) -> int:
        tag = self._new_tag()
        uniaxialMaterial(
            'Steel02', tag, Fy, E, b,
            R0, cR1, cR2, a1, a2, a3, a4, sig_init,
        )
        self.tags[name] = tag
        return tag

    def add_concrete02(
        self, name: str,
        fc: float, eps_c: float,
        fcu: Optional[float] = None, eps_u: float = -0.01,
        Lambda: float = 0.1,
        ft: Optional[float] = None,
        Ets: Optional[float] = None,
    ) -> int:
        """Concrete02 with sensible defaults derived from fc when omitted.

        - fc, eps_c : peak compressive stress / strain  (negative in compression)
        - fcu       : residual stress, defaults to 0.2*fc
        - ft        : tensile strength, defaults to -0.14*fc
        - Ets       : tension softening stiffness, defaults to ft / 0.002
        """
        if fcu is None: fcu = 0.2 * fc
        if ft  is None: ft  = -0.14 * fc
        if Ets is None: Ets = ft / 0.002

        tag = self._new_tag()
        uniaxialMaterial(
            'Concrete02', tag, fc, eps_c, fcu, eps_u, Lambda, ft, Ets,
        )
        self.tags[name] = tag
        return tag

    def add_elastic(self, name: str, E: float) -> int:
        tag = self._new_tag()
        uniaxialMaterial('Elastic', tag, E)
        self.tags[name] = tag
        return tag

    # -------------------- CSV loader -------------------  FIX: method added
    def build_from_csv(self, loader, filename: str = 'materials.csv') -> None:
        """Load material definitions from a CSV file.

        If the file does not exist, fall back to ``build_defaults()`` so the
        framework works out-of-the-box without a materials.csv.

        materials.csv schema
        --------------------
        name     : str   — lookup name used by SectionLibrary
        type     : str   — 'Steel02' | 'Concrete02' | 'Elastic'
        (remaining columns are type-specific; see module docstring)
        """
        df = loader.read(filename, optional=True)
        if df is None:
            print('[INFO] materials.csv not found — using built-in defaults')
            self.build_defaults()
            return

        # Work on a copy with lower-cased columns so we never mutate the
        # DataLoader cache and are robust to any header capitalisation.
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # Build a case-insensitive alias map so common header variants all
        # resolve to the canonical key names used below.
        # e.g. 'Material Type', 'mat_type', 'TYPE' all -> 'type'
        _ALIASES = {
            'name':  ['name', 'material', 'mat', 'material_name', 'mat_name'],
            'type':  ['type', 'material type', 'mat_type', 'material_type', 'kind'],
            'fy':    ['fy', 'yield_stress', 'fy_ksi'],
            'e':     ['e', 'es', 'ec', 'youngs_modulus', 'elastic_modulus', 'modulus'],
            'fc':    ['fc', 'fc_ksi', 'compressive_strength'],
            'eps_c': ['eps_c', 'epsc', 'strain_at_fc', 'eps1'],
        }
        col_map: dict = {}
        for canonical, aliases in _ALIASES.items():
            for col in df.columns:
                if col.lower().replace(' ', '_') in [a.replace(' ', '_') for a in aliases]:
                    col_map[canonical] = col
                    break
        # Direct 1-to-1 mapping for every remaining column not yet aliased
        for col in df.columns:
            col_map.setdefault(col, col)

        def _req(row, col):
            """Return the value of a required column (case-insensitive)."""
            key = col_map.get(col.lower())
            if key is None:
                raise KeyError(
                    f'materials.csv is missing required column "{col}". '
                    f'Available columns: {list(df.columns)}'
                )
            return row[key]

        def _get(row, col, default=None):
            """Return float value of an optional column, or default."""
            key = col_map.get(col.lower())
            if key is None:
                return default
            val = row[key]
            if val != val:   # NaN
                return default
            return float(val)

        for _, row in df.iterrows():
            name = str(_req(row, 'name')).strip()
            kind = str(_req(row, 'type')).strip()

            if kind == 'Steel02':
                self.add_steel02(
                    name,
                    Fy       = float(_req(row, 'fy')),
                    E        = float(_req(row, 'e')),
                    b        = _get(row, 'b',        0.01),
                    R0       = _get(row, 'r0',       18.0),
                    cR1      = _get(row, 'cr1',      0.925),
                    cR2      = _get(row, 'cr2',      0.15),
                    a1       = _get(row, 'a1',       0.05),
                    a2       = _get(row, 'a2',       1.0),
                    a3       = _get(row, 'a3',       0.05),
                    a4       = _get(row, 'a4',       1.0),
                    sig_init = _get(row, 'sig_init', 0.0),
                )
            elif kind == 'Concrete02':
                self.add_concrete02(
                    name,
                    fc    = float(_req(row, 'fc')),
                    eps_c = float(_req(row, 'eps_c')),
                    fcu   = _get(row, 'fcu'),
                    eps_u = _get(row, 'eps_u', -0.01),
                    Lambda= _get(row, 'lambda', 0.1),
                    ft    = _get(row, 'ft'),
                    Ets   = _get(row, 'ets'),
                )
            elif kind == 'Elastic':
                self.add_elastic(name, E=float(_req(row, 'e')))
            else:
                raise ValueError(
                    f'Unknown material type "{kind}" for material "{name}". '
                    f'Supported types: Steel02, Concrete02, Elastic.'
                )

        print(f'[OK]   Loaded {len(df)} materials from {filename}')

    # -------------------- defaults ---------------------
    def build_defaults(self) -> None:
        """Replicates the original script's material set."""
        from .config import Units as U
        self.add_steel02('steel', Fy=36.0 * U.ksi, E=29000.0 * U.ksi, b=0.10)
        self.add_concrete02('unconfined', fc=-4.0 * U.ksi, eps_c=-0.003)
        self.add_concrete02('confined',   fc=-4.5 * U.ksi, eps_c=-0.0035)
