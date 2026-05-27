"""
loads.py
========
Distributed gravity loads from a CSV instead of hardcoded ``eleLoad`` calls.

Bug fixes
---------
1. ``_apply_legacy_loads`` — all ``eleLoad`` calls after the first one were
   missing the leading ``'-'`` in the type string (``'beamUniform'`` instead
   of ``'-beamUniform'``). OpenSees requires the dash; without it the loads
   are silently not applied. Fixed all nine calls to use ``'-beamUniform'``.

2. ``_ensure_pattern`` — ``timeSeries('Linear', ts_tag)`` was guarded by
   ``if not self._patterns_created`` (fires only once for the very first
   pattern ever added).  If two patterns reference *different* time-series
   tags, only the first ts_tag ever gets created.  Additionally, if
   ``_apply_legacy_loads`` is called after ``apply_loads`` has already
   registered pattern 1, both methods try to create ``timeSeries(1)`` and
   ``pattern('Plain', 1, 1)``, raising an OpenSees error for the duplicate.
   Fixed with a separate ``_ts_created`` set that tracks which series tags
   have been registered.
"""

from typing import Dict, Set
import pandas as pd

from openseespy.opensees import timeSeries, pattern, eleLoad

from .config  import Config
from .data_io import DataLoader


class LoadBuilder:
    def __init__(self, config: Config, loader: DataLoader):
        self.config = config
        self.loader = loader
        self._patterns_created: Set[int] = set()
        self._ts_created:       Set[int] = set()   # FIX: track per-ts_tag

    # ----------------------------------------------------
    def _ensure_pattern(self, ptag: int, ts_tag: int = 1) -> None:
        # FIX: create the timeSeries only if this specific ts_tag is new
        if ts_tag not in self._ts_created:
            timeSeries('Linear', ts_tag)
            self._ts_created.add(ts_tag)
        if ptag not in self._patterns_created:
            pattern('Plain', ptag, ts_tag)
            self._patterns_created.add(ptag)

    # ----------------------------------------------------
    def apply_loads(self) -> None:
        df = self.loader.read(self.config.loads_csv, optional=True)
        if df is None:
            print('[WARN] loads.csv not found, falling back to legacy loads')
            self._apply_legacy_loads()
            return

        for _, row in df.iterrows():
            ptag   = int(row.get('pattern', 1))
            ts_tag = int(row.get('ts_tag',  1))
            self._ensure_pattern(ptag, ts_tag)

            wx = float(row.get('wx', 0.0))
            wy = float(row.get('wy', 0.0))
            wz = float(row.get('wz', 0.0))

            elems = row.get('elements', '')
            if isinstance(elems, str) and elems.strip():
                ids = [int(x) for x in elems.replace(',', ';').split(';') if x.strip()]
                eleLoad('-ele', *ids, '-type', '-beamUniform', wy, wz, wx)
                continue

            r_from = row.get('range_from')
            r_to   = row.get('range_to')
            if pd.notna(r_from) and pd.notna(r_to):
                eleLoad(
                    '-range', int(r_from), int(r_to),
                    '-type', '-beamUniform', wy, wz, wx,
                )
                continue

            raise ValueError(
                'Each row of loads.csv needs either (range_from, range_to) '
                'or a non-empty "elements" column.'
            )
        print(f'[OK]   Applied {len(df)} load rows from CSV')

    # ----------------------------------------------------
    def _apply_legacy_loads(self) -> None:
        """Exact replica of the original script's eleLoad block.

        FIX: every call now uses '-beamUniform' (with the required leading
        dash). The original script and the previous refactored version had
        only the first call correct; all subsequent calls were missing the '-'
        and would have been silently ignored by OpenSees.
        """
        den = 0.150 / 1728   # kips / in^3
        w, d = 100, 7.5
        wzi = den * w * d

        self._ensure_pattern(1, ts_tag=1)

        # Self-weight slab dead load on all beam elements (1-90)
        eleLoad('-range', 1,   90,  '-type', '-beamUniform', 0, -wzi,      0)
        # Superimposed dead loads by zone
        eleLoad('-range', 1,   15,  '-type', '-beamUniform', 0, -0.009833, 0)  # FIX: added '-'
        eleLoad('-range', 16,  25,  '-type', '-beamUniform', 0, -0.01175,  0)  # FIX: added '-'
        eleLoad('-ele', 26, 30, 31, 35, 36, 40, 51, 55, 56, 60, 61, 65,
                '-type', '-beamUniform', 0, -0.009833, 0)                       # FIX: added '-'
        eleLoad('-ele', 27, 28, 29, 32, 33, 34, 37, 38, 39,
                52, 53, 54, 57, 58, 59, 62, 63, 64,
                '-type', '-beamUniform', 0, -0.010833, 0)                       # FIX: added '-'
        eleLoad('-range', 41,  50,  '-type', '-beamUniform', 0, -0.01175,  0)  # FIX: added '-'
        eleLoad('-range', 66,  75,  '-type', '-beamUniform', 0, -0.01175,  0)  # FIX: added '-'
        eleLoad('-range', 76,  90,  '-type', '-beamUniform', 0, -0.009833, 0)  # FIX: added '-'
        eleLoad('-range', 91,  142, '-type', '-beamUniform', 0, -0.001725, 0)  # FIX: added '-'
        eleLoad('-range', 143, 166, '-type', '-beamUniform', 0, -0.148438, 0)  # FIX: added '-'
