"""
data_io.py
==========
Centralised CSV reading. Adds:
  * up-front validation of required columns
  * informative errors when a file is missing
  * caching so the same CSV isn't parsed twice
  * graceful handling of optional inputs
  * SCHEMA-TOLERANT column aliasing (e.g. 'node_index' -> 'nodetag',
    'x (m)' -> 'x') so different upstream tools can feed the framework
  * UNIT auto-detection from column-header tags like '(m)', '(ft)', '(mm)'
"""

from typing import Dict, List, Optional
import re
import pandas as pd

from .config import Config, Units


# ----------------------------------------------------------------
# Canonical column schemas. Keys are the names the rest of the
# framework uses internally; values are accepted aliases.
# ----------------------------------------------------------------
SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    'nodes': {
        'nodetag': ['nodetag', 'node_index', 'node_id', 'nodeid', 'tag', 'id'],
        'x':       ['x', 'X'],
        'y':       ['y', 'Y'],
        'z':       ['z', 'Z'],
        'mass':    ['mass', 'm'],
    },
    'beams': {
        'ID':     ['id', 'tag', 'element_id', 'eletag'],
        'node1':  ['node1', 'i', 'ni', 'inode', 'start_node'],
        'node2':  ['node2', 'j', 'nj', 'jnode', 'end_node'],
        'Sectag': ['sectag', 'section', 'section_tag', 'sec'],
        'mass':   ['mass'],
    },
    'columns': {
        'ID':     ['id', 'tag', 'element_id', 'eletag'],
        'node1':  ['node1', 'i', 'ni', 'inode', 'start_node'],
        'node2':  ['node2', 'j', 'nj', 'jnode', 'end_node'],
        'Sectag': ['sectag', 'section', 'section_tag', 'sec'],
        'mass':   ['mass'],
    },
    'equal_dof': {
        'node1': ['node1', 'master', 'master_node'],
        'node2': ['node2', 'slave',  'slave_node'],
        'D1':    ['d1', 'ux'],
        'D2':    ['d2', 'uy'],
        'D3':    ['d3', 'uz'],
        'D4':    ['d4', 'rx'],
        'D5':    ['d5', 'ry'],
        'D6':    ['d6', 'rz'],
    },
    'bc': {
        'node': ['node', 'nodetag', 'tag'],
        'ux':   ['ux', 'd1'],
        'uy':   ['uy', 'd2'],
        'uz':   ['uz', 'd3'],
        'rx':   ['rx', 'd4'],
        'ry':   ['ry', 'd5'],
        'rz':   ['rz', 'd6'],
    },
}


# Length-unit suffix detection: e.g. 'x (m)', 'x_mm', 'X [ft]'
_UNIT_PATTERNS = {
    'mm':   Units.mm,
    'cm':   10 * Units.mm,
    'm':    Units.m,
    'in':   Units.inch,
    'inch': Units.inch,
    'ft':   Units.ft,
}


def _detect_length_unit(headers: List[str]) -> Optional[float]:
    """Scan column headers for a unit tag like '(m)', '[ft]', '_mm'.

    Returns the multiplier that converts the file's units to the
    framework's internal inch units. Returns None if no tag found.
    """
    pat = re.compile(r'[\(\[\s_]([a-zA-Z]+)[\)\]\s]?$')
    for h in headers:
        m = pat.search(h.strip())
        if m:
            unit = m.group(1).lower()
            if unit in _UNIT_PATTERNS:
                return _UNIT_PATTERNS[unit]
    return None


class DataLoader:
    """Read, normalise, and validate CSV inputs."""

    def __init__(self, config: Config):
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}
        self._unit_cache: Dict[str, Optional[float]] = {}

    # ============================================================
    # Public read entry point
    # ============================================================
    def read(
        self,
        filename: str,
        required_cols: Optional[List[str]] = None,
        optional: bool = False,
        schema_key: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        path = self.config.in_data(filename)

        if filename in self._cache:
            return self._cache[filename]

        if not path.exists():
            if optional:
                print(f'[INFO] Optional file not found: {filename}  (skipping)')
                return None
            raise FileNotFoundError(f'Required CSV not found: {path}')

        df = pd.read_csv(path)
        original_headers = list(df.columns)
        df.columns = [c.strip() for c in df.columns]

        # ---- Schema normalisation (alias -> canonical) ----
        if schema_key and schema_key in SCHEMAS:
            df = self._normalise_columns(df, SCHEMAS[schema_key])
            # Unit detection uses the ORIGINAL headers
            self._unit_cache[filename] = _detect_length_unit(original_headers)

        # ---- Required-column validation ----
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f'{filename} is missing required columns: {missing}\n'
                    f'  Found columns: {list(df.columns)}'
                )

        self._cache[filename] = df

        # Pretty-print the load with detected unit if any
        unit_msg = ''
        scale = self._unit_cache.get(filename)
        if scale and abs(scale - 1.0) > 1e-9:
            unit_name = next(
                (k for k, v in _UNIT_PATTERNS.items() if abs(v - scale) < 1e-9),
                f'x{scale:g}',
            )
            unit_msg = f'  [length unit: {unit_name} -> inch]'
        print(f'[OK]   Loaded {filename:<32s} ({len(df):4d} rows){unit_msg}')
        return df

    # ============================================================
    # Helpers
    # ============================================================
    def length_scale(self, filename: str) -> float:
        """Multiplier that converts the file's length units to the
        framework's internal inch units. Defaults to 1.0 (inches)."""
        scale = self._unit_cache.get(filename)
        return scale if scale else 1.0

    @staticmethod
    def _normalise_columns(
        df: pd.DataFrame, schema: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Rename any column whose lower-cased base name (ignoring unit
        suffixes like '(m)') matches an alias, to the canonical name."""
        # Strip unit suffixes from column names for matching:
        # 'x (m)' -> 'x', 'mass_kg' -> 'mass', 'Y [ft]' -> 'y'
        stripped: Dict[str, str] = {}
        for c in df.columns:
            base = re.sub(r'\s*[\(\[].*?[\)\]]\s*$', '', c).strip().lower()
            base = re.sub(r'_(mm|cm|m|in|inch|ft)$', '', base)
            stripped[c] = base

        rename: Dict[str, str] = {}
        for canonical, aliases in schema.items():
            cand_set = {a.lower() for a in aliases} | {canonical.lower()}
            for orig_col, base in stripped.items():
                if base in cand_set and orig_col != canonical and orig_col not in rename:
                    rename[orig_col] = canonical
                    break
        if rename:
            df = df.rename(columns=rename)
        return df
