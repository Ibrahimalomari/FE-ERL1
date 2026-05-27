"""
fe_model
========
Modular OpenSeesPy framework for 3D structural FE models built from CSVs.

Typical usage
-------------
    from fe_model import Config, build_full_model

    cfg = Config(data_dir='./data')
    build_full_model(cfg)

Bug fixes
---------
* Removed ``from .slabs import SlabGenerator`` — slabs.py does not exist in
  this package, so the import raised ModuleNotFoundError at startup and
  prevented any use of the framework. The slab step is handled gracefully
  inside pipeline.py with a try/except guard. SlabGenerator is also removed
  from __all__ for the same reason.
"""

from .config    import Units, Config
from .data_io   import DataLoader
from .materials import MaterialLibrary
from .sections  import SectionLibrary
from .builder   import ModelBuilder
from .loads     import LoadBuilder
from .analysis  import AnalysisDriver
from .pipeline  import build_full_model

__all__ = [
    'Units', 'Config', 'DataLoader',
    'MaterialLibrary', 'SectionLibrary',
    'ModelBuilder', 'LoadBuilder',
    'AnalysisDriver',
    'build_full_model',
]
