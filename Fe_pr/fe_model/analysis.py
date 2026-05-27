"""
analysis.py
===========
Static gravity, modal, and pushover analyses.

Bug fixes
---------
* ``run_modal`` — called ``eigen()`` directly after ``run_gravity()`` had
  called ``wipeAnalysis()``. In OpenSeesPy, ``wipeAnalysis()`` destroys the
  solver objects (constraints handler, numberer, system, test, algorithm).
  Calling ``eigen()`` without re-creating them results in either an error or
  garbage eigenvalues. Fixed by calling ``_configure_solver()`` at the start
  of ``run_modal`` so the solver state is always valid.
"""

import time
from typing import List, Optional

from openseespy.opensees import (
    constraints, numberer, system, test,
    integrator, algorithm, analysis,
    analyze, record, setTime, loadConst,
    remove, wipeAnalysis, eigen, nodeDisp,
)

from .config import Config


class AnalysisDriver:
    """Run static / modal analyses."""

    def __init__(self, config: Config):
        self.config = config

    # ===========================================================
    # Common solver block
    # ===========================================================
    def _configure_solver(self, max_iter: Optional[int] = None) -> None:
        constraints(self.config.solver_constraints)
        numberer(self.config.solver_numberer)
        system(self.config.solver_system)
        test('NormDispIncr', self.config.tolerance,
             max_iter or self.config.max_iter * 30, 5)
        algorithm(self.config.solver_algorithm)

    # ===========================================================
    # Gravity / load-control analysis
    # ===========================================================
    def run_gravity(
        self,
        n_steps: Optional[int] = None,
        progress: bool = True,
    ) -> bool:
        n_steps = n_steps or self.config.gravity_load_steps
        inc = 1.0 / n_steps
        self._configure_solver()
        integrator('LoadControl', inc)
        analysis('Static')
        record()
        print('=== Gravity analysis ===')
        start = time.time()
        for i in range(n_steps):
            ok = analyze(1)
            if ok != 0:
                print(f'\n[FAIL] Gravity analysis diverged at step {i+1}/{n_steps}')
                return False
            if progress and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
                pct = (i + 1) * 100 / n_steps
                el  = (time.time() - start) / 60
                print(f'  step {i+1:4d}/{n_steps}   ({pct:5.1f}%)   '
                      f'elapsed {el:5.2f} min', end='\r')
        print()
        print('=== Gravity analysis complete ===')
        setTime(0.0)
        loadConst()
        remove('recorders')
        wipeAnalysis()
        return True

    # ===========================================================
    # Modal / eigen analysis
    # ===========================================================
    def run_modal(self, n_modes: int = 5) -> List[float]:
        """Run an eigenvalue analysis and return the natural periods.

        eigen() creates its own ArpackSOE internally and only needs a clean
        analysis state. Calling system('SuperLU') or any non-eigen system
        before eigen() suppresses the EigenSOE and produces the warning
        "no EigenSOE has been set". wipeAnalysis() provides the clean state
        without interfering with eigen()'s internal solver setup.
        """
        print(f'=== Modal analysis: {n_modes} modes ===')
        wipeAnalysis()   # clean state; eigen() will create its own EigenSOE
        omega2 = eigen(n_modes)
        periods = []
        for i, w2 in enumerate(omega2):
            if w2 <= 0.0:
                periods.append(float('inf'))
                print(f'  Mode {i+1}: non-positive eigenvalue ({w2:.4e}) — check model')
                continue
            T = 2.0 * 3.14159265358979 / (w2 ** 0.5)
            periods.append(T)
            print(f'  Mode {i+1}: T = {T:.4f} s   (omega^2 = {w2:.4e})')
        return periods

    # ===========================================================
    # Save node displacements
    # ===========================================================
    def save_node_displacements(
        self, node_tags: List[int], filename: str,
        dofs: List[int] = (1, 2, 3),
    ) -> None:
        import csv
        path = self.config.in_output(filename)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['node'] + [f'u{d}' for d in dofs]
            w.writerow(header)
            for tag in node_tags:
                row = [tag] + [nodeDisp(tag, d) for d in dofs]
                w.writerow(row)
        print(f'[OK]   Wrote nodal displacements to {path}')
