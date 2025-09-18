"""
This module implements the standard Verlet integration scheme.

The VerletScheme class extends IntegrationScheme with the basic second-order symplectic
Verlet method, providing simple and robust time integration for N-body systems. The
implementation delegates to the efficient _verlet_kernel in the base class while
maintaining the scheme-specific interface. It assumes the simulation has valid
accelerations and that timesteps are appropriate for system dynamics.
"""

from __future__ import annotations
from .integration_scheme_base import IntegrationScheme



class VerletScheme(IntegrationScheme):
    def _verlet(self, h: float) -> None:
        self._verlet_kernel(h)

