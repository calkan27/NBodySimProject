"""
This module implements Yoshida's fourth-order symplectic integrator.

The Yoshida4Scheme class extends the Verlet scheme to fourth-order accuracy through a
specific composition of Verlet steps with calculated coefficients. The implementation
provides higher accuracy than standard Verlet at moderate additional cost, making it
suitable for long-term integration requiring excellent energy conservation. It assumes
timesteps are small enough for the fourth-order expansion to be valid and that the
system benefits from higher-order accuracy.
"""

from __future__ import annotations
from .integration_scheme_base import IntegrationScheme



class Yoshida4Scheme(IntegrationScheme):
    def _yoshida4(self, h: float) -> None:
        h = float(h)
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w2 = -cbrt2 / (2.0 - cbrt2)
        self._verlet_kernel(w1 * h)
        self._verlet_kernel(w2 * h)
        self._verlet_kernel(w1 * h)



