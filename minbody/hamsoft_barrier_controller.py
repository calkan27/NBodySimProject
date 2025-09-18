from __future__ import annotations
from .hamsoft_utils import symplectic_reflect_eps, reflect_if_needed
import numpy as np

"""
This module manages barrier interactions for the Hamiltonian softening integrator, implementing reflection boundary conditions for the softening parameter. The HamSoftBarrier class provides reflect_and_bounce for symplectic reflection at barriers during evolution, and reflect_if_active for instantaneous reflection checks. The implementation preserves the symplectic structure of the extended phase space by properly handling momentum reversal at boundaries. It integrates with the configuration system to respect barrier policy settings and disable flags. The module assumes the parent integrator maintains valid epsilon and pi values.


"""



class HamSoftBarrier:
	def __init__(self, owner) -> None:
		self._owner = owner



	def reflect_and_bounce(self, eps: float, pi: float, h: float) -> tuple[float, float]:
		owner = getattr(self, "_owner", None)
		if owner is None:
			return float(eps), float(pi)

		sim = getattr(owner, "sim", None)
		if sim is None:
			return float(eps), float(pi)

		cfg = getattr(sim, "cfg", None)
		barrier_disabled = False
		if cfg is not None:
			barrier_disabled = bool(getattr(cfg, "disable_barrier", False))

		pol_attr = getattr(owner, "barrier_policy", "reflection")
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"

		if barrier_disabled or pol != "reflection":
			return float(eps), float(pi)

		eps_min = float(getattr(sim, "_min_softening", 0.0))
		if hasattr(sim, "manager") and sim.manager is not None:
			eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		else:
			eps_max = max(1.0, eps_min)

		mu_attr = float(getattr(owner, "mu_soft", 1.0))
		if mu_attr != 0.0:
			mu = float(mu_attr)
		else:
			mu = 1.0

		return symplectic_reflect_eps(
			float(eps),
			float(pi),
			float(eps_min),
			float(eps_max),
			float(h),
			float(mu),
		)



	def reflect_if_active(self, eps: float, pi: float) -> tuple[float, float]:
		sim = getattr(self, "_owner", None)
		if sim is not None:
			sim = getattr(sim, "sim", None)

		if sim is not None and hasattr(sim, "cfg"):
			if bool(getattr(sim.cfg, "disable_barrier", False)):
				return float(eps), float(pi)

		pol_attr = getattr(self._owner, "barrier_policy", None)
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"

		if pol != "reflection":
			return float(eps), float(pi)

		eps_min = float(getattr(sim, "_min_softening", 0.0)) if sim is not None else 0.0
		if sim is not None and hasattr(sim, "manager") and sim.manager is not None:
			eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		else:
			eps_max = max(1.0, eps_min)

		eps_r, pi_r = reflect_if_needed(float(eps), float(pi), float(eps_min), float(eps_max))
		return float(eps_r), float(pi_r)






