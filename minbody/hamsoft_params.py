"""
This module manages parameters for the Hamiltonian softening integrator through the
HamSoftParams dataclass.

It handles configuration inheritance from SimConfig, provides properties with validation
and schedule refresh triggers, maintains defaults for k_soft, mu_soft, chi_eps, k_wall,
and barrier_exponent, and coordinates parameter updates with the timestep scheduler. The
implementation ensures parameter changes properly trigger recalibration of derived
quantities like substep counts. It assumes the parent integrator maintains valid
references to the simulation and configuration objects.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass






@dataclass
class HamSoftParams:
	owner: object
	k_soft: float
	mu_soft: float
	chi_eps: float
	k_wall: float
	barrier_exponent: int

	def __init__(
		self,
		integ,
		*,
		k_soft: float = 0.0,
		mu_soft: float = 1.0,
		chi_eps: float = 1.0,
		k_wall: float = 1.0e9,
		barrier_exponent: int | None = None,
	) -> None:
		self._integ = integ

		sim = getattr(integ, "sim", None)
		if sim is not None:
			cfg = getattr(sim, "cfg", None)
		else:
			cfg = None

		if cfg is not None:
			self._k_soft = float(getattr(cfg, "k_soft", k_soft))
		else:
			self._k_soft = float(k_soft)

		mu_val = float(mu_soft)
		if mu_val == 0.0:
			self._mu_soft = 1.0
		else:
			self._mu_soft = mu_val

		if cfg is not None and hasattr(cfg, "chi_eps"):
			self._chi_eps = float(getattr(cfg, "chi_eps", chi_eps))
		else:
			self._chi_eps = float(chi_eps)

		if cfg is not None:
			self._k_wall = float(getattr(cfg, "k_wall", k_wall))
		else:
			self._k_wall = float(k_wall)

		if barrier_exponent is not None:
			self._barrier_n = int(barrier_exponent)
		else:
			if cfg is not None and hasattr(cfg, "barrier_exponent"):
				self._barrier_n = int(getattr(cfg, "barrier_exponent"))
			else:
				self._barrier_n = None

	def _refresh_schedule_now(self) -> None:

		integ = getattr(self, "_integ", None)
		if integ is None:
			return  

		dt_user = getattr(integ, "_top_dt", None)
		if isinstance(dt_user, (int, float)) and dt_user > 0.0:
			dt_user = float(dt_user)
		else:
			hsr = getattr(integ, "h_sub_ref", None)
			if isinstance(hsr, (int, float)) and hsr > 0.0:
				dt_user = float(hsr)
			else:
				sim = getattr(integ, "sim", None)
				if sim is not None:
					cfg = getattr(sim, "cfg", None)
				else:
					cfg = None
				cand = None
				if cfg is not None:
					cand = getattr(cfg, "initial_dt", None)
					if not (isinstance(cand, (int, float)) and cand > 0.0):
						cand = getattr(cfg, "max_fraction_of_dt", None)
				if isinstance(cand, (int, float)) and cand > 0.0:
					dt_user = float(cand)
				else:
					dt_user = 1.0e-2

		if getattr(integ, "_top_dt", None) is None:
			integ._top_dt = float(dt_user)

		ts = getattr(integ, "_ts", None)
		if ts is not None and hasattr(ts, "init_substep_schedule"):
			ts.init_substep_schedule(float(dt_user))


	@property
	def k_soft(self) -> float:
		return self._k_soft

	@k_soft.setter
	def k_soft(self, value: float) -> None:
		self._k_soft = float(value)
		self._refresh_schedule_now()

	@property
	def mu_soft(self) -> float:
		return self._mu_soft

	@mu_soft.setter
	def mu_soft(self, value: float) -> None:
		v = float(value)
		if v == 0.0:
			self._mu_soft = 1.0
		else:
			self._mu_soft = v
		self._refresh_schedule_now()

	@property
	def chi_eps(self) -> float:
		return self._chi_eps

	@chi_eps.setter
	def chi_eps(self, value: float) -> None:
		self._chi_eps = float(value)

	@property
	def k_wall(self) -> float:
		return self._k_wall

	@k_wall.setter
	def k_wall(self, value: float) -> None:
		self._k_wall = float(value)

	@property
	def barrier_exponent(self) -> int:
		if self._barrier_n is not None:
			return int(self._barrier_n)
		cfg = getattr(self._integ.sim, "cfg", None)
		if cfg is not None:
			n = int(getattr(cfg, "barrier_exponent", 5))
		else:
			n = 5
		return max(2, n)

	@barrier_exponent.setter
	def barrier_exponent(self, n: int) -> None:
		self._barrier_n = int(n)

