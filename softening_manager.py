from __future__ import annotations
from collections.abc import Iterable
from typing import List, Tuple
import math, collections, sys
from typing import TYPE_CHECKING
import numpy as np
import collections
from .energy_accumulator import EnergyAccumulator 
from .barrier import barrier_energy
if TYPE_CHECKING:
	from .simulation import NBodySimulation
from .hamsoft_utils import dU_depsilon_plummer
from collections import deque
import numpy as _np

"""
This critical module manages the evolution of gravitational softening parameters during simulation. The SofteningManager class tracks softening history with configurable buffer length, computes energy corrections from softening changes, coordinates with integrators for consistent updates, and supports both continuous and discrete softening updates. It provides sophisticated energy bookkeeping through Kahan summation, handles the interplay between adaptive algorithms and energy conservation, and maintains diagnostic information for analysis. The manager is central to the adaptive softening machinery and assumes close coordination with the simulation and integrator layers.

"""


_DEFAULT_SYSTEM = object()
_ABS_SOFTENING_FLOOR: float = 1.0e-12
__all__ = ["SofteningManager"]


class SofteningManager:
	def __init__(
		self,
		sim: "NBodySimulation",
		softening: float,
		min_softening: float,
		history: int = 1024,
		tol: float = 1e-12,
	):
		self.sim = sim
		self.s0 = float(max(softening, min_softening))
		self._min_softening = float(min_softening)
		self.s = self.s0
		self.s2 = self.s * self.s
		self._step_s2 = self.s2
		self.current_epsilon = self.s
		self.S_bar = 0.0
		self._step_needs_commit = False
		self._acc = EnergyAccumulator()  
		self._history_max_len = int(history)
		self._history_tol = float(tol)

		self._history = deque([self.s], maxlen=self._history_max_len)

		self._step_dE = 0.0
		self._pending_energy_delta = 0.0

		self.segments_used = 0
		self._step_finished = True
		self._step_start_s = self.s
		self._history_len_at_begin = 1

		self._substep_refresh_count = 0
		self._substep_commit_count = 0



	@property
	def softening(self) -> float:
		return self.s

	@property
	def step_s2(self) -> float:
		return self._step_s2

	@property
	def history(self):
		return list(self._history)

	@property
	def pending_energy_delta(self) -> float:
		return self._step_dE

	@staticmethod
	def _two_sum(a: float, b: float) -> tuple[float, float]:

		s = a + b
		bb = s - a
		e = (a - (s - bb)) + (b - bb)
		return s, e


	@staticmethod
	def _limited_softening(old_eps: float, proposed_eps: float, *, factor: float = 2.0) -> float:
		lower = old_eps / factor
		upper = old_eps * factor
		return max(lower, min(upper, proposed_eps))


	def _add_energy(self, dE: float) -> None:
		self._acc.add(float(dE))
		self.sim.softening_energy_delta = self._acc.total()            

	_add_to_sim_delta = _add_energy                              

	def _kahan_add_step(self, dE: float) -> None:
		self._acc.add(float(dE))
		self._step_dE             += float(dE)                   
		self._pending_energy_delta  = self._step_dE
		self.sim.softening_energy_delta = self._acc.total()            


	def _kahan_add(self, total: float, c: float, increment: float) -> tuple[float, float]:

		s, e1 = self._two_sum(total, increment)
		hi, e2 = self._two_sum(s, c)
		lo = e1 + e2
		if lo:                       
			hi, lo = self._two_sum(hi, lo)
		return hi, lo


	def _compute_energy_delta(self, s_old: float, s_new: float) -> float:

		sim = getattr(self, "sim", None)
		if sim is None:
			return 0.0

		eps_old = float(s_old)
		eps_new = float(s_new)
		if eps_old == eps_new:
			return 0.0

		G = float(getattr(sim, "G", 0.0))
		if G == 0.0:
			return 0.0

		m = getattr(sim, "_mass", None)
		q = getattr(sim, "_pos", None)
		if not isinstance(m, np.ndarray) or not isinstance(q, np.ndarray):
			return 0.0

		n = int(m.size)
		if n < 2:
			return 0.0

		q64 = np.asarray(q, dtype=np.float64)
		diff = q64[:, None, :] - q64[None, :, :]
		r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
		np.fill_diagonal(r2, np.inf)

		s2_old = eps_old * eps_old
		s2_new = eps_new * eps_new

		inv_old = 1.0 / np.sqrt(r2 + s2_old)
		inv_new = 1.0 / np.sqrt(r2 + s2_new)

		iu = np.triu_indices(n, 1)
		m64 = np.asarray(m, dtype=np.float64)
		mprod = m64[iu[0]] * m64[iu[1]]

		pair_terms = mprod * (inv_old[iu] - inv_new[iu])

		a = np.asarray(pair_terms, dtype=np.float64).ravel()
		if a.size == 0:
			ssum = 0.0
		else:
			while a.size > 1:
				even = (a.size // 2) * 2
				pair = a[:even].reshape(-1, 2).sum(axis=1, dtype=np.float64)
				if a.size % 2 == 1:
					a = np.concatenate((pair, a[-1:]), axis=0)
				else:
					a = pair
			ssum = float(a[0])

		return float(G * ssum)


	def begin_step(self) -> None:

		sim = getattr(self, "sim", None)
		mode = getattr(sim, "_integrator_mode", None) if sim is not None else None

		if mode == "ham_soft":
			eps_now = float(getattr(sim, "_epsilon", getattr(self, "s", 0.0))) if sim is not None else float(getattr(self, "s", 0.0))
			self.s = float(eps_now)
			self._step_s2 = float(self.s * self.s)
		else:
			self._pending_energy_delta = 0.0
			self._step_s2 = float(getattr(self, "s", 0.0)) ** 2

		self._history.append(float(self.s))



	def _inert_softening(self, eps_new: float) -> bool:
		integ = getattr(self.sim, "_integrator", None)
		if integ is None:
			return False

		k_soft  = float(getattr(integ, "k_soft", 0.0))

		if k_soft == 0.0:
			return True

		if float(getattr(integ, "mu_soft", 1.0)):
			mu_soft = float(getattr(integ, "mu_soft", 1.0))
		else:
			mu_soft = 1.0

		h = float(getattr(integ, "h_actual", 0.0) or 0.0)
		if h <= 0.0:
			h = float(getattr(integ, "_top_dt", 0.0) or 0.0)
		if h <= 0.0:
			h = float(getattr(self.sim.cfg, "initial_dt", 0.0) or 0.0)

		if k_soft > 0.0 and mu_soft > 0.0:
			omega_spr = math.sqrt(k_soft / mu_soft)
		else:
			omega_spr = 0.0

		eps_mach = 2.22e-16
		C_eps    = 10.0
		C_pi     = 10.0

		s_old  = float(self.s)
		pi_val = float(getattr(self.sim, "_pi", 0.0))

		pi_idle  = abs(pi_val) <= max(eps_mach, C_pi * mu_soft * omega_spr * (h * h))
		eps_idle = abs(float(eps_new) - s_old) <= max(eps_mach * abs(s_old), C_eps * (h * h))

		return bool(pi_idle and eps_idle)


	def commit_substep(self) -> None:
		sim = getattr(self, "sim", None)
		mode = getattr(sim, "_integrator_mode", None) if sim is not None else None

		if mode == "ham_soft":
			return

		pending = float(getattr(self, "_pending_energy_delta", 0.0))
		if pending != 0.0 and sim is not None:
			current = float(getattr(sim, "softening_energy_delta", 0.0))
			sim.softening_energy_delta = current + pending
			self._pending_energy_delta = 0.0


	def _magnus_average(self, eps_old: float, eps_new: float) -> float:
		return 0.5 * (float(eps_old) + float(eps_new))


	def _bookkeep_energy(self, s_old: float, s_new: float) -> None:
		if self.sim._integrator_mode == "ham_soft":
			return

		sim = self.sim                               
		Δε  = s_new - s_old

		k_soft = float(getattr(sim._integrator, "k_soft", 0.0))
		if k_soft > 0.0 and sim.cfg.use_energy_spring:
			if hasattr(sim._integrator, "_eps_target"):
				eps_star = sim._integrator._eps_target()
			else:
				eps_star = self.s0
			dE_spring = 0.5 * k_soft * (
				(s_new - eps_star) ** 2 - (s_old - eps_star) ** 2
			)
		else:
			dE_spring = 0.0

		k_wall   = float(getattr(sim.cfg, "k_wall", 1.0e9))
		eps_min  = sim._min_softening
		eps_max  = sim._max_softening
		n_exp    = int(getattr(sim.cfg, "barrier_exponent", 4))

		U_old = barrier_energy(s_old, eps_min, eps_max,
							   k_wall=k_wall, n=n_exp)
		U_new = barrier_energy(s_new, eps_min, eps_max,
							   k_wall=k_wall, n=n_exp)
		dE_barrier = U_new - U_old


		dU = dU_depsilon_plummer(sim._pos, sim._mass, sim.G, self.s)
		dE_plummer = -dU * Δε

		sim.softening_energy_delta += dE_spring + dE_barrier + dE_plummer



	def refresh_softening(self, eps, sim) -> None:
		mode = getattr(sim, "_integrator_mode", None) if sim is not None else None
		eps_new = float(eps)

		if mode == "ham_soft":
			self.s = float(eps_new)
			self._step_s2 = float(eps_new * eps_new)
			return

		eps_old = float(getattr(self, "s", 0.0))

		dE = 0.0
		func = getattr(self, "_compute_energy_correction", None)

		if (sim is not None) and callable(func):
			is_finite_old = (eps_old == eps_old) and (eps_old != float("inf")) and (eps_old != float("-inf"))
			is_finite_new = (eps_new == eps_new) and (eps_new != float("inf")) and (eps_new != float("-inf"))
			if is_finite_old and is_finite_new:
				val = func(sim, float(eps_old), float(eps_new))
				_has_np = "np" in globals()
				if _has_np:
					is_scalar_val = isinstance(val, (int, float, np.floating))
				else:
					is_scalar_val = isinstance(val, (int, float))
				if is_scalar_val:
					vf = float(val)
					if (vf == vf) and (vf != float("inf")) and (vf != float("-inf")):
						dE = vf

		self._pending_energy_delta = float(getattr(self, "_pending_energy_delta", 0.0)) + float(dE)

		self.s = float(eps_new)
		self._step_s2 = float(eps_new * eps_new)

		h = getattr(self, "history", None)
		if isinstance(h, list):
			h.append(float(self.s))
		elif hasattr(self, "_history") and isinstance(getattr(self, "_history"), list):
			getattr(self, "_history").append(float(self.s))



	def update_continuous(self, eps_new: float) -> None:
		sim = getattr(self, "sim", None)
		mode = getattr(sim, "_integrator_mode", None) if sim is not None else None

		eps_val = float(eps_new)

		self.s = float(eps_val)
		self._step_s2 = float(eps_val * eps_val)

		if mode == "ham_soft":
			return

		return


	def finish_step(self) -> None:
		sim = getattr(self, "sim", None)
		mode = getattr(sim, "_integrator_mode", None) if sim is not None else None

		if mode == "ham_soft":
			if sim is not None:
				eps_now = float(getattr(sim, "_epsilon", getattr(self, "s", 0.0)))
			else:
				eps_now = float(getattr(self, "s", 0.0))
			self.s = float(eps_now)
			self._step_s2 = float(eps_now * eps_now)
			return

		pending = float(getattr(self, "_pending_energy_delta", 0.0))
		if pending != 0.0 and sim is not None:
			current = float(getattr(sim, "softening_energy_delta", 0.0))
			sim.softening_energy_delta = current + pending
		self._pending_energy_delta = 0.0



	def validate_energy(self) -> None:
		if len(self._history) < 2:
			return
		total_dE = 0.0
		hist = list(self._history)
		for s_old, s_new in zip(hist[:-1], hist[1:]):
			total_dE += self._compute_energy_delta(s_old, s_new)
		ref = self.sim.softening_energy_delta
		if ref == 0.0:
			err = abs(total_dE - ref)
		else:
			err = abs((total_dE - ref) / ref)
		if err > 1e-10:
			print(f"[warning] energy mismatch: {err:.3g}")
			

	def update_base_softening(self, adaptive: bool) -> None:
		if adaptive:
			return

		self.s          = self.s0
		self.s2         = self.s * self.s
		self._step_s2   = self.s2
		self._step_start_s = self.s

		self._history.clear()
		self._history.append(self.s)

		self._step_dE              = 0.0
		self._pending_energy_delta = 0.0
		self.sim.softening_energy_delta = 0.0
		self.sim._max_softening = 10.0 * self.s0

	def debug_info(self) -> dict:
		return dict(
			softening=self.s,
			step_s2=self._step_s2,
			history=list(self._history),
			pending_energy_delta=self._step_dE,
			last_dE=getattr(self, 'last_dE', 0.0),
			segments_used=self.segments_used,
		)

	def mismatch_stats(self):
		return getattr(self, "_mismatch_count", 0)


	@staticmethod
	def _compute_energy_correction(system, eps_old: float, eps_new: float) -> float:
		if eps_old == eps_new:
			return 0.0  

		mode        = getattr(system, "_integrator_mode", "")
		is_ham_soft = (mode == "ham_soft")
		use_spring  = bool(system.cfg.use_energy_spring) and system._integrator.k_soft != 0.0

		G      = float(system.G)
		pos    = _np.asarray(system._pos,  dtype=_np.float64)
		m      = _np.asarray(system._mass, dtype=_np.float64)
		n      = m.size

		eps_min = float(system._min_softening)
		eps_max = float(10.0 * system.manager.s0)
		k_wall  = float(getattr(system._integrator, "k_wall", 1.0e9))
		n_exp   = system.cfg.barrier_exponent

		dE_total = 0.0

		if (not is_ham_soft) and n >= 2 and G != 0.0:
			diff = pos[:, None, :] - pos[None, :, :]
			r2   = _np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
			_np.fill_diagonal(r2, _np.inf)
			inv_old = 1.0 / _np.sqrt(r2 + eps_old ** 2)
			inv_new = 1.0 / _np.sqrt(r2 + eps_new ** 2)
			_np.fill_diagonal(inv_old, 0.0)
			_np.fill_diagonal(inv_new, 0.0)
			iu, ju = _np.triu_indices(n, 1)
			dE_total += G * float(_np.sum(
				m[iu] * m[ju] * (inv_new[iu, ju] - inv_old[iu, ju])
			))

		if (not is_ham_soft) and use_spring:
			eps_star = system.manager.s0       
			k_soft   = float(system._integrator.k_soft)
			S_old = 0.5 * k_soft * (eps_old - eps_star) ** 2
			S_new = 0.5 * k_soft * (eps_new  - eps_star) ** 2
			dE_total += (S_new - S_old)

		if not is_ham_soft:  
			Sb_old = barrier_energy(eps_old, eps_min, eps_max,
									k_wall=k_wall, n=n_exp)
			Sb_new = barrier_energy(eps_new, eps_min, eps_max,
									k_wall=k_wall, n=n_exp)
			dE_total += (Sb_new - Sb_old)

		return float(dE_total)


	@staticmethod
	def delta_potential_from_softening(
		q,
		m,
		G: float,
		eps_old: float,
		eps_new: float,
	) -> float:

		Gf = float(G)
		if Gf == 0.0:
			return 0.0

		q_arr = np.asarray(q, dtype=float)
		m_arr = np.asarray(m, dtype=float).ravel()

		if q_arr.ndim != 2:
			return 0.0
		if q_arr.shape[0] != m_arr.size:
			return 0.0
		if q_arr.shape[0] < 2:
			return 0.0

		n = int(q_arr.shape[0])

		diff = q_arr[:, None, :] - q_arr[None, :, :]
		r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
		np.fill_diagonal(r2, np.inf)

		s2_old = float(eps_old) * float(eps_old)
		s2_new = float(eps_new) * float(eps_new)

		inv_old = 1.0 / np.sqrt(r2 + s2_old)
		inv_new = 1.0 / np.sqrt(r2 + s2_new)

		iu = np.triu_indices(n, 1)
		mprod = m_arr[iu[0]] * m_arr[iu[1]]

		terms = mprod * (inv_new[iu] - inv_old[iu])

		a = np.asarray(terms, dtype=np.float64).ravel()
		if a.size == 0:
			ssum = 0.0
		else:
			while a.size > 1:
				even = (a.size // 2) * 2
				pair = a[:even].reshape(-1, 2).sum(axis=1, dtype=np.float64)
				if a.size % 2 == 1:
					a = np.concatenate((pair, a[-1:]), axis=0)
				else:
					a = pair
			ssum = float(a[0])

		delta_U = -Gf * ssum
		return float(delta_U)


	def _compact_history(self) -> None:
		if len(self._history) <= 1:
			return                                  

		newest = self._history.pop()                
		total  = sum(self._history)                 
		self._history.clear()
		self._history.extend([total, newest])       


	def softening_from_min_sep(self, min_sep: float) -> float:
		if not math.isfinite(min_sep) or min_sep <= 0.0:
			return self.s

		proposed = max(self._min_softening, min_sep / self.sim._softening_scale)
		proposed = min(proposed, 10.0 * self.s0)  
		return SofteningManager._limited_softening(self.s, proposed)

	def energy_delta_exact(self, eps_old: float, eps_new: float, q: np.ndarray, m: np.ndarray, G: float) -> float:

		q_arr = np.asarray(q, dtype=float)
		m_arr = np.asarray(m, dtype=float).ravel()

		if not isinstance(q_arr, np.ndarray):
			return 0.0
		if q_arr.ndim != 2:
			return 0.0
		if q_arr.shape[1] != 2:
			return 0.0
		if not np.all(np.isfinite(q_arr)):
			return 0.0

		n = int(q_arr.shape[0])
		if n < 2:
			return 0.0
		if m_arr.size != n:
			return 0.0
		if not np.all(np.isfinite(m_arr)):
			return 0.0
		if float(G) == 0.0:
			return 0.0

		diff = q_arr[:, None, :] - q_arr[None, :, :]
		r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
		iu = np.triu_indices(n, 1)

		m_i = m_arr[iu[0]]
		m_j = m_arr[iu[1]]
		mprod = m_i * m_j

		def _U(eps_val: float) -> float:
			e = float(eps_val)
			rsoft = np.sqrt(r2[iu] + e * e)
			term = mprod / rsoft
			u = -float(G) * float(np.sum(term))
			if not np.isfinite(u):
				return 0.0
			return float(u)

		U_old = _U(float(eps_old))
		U_new = _U(float(eps_new))
		dU = float(U_new - U_old)
		return dU

	def update_softening(self, new_eps: float, *, is_continuous: bool | None = None):
		if (is_continuous or (is_continuous is None and self.sim._integrator_mode == "ham_soft")):
			target = self.update_continuous
		else:
			target = self.refresh_softening
		return target(new_eps)


	def __repr__(self) -> str:
		return (
			f"<SofteningManager ε={self.current_epsilon:g} ΔE_step={self._step_dE:g} "
			f"ΔE_tot={self.sim.softening_energy_delta:g} segs={self.segments_used}>"
		)


