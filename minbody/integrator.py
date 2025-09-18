from __future__ import annotations
import math
from typing import TYPE_CHECKING, Tuple
import numpy as np
from .kepler_solver import UniversalVariableKeplerSolver
from .timestep_manager import TimestepManager
from .integration_scheme_base import IntegrationScheme
from .verlet_scheme import VerletScheme
from .yoshida4_scheme import Yoshida4Scheme
from .whfast_scheme import WHFastScheme
from .diagnostics import Diagnostics
from .hamsoft_flows import PhaseState

"""
This central module implements the base Integrator class that coordinates timestepping for N-body simulations. Key responsibilities include managing adaptive and fixed timestep schemes, coordinating position updates (drift) and velocity updates (kick), maintaining geometric buffers for force calculations, interfacing with specialized integration schemes, and handling substep scheduling for stability. The class provides abstractions for different integration methods while maintaining common infrastructure for buffer management, timestep control, and diagnostic collection. It integrates with the TimestepManager for stability-based step subdivision and supports both classical and Hamiltonian softening approaches. The implementation assumes the simulation maintains valid particle data and configuration settings.

"""

class Integrator:
	split_n_max: int = 16_384
	k_soft: float = 0.0
	mu_soft: float = 1.0
	chi_eps: float = 1.0

	def __init__(self, sim: "NBodySimulation", *, split_n_max: int = 10000) -> None:
		self.sim = sim
		self.split_n_max = int(split_n_max)

		self._eps_prev = None
		self._dt_prev = None
		self._last_update_tick = 0
		self._cached_min_sep = None
		self._top_dt = None
		self._substeps_in_last_step = 0
		self._refresh_calls_in_last_step = 0
		self._last_tr_hessian = 0.0

		self._pos_hash = np.int64(np.sum(sim._pos, dtype=np.float64))
		self._dV_cache_key = None
		self._eps_tick = 0

		self._uv_solver = UniversalVariableKeplerSolver()

		self._scheme: IntegrationScheme = self._make_scheme(sim._integrator_mode)

		self._ts = TimestepManager(self)

		dt_user = float(getattr(sim.cfg, "initial_dt", 1.0e-2))
		self._ts.init_substep_schedule(dt_user)

	def _make_scheme(self, mode: str) -> IntegrationScheme:
		if mode == "yoshida4":
			return Yoshida4Scheme(self)
		if mode == "whfast":
			return WHFastScheme(self)
		return VerletScheme(self)

	@property
	def h_sub_ref(self) -> float:
		return float(self._ts.h_sub_ref)

	@h_sub_ref.setter
	def h_sub_ref(self, value: float) -> None:
		self._ts.h_sub_ref = float(value)

	def step(self, dt: float) -> None:
		if dt == 0.0 or self.sim.n_bodies == 0:
			return

		sim = self.sim
		dt = float(dt)
		self._top_dt = abs(dt)

		h_sub = float(getattr(self, "h_sub_ref", 0.0) or 0.0)
		if not math.isfinite(h_sub) or h_sub <= 0.0:
			self._ts.init_substep_schedule(abs(dt))
			h_sub = float(getattr(self, "h_sub_ref", abs(dt)) or abs(dt))

		n_sub = int(max(1, min(self.split_n_max, math.ceil(abs(dt) / h_sub))))
		h_piece = dt / n_sub

		sim.manager.begin_step()
		self._substeps_in_last_step = 0
		self._refresh_calls_in_last_step = 0

		for _ in range(n_sub):
			self._strang_split_step(h_piece)
			self._substeps_in_last_step += 1

		sim.manager.finish_step()
		sim._acc_cached = False
		self._has_integrated = True

	def atomicstep(
		self,
		dt: float,
		*,
		depth: int = 0,
		do_refresh: bool = True,
	) -> None:
		self._substeps_in_last_step += 1
		self._last_update_tick += 1

		acc_old = self.sim._accel()
		self.sim._vel += 0.5 * dt * acc_old

		self.sim._pos += dt * self.sim._vel
		self._scheme.flag_positions_changed()

		acc_new = self.sim._accel()
		self.sim._vel += 0.5 * dt * acc_new
		self.sim._acc_cached = False

		_should_refresh = (
			do_refresh and
			self.sim._adaptive_softening
		)
		if _should_refresh:
			min_r = self._ts.get_cached_min_sep()
			eps_new = self.sim.manager.softening_from_min_sep(min_r)
			self.sim.manager.refresh_softening(eps_new, self.sim)
			self._refresh_calls_in_last_step += 1

		self.sim.manager.commit_substep()



	def apply_corrector(self, order: int) -> None:

		self._scheme.apply_corrector(order)

	def compute_extended_hamiltonian(self) -> float:

		diag = Diagnostics(self.sim, integrator=self)
		return diag.compute_extended_hamiltonian()

	def _current_phase(self) -> PhaseState:

		sim = self.sim
		p = (sim._mass[:, None] * sim._vel).copy()
		return PhaseState(
			q=sim._pos.copy(),
			p=p,
			epsilon=float(getattr(sim, "_epsilon", sim.manager.s)),
			pi=float(getattr(sim, "_pi", 0.0)),
			m=sim._mass.copy(),
		)

	def _barrier_n(self) -> int:

		return int(self.sim.cfg.barrier_exponent)

	def _eps_target(self, q: np.ndarray | None = None, **kwargs) -> float:

		sim = self.sim

		mgr = getattr(sim, "manager", None)
		if mgr is not None:
			s0 = getattr(mgr, "s0", None)
			if isinstance(s0, (int, float, np.floating)):
				s0f = float(s0)
				if np.isfinite(s0f) and s0f > 0.0:
					return s0f

		val = getattr(sim, "_softening_scale", None)
		if isinstance(val, (int, float, np.floating)):
			valf = float(val)
			if np.isfinite(valf) and valf > 0.0:
				return valf

		eps_cur = getattr(sim, "_epsilon", None)
		if isinstance(eps_cur, (int, float, np.floating)):
			epsf = float(eps_cur)
			if np.isfinite(epsf) and epsf > 0.0:
				return epsf

		return 0.0

	def _ensure_buffer(
		self,
		name: str,
		shape: Tuple[int, int],
		*,
		dtype: np.dtype | str = np.float64,
	) -> np.ndarray:
		return self._scheme.ensure_buffer(name, shape, dtype=dtype)

	def _strang_split_step(self, dt_piece: float) -> None:
		dt_piece = float(dt_piece)
		sim = self.sim

		do_refresh = (sim._adaptive_softening and sim._integrator_mode != "ham_soft")
		mode = getattr(sim, "_integrator_mode", "verlet")

		if mode == "yoshida4":
			self._yoshida4(dt_piece)
			if do_refresh:
				min_r = self._ts.get_cached_min_sep()
				eps_new = sim.manager.softening_from_min_sep(min_r)
				sim.manager.refresh_softening(eps_new, sim)
				self._refresh_calls_in_last_step += 1
			sim.manager.commit_substep()
			return

		if mode == "whfast":
			self._wisdom_holman(dt_piece)
			if do_refresh:
				min_r = self._ts.get_cached_min_sep()
				eps_new = sim.manager.softening_from_min_sep(min_r)
				sim.manager.refresh_softening(eps_new, sim)
				self._refresh_calls_in_last_step += 1
			sim.manager.commit_substep()
			return

		self.atomicstep(dt_piece, do_refresh=do_refresh)



	def _determine_substeps(self, dt_abs: float) -> int:
		return self._ts.determine_substeps(dt_abs)

	def _init_substep_schedule(self, dt_user: float) -> None:
		self._ts.init_substep_schedule(dt_user)

	def _enforce_stability(self, h: float) -> tuple[bool, int]:
		return self._ts.enforce_stability(h)

	def _estimate_h(self, dt_max: float) -> float:
		return self._ts.estimate_h(dt_max)

	def _predict_min_separation(self, dt: float) -> float:
		return self._ts.predict_min_separation(dt)

	def _get_cached_min_sep(self) -> float:
		return self._ts.get_cached_min_sep()

	def drift(self, h_coeff: float) -> None:
		self._scheme.drift(h_coeff)

	def kick(self, dt: float) -> None:
		self._scheme.kick(dt)

	def _apply_V_operator(self, dt: float) -> None:
		self._scheme._apply_V_operator(dt)

	def _verlet(self, h: float) -> None:
		meth = getattr(self._scheme, "_verlet", None)
		(meth or self._scheme._verlet_kernel)(h)


	def t_full(self, h: float) -> None:
		self.drift(float(h))

	def _t_full(self, h: float) -> None:
		self.t_full(float(h))

	def _yoshida4(self, h: float) -> None:
		h = float(h)
		meth = getattr(self._scheme, "_yoshida4", None)
		if callable(meth):
			meth(h)
			return
		cbrt2 = 2.0 ** (1.0 / 3.0)
		w1 = 1.0 / (2.0 - cbrt2)
		w2 = -cbrt2 / (2.0 - cbrt2)
		self._scheme._verlet_kernel(w1 * h)
		self._scheme._verlet_kernel(w2 * h)
		self._scheme._verlet_kernel(w1 * h)


	def _wisdom_holman(self, h: float) -> None:
		meth = getattr(self._scheme, "_wisdom_holman", None)
		if meth:
			meth(h)
		else:
			print("WHFast not available; falling back to Verlet")
			self._scheme._verlet_kernel(h)

	def _kepler_propagate(self, r, v, mu, dt):
		return self._uv_solver.propagate(r, v, mu, dt)


