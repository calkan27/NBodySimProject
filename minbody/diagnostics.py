from __future__ import annotations
import itertools, math
import numpy as np
from .hamsoft_energy import extended_hamiltonian  
from .barrier import barrier_energy
from typing import TYPE_CHECKING
from .geometry_cache import geometry_buffers
if TYPE_CHECKING:                       
    from .integrator import Integrator  

"""
This module computes and monitors conserved quantities and system health metrics during N-body simulations. The Diagnostics class provides methods for calculating kinetic and potential energies with proper softening treatment, extended Hamiltonian for the ham_soft integrator including spring and barrier terms, angular and linear momentum, center of mass position and velocity, energy breakdowns by component, and per-step metrics including COM drift and momentum variance. It includes specialized support for high-precision energy calculations using extended precision arithmetic, energy conservation monitoring with configurable tolerance checking, and rate-limited diagnostic printing to avoid console spam. The module deeply integrates with both classical and Hamiltonian softening integrators, handling the different energy formulations required by each approach. It assumes access to simulation state through standardized interfaces and maintains internal caches for efficiency.

"""




class Diagnostics:
	_GLOBAL_DIAG_COUNTS = {}

	def __init__(self, simulation, integrator: "Integrator" | None = None):
		self.sim = simulation
		if integrator is not None:
			self._integ = integrator
		else:
			self._integ = getattr(simulation, "_integrator", None)
		
		if self._integ:
			self._k_wall = getattr(self._integ, "k_wall", 1e9)
		else:
			self._k_wall = 1e9

		pref = getattr(simulation.cfg, "energy_tol_pref", None)
		if pref is None:
			if hasattr(simulation, "_pos") and hasattr(simulation._pos, "dtype"):
				if simulation._pos.dtype == np.float32:
					self._tol_pref = 5e-6
				else:
					self._tol_pref = 1e-7
			else:
				self._tol_pref = 1e-7
		else:
			self._tol_pref = float(pref)

		self._H0_mod = None


	def kinetic_energy(self):
		s = 0.0
		for b in self.sim.bodies:
			s += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy)
		return s

	def potential_energy(self):
		s = 0.0
		G = self.sim.G
		s2 = self.sim.manager.step_s2
		for a, b in itertools.combinations(self.sim.bodies, 2):
			dx = b.x - a.x
			dy = b.y - a.y
			r = math.sqrt(dx * dx + dy * dy + s2)
			s -= G * a.mass * b.mass / r
		return s


	def energy(self):
		sim = self.sim

		eps = float(getattr(sim, "_epsilon", sim.manager.s))

		m = sim._mass
		v = sim._vel
		T = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))

		n = sim.n_bodies
		if n >= 2 and sim.G != 0.0:
			pos = sim._pos
			diff = pos[:, None, :] - pos[None, :, :]
			r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
			np.fill_diagonal(r2, np.inf)
			r2_soft = r2 + eps * eps
			inv_r = 1.0 / np.sqrt(r2_soft)
			iu = np.triu_indices(n, 1)
			V_grav = -sim.G * float(np.sum((m[:, None] * m[None, :] * inv_r)[iu]))
		else:
			V_grav = 0.0

		integ = getattr(sim, "_integrator", None)

		if integ is not None:
			pol_raw = getattr(integ, "barrier_policy", "reflection")
			pol = pol_raw.lower() if isinstance(pol_raw, str) else "reflection"
			barrier_enabled = (pol == "soft") and (not getattr(sim.cfg, "disable_barrier", False))
			if barrier_enabled:
				eps_min = float(getattr(sim, "_min_softening", 0.0))
				eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
				k_wall = float(getattr(integ, "k_wall", 1.0e9))
				if hasattr(integ, "_barrier_n"):
					n_exp = int(integ._barrier_n())
				else:
					n_exp = int(getattr(sim.cfg, "barrier_exponent", 5))
				if (k_wall > 0.0) and (n_exp >= 2):
					S_bar = barrier_energy(eps, eps_min, eps_max, k_wall=k_wall, n=n_exp)
				else:
					S_bar = 0.0
			else:
				S_bar = 0.0
		else:
			S_bar = 0.0

		if integ is not None:
			mu_soft = float(getattr(integ, "mu_soft", 1.0))
			k_soft = float(getattr(integ, "k_soft", 0.0))
			pi = float(getattr(sim, "_pi", 0.0))

			if mu_soft != 0.0:
				K_eps = 0.5 * (pi * pi) / mu_soft
			else:
				K_eps = 0.0

			if hasattr(integ, "_eps_model") and integ._eps_model is not None:
				eps_star = integ._eps_model.eps_target(sim._pos)
			elif hasattr(integ, "_eps_target"):
				eps_star = integ._eps_target(q=sim._pos)
			else:
				eps_star = sim.manager.s0

			if not isinstance(eps_star, (int, float, np.floating)) or not np.isfinite(float(eps_star)):
				eps_star = sim.manager.s0

			eps_star = float(eps_star)

			S_spring = 0.5 * k_soft * (eps - eps_star) ** 2

			S_harm = K_eps + S_spring
		else:
			S_harm = 0.0

		H_ext = T + V_grav + S_bar + S_harm
		return float(H_ext)


	def energy_breakdown(self) -> dict:
		sim, m, v = self.sim, self.sim._mass, self.sim._vel

		T = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))

		if getattr(sim, "_integrator_mode", None) == "ham_soft":
			eps_cur = float(getattr(sim, "_epsilon", sim.manager.s))
			s2 = eps_cur * eps_cur
		else:
			s2_raw = getattr(sim.manager, "step_s2", None)
			s2_val = None
			if isinstance(s2_raw, (int, float, np.floating)):
				cand = float(s2_raw)
				if np.isfinite(cand):
					s2_val = cand
			elif isinstance(s2_raw, str):
				parsed = np.fromstring(s2_raw.strip(), dtype=float, sep=' ')
				if parsed.size == 1:
					if np.isfinite(parsed[0]):
						s2_val = float(parsed[0])
			if s2_val is None:
				eps_cur = float(getattr(sim, "_epsilon", sim.manager.s))
				s2 = eps_cur * eps_cur
			else:
				s2 = float(s2_val)

		if sim.n_bodies >= 2:
			if sim.G != 0.0:
				pos  = sim._pos
				diff = pos[:, None, :] - pos[None, :, :]
				r2   = np.einsum("ijk,ijk->ij", diff, diff)
				np.fill_diagonal(r2, np.inf)
				inv_r = 1.0 / np.sqrt(r2 + s2)
				iu    = np.triu_indices(sim.n_bodies, 1)
				V     = -float(sim.G) * float(np.sum((m[:, None] * m[None, :] * inv_r)[iu]))
			else:
				V = 0.0
		else:
			V = 0.0

		pi    = float(getattr(sim, "_pi", 0.0))
		mu    = float(getattr(sim._integrator, "mu_soft", 1.0))
		K_eps = 0.5 * (pi * pi) / mu

		k_s   = float(getattr(sim._integrator, "k_soft", 0.0))
		eps   = float(getattr(sim, "_epsilon", sim.manager.s))

		if getattr(sim, "_integrator", None) is not None:
			integ = sim._integrator
			eps_star = None

			if hasattr(integ, "_eps_model"):
				em = getattr(integ, "_eps_model")
				if em is not None:
					if hasattr(em, "eps_target"):
						val_model = em.eps_target(sim._pos)
						if isinstance(val_model, (int, float, np.floating)):
							if np.isfinite(float(val_model)):
								eps_star = float(val_model)

			if eps_star is None:
				if hasattr(integ, "_eps_target"):
					eps_star_val = integ._eps_target(q=sim._pos)
					if isinstance(eps_star_val, (int, float, np.floating)):
						if np.isfinite(float(eps_star_val)):
							eps_star = float(eps_star_val)

			if eps_star is None:
				eps_star = float(sim.manager.s0)
		else:
			eps_star = float(sim.manager.s0)

		if k_s > 0.0:
			PE_spring = 0.5 * k_s * (eps - eps_star) ** 2
		else:
			PE_spring = 0.0

		return dict(T=T, V=V, K_eps=K_eps, PE_spring=PE_spring, H=T + V + K_eps + PE_spring)





	def step_metrics(self, megno_slope_history: list[float] | None = None) -> dict:
		sim, m, pos, vel = self.sim, self.sim._mass, self.sim._pos, self.sim._vel

		com_vec   = np.sum(m[:, None] * pos, axis=0)
		com_drift = float(np.linalg.norm(com_vec))

		eps = getattr(sim, "_epsilon", sim.manager.s)
		pi  = getattr(sim, "_pi", 0.0)
		mu  = getattr(sim._integrator, "mu_soft", 1.0)
		J_eps   = float(eps * pi / mu)
		if mu*eps or pi:
			theta_e = math.atan2(pi, mu * eps)
		else:
			theta_e = float("nan")

		L_i   = m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
		L_tot = float(np.sum(L_i))
		var_L = float(np.var(L_i))
		if not hasattr(self, "_L0"):
			self._L0 = L_tot
		if self._L0 and L_tot:
			cos_theta = (L_tot * self._L0) / (abs(L_tot) * abs(self._L0))
		else:
			cos_theta = float("nan")

		tr_hess = getattr(sim._integrator, "_last_tr_hessian", float("nan"))

		if megno_slope_history:
			med_megno = float(np.median(megno_slope_history))
		else:
			med_megno = float("nan")

		e_parts = self.energy_breakdown()

		return dict(
			com_drift=com_drift,
			J_eps=J_eps,
			L_tot=L_tot,
			var_L=var_L,
			cos_theta=cos_theta,
			tr_hessian=tr_hess,
			megno_slope_med=med_megno,
			theta_eps=theta_e,
			**e_parts,
		)


	def energy_guard(self, dt: float) -> None:
		cfg = self.sim.cfg
		if not getattr(cfg, "enable_runtime_guard", False):
			return

		interval = int(getattr(cfg, "invariant_check_interval", 2000))
		self._step_idx = getattr(self, "_step_idx", 0) + 1
		if self._step_idx % interval:
			return

		if not hasattr(self, "_phase_ref"):
			self._phase_ref = self._integ._current_phase()
			self._geom_buf  = {}

			eps_min = self.sim._min_softening
			eps_max = 10.0 * self.sim.manager.s0

			H0 = extended_hamiltonian(
				self._phase_ref,
				G=self.sim.G,
				k_soft=self._integ.k_soft,
				mu_soft=self._integ.mu_soft,
				eps_star=self._integ._eps_target(),   
				eps_min=eps_min,
				eps_max=eps_max,
				integrator=self._integ,               
				barrier_enabled=not self.sim.cfg.disable_barrier,
			)
			self._H0_mod = H0
			return

		phase_now = self._integ._current_phase()

		eps_now  = phase_now.epsilon
		eps_min  = self.sim._min_softening
		eps_max  = 10.0 * self.sim.manager.s0

		use_barrier = False
		integ = self._integ
		if integ is not None:
			pol = getattr(integ, "barrier_policy", "reflection")
			if isinstance(pol, str) and pol.lower() == "soft":
				if not self.sim.cfg.disable_barrier:
					use_barrier = True

		if use_barrier:
			k_wall = float(getattr(integ, "k_wall", 0.0))
			n_exp  = int(integ._barrier_n()) if hasattr(integ, "_barrier_n") else 0
			if k_wall > 0.0 and n_exp >= 2:
				U_bar = barrier_energy(eps_now, eps_min, eps_max, k_wall=k_wall, n=n_exp)
			else:
				U_bar = 0.0
		else:
			U_bar = 0.0

		m   = phase_now.m
		p   = phase_now.p
		T   = 0.5 * float(np.sum(np.sum(p * p, axis=1) / m))

		q = phase_now.q
		n_bodies = q.shape[0]
		if n_bodies < 2 or self.sim.G == 0.0:
			U = 0.0
		else:
			if "diff" not in self._geom_buf or self._geom_buf["diff"].shape[0] != n_bodies:
				diff, r2, _ = geometry_buffers(q, phase_now.epsilon)
				self._geom_buf.update(diff=diff, r2=r2)
			else:
				diff = self._geom_buf["diff"]
				r2   = self._geom_buf["r2"]
				diff[:] = q[:, None, :] - q[None, :, :]
				np.einsum("ijk,ijk->ij", diff, diff, out=r2)
				r2 += phase_now.epsilon ** 2
				np.fill_diagonal(r2, np.inf)

			iu        = np.triu_indices(n_bodies, 1)
			inv_r     = 1.0 / np.sqrt(r2[iu])
			m_i, m_j  = m[iu[0]], m[iu[1]]
			U         = -self.sim.G * float(np.sum(m_i * m_j * inv_r))

		k_soft   = self._integ.k_soft
		eps_star = self._integ._eps_target()
		if k_soft > 0.0:
			S = 0.5 * k_soft * (eps_now - eps_star) ** 2
		else:
			S = 0.0
		K_eps = 0.5 * (phase_now.pi ** 2) / self._integ.mu_soft

		H_now = T + U + S + U_bar + K_eps

		tol = self._tol_pref * dt * dt
		if abs(H_now - self._H0_mod) > tol:
			print(
				f"[energy_guard] |ΔH_ext| = {abs(H_now - self._H0_mod):.3e}"
				f" > tol = {tol:.3e}"
			)
			return


	def _rate_limited_diag_print(self, key: str, msg: str) -> None:

		sim = getattr(self, "sim", None)
		cfg = getattr(sim, "cfg", None)

		if cfg is None:
			enabled = True
		else:
			enabled = bool(getattr(cfg, "diag_prints", True))
		if not enabled:
			return

		if cfg is None:
			limit = 3
		else:
			limit = int(getattr(cfg, "diag_print_limit", 3))
		if cfg is None:
			interval = 1000
		else:
			interval = int(getattr(cfg, "diag_print_interval", 1000))
		if limit < 0:
			limit = 0
		if interval < 1:
			interval = 1

		counts = Diagnostics._GLOBAL_DIAG_COUNTS
		c = counts.get(key, 0) + 1
		counts[key] = c

		if (c <= limit) or (c % interval == 0):
			if c <= limit:
				suffix = ""
			else:
				suffix = f" (occurrence #{c})"
			print(msg + suffix)


	def _kahan_sum_hp(self, arr_hp):
		a = np.asarray(arr_hp).ravel()
		if hasattr(a, "dtype"):
			dtype = a.dtype.type
		else:
			dtype = float
		s = dtype(0.0)
		c = dtype(0.0)
		i = 0
		n = a.size
		while i < n:
			x = a[i]
			y = x - c
			t = s + y
			c = (t - s) - y
			s = t
			i += 1
		return s

	def _clip_and_cast_hp(self, x_hp, name: str, f64_max_hp, f64_max) -> float:
		if not np.isfinite(x_hp):
			self._rate_limited_diag_print(name, f"[diag] non-finite {name}; setting to 0.0")
			return 0.0
		if x_hp > f64_max_hp:
			self._rate_limited_diag_print(name, f"[diag] clipped {name} to +f64_max")
			return f64_max
		if x_hp < -f64_max_hp:
			self._rate_limited_diag_print(name, f"[diag] clipped {name} to -f64_max")
			return -f64_max
		return float(x_hp)



	def compute_extended_hamiltonian(self) -> float:
		sim    = self.sim
		integ  = self._integ or getattr(sim, "_integrator", None)

		if hasattr(np, "float128"):
			hp = np.float128
		else:
			hp = np.longdouble
		f64_max = np.finfo(float).max
		f64_max_hp = hp(f64_max)

		m_hp = np.asarray(sim._mass, dtype=hp)
		v_hp = np.asarray(sim._vel,  dtype=hp)
		v2_hp = np.sum(v_hp * v_hp, axis=1, dtype=hp)
		T_hp = self._kahan_sum_hp(hp(0.5) * m_hp * v2_hp)

		n = sim.n_bodies
		eps = hp(getattr(sim, "_epsilon", sim.manager.s))
		if n >= 2 and sim.G != 0.0:
			pos_hp = np.asarray(sim._pos, dtype=hp)
			diff = pos_hp[:, None, :] - pos_hp[None, :, :]
			r2   = np.einsum("ijk,ijk->ij", diff, diff, optimize=True, dtype=hp)
			iu   = np.triu_indices(n, 1)
			r2_iu = r2[iu] + eps * eps
			r2_iu = np.where(r2_iu > hp(0.0), r2_iu, hp(1e-300))
			inv_r = hp(1.0) / np.sqrt(r2_iu, dtype=hp)
			m_i = m_hp[iu[0]]
			m_j = m_hp[iu[1]]
			pair_terms = m_i * m_j * inv_r
			V_hp = hp(-sim.G) * self._kahan_sum_hp(pair_terms)
		else:
			V_hp = hp(0.0)

		if integ:
			k_soft = hp(getattr(integ, "k_soft", 0.0))
		else:
			k_soft = hp(0.0)
		if integ:
			mu_soft = hp(getattr(integ, "mu_soft", 1.0))
		else:
			mu_soft = hp(1.0)

		eps_star_val = None
		if integ is not None:
			em = getattr(integ, "_eps_model", None)
			if em is not None and hasattr(em, "eps_target"):
				eps_star_val = em.eps_target(sim._pos)
			elif hasattr(integ, "_eps_target"):
				eps_star_val = integ._eps_target(q=sim._pos)

		if not isinstance(eps_star_val, (int, float, np.floating)) or not np.isfinite(float(eps_star_val)):
			eps_star_val = float(sim.manager.s0)

		eps_star = hp(float(eps_star_val))

		pi_hp = hp(getattr(sim, "_pi", 0.0))
		if mu_soft == 0.0 or (not np.isfinite(mu_soft)):
			return 1e300

		K_eps_hp  = hp(0.5) * (pi_hp * pi_hp) / mu_soft
		delta_eps = eps - eps_star
		S_spr_hp  = hp(0.5) * self._kahan_sum_hp(np.array([k_soft * (delta_eps * delta_eps)], dtype=hp))

		S_bar_hp = hp(0.0)
		if integ is not None:
			pol_raw = getattr(integ, "barrier_policy", "reflection")
			if isinstance(pol_raw, str):
				pol = pol_raw.lower()
			else:
				pol = "reflection"
			if pol == "soft" and not getattr(sim.cfg, "disable_barrier", False):
				k_wall_attr = getattr(integ, "k_wall", 0.0)
				n_exp_val   = 2
				if hasattr(integ, "_barrier_n"):
					bn = integ._barrier_n()
					if isinstance(bn, (int, float, np.floating)):
						n_exp_val = int(bn)

				k_wall = float(k_wall_attr)
				if np.isfinite(k_wall) and (k_wall > 0.0) and (n_exp_val >= 2):
					eps_min = float(getattr(sim, "_min_softening", 0.0))
					eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
					eps_now = float(eps)
					S_bar_f = barrier_energy(eps_now, eps_min, eps_max, k_wall=float(k_wall), n=int(n_exp_val))
					S_bar_hp = hp(S_bar_f)

		T     = self._clip_and_cast_hp(T_hp,     "T",      f64_max_hp, f64_max)
		V     = self._clip_and_cast_hp(V_hp,     "V",      f64_max_hp, f64_max)
		K_eps = self._clip_and_cast_hp(K_eps_hp, "K_eps",  f64_max_hp, f64_max)
		S_spr = self._clip_and_cast_hp(S_spr_hp, "S_spring", f64_max_hp, f64_max)
		S_bar = self._clip_and_cast_hp(S_bar_hp, "S_bar",  f64_max_hp, f64_max)

		return T + V + K_eps + S_spr + S_bar



	def angular_momentum(self):
		s = 0.0
		for b in self.sim.bodies:
			s += b.mass * (b.x * b.vy - b.y * b.vx)
		return s

	def linear_momentum(self):
		px = 0.0
		py = 0.0
		for b in self.sim.bodies:
			px += b.mass * b.vx
			py += b.mass * b.vy
		return px, py

	def center_of_mass(self):
		M = 0.0
		for b in self.sim.bodies:
			M += b.mass
		if M == 0.0:
			return (0.0, 0.0), (0.0, 0.0)
		xs = 0.0
		ys = 0.0
		for b in self.sim.bodies:
			xs += b.mass * b.x
			ys += b.mass * b.y
		x_cm = xs / M
		y_cm = ys / M
		px, py = self.linear_momentum()
		vx_cm = px / M
		vy_cm = py / M
		return (x_cm, y_cm), (vx_cm, vy_cm)
