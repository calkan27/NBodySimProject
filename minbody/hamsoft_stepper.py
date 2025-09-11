from __future__ import annotations
import numpy as np
from .hamsoft_flows import PhaseState, spring_oscillation
from .forces import dV_d_epsilon, gravitational_force as _grav_force, softened_forces as _soft_forces
from .barrier import barrier_force, barrier_curvature
from .hamsoft_utils import reflect_if_needed, reflect_and_limit_eps
from .hamsoft_flows import _osc_segment_update, _compute_eps_wall_hits, s_half_momentum_increment, pi_half_kick, s_flow
from .potential import dU_d_eps as _dU_d_eps


"""
This module implements the core stepping methods for the Hamiltonian softening integrator. The HamSoftStepper class provides s_half for half-step softening evolution, v_half_kick for velocity updates with current epsilon, t_drift for position updates, strang_step for complete Strang-split timesteps, and various specialized methods like s_full for testing. The implementation carefully manages the interplay between softening evolution and particle dynamics, handles both reflection and soft barrier boundary conditions, and maintains diagnostic information about each substep. It assumes the parent integrator provides valid force calculations and epsilon target functions.



"""





class HamSoftStepper:
	def __init__(self, owner) -> None:
		self.integ = owner

	def _get_j_max_cap(self) -> float:

		default = 0.02
		cfg = getattr(self.integ.sim, "cfg", None)
		if cfg is None:
			return float(default)

		val = getattr(cfg, "j_max_cap", default)
		if isinstance(val, (int, float, np.floating)):
			v = float(val)
			if np.isfinite(v) and v > 0.0:
				return float(v)
		return float(default)

	def _sflow_half(self, h: float) -> None:
		sim = self.integ.sim
		dt_half = 0.5 * float(h)

		m = sim._mass
		q = sim._pos
		p = (m[:, None] * sim._vel).copy()

		state_in = PhaseState(
			q=q.copy(),
			p=p.copy(),
			epsilon=float(getattr(sim, "_epsilon", sim.manager.s)),
			pi=float(getattr(sim, "_pi", 0.0)),
			m=m.copy(),
		)

		state_out = spring_oscillation(
			state_in,
			dt_half,
			float(getattr(self.integ, "k_soft", 0.0)),
			mu=float(getattr(self.integ, "mu_soft", 1.0)),
			integrator=self.integ,
			q_frozen=q.copy(),
		)

		pol = getattr(self.integ, "barrier_policy", "reflection")
		barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))
		eps_fin = float(state_out.epsilon)
		pi_fin  = float(state_out.pi)
		if isinstance(pol, str) and pol.lower() == "reflection" and (not barrier_disabled):
			eps_min = float(getattr(sim, "_min_softening", 0.0))
			eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
			e_r, p_r = reflect_if_needed(float(eps_fin), float(pi_fin), float(eps_min), float(eps_max))
			eps_fin, pi_fin = float(e_r), float(p_r)

		sim._vel[:]  = state_out.p / m[:, None]
		sim._epsilon = float(eps_fin)
		sim._pi      = float(pi_fin)
		sim.manager.update_continuous(sim._epsilon)
		sim._acc_cached = False

		return

	def s_half(self, h: float) -> None:
		sim = self.integ.sim
		h_f = float(h)

		eps_now = float(getattr(sim, "_epsilon", sim.manager.s))
		pi_now  = float(getattr(sim, "_pi", 0.0))

		pol_attr = getattr(self.integ, "barrier_policy", "reflection")
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"

		barrier_disabled = False
		if hasattr(sim, "cfg"):
			barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))

		if pol == "reflection":
			if not barrier_disabled:
				eps_can, pi_can = self.integ._barrier.reflect_and_bounce(float(eps_now), float(pi_now), 0.0)
			else:
				eps_can, pi_can = float(eps_now), float(pi_now)
		else:
			eps_can, pi_can = float(eps_now), float(pi_now)

		sim._epsilon = float(eps_can)
		sim._pi      = float(pi_can)
		sim.manager.update_continuous(sim._epsilon)

		if bool(getattr(sim.cfg, "freeze_s_subsystem", False)):
			if hasattr(self.integ, "_last_s_info"):
				self.integ._last_s_half = dict(self.integ._last_s_info)
			else:
				self.integ._last_s_half = {}
			return

		q_frozen   = sim._pos.copy()
		eps0_input = float(sim._epsilon)
		pi0_input  = float(sim._pi)
		mu_eff     = float(getattr(self.integ, "mu_soft", 1.0))
		k_soft_val = float(getattr(self.integ, "k_soft", 0.0))
		dt_half    = 0.5 * h_f

		self._sflow_half(h_f)

		info = getattr(self.integ, "_last_s_info", None)
		has_complete = False
		if isinstance(info, dict):
			if ("I_tau" in info) and ("J" in info) and ("grad_used" in info):
				self.integ._last_s_half = dict(info)
				has_complete = True

		if not has_complete:
			if hasattr(self.integ, "_eps_target"):
				eps_star_val = self.integ._eps_target(q_frozen)
			else:
				eps_star_val = sim.manager.s0
			if isinstance(eps_star_val, (int, float, np.floating)):
				eps_star = float(eps_star_val) if np.isfinite(float(eps_star_val)) else float(sim.manager.s0)
			else:
				eps_star = float(sim.manager.s0)

			g_try = self.integ._grad_eps_target(q_frozen)
			if isinstance(g_try, np.ndarray) and g_try.shape == q_frozen.shape:
				grad = np.asarray(g_try, dtype=float)
				if not np.all(np.isfinite(grad)):
					grad = np.where(np.isfinite(grad), grad, 0.0)
			else:
				grad = np.zeros_like(q_frozen, dtype=float)

			barrier_enabled = False
			include_curv = False
			if isinstance(pol, str) and pol == "soft":
				if hasattr(sim, "cfg"):
					if not bool(getattr(sim.cfg, "disable_barrier", False)):
						barrier_enabled = True
			if hasattr(sim, "cfg"):
				include_curv = bool(getattr(sim.cfg, "include_barrier_curvature_in_S", False))

			k_eff = float(k_soft_val)
			if barrier_enabled and include_curv:
				eps_min = float(getattr(sim, "_min_softening", 0.0))
				if hasattr(sim, "manager"):
					eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
				else:
					eps_max = max(1.0, eps_min)
				k_wall = float(getattr(self.integ, "k_wall", 0.0))
				n_exp = 2
				if hasattr(self.integ, "_barrier_n"):
					nv = self.integ._barrier_n()
					if isinstance(nv, (int, float, np.floating)):
						n_exp = int(nv)
				if (k_wall > 0.0) and (n_exp >= 2):
					Bpp = float(barrier_curvature(float(eps0_input), float(eps_min), float(eps_max),
												  k_wall=float(k_wall), n=int(n_exp)))
					if np.isfinite(Bpp):
						if Bpp >= 0.0:
							k_eff = float(k_soft_val + Bpp)

			if (k_eff > 0.0) and (mu_eff > 0.0):
				omega = float(np.sqrt(k_eff / mu_eff))
			else:
				omega = 0.0
			theta = float(omega * dt_half)


			Delta0 = float(eps0_input - eps_star)
			if (omega != 0.0) and (mu_eff != 0.0):
				if abs(theta) < 1.0e-8:
					th  = theta
					th2 = th * th
					th3 = th2 * th
					th4 = th2 * th2
					th5 = th4 * th
					sin_th = th - th3 / 6.0 + th5 / 120.0
					cos_th = 1.0 - th2 / 2.0 + th4 / 24.0
				else:
					sin_th = float(np.sin(theta))
					cos_th = float(np.cos(theta))
				denom = float(mu_eff * omega * omega)
				if denom != 0.0:
					I_tau = float((Delta0 / omega) * sin_th + (pi0_input / denom) * (1.0 - cos_th))
				else:
					I_tau = 0.0
			else:
				I_tau = 0.0

			J_val = float(k_soft_val) * float(I_tau)

			if pol == "soft":
				b1 = 0.0
				b2 = 0.0
			else:
				b1 = 0.0
				b2 = 0.0

			self.integ._last_s_half = {
				"I_tau": float(I_tau),
				"J": float(J_val),
				"grad_used": grad.copy(),
				"eps_star": float(eps_star),
				"omega": float(omega),
				"theta": float(theta),
				"k_eff": float(k_eff),
				"barrier_kick1": float(b1),
				"barrier_kick2": float(b2),
			}

		return



	def t_drift(self, h: float) -> None:
		self.integ.drift(float(h))
		return


	def strang_step(self, h: float) -> None:
		h_f = float(h)
		sim = self.integ.sim

		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))
		pi0  = float(getattr(sim, "_pi", 0.0))

		pol_attr = getattr(self.integ, "barrier_policy", "reflection")
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"
		barrier_disabled = bool(getattr(sim, "cfg", None) and getattr(sim.cfg, "disable_barrier", False))

		if (pol == "reflection") and (not barrier_disabled):
			eps_c0, pi_c0 = self.integ._barrier.reflect_and_bounce(float(eps0), float(pi0), 0.0)
			sim._epsilon = float(eps_c0)
			sim._pi      = float(pi_c0)
		else:
			sim._epsilon = float(eps0)
			sim._pi      = float(pi0)
		sim.manager.update_continuous(sim._epsilon)

		validate_s_only = bool(getattr(sim.cfg, "_validate_S_only", False)) if hasattr(sim, "cfg") else False
		if validate_s_only:
			self.s_half(h_f)
			self.s_half(h_f)
			eps_e = float(getattr(sim, "_epsilon", sim.manager.s))
			pi_e  = float(getattr(sim, "_pi", 0.0))
			if (pol == "reflection") and (not barrier_disabled):
				eps_ce, pi_ce = self.integ._barrier.reflect_and_bounce(float(eps_e), float(pi_e), 0.0)
				sim._epsilon = float(eps_ce)
				sim._pi      = float(pi_ce)
			else:
				sim._epsilon = float(eps_e)
				sim._pi      = float(pi_e)
			sim.manager.update_continuous(sim._epsilon)
			return

		self.s_half(h_f)

		eps_for_v1 = float(getattr(sim, "_epsilon", sim.manager.s))
		self.v_half_kick(h_f, eps_override=float(eps_for_v1))

		self.t_drift(h_f)

		eps_for_v2 = float(getattr(sim, "_epsilon", sim.manager.s))
		self.v_half_kick(h_f, eps_override=float(eps_for_v2))

		self.s_half(h_f)

		eps1 = float(getattr(sim, "_epsilon", sim.manager.s))
		pi1  = float(getattr(sim, "_pi", 0.0))
		if (pol == "reflection") and (not barrier_disabled):
			eps_c1, pi_c1 = self.integ._barrier.reflect_and_bounce(float(eps1), float(pi1), 0.0)
			sim._epsilon = float(eps_c1)
			sim._pi      = float(pi_c1)
		else:
			sim._epsilon = float(eps1)
			sim._pi      = float(pi1)
		sim.manager.update_continuous(sim._epsilon)
		return

	  
	def s_full(self, h: float) -> None:
		sim = self.integ.sim
		h_f = float(h)

		m = sim._mass
		q = sim._pos
		p = (m[:, None] * sim._vel).copy()

		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))
		pi0  = float(getattr(sim, "_pi", 0.0))

		k_soft = float(getattr(self.integ, "k_soft", 0.0))
		mu     = float(getattr(self.integ, "mu_soft", 1.0))
		chi_eps = float(getattr(self.integ, "chi_eps", 1.0))

		eps_star_val = self.integ._eps_target(q)
		eps_star = float(eps_star_val) if isinstance(eps_star_val, (int, float, np.floating)) else float(sim.manager.s0)

		g_try = self.integ._grad_eps_target(q)
		if isinstance(g_try, np.ndarray) and g_try.shape == q.shape:
			grad = np.asarray(g_try, dtype=float)
			if not np.all(np.isfinite(grad)):
				grad = np.where(np.isfinite(grad), grad, 0.0)
		else:
			grad = np.zeros_like(q, dtype=float)

		grad_row_norm = 0.0
		if grad.size > 0:
			row_norms = np.sqrt(np.sum(grad * grad, axis=1))
			if row_norms.size > 0:
				grad_row_norm = float(np.max(row_norms))
		self.integ._last_eps_star_info = {
			"eps_star": float(eps_star),
			"grad": grad.copy(),
			"grad_norm_max": float(grad_row_norm),
		}

		if (not np.isfinite(k_soft)) or (not np.isfinite(mu)) or (k_soft <= 0.0) or (mu <= 0.0) or (h_f == 0.0):
			p_new = p
			eps_rot = float(eps0 + (pi0 / mu) * h_f) if mu != 0.0 else float(eps0)
			pi_rot  = float(pi0)
			theta_log = 0.0
			sin_theta = 0.0
			cos_theta = 1.0
			one_minus_cos = 0.0
		else:
			omega = float(np.sqrt(k_soft / mu))
			theta = float(omega * h_f)
			if abs(theta) < 1.0e-8:
				th  = theta
				th2 = th * th
				th3 = th2 * th
				th4 = th2 * th2
				th5 = th4 * th
				sin_theta = float(th  - th3 / 6.0  + th5 / 120.0)
				cos_theta = float(1.0 - th2 / 2.0  + th4 / 24.0)
			else:
				sin_theta = float(np.sin(theta))
				cos_theta = float(np.cos(theta))
			one_minus_cos = float(1.0 - cos_theta)

			Delta0   = float(eps0 - eps_star)
			mu_omega = float(np.sqrt(mu * k_soft))

			if (omega != 0.0) and (mu != 0.0):
				J_unscaled = float((Delta0 / omega) * sin_theta + (pi0 / (mu * omega * omega)) * one_minus_cos)
				delta_t    = float(Delta0 * cos_theta + (pi0 / (mu * omega)) * sin_theta)
				eta_t      = float(pi0 * cos_theta   - mu_omega * Delta0 * sin_theta)
			else:
				J_unscaled = 0.0
				delta_t    = float(Delta0)
				eta_t      = float(pi0)

			J     = float(chi_eps) * float(k_soft) * float(J_unscaled)
			p_new = p + float(J) * grad
			eps_rot = float(eps_star + delta_t)
			pi_rot  = float(eta_t)
			theta_log = float(theta)

		eps_min = float(getattr(sim, "_min_softening", 0.0))
		eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0)) if hasattr(sim, "manager") else max(1.0, eps_min)

		pol_attr = getattr(self.integ, "barrier_policy", None)
		pol = pol_attr.lower() if isinstance(pol_attr, str) else "reflection"
		barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))

		if (pol == "reflection") and (not barrier_disabled):
			eps_fin, pi_fin = reflect_if_needed(float(eps_rot), float(pi_rot), float(eps_min), float(eps_max))
		else:
			eps_fin, pi_fin = float(eps_rot), float(pi_rot)

		sim._vel[:]  = p_new / m[:, None]
		sim._epsilon = float(eps_fin)
		sim._pi      = float(pi_fin)
		sim.manager.update_continuous(sim._epsilon)
		sim._acc_cached = False

		self.integ._last_s_trig = {
			"theta": float(theta_log if (k_soft > 0.0 and mu > 0.0 and h_f != 0.0) else 0.0),
			"sin": float(sin_theta),
			"cos": float(cos_theta),
			"one_minus_cos": float(one_minus_cos),
		}


	def s_full_centered(self, h: float) -> None:

		sim = self.integ.sim
		h_f = float(h)

		m = sim._mass
		q = sim._pos
		p = (m[:, None] * sim._vel).copy()

		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))
		pi0  = float(getattr(sim, "_pi", 0.0))

		k_soft = float(getattr(self.integ, "k_soft", 0.0))
		mu     = float(getattr(self.integ, "mu_soft", 1.0))

		eps_star_val = self.integ._eps_target(q)
		if isinstance(eps_star_val, (int, float, np.floating)):
			if np.isfinite(float(eps_star_val)):
				eps_star = float(eps_star_val)
			else:
				eps_star = float(sim.manager.s0)
		else:
			eps_star = float(sim.manager.s0)

		g_try = self.integ._grad_eps_target(q)
		if isinstance(g_try, np.ndarray) and g_try.shape == q.shape:
			grad = np.asarray(g_try, dtype=float)
			if not np.all(np.isfinite(grad)):
				grad = np.where(np.isfinite(grad), grad, 0.0)
		else:
			grad = np.zeros_like(q, dtype=float)

		eps_min = float(getattr(sim, "_min_softening", 0.0))
		eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		if eps_max < eps_min:
			tmp = eps_min
			eps_min = eps_max
			eps_max = tmp
		if eps_star < eps_min:
			eps_star = float(eps_min)
			grad[:] = 0.0
		elif eps_star > eps_max:
			eps_star = float(eps_max)
			grad[:] = 0.0

		if (k_soft <= 0.0) or (mu <= 0.0) or (h_f == 0.0):
			if mu != 0.0:
				eps_new = float(eps0 + (pi0 / mu) * h_f)
			else:
				eps_new = float(eps0)
			pi_new = float(pi0)
			p_new  = p

		else:
			hp = np.longdouble if not hasattr(np, "float128") else np.float128
			mu_hp = hp(mu)
			ks_hp = hp(k_soft)
			h_hp  = hp(h_f)

			omega_hp = np.sqrt(ks_hp / mu_hp) if (float(ks_hp) > 0.0 and float(mu_hp) > 0.0) else hp(0.0)
			theta_hp = omega_hp * h_hp
			abs_th = float(np.abs(theta_hp))

			if abs_th < 1.0e-8:
				th  = theta_hp
				th2 = th * th
				th3 = th2 * th
				th4 = th2 * th2
				th5 = th4 * th
				sin_theta_hp = th  - th3 / hp(6.0)  + th5 / hp(120.0)
				cos_theta_hp = hp(1.0) - th2 / hp(2.0) + th4 / hp(24.0)
			else:
				sin_theta_hp = np.sin(theta_hp)
				cos_theta_hp = np.cos(theta_hp)

			if float(theta_hp) == 0.0:
				sinc_hp = hp(1.0)
				omc_over_theta_hp = hp(0.0)
			else:
				sinc_hp = sin_theta_hp / theta_hp
				omc_over_theta_hp = (hp(1.0) - cos_theta_hp) / theta_hp

			delta0_hp = hp(eps0) - hp(eps_star)
			eta0_hp   = hp(pi0)
			mu_omega_hp = np.sqrt(mu_hp * ks_hp)

			zeta0_hp   = eta0_hp / mu_omega_hp if float(mu_omega_hp) != 0.0 else hp(0.0)
			delta_h_hp = delta0_hp * cos_theta_hp + zeta0_hp * sin_theta_hp
			pi_h_hp    = eta0_hp * cos_theta_hp   - mu_omega_hp * delta0_hp * sin_theta_hp
			eps_h_hp   = hp(eps_star) + delta_h_hp

			I_tau_hp = h_hp * (delta0_hp * sinc_hp + zeta0_hp * omc_over_theta_hp)
			coef_hp  = ks_hp * I_tau_hp   

			p_new  = p + float(coef_hp) * grad
			eps_new = float(eps_h_hp)
			pi_new  = float(pi_h_hp)

		sim._vel[:]  = p_new / m[:, None]
		sim._epsilon = float(eps_new)
		sim._pi      = float(pi_new)
		sim.manager.update_continuous(sim._epsilon)
		sim._acc_cached = False

		pol_attr = getattr(self.integ, "barrier_policy", None)
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"
		if not getattr(sim.cfg, "disable_barrier", False):
			if pol == "reflection":
				eps_fin, pi_fin = reflect_if_needed(float(sim._epsilon), float(sim._pi), float(eps_min), float(eps_max))
				sim._epsilon = float(eps_fin)
				sim._pi      = float(pi_fin)
				sim.manager.update_continuous(sim._epsilon)
		return


	

	
	def ts_centered(self, h: float) -> None:
		h_f = float(h)
		self.integ.drift(h_f)
		return


	def v_half_kick(self, h: float, eps_override: float | None = None) -> None:
		sim = self.integ.sim
		h_half = 0.5 * float(h)

		pol_attr = getattr(self.integ, "barrier_policy", None)
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"

		use_test_override = False
		if hasattr(sim, "cfg"):
			use_test_override = bool(getattr(sim.cfg, "_allow_v_eps_override", False))

		if eps_override is not None:
			eps_used = float(eps_override)
		else:
			if use_test_override:
				forced = getattr(self.integ, "force_epsilon_override", None)
				if isinstance(forced, (int, float, np.floating)):
					eps_used = float(forced)
				else:
					has_op = hasattr(self.integ, "_epsilon_for_v_operator")
					if has_op:
						eps_used = float(self.integ._epsilon_for_v_operator(float(getattr(self.integ, "epsilon", sim.manager.s))))
					else:
						eps_used = float(getattr(sim, "_epsilon", sim.manager.s))
			else:
				eps_used = float(getattr(sim, "_epsilon", sim.manager.s))

		m = sim._mass
		q = sim._pos
		n = sim.n_bodies
		Gf = float(sim.G)

		p = (m[:, None] * sim._vel).copy()
		if n >= 2:
			if Gf != 0.0:
				F = _grav_force(q, m, eps=float(eps_used), G=float(Gf))
			else:
				F = np.zeros_like(q, dtype=float)
		else:
			F = np.zeros_like(q, dtype=float)
		p = p + float(h_half) * F
		sim._vel[:] = p / m[:, None]
		sim._last_force_eps = float(eps_used)
		self.integ._last_force_eps = float(eps_used)
		sim._acc_cached = False

		if bool(getattr(sim.cfg, "freeze_s_subsystem", False)):
			self.integ._last_vkick = {
				"epsilon_used": float(eps_used),
				"dVgrav_deps": 0.0,
				"dSbar_deps": 0.0,
				"dV_total_deps": 0.0,
				"dt_half": float(h_half),
			}
			return

		em = getattr(self.integ, "_eps_model", None)
		if em is not None:
			if hasattr(em, "eps_target"):
				eps_star_val = em.eps_target(q)
			else:
				eps_star_val = sim.manager.s0
		else:
			if hasattr(self.integ, "_eps_target"):
				eps_star_val = self.integ._eps_target(q)
			else:
				eps_star_val = sim.manager.s0

		if isinstance(eps_star_val, (int, float, np.floating)):
			if np.isfinite(float(eps_star_val)):
				eps_star = float(eps_star_val)
			else:
				eps_star = float(sim.manager.s0)
		else:
			eps_star = float(sim.manager.s0)

		k_soft_val = float(getattr(self.integ, "k_soft", 0.0))
		pi_now = float(self.integ.pi)

		dUbar_deps = 0.0
		barrier_disabled = False
		if hasattr(sim, "cfg"):
			barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))
		if pol == "soft":
			if not barrier_disabled:
				eps_min_b = float(getattr(sim, "_min_softening", 0.0))
				if hasattr(sim, "manager"):
					eps_max_b = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
				else:
					eps_max_b = max(1.0, float(eps_min_b))
				k_wall_b = float(getattr(self.integ, "k_wall", 0.0))
				n_exp_b = 2
				if hasattr(self.integ, "_barrier_n"):
					nv_b = self.integ._barrier_n()
					if isinstance(nv_b, (int, float, np.floating)):
						n_exp_b = int(nv_b)
				if (k_wall_b > 0.0):
					if (n_exp_b >= 2):
						F_bar = barrier_force(float(eps_used), float(eps_min_b), float(eps_max_b),
											  k_wall=float(k_wall_b), n=int(n_exp_b))
						dUbar_deps = -float(F_bar)

		pi_new, dU_ret, _k_unused = pi_half_kick(
			q, m, Gf, float(eps_used), float(pi_now), float(eps_star),
			float(k_soft_val), float(h_half),
			dB_d_eps=float(dUbar_deps),
		)
		self.integ.pi = float(pi_new)

		dV_total_deps = float(dU_ret + dUbar_deps)
		self.integ._last_vkick = {
			"epsilon_used": float(eps_used),
			"dVgrav_deps": float(dU_ret),
			"dSbar_deps": float(dUbar_deps),
			"dV_total_deps": float(dV_total_deps),
			"dt_half": float(h_half),
		}
		return


