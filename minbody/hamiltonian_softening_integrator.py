from __future__ import annotations
import math
import numpy as np
from .integrator import Integrator
from .hamsoft_params import HamSoftParams
from .hamsoft_eps_model import EpsilonModel
from .hamsoft_barrier_controller import HamSoftBarrier
from .hamsoft_stepper import HamSoftStepper
from .barrier import barrier_force
from .hamsoft_constants import LAMBDA_SOFTENING as _LAM_SOFT
from .softening import grad_eps_target as _grad_ss, eps_target as _eps_target_ss
from .barrier import barrier_curvature
from .potential import dU_d_eps as _dU_d_eps
from .forces import gravitational_force  
from .forces import dV_d_epsilon as _dVdeps
from .hamsoft_flows import s_flow as _s_flow

""""
This advanced integrator implements a symplectic integration scheme for N-body systems with adaptive softening as a Hamiltonian degree of freedom. The HamiltonianSofteningIntegrator extends the base Integrator with epsilon (softening) and pi (conjugate momentum) as dynamical variables, implements Strang splitting for the coupled particle-softening dynamics, includes spring potentials to guide epsilon toward optimal values, supports both soft barrier and reflection boundary conditions, and provides extensive calibration methods for the effective mass mu_soft. The integrator uses multiple internal classes (HamSoftParams, EpsilonModel, HamSoftBarrier, HamSoftStepper) to organize functionality. It maintains detailed diagnostics of the extended phase space evolution and carefully manages the interplay between gravitational dynamics and softening evolution. The implementation assumes the simulation uses compatible configuration settings and that the softening degree of freedom is physically meaningful for the system.


"""




class HamiltonianSofteningIntegrator(Integrator):

	k_soft: float = 0.0
	mu_soft: float = 1.0
	chi_eps: float = 1.0
	k_wall: float = 1.0e9

	def __init__(self, sim, *, split_n_max: int = 10000, force_adaptive_timestep: bool = False) -> None:
		super().__init__(sim, split_n_max=split_n_max)

		cap = int(getattr(sim.cfg, "split_n_max", split_n_max))
		self.split_n_max = max(1, cap)

		self._params = HamSoftParams(self)
		self.k_soft  = float(getattr(self._params, "k_soft", 0.0) or 0.0)
		self.mu_soft = float(getattr(self._params, "mu_soft", 1.0) or 1.0)
		self.chi_eps = float(getattr(self._params, "chi_eps", 1.0) or 1.0)
		self.k_wall  = float(getattr(self._params, "k_wall", 1.0e9) or 1.0e9)

		cfg_cap_raw = getattr(sim.cfg, "split_n_max", None)
		if isinstance(cfg_cap_raw, (int, float, np.floating)):
			cfg_cap = int(cfg_cap_raw)
			if cfg_cap > 0:
				self.split_n_max = int(cfg_cap)

		self._eps_model = EpsilonModel(owner=self)

		self._barrier_policy_override = None

		self._eps_model.calibrate_from_initial_conditions()

		_fixed_override = False
		_fixed_value = None
		cfg_ref = getattr(sim, "cfg", None)
		if cfg_ref is not None:
			if bool(getattr(cfg_ref, "fixed_eps_star", False)):
				v_fixed = getattr(cfg_ref, "eps_star_value", None)
				if isinstance(v_fixed, (int, float, np.floating)):
					vf = float(v_fixed)
					if np.isfinite(vf):
						sim._epsilon = float(vf)
						if hasattr(sim, "manager"):
							if hasattr(sim.manager, "update_continuous"):
								sim.manager.update_continuous(sim._epsilon)
						sim._pi = 0.0
						_fixed_override = True
						_fixed_value = float(vf)

		sim = self.sim
		use_soft_attr = None
		if hasattr(sim, "cfg"):
			if hasattr(sim.cfg, "use_soft_barrier"):
				use_soft_attr = bool(getattr(sim.cfg, "use_soft_barrier", False))
			else:
				use_soft_attr = None
		else:
			use_soft_attr = None

		_barrier_disabled = False
		if hasattr(sim, "cfg"):
			_barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))

		if bool(use_soft_attr):
			if not _barrier_disabled:
				self._barrier_policy = "soft"
			else:
				self._barrier_policy = "reflection"
		else:
			self._barrier_policy = "reflection"

		G = float(sim.G)
		M_tot = float(np.sum(sim._mass)) if isinstance(sim._mass, np.ndarray) else 0.0
		eps_min = float(getattr(sim, "_min_softening", sim.manager.s0 * 0.1))
		if not np.isfinite(eps_min) or eps_min <= 0.0:
			eps_min = max(sim.manager.s0 * 0.1, 1.0e-12)

		if not np.isfinite(self.k_soft) or self.k_soft <= 0.0:
			c_k = 8.0
			self.k_soft = c_k * G * M_tot * M_tot / (eps_min * eps_min * eps_min)

		self._calibrate_mu_from_timescales()

		self._barrier    = HamSoftBarrier(self)
		self._hs_stepper = HamSoftStepper(self)
		self._stepper    = self._hs_stepper

		self.force_epsilon_override = None
		if _fixed_override:
			self.force_epsilon_override = float(_fixed_value)

		self._frozen_schedule_active = False
		self._frozen_h   = None
		self._frozen_n_sub = None
		self._omega_spr0 = None

		self.force_adaptive_timestep = bool(force_adaptive_timestep)
		self.sim._adaptive_timestep  = bool(force_adaptive_timestep)

		self._freeze_production_schedule(float(getattr(sim.cfg, "initial_dt", 1.0e-2)))

		self._init_substep_schedule(float(getattr(sim.cfg, "initial_dt", 1.0e-2)))
		self.refreshes_per_strang_substep = 0



	def _calibrate_mu_from_pi_budget(self, dt: float) -> None:
		sim = self.sim

		dt_abs = float(abs(float(dt)))
		if (not np.isfinite(dt_abs)) or (dt_abs <= 0.0):
			if isinstance(getattr(self, "_top_dt", None), (int, float, np.floating)):
				cand = float(abs(self._top_dt))
			else:
				cand = float(abs(getattr(sim.cfg, "initial_dt", 1.0e-2))) if hasattr(sim, "cfg") else 1.0e-2
			if (not np.isfinite(cand)) or (cand <= 0.0):
				dt_abs = 1.0e-2
			else:
				dt_abs = cand

		q = np.asarray(sim._pos, dtype=float)
		m = np.asarray(sim._mass, dtype=float)
		Gf = float(sim.G)

		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))
		pi0  = float(getattr(sim, "_pi", 0.0))

		eps_star_val = self._eps_target(q=q)
		if isinstance(eps_star_val, (int, float, np.floating)):
			eps_star = float(eps_star_val) if np.isfinite(float(eps_star_val)) else float(sim.manager.s0)
		else:
			eps_star = float(sim.manager.s0)

		delta = float(eps0 - eps_star)

		if q.ndim == 2 and q.shape[0] >= 2 and Gf != 0.0:
			dUgrav_deps = float(_dVdeps(q, m, float(eps0), float(Gf)))
		else:
			dUgrav_deps = 0.0

		dUbar_deps = 0.0
		pol_attr = getattr(self, "barrier_policy", None)
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"
		barrier_enabled = False
		if pol == "soft":
			if hasattr(sim, "cfg"):
				if not bool(getattr(sim.cfg, "disable_barrier", False)):
					barrier_enabled = True
		if barrier_enabled:
			eps_min = float(getattr(sim, "_min_softening", 0.0))
			eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
			k_wall_val = float(getattr(self, "k_wall", 1.0e9))
			n_eff = 0
			if hasattr(self, "_barrier_n"):
				n_val = self._barrier_n()
				if isinstance(n_val, (int, float, np.floating)):
					n_eff = int(n_val)
			if n_eff < 2:
				if hasattr(sim, "cfg"):
					n_eff = int(getattr(sim.cfg, "barrier_exponent", 5))
				else:
					n_eff = 5
			if (k_wall_val > 0.0) and (n_eff >= 2):
				F_bar = float(barrier_force(float(eps0), float(eps_min), float(eps_max),
											k_wall=float(k_wall_val), n=int(n_eff)))
				dUbar_deps = -float(F_bar)

		d_total = float(dUgrav_deps + dUbar_deps)
		dpi_macro = -float(d_total) * float(dt_abs)

		k = float(getattr(self, "k_soft", 0.0))
		if (not np.isfinite(k)) or (k <= 0.0):
			return

		chi_pi = 0.2
		if hasattr(sim, "cfg"):
			val = getattr(sim.cfg, "chi_pi", 0.2)
			chi_pi = float(val)
		if (not np.isfinite(chi_pi)) or (chi_pi <= 0.0):
			chi_pi = 0.2

		bound = float(chi_pi) * float(np.sqrt(k)) * float(abs(delta))

		theta_imp = getattr(sim.cfg, "theta_imp", 0.5)
		if not isinstance(theta_imp, (int, float, np.floating)):
			theta_imp = 0.5
		theta_imp = float(theta_imp)
		if (not np.isfinite(theta_imp)) or (theta_imp <= 0.0):
			theta_imp = 0.5

		mu_macro = float(k) * (float(dt_abs) / float(theta_imp)) ** 2

		mu_cur_attr = getattr(self, "mu_soft", 1.0)
		if isinstance(mu_cur_attr, (int, float, np.floating)):
			mu_cur = float(mu_cur_attr)
		else:
			mu_cur = 1.0
		if (not np.isfinite(mu_cur)) or (mu_cur <= 0.0):
			mu_cur = 1.0
		if mu_cur < float(mu_macro):
			self.mu_soft = float(mu_macro)

		if abs(float(dpi_macro)) > float(bound):
			pass
		return




	def _calibrate_mu_from_timescales(self) -> None:
		sim = self.sim
		G = float(sim.G)
		q0 = np.asarray(sim._pos, dtype=float)
		m = np.asarray(sim._mass, dtype=float)
		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))

		tau_grav = np.inf
		n = int(q0.shape[0]) if isinstance(q0, np.ndarray) and q0.ndim == 2 else 0
		if n >= 2 and G != 0.0:
			i = 0
			while i < n - 1:
				j = i + 1
				while j < n:
					dx = float(q0[j, 0] - q0[i, 0])
					dy = float(q0[j, 1] - q0[i, 1])
					r2 = dx*dx + dy*dy + eps0*eps0
					if r2 > 0.0 and np.isfinite(r2):
						r = float(np.sqrt(r2))
						omega_ij = np.sqrt(G * (float(m[i]) + float(m[j])) / (r2 * r))
						if np.isfinite(omega_ij) and omega_ij > 0.0:
							t = 1.0 / omega_ij
							if t < tau_grav:
								tau_grav = t
					j = j + 1
				i = i + 1
		if (not np.isfinite(tau_grav)) or (tau_grav <= 0.0):
			tau_grav = 1.0

		k = float(getattr(self, "k_soft", 0.0))
		if (not np.isfinite(k)) or (k <= 0.0):
			k = 0.0
		c_omega = 8.0
		omega_spr = float(c_omega) / float(tau_grav) if tau_grav > 0.0 else 0.0

		if omega_spr > 0.0:
			mu_choice = k / (omega_spr * omega_spr) if k > 0.0 else 1.0
		else:
			mu_choice = 1.0

		if (not np.isfinite(mu_choice)) or (mu_choice <= 0.0):
			mu_choice = 1.0
		self.mu_soft = float(mu_choice)

		self._omega_spr0 = float(omega_spr)
		return



	def report_epsilon_policies(self) -> dict:

		if hasattr(self, "_last_eps_eff_eom"):
			eom_eps = float(self._last_eps_eff_eom)
		else:
			eom_eps = float("nan")

		vkick_eps = float("nan")

		lv = getattr(self, "_last_vkick", None)
		if isinstance(lv, dict):
			if "epsilon_used" in lv:
				val = lv.get("epsilon_used")
				if isinstance(val, (int, float, np.floating)):
					vkick_eps = float(val)

		if (not np.isfinite(vkick_eps)) and hasattr(self, "_last_force_eps"):
			val2 = getattr(self, "_last_force_eps")
			if isinstance(val2, (int, float, np.floating)):
				vkick_eps = float(val2)

		return {
			"eom_eps_eff": eom_eps,
			"vkick_eps_eff": vkick_eps,
		}


	@property
	def soft_mgr(self):
		return self.sim.manager

	def s_half(self, h: float) -> None:
		self._stepper.s_half(float(h))

	def _s_half(self, h: float) -> None:
		self._stepper.s_half(float(h))

	def _v_half_kick(self, h: float) -> None:
		self._stepper.v_half_kick(float(h))

	def _last_vkick_probe(self):
		rec = getattr(self, "_last_vkick", None)
		if isinstance(rec, dict):
			out = dict(rec)
			if "epsilon_used" in out:
				out["eps_used"] = float(out["epsilon_used"])
			elif "eps_used" in out:
				out["epsilon_used"] = float(out["eps_used"])
			else:
				fallback_eps = None
				val = getattr(self, "_last_force_eps", None)
				if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
					fallback_eps = float(val)
				else:
					fallback_eps = float(getattr(self.sim, "_epsilon", self.sim.manager.s))
				out["epsilon_used"] = float(fallback_eps)
				out["eps_used"]     = float(fallback_eps)
			return out
		pos = getattr(self.sim, "_pos", None)
		if isinstance(pos, np.ndarray):
			qshape = tuple(pos.shape)
		else:
			qshape = (0, 0)
		eps_now = float(getattr(self.sim, "_epsilon", self.sim.manager.s))
		return dict(
			q_ref_shape=qshape,
			epsilon_used=eps_now,
			eps_used=eps_now,
			dU_d_eps_used=0.0,
			F_bar_used=0.0,
			dt2=0.0,
		)

	@property
	def epsilon(self) -> float:
		return float(self.sim._epsilon)

	@epsilon.setter
	def epsilon(self, value: float) -> None:
		self.sim._epsilon = float(value)
		self.sim.manager.update_continuous(self.sim._epsilon)

	@property
	def pi(self) -> float:
		return float(self.sim._pi)

	@pi.setter
	def pi(self, value: float) -> None:
		self.sim._pi = float(value)

	@property
	def k_soft(self) -> float:
		return float(self._params.k_soft)

	@k_soft.setter
	def k_soft(self, v: float) -> None:
		self._params.k_soft = float(v)

	@property
	def mu_soft(self) -> float:
		return float(self._params.mu_soft)

	@mu_soft.setter
	def mu_soft(self, v: float) -> None:
		self._params.mu_soft = float(v)

	@property
	def chi_eps(self) -> float:
		val = getattr(self._params, "chi_eps", 1.0)
		if isinstance(val, (int, float, np.floating)):
			vf = float(val)
		else:
			vf = 1.0
		if (not np.isfinite(vf)) or (vf <= 0.0):
			return 1.0
		if vf > 1.0:
			return 1.0
		return float(vf)

	@chi_eps.setter
	def chi_eps(self, v: float) -> None:
		if isinstance(v, (int, float, np.floating)):
			x = float(v)
		else:
			x = 1.0
		if (not np.isfinite(x)) or (x <= 0.0):
			x = 1.0
		if x > 1.0:
			x = 1.0
		self._params.chi_eps = float(x)


	@property
	def k_wall(self) -> float:
		return float(self._params.k_wall)

	@k_wall.setter
	def k_wall(self, v: float) -> None:
		self._params.k_wall = float(v)


	def _barrier_n(self) -> int:
		n = int(getattr(self._params, "barrier_exponent", 0) or 0)
		if n <= 1:
			n = int(getattr(self.sim.cfg, "barrier_exponent", 5))
		return n

	@property
	def barrier_policy(self) -> str:
		ov = getattr(self, "_barrier_policy_override", None)
		if isinstance(ov, str):
			low = ov.lower()
			if low == "soft":
				return "soft"
			if low == "reflection":
				return "reflection"

		sim_ref = getattr(self, "sim", None)
		if sim_ref is not None:
			use_soft = bool(getattr(sim_ref.cfg, "use_soft_barrier", False))
			barrier_disabled = bool(getattr(sim_ref.cfg, "disable_barrier", False))
			if use_soft:
				if not barrier_disabled:
					return "soft"
				else:
					return "reflection"
			else:
				return "reflection"

		val = getattr(self, "_barrier_policy", None)
		if isinstance(val, str):
			low = val.lower()
			if low == "soft":
				return "soft"
		return "reflection"

	@barrier_policy.setter
	def barrier_policy(self, v: str) -> None:
		if isinstance(v, str):
			low = v.lower()
			if low == "soft" or low == "reflection":
				self._barrier_policy_override = low
				self._barrier_policy = low
				return
		print("HamiltonianSofteningIntegrator.barrier_policy must be 'reflection' or 'soft'; keeping previous value")











	def step(self, dt: float) -> None:
		sim = self.sim
		if dt == 0.0 or sim.n_bodies == 0:
			return

		dt = float(dt)
		self._top_dt = float(abs(dt))

		n_pred = int(self.strang_substeps(dt))
		if n_pred < 1:
			n_pred = 1

		cap_attr = getattr(self, "split_n_max", n_pred)
		cap = int(cap_attr)
		if cap < 1:
			cap = 1

		sim._freeze_softening_for_hamsoft()
		sim.manager.begin_step()

		self._substeps_in_last_step = 0
		self._refresh_calls_in_last_step = 0
		self._total_substeps_in_last_step = 0

		h_piece = dt / float(n_pred)

		if n_pred <= cap:
			i = 0
			while i < n_pred:
				self._hs_stepper.strang_step(h_piece)
				i = i + 1
			self._substeps_in_last_step = int(n_pred)
			self._total_substeps_in_last_step = int(n_pred)
		else:
			n_chunks = int(math.ceil(n_pred / float(cap)))

			total_done = 0
			c = 0
			while c < n_chunks:
				if c < n_chunks - 1:
					n_this = int(cap)
				else:
					n_this = int(n_pred - cap * (n_chunks - 1))
					if n_this < 1:
						n_this = 1

				j = 0
				while j < n_this:
					self._hs_stepper.strang_step(h_piece)
					j = j + 1

				self._substeps_in_last_step = int(n_this)
				total_done = total_done + int(n_this)
				c = c + 1

			self._total_substeps_in_last_step = int(total_done)

		sim.manager.finish_step()
		sim._unfreeze_softening_for_hamsoft()

		sim._acc_cached = False
		self._has_integrated = True

	def _barrier_force_if_active(self, eps: float) -> float:
		sim = self.sim

		if getattr(sim.cfg, "disable_barrier", False):
			return 0.0

		k_wall = float(getattr(self, "k_wall", 0.0))
		if not (np.isfinite(k_wall) and k_wall > 0.0):
			return 0.0

		n_exp = 0
		n_fun = getattr(self, "_barrier_n", None)
		if callable(n_fun):
			n_val = n_fun()
			if isinstance(n_val, (int, float, np.floating)):
				n_exp = int(n_val)
		if n_exp < 2:
			return 0.0

		pol = self.barrier_policy
		if pol == "reflection":
			return 0.0

		eps_min = float(getattr(sim, "_min_softening", 0.0))
		eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		return float(barrier_force(float(eps), eps_min, eps_max, k_wall=k_wall, n=int(n_exp)))



	def eps_star_and_grad(self, q: np.ndarray | None = None) -> tuple[float, np.ndarray]:

		sim = self.sim

		if not hasattr(self, "_eps_model") or (self._eps_model is None):
			self._eps_model = EpsilonModel(self)

		if q is None:
			if isinstance(getattr(sim, "_pos", None), np.ndarray):
				q_ref = np.asarray(sim._pos, dtype=float)
			else:
				q_ref = np.zeros((0, 2), dtype=float)
		else:
			q_ref = np.asarray(q, dtype=float)

		eps_star_val, grad_val = self._eps_model.eps_star_and_grad(q_ref)

		if isinstance(eps_star_val, (int, float, np.floating)):
			ef = float(eps_star_val)
			if not np.isfinite(ef):
				if hasattr(sim, "manager") and hasattr(sim.manager, "s0"):
					eps_star = float(sim.manager.s0)
				else:
					eps_star = float(getattr(sim, "_epsilon", 0.0))
			else:
				eps_star = float(ef)
		else:
			if hasattr(sim, "manager") and hasattr(sim.manager, "s0"):
				eps_star = float(sim.manager.s0)
			else:
				eps_star = float(getattr(sim, "_epsilon", 0.0))

		if isinstance(grad_val, np.ndarray) and grad_val.shape == q_ref.shape:
			grad = np.asarray(grad_val, dtype=float)
			if not np.all(np.isfinite(grad)):
				grad = np.where(np.isfinite(grad), grad, 0.0)
		else:
			grad = np.zeros_like(q_ref, dtype=float)

		return float(eps_star), grad





	def _eps_target(self, q: np.ndarray | None = None, **kwargs) -> float:

		sim = self.sim

		if not hasattr(self, "_eps_model") or (self._eps_model is None):
			self._eps_model = EpsilonModel(self)

		if q is None:
			q_ref = getattr(sim, "_pos", None)
			if isinstance(q_ref, np.ndarray):
				q_ref = np.asarray(q_ref, dtype=float)
			else:
				q_ref = None
		else:
			q_ref = np.asarray(q, dtype=float)

		val = None
		if hasattr(self._eps_model, "eps_target"):
			val = self._eps_model.eps_target(q_ref)

		if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
			return float(val)

		if hasattr(sim, "manager") and hasattr(sim.manager, "s0"):
			return float(sim.manager.s0)

		cur = getattr(sim, "_epsilon", 0.0)
		return float(cur)




	def _grad_eps_target(self, q: np.ndarray | None = None, **kwargs) -> np.ndarray:
		if q is None:
			pos_ref = getattr(self.sim, "_pos", None)
			if isinstance(pos_ref, np.ndarray):
				q_ref = np.asarray(pos_ref, dtype=float)
			else:
				q_ref = np.zeros((0, 2), dtype=float)
		else:
			q_ref = np.asarray(q, dtype=float)

		if hasattr(self, "_eps_model") and (self._eps_model is not None):
			g_raw = self._eps_model._production_grad(q_ref)
		else:
			g_raw = None

		if not isinstance(g_raw, np.ndarray):
			grad_out = np.zeros_like(q_ref, dtype=float)
		else:
			if g_raw.shape != q_ref.shape:
				grad_out = np.zeros_like(q_ref, dtype=float)
			else:
				g_tmp = np.asarray(g_raw, dtype=float)
				if not np.all(np.isfinite(g_tmp)):
					g_tmp = np.where(np.isfinite(g_tmp), g_tmp, 0.0)

				g_ref = None
				if q_ref.ndim == 2:
					if q_ref.shape[0] >= 2:
						alpha_align = 1.0
						lam_align = float(getattr(self.sim.cfg, "lambda_softening", _LAM_SOFT))
						if hasattr(self, "_eps_model"):
							if hasattr(self._eps_model, "_cfg_args"):
								a_l = self._eps_model._cfg_args()
								if isinstance(a_l, tuple):
									if len(a_l) == 2:
										# a_l = (alpha, lam)
										if isinstance(a_l[0], (int, float, np.floating)):
											alpha_align = float(a_l[0])
										if isinstance(a_l[1], (int, float, np.floating)):
											lam_align = float(a_l[1])

						try_ref = None
						try_ref = _grad_ss(q_ref, alpha=float(alpha_align), lam=float(lam_align))
						if isinstance(try_ref, np.ndarray):
							if try_ref.shape == q_ref.shape:
								gr = np.asarray(try_ref, dtype=float)
								if np.all(np.isfinite(gr)):
									g_ref = gr

				if isinstance(g_ref, np.ndarray):
					dot_total = float(np.sum(g_tmp * g_ref))
					if np.isfinite(dot_total):
						if dot_total < 0.0:
							g_tmp = -g_tmp

				grad_out = g_tmp

		eps_star_val = self._eps_target(q=q_ref)
		if isinstance(eps_star_val, (int, float, np.floating)):
			eps_star = float(eps_star_val)
		else:
			eps_star = float(getattr(self.sim.manager, "s0", 0.0))

		if not np.isfinite(eps_star):
			eps_star = float(getattr(self.sim.manager, "s0", 0.0))

		gmax = 0.0
		if isinstance(grad_out, np.ndarray):
			if grad_out.size > 0:
				row_norms = np.sqrt(np.sum(grad_out * grad_out, axis=1))
				if isinstance(row_norms, np.ndarray):
					if row_norms.size > 0:
						gmax = float(np.max(row_norms))

		self._last_eps_star_info = {
			"eps_star": float(eps_star),
			"grad": grad_out.copy(),
			"grad_norm_max": float(gmax),
		}

		return grad_out



	def _init_substep_schedule(self, dt_user: float) -> None:
		self._ts.init_substep_schedule(float(dt_user))


	def apply_corrector(self, order: int) -> None:
		return


	def _get_stepper(self):
		stepper = getattr(self, "_hamsoft_stepper", None)
		if (stepper is None) or (getattr(stepper, "integ", None) is not self):
			self._hamsoft_stepper = HamSoftStepper(self)
			stepper = self._hamsoft_stepper
		return stepper


	def v_half_kick(self, h: float) -> None:
		self._get_stepper().v_half_kick(h)


	def t_drift(self, h: float) -> None:
		self._get_stepper().t_drift(h)


	def _t_drift(self, h: float) -> None:
		self.t_drift(h)

	def _strang_split_step(self, h: float) -> None:
		self._hs_stepper.strang_step(float(h))



	def strang_substeps(self, dt: float) -> int:
		dt_abs = abs(float(dt))
		if dt_abs == 0.0:
			n_zero = 1
			self._theta_sub_half_last = 0.0
			self._last_strang_schedule_info = {
				"dt": 0.0,
				"n_sub": int(n_zero),
				"h_piece": 0.0,
				"omega_eff": float(getattr(self, "_omega_spr0", 0.0) or 0.0),
				"theta_sub_half": 0.0,
				"k_soft": float(getattr(self, "k_soft", 0.0)),
				"mu_soft": float(getattr(self, "mu_soft", 1.0)),
				"chi_g_used": float(getattr(self.sim.cfg, "safety_factor", 0.20)),
				"barrier_policy": self.barrier_policy,
				"h_sub_ref": float(getattr(self, "h_sub_ref", 0.0) or 0.0),
			}
			return int(n_zero)

		self._calibrate_mu_from_pi_budget(dt_abs)

		cfg_ref = getattr(self, "sim", None)
		cfg_obj = getattr(cfg_ref, "cfg", None) if cfg_ref is not None else None
		validate_s_only = bool(getattr(cfg_obj, "_validate_S_only", False)) if (cfg_obj is not None) else False
		if validate_s_only:
			k = float(getattr(self, "k_soft", 0.0))
			mu = float(getattr(self, "mu_soft", 1.0))
			if (not np.isfinite(mu)) or (mu <= 0.0):
				mu = 1.0
			if k > 0.0:
				omega_eff = float(np.sqrt(k / mu))
			else:
				omega_eff = 0.0
			h_piece = float(dt_abs)
			theta_sub_half = 0.5 * float(omega_eff) * float(h_piece)

			self._theta_sub_half_last = float(theta_sub_half)
			self._last_strang_schedule_info = {
				"dt": float(dt_abs),
				"n_sub": 1,
				"h_piece": float(h_piece),
				"omega_eff": float(omega_eff),
				"theta_sub_half": float(theta_sub_half),
				"k_soft": float(getattr(self, "k_soft", 0.0)),
				"mu_soft": float(getattr(self, "mu_soft", 1.0)),
				"chi_g_used": float(getattr(self.sim.cfg, "safety_factor", 0.20)),
				"barrier_policy": self.barrier_policy,
				"h_sub_ref": float(h_piece),
			}
			return 1

		macro_frozen = bool(getattr(self, "_macro_schedule_frozen", False))
		if macro_frozen:
			prev_dt = getattr(self, "_macro_dt_frozen", None)
			prev = float(prev_dt) if isinstance(prev_dt, (int, float, np.floating)) else None
			if (prev is not None):
				if prev > 0.0:
					rel = abs(dt_abs - prev) / prev
					if rel <= 0.01:
						n_frozen_attr = getattr(self, "_frozen_n_sub", None)
						n_frozen = int(n_frozen_attr) if isinstance(n_frozen_attr, (int, float, np.floating)) else 1
						if n_frozen < 1:
							n_frozen = 1
						omega_eff = float(getattr(self, "_omega_spr0", 0.0) or 0.0)
						h_piece = float(dt_abs) / float(n_frozen)

						theta_sub_half = 0.5 * float(omega_eff) * float(h_piece)
						self._theta_sub_half_last = float(theta_sub_half)
						self._last_strang_schedule_info = {
							"dt": float(dt_abs),
							"n_sub": int(n_frozen),
							"h_piece": float(h_piece),
							"omega_eff": float(omega_eff),
							"theta_sub_half": float(theta_sub_half),
							"k_soft": float(getattr(self, "k_soft", 0.0)),
							"mu_soft": float(getattr(self, "mu_soft", 1.0)),
							"chi_g_used": float(getattr(self.sim.cfg, "safety_factor", 0.20)),
							"barrier_policy": self.barrier_policy,
							"h_sub_ref": float(getattr(self, "h_sub_ref", 0.0) or 0.0),
						}
						return int(n_frozen)
			self._freeze_production_schedule(dt_abs)
		else:
			self._freeze_production_schedule(dt_abs)

		n_attr = getattr(self, "_frozen_n_sub", None)
		n_frozen = int(n_attr) if isinstance(n_attr, (int, float, np.floating)) else 1
		if n_frozen < 1:
			n_frozen = 1

		omega_eff = float(getattr(self, "_omega_spr0", 0.0) or 0.0)
		h_piece = float(dt_abs) / float(n_frozen)
		theta_sub_half = 0.5 * float(omega_eff) * float(h_piece)

		self._theta_sub_half_last = float(theta_sub_half)
		self._last_strang_schedule_info = {
			"dt": float(dt_abs),
			"n_sub": int(n_frozen),
			"h_piece": float(h_piece),
			"omega_eff": float(omega_eff),
			"theta_sub_half": float(theta_sub_half),
			"k_soft": float(getattr(self, "k_soft", 0.0)),
			"mu_soft": float(getattr(self, "mu_soft", 1.0)),
			"chi_g_used": float(getattr(self.sim.cfg, "safety_factor", 0.20)),
			"barrier_policy": self.barrier_policy,
			"h_sub_ref": float(getattr(self, "h_sub_ref", 0.0) or 0.0),
		}
		return int(n_frozen)








	def canonical_eom(self):
		sim = self.sim

		q = sim._pos
		m = sim._mass
		v = sim._vel
		p = m[:, None] * v

		eps = float(getattr(sim, "_epsilon", sim.manager.s))
		pi  = float(getattr(sim, "_pi", 0.0))

		n = int(sim.n_bodies)
		d = 2

		qdot = np.zeros((n, d), dtype=float)
		i = 0
		while i < n:
			mi = float(m[i])
			if mi != 0.0:
				qdot[i] = p[i] / mi
			else:
				qdot[i] = 0.0 * p[i]
			i = i + 1

		def _sanitize_grad(g, q_ref):
			g_arr = np.asarray(g, dtype=float)
			if not isinstance(g_arr, np.ndarray):
				return np.zeros_like(q_ref, dtype=float)
			if g_arr.shape != q_ref.shape:
				return np.zeros_like(q_ref, dtype=float)
			if not np.all(np.isfinite(g_arr)):
				g_arr = np.where(np.isfinite(g_arr), g_arr, 0.0)
			return g_arr

		eps_for_V = self._epsilon_for_v_operator(eps)
		self._last_eps_eff_eom = float(eps_for_V)

		if n >= 2 and float(sim.G) != 0.0:
			F_grav = gravitational_force(q, m, eps=float(eps_for_V), G=float(sim.G))
			dVgrav_deps = float(_dVdeps(q, m, float(eps_for_V), float(sim.G)))
		else:
			F_grav = np.zeros((n, d), dtype=float)
			dVgrav_deps = 0.0

		eps_star = float(self._eps_target(q=q))
		grad_eps_star = _sanitize_grad(self._grad_eps_target(q), q)

		k_soft = float(self.k_soft)
		Delta  = float(eps - eps_star)

		pdot = F_grav + float(k_soft) * float(Delta) * grad_eps_star

		if float(self.mu_soft) != 0.0:
			epsdot = float(pi) / float(self.mu_soft)
		else:
			epsdot = 0.0

		dUbar_deps = 0.0
		pol = self.barrier_policy
		if isinstance(pol, str):
			pol_low = pol.lower()
		else:
			pol_low = "reflection"

		if pol_low == "soft":
			cfg = getattr(sim, "cfg", None)
			barrier_disabled = bool(getattr(cfg, "disable_barrier", False)) if cfg is not None else False
			if not barrier_disabled:
				k_wall = float(getattr(self, "k_wall", 0.0))
				n_eff = 0
				if hasattr(self, "_barrier_n"):
					n_val = self._barrier_n()
					if isinstance(n_val, (int, float, np.floating)):
						n_eff = int(n_val)
				if n_eff < 2:
					n_eff = int(getattr(sim.cfg, "barrier_exponent", 5))
				if (k_wall > 0.0) and (n_eff >= 2):
					eps_min = float(getattr(sim, "_min_softening", 0.0))
					eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
					F_bar = float(barrier_force(float(eps), float(eps_min), float(eps_max),
												k_wall=float(k_wall), n=int(n_eff)))
					dUbar_deps = -F_bar

		pidot = -float(dVgrav_deps) - float(k_soft) * float(Delta) - float(dUbar_deps)

		return qdot, pdot, float(epsdot), float(pidot)



	def _freeze_production_schedule(self, dt_user: float) -> None:
		sim  = self.sim
		q0   = np.asarray(sim._pos, dtype=float)
		m    = np.asarray(sim._mass, dtype=float)
		eps0 = float(getattr(sim, "_epsilon", sim.manager.s))
		G    = float(sim.G)

		dt_abs = float(abs(float(dt_user)))
		if (not np.isfinite(dt_abs)) or (dt_abs <= 0.0):
			dt_abs = 1.0e-2

		tau_grav = np.inf
		n = int(q0.shape[0]) if isinstance(q0, np.ndarray) and q0.ndim == 2 else 0
		if n >= 2:
			if G != 0.0:
				i = 0
				while i < n - 1:
					j = i + 1
					while j < n:
						dx = float(q0[j, 0] - q0[i, 0])
						dy = float(q0[j, 1] - q0[i, 1])
						r2 = dx * dx + dy * dy + eps0 * eps0
						if r2 > 0.0:
							if np.isfinite(r2):
								r = float(np.sqrt(r2))
								omega_ij = np.sqrt(G * (float(m[i]) + float(m[j])) / (r2 * r))
								if np.isfinite(omega_ij):
									if omega_ij > 0.0:
										t = 1.0 / omega_ij
										if t < tau_grav:
											tau_grav = t
						j = j + 1
					i = i + 1
		if (not np.isfinite(tau_grav)) or (tau_grav <= 0.0):
			tau_grav = float(dt_abs)

		need_cal = False
		if not hasattr(self, "_omega_spr0"):
			need_cal = True
		else:
			w0 = getattr(self, "_omega_spr0", None)
			if isinstance(w0, (int, float, np.floating)):
				if not np.isfinite(float(w0)):
					need_cal = True
			else:
				need_cal = True
		if need_cal:
			self._calibrate_mu_from_timescales()

		omega_spr = getattr(self, "_omega_spr0", 0.0)
		if isinstance(omega_spr, (int, float, np.floating)):
			omega_spr = float(omega_spr)
		else:
			omega_spr = 0.0
		if (not np.isfinite(omega_spr)) or (omega_spr <= 0.0):
			c_omega = 8.0
			if tau_grav > 0.0:
				omega_spr = float(c_omega) / float(tau_grav)
			else:
				omega_spr = 0.0
			self._omega_spr0 = float(omega_spr)

		theta_cap = float(getattr(sim.cfg, "theta_cap", 0.1))
		if (not np.isfinite(theta_cap)) or (theta_cap <= 0.0):
			theta_cap = 0.1

		chi = 0.9
		h_theta_grav = float(chi) * float(tau_grav)
		if omega_spr > 0.0:
			h_theta_osc = float(theta_cap) / float(omega_spr)
		else:
			h_theta_osc = float("inf")

		if np.isfinite(h_theta_osc):
			if h_theta_osc > 0.0:
				h_theta = float(min(h_theta_grav, h_theta_osc))
			else:
				h_theta = float(h_theta_grav)
		else:
			h_theta = float(h_theta_grav)

		pol_attr = getattr(self, "barrier_policy", "reflection")
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"
		barrier_disabled = False
		if hasattr(sim, "cfg"):
			barrier_disabled = bool(getattr(sim.cfg, "disable_barrier", False))
		include_barrier = False
		if pol == "soft":
			if not barrier_disabled:
				include_barrier = True

		h_pi = self._estimate_pi_budget_h(q0, float(eps0), float(dt_abs), include_barrier=bool(include_barrier))
		if (not np.isfinite(h_pi)) or (h_pi <= 0.0):
			h_pi = float(dt_abs)

		h_sub = float(min(h_theta, h_pi))
		if (not np.isfinite(h_sub)) or (h_sub <= 0.0):
			h_sub = float(dt_abs)

		if h_sub > 0.0:
			n_sub = int(np.ceil(dt_abs / float(h_sub)))
		else:
			n_sub = 1
		if n_sub < 1:
			n_sub = 1

		self.h_sub_ref = float(dt_abs) / float(n_sub)
		self._frozen_schedule_active = True
		self._frozen_h = float(dt_abs)
		self._frozen_n_sub = int(n_sub)
		self._frozen_macro_h = float(self._frozen_n_sub) * float(self.h_sub_ref)
		self._macro_schedule_frozen = True
		self._macro_dt_frozen = float(dt_abs)

		omega_eff = float(omega_spr)
		theta_sub_half = 0.5 * float(omega_eff) * float(self.h_sub_ref)
		self._last_strang_schedule_info = {
			"dt": float(dt_abs),
			"n_sub": int(n_sub),
			"h_piece": float(self.h_sub_ref),
			"omega_eff": float(omega_eff),
			"theta_sub_half": float(theta_sub_half),
			"k_soft": float(getattr(self, "k_soft", 0.0)),
			"mu_soft": float(getattr(self, "mu_soft", 1.0)),
			"chi_g_used": 0.9,
			"barrier_policy": self.barrier_policy,
			"h_sub_ref": float(self.h_sub_ref),
			"h_theta": float(h_theta),
			"h_pi": float(h_pi),
		}
		return





	def _estimate_pi_budget_h(self, q: np.ndarray, eps: float, dt_abs: float, *, include_barrier: bool) -> float:
		sim = self.sim

		dt_f = float(abs(float(dt_abs)))
		if (not np.isfinite(dt_f)) or (dt_f <= 0.0):
			dt_f = 1.0e-2

		chi_pi_attr = getattr(sim.cfg, "chi_pi", 0.2)
		if isinstance(chi_pi_attr, (int, float, np.floating)):
			chi_pi = float(chi_pi_attr)
		else:
			chi_pi = 0.2
		if (not np.isfinite(chi_pi)) or (chi_pi <= 0.0):
			chi_pi = 0.2

		k_attr = getattr(self, "k_soft", 0.0)
		if isinstance(k_attr, (int, float, np.floating)):
			k = float(k_attr)
		else:
			k = 0.0
		if (not np.isfinite(k)) or (k <= 0.0):
			return float(dt_f)

		eps_star_val = self._eps_target(q=q)
		if isinstance(eps_star_val, (int, float, np.floating)):
			if np.isfinite(float(eps_star_val)):
				eps_star = float(eps_star_val)
			else:
				eps_star = float(sim.manager.s0)
		else:
			eps_star = float(sim.manager.s0)

		Delta = float(eps) - float(eps_star)
		s0_attr = getattr(sim.manager, "s0", 0.0)
		if isinstance(s0_attr, (int, float, np.floating)):
			s0 = float(s0_attr)
		else:
			s0 = 0.0
		if (not np.isfinite(s0)) or (s0 <= 0.0):
			s0 = 1.0
		delta_eff = float(max(abs(float(Delta)), 1.0e-4 * float(s0)))

		G = float(sim.G)
		m = np.asarray(sim._mass, dtype=float)
		q_arr = np.asarray(q, dtype=float)

		n = 0
		if isinstance(q_arr, np.ndarray):
			if q_arr.ndim == 2:
				n = int(q_arr.shape[0])

		if (n >= 2) and (G != 0.0):
			dVdeps = float(_dVdeps(q_arr, m, float(eps), float(G)))
		else:
			dVdeps = 0.0

		dBdeps = 0.0
		if bool(include_barrier):
			eps_min = float(getattr(sim, "_min_softening", 0.0))
			if hasattr(sim, "_max_softening"):
				eps_max = float(getattr(sim, "_max_softening"))
			else:
				eps_max = 10.0 * float(getattr(sim.manager, "s0", 1.0))
			k_wall_attr = getattr(self, "k_wall", 0.0)
			if isinstance(k_wall_attr, (int, float, np.floating)):
				k_wall = float(k_wall_attr)
			else:
				k_wall = 0.0
			n_exp = 0
			if hasattr(self, "_barrier_n"):
				nv = self._barrier_n()
				if isinstance(nv, (int, float, np.floating)):
					n_exp = int(nv)
			if n_exp < 2:
				n_exp_cfg = getattr(sim.cfg, "barrier_exponent", 5)
				if isinstance(n_exp_cfg, (int, float, np.floating)):
					n_exp = int(n_exp_cfg)
				else:
					n_exp = 5
			if (k_wall > 0.0) and (n_exp >= 2):
				F_bar = float(barrier_force(float(eps), float(eps_min), float(eps_max), k_wall=float(k_wall), n=int(n_exp)))
				dBdeps = -float(F_bar)
			else:
				dBdeps = 0.0

		deps_total = float(dVdeps + dBdeps)
		deps_eff = float(max(abs(deps_total), 1.0e-16))

		sqrtk = 0.0
		if k > 0.0:
			sqrtk = float(np.sqrt(k))
		h_pi = (2.0 * float(chi_pi) * float(sqrtk) * float(delta_eff)) / float(deps_eff)

		if (not np.isfinite(h_pi)) or (h_pi < 0.0):
			h_pi = float(dt_f)

		return float(h_pi)


	def _epsilon_for_v_operator(self, eps: float) -> float:
		eps_eff = float(eps)

		pol_attr = getattr(self, "barrier_policy", None)
		if isinstance(pol_attr, str):
			pol = pol_attr.lower()
		else:
			pol = "reflection"
		if pol == "soft":
			self._last_eps_eff_vpolicy = float(eps_eff)
			return float(eps_eff)


		self._last_eps_eff_vpolicy = float(eps_eff)
		return float(eps_eff)



	def last_eps_star_probe(self) -> dict:
		info = getattr(self, "_last_eps_star_info", None)

		if isinstance(info, dict):
			eps_val = info.get("eps_star", 0.0)
			if isinstance(eps_val, (int, float, np.floating)):
				eps_out = float(eps_val)
			else:
				eps_out = 0.0

			gmax = info.get("grad_norm_max", None)
			if isinstance(gmax, (int, float, np.floating)):
				gn_out = float(gmax)
			else:
				grad_mat = info.get("grad", None)
				if isinstance(grad_mat, np.ndarray):
					if grad_mat.ndim == 2:
						if grad_mat.shape[0] > 0:
							norms = np.sqrt(np.sum(grad_mat * grad_mat, axis=1))
							if isinstance(norms, np.ndarray) and norms.size > 0:
								gn_out = float(np.max(norms))
							else:
								gn_out = 0.0
						else:
							gn_out = 0.0
					else:
						gn_out = 0.0
				else:
					gn_out = 0.0

			return {"eps_star": float(eps_out), "grad_norm_max": float(gn_out)}

		return {"eps_star": 0.0, "grad_norm_max": 0.0}

