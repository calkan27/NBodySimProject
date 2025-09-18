"""
This module implements adaptive timestep control for numerical stability.

The TimestepManager class determines substep counts based on multiple criteria including
gravitational timescales from particle separations, harmonic oscillator periods for
softening dynamics, momentum-based epsilon drift limits, and gradient-based impulse
constraints. It provides methods for initial schedule calculation, runtime stability
enforcement, and minimum separation prediction. The implementation balances accuracy
requirements with computational efficiency through intelligent step subdivision. It
assumes access to current simulation state and force calculations.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .integrator import Integrator





class TimestepManager:

	def __init__(self, integrator: "Integrator") -> None:
		self.integ = integrator
		self.h_sub_ref: float = 0.0
		self._schedule_log = None      
		self._last_schedule = None
		self._replay = False

	def get_cached_min_sep(self) -> float:
		if self.integ._cached_min_sep is None:
			self.integ._cached_min_sep = self.integ.sim._get_min_separation()
		return self.integ._cached_min_sep

	def determine_substeps(self, dt_abs: float) -> int:
		dt_abs = float(abs(dt_abs))
		if dt_abs == 0.0 or self.integ.sim.n_bodies < 2 or self.integ.sim.G == 0.0:
			return 1

		sim = self.integ.sim
		pos = sim._pos
		m = sim._mass
		G = float(sim.G)
		chi = 0.9

		diff = pos[:, None, :] - pos[None, :, :]
		r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
		np.fill_diagonal(r2, np.inf)
		r = np.sqrt(r2)

		denom = G * (m[:, None] + m[None, :])
		r3 = r * r * r

		valid = np.isfinite(r3) & np.isfinite(denom) & (denom > 0.0)
		tau_ij = np.full_like(r3, np.inf, dtype=float)
		tau_ij[valid] = np.sqrt(r3[valid] / denom[valid])
		if np.any(np.isfinite(tau_ij)):
			tau_grav = float(np.min(tau_ij))
		else:
			tau_grav = math.inf

		k_soft = float(getattr(self.integ, "k_soft", 0.0))
		mu_soft = float(getattr(self.integ, "mu_soft", 1.0))
		theta_cp = float(getattr(sim.cfg, "theta_cap", 0.0) or 0.0)
		if k_soft > 0.0 and mu_soft > 0.0 and theta_cp > 0.0:
			omega_spr = math.sqrt(k_soft / mu_soft)
			tau_spr = theta_cp / omega_spr
		else:
			tau_spr = math.inf

		pi_mom = float(getattr(sim, "_pi", 0.0))
		eps_min = float(getattr(sim, "_min_softening", 0.0))
		eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		eps_safe = 0.1 * (eps_max - eps_min)
		if pi_mom != 0.0 and mu_soft != 0.0:
			tau_eps = chi * eps_safe / abs(pi_mom / mu_soft)
		else:
			tau_eps = math.inf

		theta_imp = 0.1
		eps_p = 1e-12

		vel = sim._vel
		p = m[:, None] * vel
		p_norm = np.sqrt(np.sum(p * p, axis=1))
		if p_norm.size:
			p_max = float(np.max(p_norm))
		else:
			p_max = 0.0
		if not np.isfinite(p_max):
			p_max = 0.0

		eps_cur = float(getattr(sim, "_epsilon", sim.manager.s))
		eps_star_fun = getattr(self.integ, "_eps_target", None)
		if callable(eps_star_fun):
			eps_star = float(eps_star_fun(q=pos))
		else:
			eps_star = float(sim.manager.s0)

		grad_fun = getattr(self.integ, "_grad_eps_target", None)
		if callable(grad_fun):
			grad = grad_fun(pos)
			if np.all(np.isfinite(grad)):
				grad_norm = float(np.linalg.norm(grad))
			else:
				grad_norm = 0.0
		else:
			grad_norm = 0.0

		delta = abs(eps_cur - eps_star)
		if k_soft > 0.0 and grad_norm > 0.0 and delta > 0.0:
			den = k_soft * delta * grad_norm
			if den > 0.0 and np.isfinite(den):
				tau_imp = (2.0 * theta_imp * (p_max + eps_p)) / den
			else:
				tau_imp = math.inf
		else:
			tau_imp = math.inf

		h_req = min(chi * tau_grav, tau_spr, tau_eps, tau_imp)
		if not math.isfinite(h_req) or h_req <= 0.0:
			return 1

		n_sub = int(math.ceil(dt_abs / h_req))
		split_cap = int(getattr(self.integ, "split_n_max", n_sub))

		if split_cap > 0:
			n_sub = min(max(1, n_sub), split_cap)
		else:
			n_sub = max(1, n_sub)

		return n_sub


	def init_substep_schedule(self, dt_user: float) -> None:
		sim = self.integ.sim
		dt_user = abs(float(dt_user))
		chi = 0.9

		if sim.n_bodies < 2 or sim.G == 0.0:
			tau_grav = math.inf
		else:
			pos = np.asarray(sim._pos, dtype=float)
			mass = np.asarray(sim._mass, dtype=float)

			diff = pos[:, None, :] - pos[None, :, :]
			r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
			np.fill_diagonal(r2, np.inf)
			r = np.sqrt(r2)
			r3 = r * r * r

			denom = float(sim.G) * (mass[:, None] + mass[None, :])

			valid = np.isfinite(r3) & np.isfinite(denom) & (denom > 0.0)
			tau_ij = np.full_like(r3, np.inf, dtype=float)
			tau_ij[valid] = np.sqrt(r3[valid] / denom[valid])

			if np.any(np.isfinite(tau_ij)):
				tau_grav = float(np.min(tau_ij))
			else:
				tau_grav = math.inf

		k_soft = float(getattr(self.integ, "k_soft", 0.0))
		mu_soft = float(getattr(self.integ, "mu_soft", 1.0))
		if (k_soft > 0.0) and (mu_soft > 0.0):
			omega = math.sqrt(k_soft / mu_soft)
			cfg = getattr(sim, "cfg", None)
			if cfg is not None:
				theta_raw = getattr(cfg, "theta_cap", None)
			else:
				theta_raw = None
			if isinstance(theta_raw, (int, float, np.floating)) and np.isfinite(float(theta_raw)) and float(theta_raw) > 0.0:
				theta_cap = float(theta_raw)
			else:
				theta_cap = 0.25
			if omega > 0.0:
				tau_spr = theta_cap / omega
			else:
				tau_spr = math.inf
		else:
			tau_spr = math.inf

		pi_mom = float(getattr(sim, "_pi", 0.0))
		eps_min = float(getattr(sim, "_min_softening", 0.0))
		eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
		eps_safe = 0.1 * max(eps_max - eps_min, 0.0)
		if (pi_mom != 0.0) and (mu_soft != 0.0):
			if eps_safe > 0.0:
				tau_eps = chi * eps_safe / abs(pi_mom / mu_soft)
			else:
				tau_eps = math.inf
		else:
			tau_eps = math.inf

		theta_imp = 0.1
		eps_p = 1e-12

		vel = sim._vel
		mass = sim._mass
		p = mass[:, None] * vel
		p_norm = np.sqrt(np.sum(p * p, axis=1))
		if p_norm.size:
			p_max = float(np.max(p_norm))
		else:
			p_max = 0.0
		if not np.isfinite(p_max):
			p_max = 0.0

		eps_cur = float(getattr(sim, "_epsilon", sim.manager.s))
		eps_star_fun = getattr(self.integ, "_eps_target", None)
		if callable(eps_star_fun):
			eps_star = float(eps_star_fun(q=sim._pos))
		else:
			eps_star = float(sim.manager.s0)

		grad_fun = getattr(self.integ, "_grad_eps_target", None)
		if callable(grad_fun):
			grad = grad_fun(sim._pos)
			if np.all(np.isfinite(grad)):
				grad_norm = float(np.linalg.norm(grad))
			else:
				grad_norm = 0.0
		else:
			grad_norm = 0.0

		delta = abs(eps_cur - eps_star)
		if k_soft > 0.0 and grad_norm > 0.0 and delta > 0.0:
			den = k_soft * delta * grad_norm
			if den > 0.0 and np.isfinite(den):
				tau_imp = (2.0 * theta_imp * (p_max + eps_p)) / den
			else:
				tau_imp = math.inf
		else:
			tau_imp = math.inf

		h_sub = min(chi * tau_grav, tau_spr, tau_eps, tau_imp)
		if not math.isfinite(h_sub) or h_sub <= 0.0:
			if dt_user > 0.0:
				h_sub = dt_user
			else:
				h_sub = 1.0

		cap = int(getattr(self.integ, "split_n_max", 0))
		if cap > 0:
			n_need = math.ceil(dt_user / max(h_sub, 1e-30))
			if n_need > cap:
				h_sub = dt_user / cap

		self.h_sub_ref = float(h_sub)

	def enforce_stability(self, h: float) -> tuple[bool, int]:
		if not self.integ.sim._adaptive_timestep:
			return False, 1
		h_abs = float(abs(h))
		h_req = self.estimate_h(h_abs)
		trigger = 1.2
		if h_abs <= trigger * h_req:
			return False, 1
		n_sub = math.ceil(h_abs / h_req)
		n_sub = min(n_sub, self.integ.split_n_max)
		if n_sub < 2:
			return False, 1
		return True, n_sub

	def estimate_h(self, dt_max: float) -> float:
		sim = self.integ.sim
		s2 = sim.manager.step_s2
		eps = math.sqrt(s2)
		acc = sim._compute_accelerations(pos=sim._pos, s2=s2)
		a_max = float(((acc ** 2).sum(axis=1)).max() ** 0.5)
		v_max = float(((sim._vel ** 2).sum(axis=1)).max() ** 0.5)
		if a_max <= 0 or not math.isfinite(a_max):
			h = float(dt_max)
		else:
			c = sim.cfg.safety_factor
			h_eps = c * math.sqrt(eps / a_max)
			h_curv = c * v_max / max(a_max, 1e-18)
			h_dyn = c * eps / max(v_max, 1e-12)
			h = min(h_eps, h_curv, h_dyn)
			min_sep = self.get_cached_min_sep()
			if math.isfinite(min_sep) and v_max > 0.0:
				h_sep = sim.cfg.safety_factor * min_sep / v_max
				h = min(h, h_sep)
		h = max(h, 1e-8 * dt_max)
		if self.integ._dt_prev is not None:
			h = min(h, 2.0 * self.integ._dt_prev)
		self.integ._dt_prev = h
		return float(max(h, 1e-16))

	def predict_min_separation(self, dt: float) -> float:
		pos = self.integ.sim._pos
		vel = self.integ.sim._vel
		if len(pos) < 2:
			return float("inf")

		r0 = pos[:, None, :] - pos[None, :, :]
		dv = vel[:, None, :] - vel[None, :, :]
		dt = abs(float(dt))

		d_now = np.linalg.norm(r0, axis=-1)
		d_dt = np.linalg.norm(r0 + dv * dt, axis=-1)

		vv = np.sum(dv * dv, axis=-1) + 1e-30
		rv = np.sum(r0 * dv, axis=-1)
		t_star = -rv / vv
		mask = (t_star > 0.0) & (t_star < dt)
		r_star = np.linalg.norm(r0 + dv * t_star[..., None], axis=-1)
		d_min = np.where(mask, np.minimum(np.minimum(d_now, d_dt), r_star),
						 np.minimum(d_now, d_dt))

		np.fill_diagonal(d_min, np.inf)
		return float(max(d_min.min(), 1e-12))

