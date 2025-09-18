"""
This module implements adaptive models for the optimal softening parameter epsilon_star.

The EpsilonModel class provides multiple strategies including SPH-inspired kernel
density estimation for particle-based epsilon, iterative solver for smoothing length
determination, gradient computation via finite differences or analytical expressions,
and legacy compatibility modes. The model calibrates from initial conditions to set
appropriate bounds and scaling, supports both production and validation modes with
different algorithms, and maintains internal state for efficient repeated evaluations.
It assumes access to particle positions and masses and integrates deeply with the
gradient-based forces in the Hamiltonian formulation.
"""

from __future__ import annotations
import numpy as np
from .hamsoft_constants import LAMBDA_SOFTENING as _LAM_SOFT
from .softening import grad_eps_target as _grad_ss, eps_target as _eps_target_ss







class EpsilonModel:
	def __init__(self, owner) -> None:
		self._owner = owner
		self._alpha_run = None      
		self._hi_last = None        
		self._cmin = 0.25           

	def _cfg_args(self):
		cfg = getattr(self._owner.sim, "cfg", None)

		if cfg is not None:
			alpha = float(getattr(cfg, "alpha", 1.0))
		else:
			alpha = 1.0
		if not np.isfinite(alpha) or alpha <= 0.0:
			alpha = 1.0

		if cfg is not None:
			lam = float(getattr(cfg, "lambda_softening", _LAM_SOFT))
		else:
			lam = float(_LAM_SOFT)
		if not np.isfinite(lam) or lam <= 0.0:
			lam = float(_LAM_SOFT)

		return alpha, lam

	def eps_target_raw(self, q) -> float:
		alpha, lam = self._cfg_args()
		if q is not None and callable(_eps_target_ss):
			q_arr = np.asarray(q, dtype=float)
			if isinstance(q_arr, np.ndarray) and q_arr.ndim == 2 and q_arr.shape[1] == 2:
				if not np.all(np.isfinite(q_arr)):
					q_arr = np.where(np.isfinite(q_arr), q_arr, 0.0)
				val = _eps_target_ss(q_arr, alpha=float(alpha), lam=float(lam))
				if isinstance(val, (int, float, np.floating)):
					vf = float(val)
					if np.isfinite(vf):
						return vf
		owner = getattr(self, "_owner", None)
		sim = getattr(owner, "sim", None) if owner is not None else None
		mgr = getattr(sim, "manager", None) if sim is not None else None
		s0_val = getattr(mgr, "s0", None) if mgr is not None else None
		if isinstance(s0_val, (int, float, np.floating)):
			s0f = float(s0_val)
			if np.isfinite(s0f):
				return s0f
		eps_cur = getattr(sim, "_epsilon", None) if sim is not None else None
		if isinstance(eps_cur, (int, float, np.floating)):
			ef = float(eps_cur)
			if np.isfinite(ef):
				return ef
		return 0.0

	def eps_target(self, q=None) -> float:
		owner = getattr(self, "_owner", None)
		sim = getattr(owner, "sim", None) if owner is not None else None
		cfg = getattr(sim, "cfg", None) if sim is not None else None
		if cfg is not None and bool(getattr(cfg, "fixed_eps_star", False)):
			v_fixed = getattr(cfg, "eps_star_value", None)
			if isinstance(v_fixed, (int, float, np.floating)):
				return float(v_fixed)
			return float(getattr(getattr(sim, "manager", None), "s0", 0.0))
		use_legacy = bool(getattr(cfg, "use_legacy_eps_star", False)) if cfg is not None else False
		if use_legacy:
			return float(self.eps_target_raw(q))
		return float(self.eps_target_production(q))



	def eps_star_and_grad(self, q: np.ndarray, *, h_rel: float = 1e-5, h_abs: float = 1e-10) -> tuple[float, np.ndarray]:
		q_arr = np.asarray(q, dtype=float)

		owner = getattr(self, "_owner", None)
		if owner is not None:
			sim = getattr(owner, "sim", None)
		else:
			sim = None

		if not isinstance(q_arr, np.ndarray):
			eps_star = self.eps_target_production(q)
			return float(eps_star), np.zeros((0, 2), dtype=float)

		if q_arr.ndim != 2:
			eps_star = self.eps_target_production(q_arr)
			shape0 = int(q_arr.shape[0]) if hasattr(q_arr, "shape") and len(q_arr.shape) >= 1 else 0
			return float(eps_star), np.zeros((shape0, 2), dtype=float)

		if q_arr.shape[1] != 2:
			eps_star = self.eps_target_production(q_arr)
			return float(eps_star), np.zeros((int(q_arr.shape[0]), 2), dtype=float)

		if sim is None:
			eps_star = self.eps_target_production(q_arr)
			return float(eps_star), np.zeros_like(q_arr, dtype=float)

		m = getattr(sim, "_mass", None)
		if not isinstance(m, np.ndarray):
			eps_star = self.eps_target_production(q_arr)
			return float(eps_star), np.zeros_like(q_arr, dtype=float)

		if int(m.size) != int(q_arr.shape[0]):
			eps_star = self.eps_target_production(q_arr)
			return float(eps_star), np.zeros_like(q_arr, dtype=float)

		eps_star = self.eps_target_production(q_arr)

		grad = np.zeros_like(q_arr, dtype=float)

		n = int(q_arr.shape[0])
		d = int(q_arr.shape[1])

		i = 0
		while i < n:
			a = 0
			while a < d:
				x = float(q_arr[i, a])

				scale = abs(x)
				if scale < 1.0:
					scale = 1.0
				h = float(h_rel) * float(scale)
				if h < float(h_abs):
					h = float(h_abs)

				q_plus = q_arr.copy()
				q_minus = q_arr.copy()

				q_plus[i, a] = float(x + h)
				q_minus[i, a] = float(x - h)

				f_p = self.eps_target_production(q_plus)
				f_m = self.eps_target_production(q_minus)

				if isinstance(f_p, (int, float, np.floating)):
					fp = float(f_p)
				else:
					fp = float("nan")

				if isinstance(f_m, (int, float, np.floating)):
					fm = float(f_m)
				else:
					fm = float("nan")

				denom = 2.0 * float(h)
				if denom != 0.0:
					g = (fp - fm) / denom
				else:
					g = 0.0

				if np.isfinite(g):
					grad[i, a] = float(g)
				else:
					grad[i, a] = 0.0

				a = a + 1
			i = i + 1

		if not np.all(np.isfinite(grad)):
			grad = np.where(np.isfinite(grad), grad, 0.0)

		gmax = 0.0
		if grad.size > 0:
			row_norms = np.sqrt(np.sum(grad * grad, axis=1))
			if isinstance(row_norms, np.ndarray):
				if row_norms.size > 0:
					gmax = float(np.max(row_norms))

		r_median = 0.0
		if q_arr.shape[0] >= 2:
			diff = q_arr[:, None, :] - q_arr[None, :, :]
			r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
			iu = np.triu_indices(int(q_arr.shape[0]), 1)
			if iu[0].size > 0:
				r_vals = np.sqrt(r2[iu])
				if isinstance(r_vals, np.ndarray):
					if r_vals.size > 0:
						r_median = float(np.median(r_vals))

		need_fallback = False
		if gmax <= 1.0e-12:
			need_fallback = True
		else:
			if gmax <= 1.0e-9 * float(r_median):
				need_fallback = True

		if need_fallback:
			g_ana = self._production_grad(q_arr)
			if isinstance(g_ana, np.ndarray):
				if g_ana.shape == q_arr.shape:
					g_use = np.asarray(g_ana, dtype=float)
					if not np.all(np.isfinite(g_use)):
						g_use = np.where(np.isfinite(g_use), g_use, 0.0)

					alpha_ref, lam_ref = self._cfg_args()
					g_ref = _grad_ss(q_arr, alpha=float(alpha_ref), lam=float(lam_ref))
					if isinstance(g_ref, np.ndarray):
						if g_ref.shape == q_arr.shape:
							if np.all(np.isfinite(g_ref)):
								dot_total = float(np.sum(g_use * g_ref))
								if np.isfinite(dot_total):
									if dot_total < 0.0:
										g_use = -g_use

					grad = g_use
				else:
					grad = np.zeros_like(q_arr, dtype=float)
			else:
				grad = np.zeros_like(q_arr, dtype=float)

		return float(eps_star), grad.astype(float)





	def eps_target_production(self, q=None) -> float:
		owner = getattr(self, "_owner", None)
		sim = getattr(owner, "sim", None) if owner is not None else None
		if sim is None:
			return 0.0

		if isinstance(q, np.ndarray):
			q_ref = np.asarray(q, dtype=float)
		elif isinstance(sim._pos, np.ndarray):
			q_ref = np.asarray(sim._pos, dtype=float)
		else:
			return float(getattr(sim.manager, "s0", 0.0))

		m = np.asarray(sim._mass, dtype=float)
		if q_ref.ndim != 2 or q_ref.shape[1] != 2 or m.size != q_ref.shape[0]:
			return float(getattr(sim.manager, "s0", 0.0))

		alpha_temp = self._alpha_for_run()

		h = self._solve_hi(q_ref, m, alpha_temp)
		if not isinstance(h, np.ndarray) or h.size == 0:
			return float(getattr(sim.manager, "s0", 0.0))

		t = -h / float(alpha_temp)
		t_max = float(np.max(t)) if t.size > 0 else 0.0
		s = 0.0
		i = 0
		n = int(t.size)
		while i < n:
			s = s + float(np.exp(t[i] - t_max))
			i = i + 1
		if s <= 0.0 or not np.isfinite(s):
			eps_star = float(getattr(sim.manager, "s0", 0.0))
		else:
			eps_star = -float(alpha_temp) * (t_max + float(np.log(s)))

		policy = getattr(owner, "barrier_policy", "reflection")
		if (not getattr(sim.cfg, "disable_barrier", False)) and isinstance(policy, str) and policy.lower() == "soft":
			eps_min = float(getattr(sim, "_min_softening", 0.0))
			eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
			if eps_max < eps_min:
				tmp = eps_min
				eps_min = eps_max
				eps_max = tmp
			if eps_star < eps_min:
				eps_star = float(eps_min)
			elif eps_star > eps_max:
				eps_star = float(eps_max)

		return float(eps_star)


	def _kernel_W(self, r: float, h: float) -> float:
		h_use = float(max(h, 1.0e-12))
		c = 1.0 / (np.pi * h_use * h_use)
		z = float(np.exp(-(r * r) / (h_use * h_use)))
		return float(c * z)

	def _kernel_dW_dh(self, r: float, h: float) -> float:
		h_use = float(max(h, 1.0e-12))
		W = self._kernel_W(r, h_use)
		term = (-2.0 / h_use) + (2.0 * r * r) / (h_use * h_use * h_use)
		return float(W * term)

	def _kernel_gradW_vec(self, rij_vec: np.ndarray, h: float) -> np.ndarray:
		h_use = float(max(h, 1.0e-12))
		dx = float(rij_vec[0])
		dy = float(rij_vec[1])
		r2 = dx * dx + dy * dy
		r = float(np.sqrt(r2))
		W = self._kernel_W(r, h_use)
		coef = -2.0 * W / (h_use * h_use)
		return np.array([coef * dx, coef * dy], dtype=float)


	
	def _solve_hi(self, q: np.ndarray, m: np.ndarray, alpha: float) -> np.ndarray:
		n = int(q.shape[0])
		if n == 0:
			return np.zeros(0, dtype=float)

		sim = getattr(self._owner, "sim", None)
		if sim is not None:
			eps_min = float(getattr(sim, "_min_softening", 0.0))
			if hasattr(sim, "manager") and sim.manager is not None:
				eps_max = float(getattr(sim, "_max_softening", 10.0 * sim.manager.s0))
			else:
				eps_max = max(1.0, eps_min)
		else:
			eps_min = 0.0
			eps_max = 1.0
		if eps_max < eps_min:
			tmp = eps_min
			eps_min = eps_max
			eps_max = tmp
		eps_floor = max(eps_min, 1.0e-12)
		eps_cap   = max(eps_floor, eps_max)

		if sim is not None:
			if hasattr(sim, "_epsilon"):
				h0 = float(getattr(sim, "_epsilon"))
			else:
				if hasattr(sim, "manager") and sim.manager is not None:
					h0 = float(sim.manager.s)
				else:
					h0 = 1.0
		else:
			h0 = 1.0
		if not np.isfinite(h0) or h0 <= 0.0:
			h0 = 1.0
		if h0 < eps_floor:
			h0 = float(eps_floor)
		if h0 > eps_cap:
			h0 = float(eps_cap)

		h = np.full(n, float(h0), dtype=float)

		max_iter = 8
		tol = 1.0e-6
		it = 0
		while it < max_iter:
			Sigma = np.zeros(n, dtype=float)
			i = 0
			while i < n:
				hj = float(max(h[i], 1.0e-12))
				j = 0
				while j < n:
					if j != i:
						dx = float(q[i, 0] - q[j, 0])
						dy = float(q[i, 1] - q[j, 1])
						r = float(np.hypot(dx, dy))
						c = 1.0 / (np.pi * hj * hj)
						wij = float(c * np.exp(-(r * r) / (hj * hj)))
						Sigma[i] = Sigma[i] + float(m[j]) * wij
					j = j + 1
				i = i + 1

			eta = self._eta_for_run()
			changed = 0.0
			i = 0
			while i < n:
				Si = float(max(Sigma[i], 1.0e-30))
				mi = float(m[i])
				h_new = float(eta) * float(np.sqrt(mi / Si))
				if not np.isfinite(h_new) or h_new <= 0.0:
					h_new = h[i]
				if h_new < eps_floor:
					h_new = float(eps_floor)
				elif h_new > eps_cap:
					h_new = float(eps_cap)
				rel = abs(h_new - h[i]) / max(h[i], 1.0e-12)
				if rel > changed:
					changed = rel
				h[i] = h_new
				i = i + 1

			if changed < tol:
				break
			it = it + 1

		return h

	def grad_eps_target(self, q=None) -> np.ndarray:
		owner = getattr(self, "_owner", None)
		if owner is not None:
			sim = getattr(owner, "sim", None)
		else:
			sim = None

		if isinstance(q, np.ndarray):
			q_ref = np.asarray(q, dtype=float)
		elif sim is not None and isinstance(getattr(sim, "_pos", None), np.ndarray):
			q_ref = np.asarray(sim._pos, dtype=float)
		else:
			return np.zeros((0, 2), dtype=float)

		valid_matrix = (
			isinstance(q_ref, np.ndarray)
			and q_ref.ndim == 2
			and q_ref.shape[1] == 2
			and np.all(np.isfinite(q_ref))
		)
		if not valid_matrix:
			if hasattr(q_ref, "shape") and len(q_ref.shape) >= 1:
				n_guess = int(q_ref.shape[0])
			else:
				n_guess = 0
			return np.zeros((n_guess, 2), dtype=float)

		n = int(q_ref.shape[0])
		if n < 2:
			return np.zeros((n, 2), dtype=float)

		if sim is not None:
			cfg = getattr(sim, "cfg", None)
		else:
			cfg = None

		if cfg is not None:
			use_legacy = bool(getattr(cfg, "use_legacy_eps_star", False))
		else:
			use_legacy = False

		if use_legacy:
			return self._legacy_grad(q_ref)

		return self._production_grad(q_ref)
	



	def _production_grad(self, q: np.ndarray) -> np.ndarray:
		sim = self._owner.sim
		m = np.asarray(sim._mass, dtype=float)

		q_ref = np.asarray(q, dtype=float)
		if not isinstance(q_ref, np.ndarray):
			return np.zeros((0, 2), dtype=float)
		if q_ref.ndim != 2:
			n_guess = int(q_ref.shape[0]) if hasattr(q_ref, "shape") and len(q_ref.shape) >= 1 else 0
			return np.zeros((n_guess, 2), dtype=float)
		if q_ref.shape[1] != 2:
			return np.zeros((int(q_ref.shape[0]), 2), dtype=float)

		n = int(q_ref.shape[0])
		if n < 2:
			return np.zeros((n, 2), dtype=float)
		if m.size != n:
			return np.zeros((n, 2), dtype=float)

		alpha_temp = self._alpha_for_run()

		h = self._solve_hi(q_ref, m, float(alpha_temp))
		if not isinstance(h, np.ndarray) or h.size != n:
			return np.zeros((n, 2), dtype=float)

		if sim is not None:
			eps_min_rt = float(getattr(sim, "_min_softening", 0.0))
		else:
			eps_min_rt = 0.0
		eps_floor = float(max(eps_min_rt, 1.0e-12))
		h_clamp_min = float(max(1.0e-12, 0.1 * eps_floor))

		t = -h / float(alpha_temp)
		t_max = float(np.max(t)) if t.size > 0 else 0.0
		denom = 0.0
		i = 0
		while i < n:
			denom = denom + float(np.exp(t[i] - t_max))
			i = i + 1
		if denom <= 0.0 or not np.isfinite(denom):
			return np.zeros((n, 2), dtype=float)

		omega = np.zeros(n, dtype=float)
		i = 0
		while i < n:
			omega[i] = float(np.exp(t[i] - t_max)) / float(denom)
			i = i + 1

		Sigma = np.zeros(n, dtype=float)
		Sd = np.zeros(n, dtype=float)
		i = 0
		while i < n:
			hj = float(max(h[i], h_clamp_min))
			j = 0
			while j < n:
				if j != i:
					dx = float(q_ref[i, 0] - q_ref[j, 0])
					dy = float(q_ref[i, 1] - q_ref[j, 1])
					r = float(np.hypot(dx, dy))
					c = 1.0 / (np.pi * hj * hj)
					Wij = float(c * np.exp(-(r * r) / (hj * hj)))
					dWh = Wij * ( -2.0 / hj + 2.0 * r * r / (hj * hj * hj) )
					Sigma[i] = Sigma[i] + float(m[j]) * Wij
					Sd[i]    = Sd[i]    + float(m[j]) * dWh
				j = j + 1
			i = i + 1

		tiny = 1.0e-30
		P = np.zeros(n, dtype=float)
		i = 0
		while i < n:
			Si = float(max(Sigma[i], tiny))
			hj = float(max(h[i], h_clamp_min))
			Omega_i = 1.0 + float(hj) * float(Sd[i]) / (2.0 * Si)
			if (not np.isfinite(Omega_i)) or (Omega_i == 0.0):
				Omega_i = 1.0
			P[i] = -hj / (2.0 * Si * Omega_i)
			i = i + 1

		g = np.zeros((n, 2), dtype=float)
		i = 0
		while i < n:
			hj = float(max(h[i], h_clamp_min))
			s_i = -float(omega[i]) * float(P[i])
			j = 0
			while j < n:
				if j != i:
					rij_x = float(q_ref[i, 0] - q_ref[j, 0])
					rij_y = float(q_ref[i, 1] - q_ref[j, 1])
					r2 = rij_x * rij_x + rij_y * rij_y
					c = 1.0 / (np.pi * hj * hj)
					Wij = float(c * np.exp(-(r2) / (hj * hj)))
					coef = -2.0 * Wij / (hj * hj)
					vx = coef * rij_x
					vy = coef * rij_y

					g[i, 0] = g[i, 0] + s_i * float(m[j]) * float(vx)
					g[i, 1] = g[i, 1] + s_i * float(m[j]) * float(vy)
					g[j, 0] = g[j, 0] - s_i * float(m[j]) * float(vx)
					g[j, 1] = g[j, 1] - s_i * float(m[j]) * float(vy)
				j = j + 1
			i = i + 1

		if not np.all(np.isfinite(g)):
			g = np.where(np.isfinite(g), g, 0.0)
		return np.asarray(g, dtype=float)





	def _legacy_grad(self, q: np.ndarray) -> np.ndarray:
		sim = self._owner.sim
		m = np.asarray(sim._mass, dtype=float)
		alpha, lam = self._cfg_args()

		n = int(q.shape[0])
		if n < 2:
			return np.zeros((n, 2), dtype=float)


		pairs_i = []
		pairs_j = []
		rlist = []
		i = 0
		while i < n - 1:
			j = i + 1
			while j < n:
				dx = float(q[i, 0] - q[j, 0])
				dy = float(q[i, 1] - q[j, 1])
				r = float(np.hypot(dx, dy))
				pairs_i.append(i)
				pairs_j.append(j)
				rlist.append(r)
				j = j + 1
			i = i + 1

		P = len(rlist)
		if P == 0:
			return np.zeros((n, 2), dtype=float)

		r_arr = np.asarray(rlist, dtype=float)
		r_max = float(np.max(r_arr)) if np.all(np.isfinite(r_arr)) else 0.0

		num = np.zeros(P, dtype=float)
		k = 0
		while k < P:
			num[k] = float(np.exp(-(r_arr[k] - r_max) / float(alpha)))
			k = k + 1
		den = float(np.sum(num))
		if den <= 0.0 or not np.isfinite(den):
			return np.zeros((n, 2), dtype=float)
		w = num / den


		t = -r_arr / float(alpha)
		t_max = float(np.max(t))
		s = 0.0
		k = 0
		while k < P:
			s = s + float(np.exp(t[k] - t_max))
			k = k + 1
		if s <= 0.0 or not np.isfinite(s):
			sigma = 0.5
		else:
			L = t_max + float(np.log(s))
			exp_term = float(np.exp(float(lam) * float(L)))
			sigma = 1.0 / (1.0 + exp_term)

		g = np.zeros((n, 2), dtype=float)
		k = 0
		while k < P:
			i_idx = int(pairs_i[k])
			j_idx = int(pairs_j[k])
			ri = float(r_arr[k])

			if ri > 0.0 and np.isfinite(ri):
				dx = float(q[i_idx, 0] - q[j_idx, 0])
				dy = float(q[i_idx, 1] - q[j_idx, 1])
				ux = dx / ri
				uy = dy / ri
				w_pair = float(sigma) * float(w[k])

				g[i_idx, 0] = g[i_idx, 0] + w_pair * ux
				g[i_idx, 1] = g[i_idx, 1] + w_pair * uy
				g[j_idx, 0] = g[j_idx, 0] - w_pair * ux
				g[j_idx, 1] = g[j_idx, 1] - w_pair * uy
			k = k + 1

		if not np.all(np.isfinite(g)):
			g = np.where(np.isfinite(g), g, 0.0)

		return np.asarray(g, dtype=float)

	def calibrate_from_initial_conditions(self) -> None:
		owner = getattr(self, "_owner", None)
		sim = getattr(owner, "sim", None) if owner is not None else None
		if sim is None:
			return

		cfg = getattr(sim, "cfg", None)
		is_fixed = False
		if cfg is not None:
			is_fixed = bool(getattr(cfg, "fixed_eps_star", False))
		if is_fixed:
			v_fixed = getattr(cfg, "eps_star_value", None)
			if isinstance(v_fixed, (int, float, np.floating)):
				vf = float(v_fixed)
				if np.isfinite(vf):
					sim._epsilon = float(vf)
					eps_min_cur = float(getattr(sim, "_min_softening", 0.0))
					if not np.isfinite(eps_min_cur):
						eps_min_cur = 0.0
					if eps_min_cur > vf:
						sim._min_softening = float(vf)
					sim.manager.update_continuous(sim._epsilon)
					return

		q0 = np.asarray(sim._pos, dtype=float)
		m  = np.asarray(sim._mass, dtype=float)
		if not (isinstance(q0, np.ndarray) and q0.ndim == 2 and q0.shape[1] == 2 and m.size == q0.shape[0]):
			return

		alpha_cfg = getattr(sim.cfg, "alpha", None)
		if isinstance(alpha_cfg, (int, float, np.floating)) and float(alpha_cfg) > 0.0:
			alpha_seed = float(alpha_cfg)
		else:
			eps_cur = float(getattr(sim, "_epsilon", sim.manager.s))
			alpha_seed = max(eps_cur, 1.0e-12)

		h0 = self._solve_hi(q0, m, float(alpha_seed))
		if not isinstance(h0, np.ndarray) or h0.size == 0:
			return

		med_h = float(np.median(h0))
		if not np.isfinite(med_h) or med_h <= 0.0:
			med_h = float(alpha_seed)

		self._alpha_run = 0.3 * med_h
		if not np.isfinite(self._alpha_run) or self._alpha_run <= 0.0:
			self._alpha_run = float(alpha_seed)

		eps_floor = self._cmin * med_h

		eps_min0 = float(getattr(sim, "_min_softening", 0.0))
		if not np.isfinite(eps_min0) or eps_min0 < 0.0:
			eps_min0 = 0.0

		eps_max_cur_attr = getattr(sim, "_max_softening", None)
		if isinstance(eps_max_cur_attr, (int, float, np.floating)):
			eps_max_cur = float(eps_max_cur_attr)
		else:
			eps_max_cur = float(10.0 * sim.manager.s0)
		if not np.isfinite(eps_max_cur) or eps_max_cur <= 0.0:
			eps_max_cur = float(10.0 * sim.manager.s0)

		candidate_floor = float(eps_floor)
		if not np.isfinite(candidate_floor):
			candidate_floor = float(eps_min0)

		if candidate_floor > eps_max_cur:
			candidate_floor = float(eps_max_cur)

		if eps_min0 >= candidate_floor:
			eps_min_new = float(eps_min0)
		else:
			eps_min_new = float(candidate_floor)

		if eps_min_new > eps_max_cur:
			eps_min_new = float(eps_max_cur)

		sim._min_softening = float(eps_min_new)

		eps_now = float(getattr(sim, "_epsilon", sim.manager.s))
		if eps_now < sim._min_softening:
			sim._epsilon = float(sim._min_softening)
			sim.manager.update_continuous(sim._epsilon)

		self._hi_last = h0




	def _alpha_for_run(self) -> float:
		if isinstance(self._alpha_run, (int, float, np.floating)) and self._alpha_run > 0.0:
			return float(self._alpha_run)
		owner = getattr(self, "_owner", None)
		sim = getattr(owner, "sim", None) if owner is not None else None
		if sim is not None and hasattr(sim, "cfg"):
			val = getattr(sim.cfg, "alpha", None)
			if isinstance(val, (int, float, np.floating)) and float(val) > 0.0:
				return float(val)
		return 1.0

	def _eta_for_run(self) -> float:
		sim_wrap = getattr(self, "_owner", None)
		sim = getattr(sim_wrap, "sim", None) if sim_wrap is not None else None
		if sim is not None and hasattr(sim, "cfg"):
			val = getattr(sim.cfg, "eta", None)
			if isinstance(val, (int, float, np.floating)) and float(val) > 0.0:
				return float(val)
		return 1.35

