from __future__ import annotations
import numpy as np
from .integration_scheme_base import IntegrationScheme

"""
This module implements the Wisdom-Holman fast integration scheme for hierarchical N-body systems. The WHFastScheme class provides Kepler drift for Jacobi coordinate evolution, interaction potential calculations in the democratic heliocentric frame, and symplectic composition of Kepler and interaction steps. The implementation significantly accelerates integration for systems with dominant central masses by analytically solving the Keplerian motion. It requires a clear mass hierarchy and is incompatible with adaptive softening. The scheme assumes Jacobi coordinates are well-defined and that the system has a dominant central body.
"""

class WHFastScheme(IntegrationScheme):

	def _wh_kepler_drift(self, dt: float) -> None:
		sim = self.integ.sim
		m = sim._mass
		cum = np.cumsum(m)
		jac_pos, jac_vel = sim.to_jacobi()
		jac_pos[0] += jac_vel[0] * dt
		n = len(m)
		for i in range(1, n):
			mu = sim.G * (cum[i - 1] + m[i])
			r_new, v_new = self._kepler_propagate(jac_pos[i], jac_vel[i], mu, dt)
			jac_pos[i] = r_new
			jac_vel[i] = v_new
		pos, vel = sim.from_jacobi(jac_pos, jac_vel)
		sim._pos = pos
		sim._vel = vel
		self.flag_positions_changed()

	def _wh_interaction_accel(self) -> np.ndarray:
		sim = self.integ.sim
		n = sim.n_bodies
		if n < 2:
			return np.zeros_like(sim._pos)
		m = sim._mass
		cum_mass = np.cumsum(m)
		jac_pos, jac_vel = sim.to_jacobi()
		pos = sim._pos
		acc = np.zeros_like(pos)
		G = sim.G
		s2 = sim.manager.step_s2
		if n >= 2:
			for i in range(2, n):
				r_jac = jac_pos[i]
				r_jac_norm2 = np.dot(r_jac, r_jac) + s2
				if r_jac_norm2 > 0:
					acc_jac_i = G * cum_mass[i - 1] * r_jac / (r_jac_norm2 ** 1.5)
					for k in range(i):
						acc[k] -= m[i] * acc_jac_i * (m[k] / cum_mass[i - 1])
					acc[i] += cum_mass[i - 1] * acc_jac_i
		for i in range(n):
			for j in range(i + 1, n):
				if not (i == 0 and j > 0):
					dr = pos[j] - pos[i]
					r2 = np.dot(dr, dr) + s2
					inv_r3 = r2 ** -1.5
					force = G * dr * inv_r3
					acc[i] -= m[j] * force
					acc[j] += m[i] * force
		return acc

	def _wisdom_holman(self, h: float) -> None:
		sim = self.integ.sim
		if sim._in_integration:
			return

		if sim._adaptive_softening:
			print("WHFast incompatible with adaptive softening")
			return

		sim._in_integration = True
		dt2 = 0.5 * h

		self._wh_kepler_drift(dt2)

		acc_int = self._wh_interaction_accel()
		sim._acc[:] = acc_int
		sim._acc_cached = True
		self.kick(h)

		self._wh_kepler_drift(dt2)

		sim._in_integration = False
		sim._acc_cached = False

	def apply_corrector(self, order: int) -> None:

		if order is None or int(order) <= 0:
			return
		sim = self.integ.sim
		if sim.n_bodies < 2 or sim.G == 0.0:
			return

		h_ref = 0.0

		top_dt = getattr(self.integ, "_top_dt", None)
		if isinstance(top_dt, (int, float, np.floating)):
			val = float(abs(top_dt))
			if np.isfinite(val) and val > 0.0:
				h_ref = val

		if not (np.isfinite(h_ref) and h_ref > 0.0):
			hsr = getattr(self.integ, "h_sub_ref", None)
			if isinstance(hsr, (int, float, np.floating)):
				val = float(abs(hsr))
				if np.isfinite(val) and val > 0.0:
					h_ref = val

		if not (np.isfinite(h_ref) and h_ref > 0.0):
			return

		acc_int = self._wh_interaction_accel()
		sim._vel += 0.5 * h_ref * acc_int
		sim._acc_cached = False

