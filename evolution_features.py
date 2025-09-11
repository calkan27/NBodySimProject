import math
import numpy as np
from typing import Dict
from .diagnostics import Diagnostics
from .dynamical_features import DynamicalFeatures
from .tangent_map import TangentMap


"""
This module computes time-evolution features that characterize the dynamical stability of N-body systems. The EvolutionFeatures class calculates the MEGNO (Mean Exponential Growth of Nearby Orbits) chaos indicator through variational equation integration, estimates Lyapunov time from MEGNO convergence, and combines static dynamical features with evolution metrics. The compute_megno method integrates tangent space perturbations to measure trajectory divergence rates, while extract_evolution_features provides a focused subset for ML applications. The module assumes access to a valid simulation state and uses the TangentMap for variational dynamics calculations.


"""




class EvolutionFeatures:
	def __init__(self, sim, n_samples: int = 20, dt: float = 0.01):
		self.sim = sim
		self.n_samples = n_samples
		self.dt = dt
		self.diagnostics = Diagnostics(sim)
		self._tmap = TangentMap(sim)

	def compute_megno(self, n_steps: int, dt: float) -> tuple[float, float]:
		n = self.sim.n_bodies
		m = self.sim._mass
		delta_r = np.random.randn(n, 2)
		com = np.sum(m[:, None] * delta_r, axis=0) / np.sum(m)
		delta_r -= com
		delta_r /= np.linalg.norm(delta_r)
		delta_v = np.random.randn(n, 2)
		com_v = np.sum(m[:, None] * delta_v, axis=0) / np.sum(m)
		delta_v -= com_v
		delta_v /= np.linalg.norm(delta_v)
		t = 0.0
		accum = 0.0
		for _ in range(n_steps):
			self.sim.step(dt)
			delta_r += delta_v * dt
			delta_a = self._tmap.variational_accel(delta_r)
			delta_v += delta_a * dt
			t += dt
			norm_r = float(np.linalg.norm(delta_r))
			if norm_r < 1e-12:
				delta_r /= norm_r
				delta_v /= norm_r
				norm_r = 1.0
			norm_v = float(np.linalg.norm(delta_v))
			ratio = norm_v / norm_r
			accum += ratio * t * dt
		Y = 2.0 * accum / t
		if Y == 0.0:
			lyap = math.inf
		else:
			lyap = t / abs(Y)
		return float(Y), float(lyap)

	def extract_evolution_features(self) -> Dict[str, float]:
		feats = self.extract_all()
		keys = ('MEGNO', 'lyapunov_time', 'current_total_energy')
		out: Dict[str, float] = {}
		for k in keys:
			out[k] = feats[k]
		return out


	def extract_all(self) -> Dict[str, float]:
		features = DynamicalFeatures(self.sim).extract_all()
		megno, lyap_time = self.compute_megno(self.n_samples, self.dt)
		E = self.diagnostics.energy()
		features.update({
			'MEGNO': megno,
			'lyapunov_time': lyap_time,
			'current_total_energy': E,
		})
		return features

