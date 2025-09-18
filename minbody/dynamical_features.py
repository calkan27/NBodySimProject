"""
This module extracts physics-based features from N-body configurations for machine
learning applications.

The DynamicalFeatures class computes comprehensive metrics including mass distribution
statistics (total mass, variance, max ratio, center offset), pairwise distances
(mean/std separation, min/max distances, separation ratios), velocity characteristics
(mean/max speeds, relative velocities), energy components (kinetic, potential, virial
ratio, binding state), angular momentum properties (total, specific, variance), and
softening parameter statistics from the manager's history. The extract_all method
combines all features into a dictionary suitable for ML pipelines. The module assumes
the simulation is in a valid state with accessible position, velocity, and mass arrays.
"""

import numpy as np
from typing import Dict
from .simulation import NBodySimulation
from .diagnostics import Diagnostics



class DynamicalFeatures:
	def __init__(self, sim: NBodySimulation):
		self.sim = sim
		self.diagnostics = Diagnostics(sim)

	def extract_all(self) -> Dict[str, float]:
		features = {}
		features.update(self._extract_mass_features())
		features.update(self._extract_distance_features())
		features.update(self._extract_velocity_features())
		features.update(self._extract_energy_features())
		features.update(self._extract_angular_features())
		features.update(self._extract_softening_features())
		return features

	def _extract_mass_features(self) -> Dict[str, float]:
		masses = self.sim._mass
		if np.min(masses) > 0:
			mass_ratio_max = float(np.max(masses) / np.min(masses))
		else:
			mass_ratio_max = 1.0
		return {
			'total_mass': float(np.sum(masses)),
			'mass_variance': float(np.var(masses)),
			'mass_ratio_max': mass_ratio_max,
			'mass_center_offset': float(self._compute_mass_center_offset())
		}

	def _extract_distance_features(self) -> Dict[str, float]:
		distances = []
		min_dist = float('inf')
		max_dist = 0.0
		for i in range(self.sim.n_bodies):
			for j in range(i + 1, self.sim.n_bodies):
				dx = self.sim._pos[j, 0] - self.sim._pos[i, 0]
				dy = self.sim._pos[j, 1] - self.sim._pos[i, 1]
				r = np.sqrt(dx * dx + dy * dy)
				distances.append(r)
				min_dist = min(min_dist, r)
				max_dist = max(max_dist, r)
		if distances:
			mean_dist = float(np.mean(distances))
			std_dist = float(np.std(distances))
		else:
			mean_dist = 0.0
			std_dist = 0.0
			min_dist = 0.0
		if min_dist > 0:
			separation_ratio = max_dist / min_dist
		else:
			separation_ratio = 1.0
		return {
			'mean_separation': mean_dist,
			'std_separation': std_dist,
			'min_separation': min_dist,
			'max_separation': max_dist,
			'separation_ratio': separation_ratio
		}

	def _extract_velocity_features(self) -> Dict[str, float]:
		velocities = self.sim._vel
		speeds = np.sqrt(np.sum(velocities**2, axis=1))
		rel_velocities = []
		for i in range(self.sim.n_bodies):
			for j in range(i + 1, self.sim.n_bodies):
				dvx = velocities[j, 0] - velocities[i, 0]
				dvy = velocities[j, 1] - velocities[i, 1]
				dv = np.sqrt(dvx * dvx + dvy * dvy)
				rel_velocities.append(dv)
		if rel_velocities:
			mean_relative_velocity = float(np.mean(rel_velocities))
		else:
			mean_relative_velocity = 0.0
		if rel_velocities:
			max_relative_velocity = float(np.max(rel_velocities))
		else:
			max_relative_velocity = 0.0
		return {
			'mean_speed': float(np.mean(speeds)),
			'std_speed': float(np.std(speeds)),
			'max_speed': float(np.max(speeds)),
			'mean_relative_velocity': mean_relative_velocity,
			'max_relative_velocity': max_relative_velocity
		}

	def _extract_energy_features(self) -> Dict[str, float]:
		KE = self.diagnostics.kinetic_energy()
		PE = self.diagnostics.potential_energy()
		E_total = KE + PE
		if PE != 0:
			virial_ratio = 2 * KE / abs(PE)
		else:
			virial_ratio = 0.0
		return {
			'kinetic_energy': KE,
			'potential_energy': PE,
			'total_energy': E_total,
			'virial_ratio': virial_ratio,
			'energy_per_mass': E_total / np.sum(self.sim._mass),
			'is_bound': float(E_total < 0)
		}

	def _extract_angular_features(self) -> Dict[str, float]:
		L_total = self.diagnostics.angular_momentum()
		specific_angular_momenta = []
		for i in range(self.sim.n_bodies):
			li = self.sim._mass[i] * (
				self.sim._pos[i, 0] * self.sim._vel[i, 1] - 
				self.sim._pos[i, 1] * self.sim._vel[i, 0]
			)
			specific_angular_momenta.append(abs(li) / self.sim._mass[i])
		return {
			'total_angular_momentum': abs(L_total),
			'mean_specific_angular_momentum': float(np.mean(specific_angular_momenta)),
			'angular_momentum_variance': float(np.var(specific_angular_momenta))
		}

	def _compute_mass_center_offset(self) -> float:
		com_pos, _ = self.diagnostics.center_of_mass()
		return np.sqrt(com_pos[0]**2 + com_pos[1]**2)

	def _extract_softening_features(self) -> Dict[str, float]:
		if hasattr(self.sim, 'manager') and self.sim.manager is not None:
			debug_info = self.sim.manager.debug_info()
			history = debug_info.get('history', [])
			if history:
				return {
					'softening_mean': float(np.mean(history)),
					'softening_std': float(np.std(history))
				}
		return {
			'softening_mean': float(self.sim.manager.softening),
			'softening_std': 0.0
		}
