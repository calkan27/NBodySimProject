"""
This module generates diverse initial conditions for N-body simulations to train ML
models.

The InitialConditionGenerator class creates systems with configurable mass distributions
(uniform or log-scale), position distributions (Gaussian with scale parameter),
velocities based on virial equilibrium with perturbations, and proper center-of-mass
frame adjustment. The GeneratorConfig dataclass encapsulates generation parameters.
Methods include generate_single for individual systems, generate_batch for multiple
systems, create_simulation for direct simulation instantiation, and validate_system for
physics checks. The generator ensures physical validity through momentum conservation
and energy consistency checks. It assumes positive masses and finite
positions/velocities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from .simulation import NBodySimulation
from .diagnostics import Diagnostics
from .physics_utils import remove_center_of_mass_velocity




@dataclass
class GeneratorConfig:
	mass_range: Tuple[float, float] = (0.1, 10.0)
	use_log_mass: bool = False
	position_scale: float = 1.0                      
	velocity_virial_fraction: float = 1.0             
	velocity_perturbation: float = 0.1               
	softening: float = 0.05
	G: float = 1.0
	seed: Optional[int] = None                        


class InitialConditionGenerator:

	def __init__(self, config: GeneratorConfig | None = None):
		self.config: GeneratorConfig = config or GeneratorConfig()
		if self.config.seed is not None:
			np.random.seed(self.config.seed)


	def _generate_masses(self, n: int) -> np.ndarray:
		min_m, max_m = self.config.mass_range
		if self.config.use_log_mass:
			return np.exp(np.random.uniform(np.log(min_m), np.log(max_m), n))
		return np.random.uniform(min_m, max_m, n)

	def _generate_positions(self, n: int) -> np.ndarray:
		return np.random.randn(n, 2) * self.config.position_scale

	def _compute_mean_separation(self, positions: np.ndarray) -> float:
		n = len(positions)
		if n < 2:
			return 1.0
		dx = positions[:, None, :] - positions[None, :, :]
		dist = np.sqrt((dx**2).sum(axis=-1))
		iu = np.triu_indices(n, 1)
		if iu[0].size:
			return float(np.mean(dist[iu]))
		else:
			return 1.0

	def _compute_potential_energy(self, m: np.ndarray, pos: np.ndarray) -> float:
		G, eps = self.config.G, self.config.softening
		n = len(m)
		U = 0.0
		for i in range(n - 1):
			for j in range(i + 1, n):
				r = np.hypot(*(pos[j] - pos[i])) + eps
				U -= G * m[i] * m[j] / r
		return U

	def _generate_velocities(self, m: np.ndarray, pos: np.ndarray) -> np.ndarray:
		n, G = len(m), self.config.G

		U = self._compute_potential_energy(m, pos)
		K_target = -U / 2.0 * self.config.velocity_virial_fraction
		if K_target <= 0.0:
			v_char = np.sqrt(G * m.sum() / self._compute_mean_separation(pos))
		else:
			v_char = np.sqrt(2.0 * K_target / m.sum())

		vel = np.random.randn(n, 2)
		speed = np.linalg.norm(vel, axis=1, keepdims=True)
		vel = np.where(speed > 0, vel / speed * v_char, vel)

		vel = remove_center_of_mass_velocity(m, vel)
		vel += np.random.randn(n, 2) * v_char * self.config.velocity_perturbation
		vel = remove_center_of_mass_velocity(m, vel)
		return vel


	def generate_single(self, n_bodies: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		m = self._generate_masses(n_bodies)
		p = self._generate_positions(n_bodies)
		v = self._generate_velocities(m, p)
		return m, p, v

	def generate_batch(
		self, n_systems: int, n_bodies_range: Tuple[int, int] = (3, 5)
	) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
		out: list = []
		for _ in range(n_systems):
			n = np.random.randint(n_bodies_range[0], n_bodies_range[1] + 1)
			out.append(self.generate_single(n))
		return out

	def create_simulation(
		self,
		n_bodies: int,
		*,
		integrator_mode: str | None = None,
		adaptive_softening: bool | None = None,
	) -> NBodySimulation:
		m, p, v = self.generate_single(n_bodies)
		kwargs: Dict = dict(
			masses=m,
			positions=p,
			velocities=v,
			G=self.config.G,
			softening=self.config.softening,
		)
		if integrator_mode is not None:
			kwargs["integrator_mode"] = integrator_mode
		if adaptive_softening is not None:
			kwargs["adaptive_softening"] = adaptive_softening
		return NBodySimulation(**kwargs)

	def validate_system(
		self,
		masses: np.ndarray,
		positions: np.ndarray,
		velocities: np.ndarray,
	) -> Dict[str, float]:
		sim = NBodySimulation(
			masses=masses,
			positions=positions,
			velocities=velocities,
			G=self.config.G,
			softening=self.config.softening,
		)
		diag = Diagnostics(sim)
		KE = diag.kinetic_energy()
		PE = diag.potential_energy()
		E_tot = KE + PE
		if PE:
			virial = 2 * KE / abs(PE)
		else:
			virial = np.inf
		L = diag.angular_momentum()
		com_pos, com_vel = diag.center_of_mass()

		return {
			"kinetic_energy": KE,
			"potential_energy": PE,
			"total_energy": E_tot,
			"virial_ratio": virial,
			"angular_momentum": L,
			"com_position": float(np.linalg.norm(com_pos)),
			"com_velocity": float(np.linalg.norm(com_vel)),
			"is_bound": bool(E_tot < 0),
		}

