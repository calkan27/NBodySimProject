import numpy as np
from typing import Tuple
from .physics_utils import remove_center_of_mass_velocity

"""
This module provides specialized initial condition generators for specific N-body configurations. The SpecializedGenerators class offers static methods for hierarchical triple systems with configurable mass and separation ratios, equal-mass polygon configurations with rotation, and other specialized test cases. These generators produce well-defined, reproducible configurations useful for testing and validation. Each generator ensures momentum conservation and appropriate scaling of velocities for orbital stability. The module assumes generated systems will be used for stability analysis and ML training.

"""


class SpecializedGenerators:
    
	@staticmethod
	def generate_hierarchical_triple(
			mass_ratio1: float = 1.0,
			mass_ratio2: float = 0.5,
			separation_ratio: float = 10.0,
			G: float = 1.0,
			*,
			integrator_mode: str | None = None,
			adaptive_softening: bool | None = None
		) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		
		m1 = 1.0
		m2 = mass_ratio1
		m3 = mass_ratio2
		masses = np.array([m1, m2, m3])
		
		a_inner = 1.0
		
		x1 = -m2 * a_inner / (m1 + m2)
		x2 = m1 * a_inner / (m1 + m2)
		
		a_outer = max(separation_ratio * a_inner, 5.0 * a_inner)
		
		positions = np.array([
			[x1, 0.0],
			[x2, 0.0],
			[a_outer, 0.0]
		])
		
		v_inner = np.sqrt(G * (m1 + m2) / a_inner)
		vy1 = -m2 * v_inner / (m1 + m2)
		vy2 =  m1 * v_inner / (m1 + m2)
		
		v_outer = np.sqrt(G * (m1 + m2 + m3) / a_outer)
		
		velocities = np.array([
			[0.0, vy1],
			[0.0, vy2],
			[0.0, v_outer]
		])
		
		velocities = remove_center_of_mass_velocity(masses, velocities)
		return masses, positions, velocities
	
	@staticmethod
	def generate_equal_mass_polygon(
		n_bodies: int,
		radius: float = 1.0,
		rotation_fraction: float = 0.5,
		G: float = 1.0,
		*,
		integrator_mode: str | None = None,
		adaptive_softening: bool | None = None
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		
		masses = np.ones(n_bodies)
		
		angles = np.linspace(0.0, 2.0 * np.pi, n_bodies, endpoint=False)
		positions = np.column_stack([
			radius * np.cos(angles),
			radius * np.sin(angles)
		])
		
		total_mass = float(np.sum(masses))
		v_scale = np.sqrt(G * total_mass / radius) * rotation_fraction
		
		velocities = np.column_stack([
			-v_scale * np.sin(angles),
			 v_scale * np.cos(angles)
		])
		
		velocities = remove_center_of_mass_velocity(masses, velocities)
		return masses, positions, velocities

