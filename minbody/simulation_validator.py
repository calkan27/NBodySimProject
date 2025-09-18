"""
This module provides validation utilities for N-body simulation states.

The SimulationValidator class offers static methods to check state validity (positive
masses, finite values, correct dimensions) and report detailed diagnostics for invalid
states. The validation covers all state components and helps identify configuration
errors before simulation starts. It assumes states should represent physically
meaningful configurations and provides informative error messages for debugging.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple
import numpy as np





Vec2 = Tuple[float, float]


class SimulationValidator:
	@staticmethod
	def state_is_valid(
		masses: Sequence[float],
		positions: Sequence[Vec2],
		velocities: Sequence[Vec2],
		softening: float,
		min_softening: float,
	) -> bool:


		if masses is None or positions is None or velocities is None:
			return False

		m = np.asarray(masses, dtype=float).ravel()
		r = np.asarray(positions, dtype=float)
		v = np.asarray(velocities, dtype=float)

		if r.ndim != 2 or v.ndim != 2 or r.shape != v.shape:
			return False
		if r.shape[0] != m.size or r.shape[1] != 2:  
			return False

		for m_i in m:
			if not (m_i > 1e-16 and math.isfinite(m_i)):
				return False

		if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
			return False

		if not len(masses) == len(positions) == len(velocities):
			return False
		
		for m in masses:
			if not m > 0.0:
				return False
			if not math.isfinite(m):
				return False
		
		for pair in positions:
			if len(pair) != 2:
				return False
			x = pair[0]
			y = pair[1]
			if not math.isfinite(x):
				return False
			if not math.isfinite(y):
				return False
		
		for pair in velocities:
			if len(pair) != 2:
				return False
			x = pair[0]
			y = pair[1]
			if not math.isfinite(x):
				return False
			if not math.isfinite(y):
				return False
		
		if softening < 0.0 or min_softening < 0.0:
			return False
		
		return True

	@staticmethod
	def report_invalid_state(
		label: str,
		masses=None,
		positions=None,
		velocities=None,
		softening=None,
		min_softening=None,
	) -> None:

		print(f"[invalid] {label}")
		if masses is not None:
			print("masses", masses)
		if positions is not None:
			print("positions", positions)
			if positions:
				for i, pos in enumerate(positions):
					if len(pos) != 2:
						print(f"  position[{i}] has {len(pos)} dimensions (expected 2)")
		if velocities is not None:
			print("velocities", velocities)
			if velocities:
				for i, vel in enumerate(velocities):
					if len(vel) != 2:
						print(f"  velocity[{i}] has {len(vel)} dimensions (expected 2)")
		if softening is not None:
			print("softening", softening)
		if min_softening is not None:
			print("min_softening", min_softening)

