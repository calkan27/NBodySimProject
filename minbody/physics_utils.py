import numpy as np
from typing import Tuple

"""
This module provides implements remove_center_of_mass_velocity which computes and subtracts the center-of-mass velocity from a velocity array, preserving momentum conservation in the CM frame. The function handles edge cases like single particles or zero total mass gracefully. It serves as a foundation for additional physics utilities and assumes mass and velocity arrays have compatible dimensions.


"""

def remove_center_of_mass_velocity(
	masses: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
	if len(masses) == 1:
		return velocities.copy()
	total_mass = float(np.sum(masses))
	if total_mass == 0 or velocities.size == 0:
		return velocities.copy()
	v_cm = np.sum(masses[:, None] * velocities, axis=0) / total_mass
	return velocities - v_cm

