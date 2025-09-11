from __future__ import annotations

import os
from typing import Final

"""
This module defines physical and numerical constants for the Hamiltonian softening scheme. It includes LAMBDA_SOFTENING for controlling softening length scales (with environment variable override support), CHI_EPS for momentum damping coefficients, and LAMBDA_SIGMA_STAR as an alias for compatibility. The module provides centralized constant management with sensible defaults while allowing runtime configuration through environment variables. It assumes constants are used consistently across the ham_soft subsystem.


"""




def _parse_lambda(default: float = 10.0) -> float:
	env_val = os.getenv("LAMBDA_SOFTENING", "")
	if env_val.strip() != "":
		if env_val.replace(".", "", 1).replace("-", "", 1).isdigit():
			val = float(env_val)
			if val > 0.0:
				return val
	return default




LAMBDA_SOFTENING: Final[float] = float(os.getenv("LAMBDA_SOFTENING", .3))
CHI_EPS: float = 0.9

LAMBDA_SIGMA_STAR: float = LAMBDA_SOFTENING

