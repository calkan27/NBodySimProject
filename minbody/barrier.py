"""
This module implements soft barrier potentials for constraining the softening parameter
epsilon within specified bounds.

The barrier_energy function computes a power-law repulsive potential that grows rapidly
as epsilon approaches the boundaries, barrier_force provides the negative gradient
(force) of this potential, and barrier_curvature calculates the second derivative for
use in effective spring constant calculations. All functions handle scalar and array
inputs uniformly through the _as_float_array helper, use configurable wall stiffness
(k_wall) and exponent (n) parameters, and include extensive validation to return zero
for invalid inputs. The module integrates with the Hamiltonian softening integrator to
prevent epsilon from violating physical bounds during adaptive softening evolution. It
assumes positive wall stiffness and integer exponents >= 2 for proper barrier behavior.
"""

from __future__ import annotations
import numpy as np
from typing import Union
import math







Number = Union[float, np.ndarray]

__all__ = ["barrier_energy", "barrier_force", "barrier_curvature"]

def _as_float_array(x):
    a = np.asarray(x, dtype=float)
    return a, (a.ndim == 0)

def barrier_energy(eps: Number,
                   eps_min: float,
                   eps_max: float,
                   *,
                   k_wall: float = 1.0e9,
                   n: int = 5) -> Number:

    x, is_scalar = _as_float_array(eps)
    kf = float(k_wall)
    n_int = int(n)

    if (not np.isfinite(kf)) or (kf <= 0.0) or (n_int < 2) \
       or (not np.isfinite(float(eps_min))) or (not np.isfinite(float(eps_max))):
        return float(0.0) if is_scalar else np.zeros_like(x, dtype=float)

    a = float(eps_min)
    b = float(eps_max)
    if b < a:
        tmp = a
        a = b
        b = tmp

    left  = np.maximum(0.0, a - x)
    right = np.maximum(0.0, x - b)

    power = n_int - 1
    U = (kf / float(power)) * (left ** power + right ** power)

    return float(U) if is_scalar else np.asarray(U, dtype=float)


def barrier_force(eps: Number,
                  eps_min: float,
                  eps_max: float,
                  *,
                  k_wall: float = 1.0e9,
                  n: int = 5) -> Number:

    x, is_scalar = _as_float_array(eps)

    kf = float(k_wall)
    n_int = int(n)

    if (not np.isfinite(kf)) or (kf <= 0.0):
        if is_scalar:
            return float(0.0)
        else:
            return np.zeros_like(x, dtype=float)

    if n_int < 2:
        if is_scalar:
            return float(0.0)
        else:
            return np.zeros_like(x, dtype=float)

    a = np.maximum(0.0, float(eps_min) - x)   
    b = np.maximum(0.0, x - float(eps_max))   

    e = n_int - 2

    left = np.zeros_like(x, dtype=float)
    right = np.zeros_like(x, dtype=float)

    mask_a = a > 0.0
    mask_b = b > 0.0

    if e == 0:
        left[mask_a] = 1.0
        right[mask_b] = 1.0
    else:
        left[mask_a] = a[mask_a] ** e
        right[mask_b] = b[mask_b] ** e

    F = kf * (left - right)

    if is_scalar:
        return float(F)
    else:
        return np.asarray(F, dtype=float)


def barrier_curvature(eps: Number,
                      eps_min: float,
                      eps_max: float,
                      *,
                      k_wall: float = 1.0e9,
                      n: int = 5) -> Number:

    x, is_scalar = _as_float_array(eps)
    kf = float(k_wall)
    n_int = int(n)

    if (not np.isfinite(kf)) or (kf <= 0.0) or (n_int < 2) or (not np.isfinite(float(eps_min))) or (not np.isfinite(float(eps_max))):
        return float(0.0) if is_scalar else np.zeros_like(x, dtype=float)

    if n_int == 2:
        return float(0.0) if is_scalar else np.zeros_like(x, dtype=float)

    a = float(eps_min)
    b = float(eps_max)
    if b < a:
        tmp = a
        a = b
        b = tmp

    left  = np.maximum(0.0, a - x)
    right = np.maximum(0.0, x - b)

    power = n_int - 3  
    K = kf * float(n_int - 2) * (left ** power + right ** power)

    return float(K) if is_scalar else np.asarray(K, dtype=float)
