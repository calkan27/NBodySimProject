"""
This module implements gravitational force calculations with softening for close
encounter handling.

The gravitational_force function computes pairwise Plummer-softened gravitational forces
using optimized numpy operations, while dV_d_epsilon calculates the derivative of
potential energy with respect to the softening parameter. The module uses
geometry_buffers for efficient distance calculations, handles edge cases like zero
gravity or single particles, and maintains numerical stability through careful infinity
handling in distance matrices. The softened_forces function provides an alternative
interface with explicit array type checking. All functions assume 2D position arrays and
positive masses.
"""

from __future__ import annotations
import numpy as np
from .geometry_cache import geometry_buffers
from numpy.typing import NDArray









def _geometry(q: np.ndarray, eps: float):
    dr, r2, inv_r3 = geometry_buffers(q, eps)   
    r2_soft = r2 + eps * eps
    np.fill_diagonal(r2_soft, np.inf)
    return dr, r2_soft, inv_r3


def softened_forces(
    q: NDArray[np.floating],
    m: NDArray[np.floating],
    G: float,
    eps: float,
) -> NDArray[np.floating]:

    q_arr = np.asarray(q, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        return np.zeros_like(q_arr, dtype=float)
    if m_arr.size != q_arr.shape[0]:
        return np.zeros_like(q_arr, dtype=float)
    if q_arr.shape[0] < 2:
        return np.zeros_like(q_arr, dtype=float)
    if float(G) == 0.0:
        return np.zeros_like(q_arr, dtype=float)

    dr, r2, inv_r3 = geometry_buffers(q_arr, float(eps))

    pair_coeff = -(float(G) * (m_arr[:, None] * m_arr[None, :]))[..., None]
    F_pair = pair_coeff * inv_r3[..., None] * dr
    F = np.sum(F_pair, axis=1)
    return np.asarray(F, dtype=float)



def gravitational_force(q: np.ndarray,
                        m: np.ndarray,
                        eps: float = 0.0,
                        G: float = 1.0) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)

    if q.shape[0] < 2 or G == 0.0:
        return np.zeros_like(q)

    dr, _, inv_r3 = _geometry(q, eps)
    F_pair = -(G * m[:, None] * m[None, :])[..., None] * inv_r3[..., None] * dr
    return F_pair.sum(axis=1)                           

def dV_d_epsilon(
    q: NDArray[np.floating],
    m: NDArray[np.floating],
    eps: float,
    G: float = 1.0,
) -> float:

    q_arr = np.asarray(q, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        return 0.0
    if m_arr.size != q_arr.shape[0]:
        return 0.0
    if q_arr.shape[0] < 2:
        return 0.0
    if float(G) == 0.0:
        return 0.0

    eps_f = float(eps)
    if eps_f == 0.0:
        return 0.0

    diff = q_arr[:, None, :] - q_arr[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)

    r2_soft = r2 + eps_f * eps_f
    np.fill_diagonal(r2_soft, np.inf)

    r32 = np.power(r2_soft, 1.5)

    iu = np.triu_indices(q_arr.shape[0], 1)
    mprod = (m_arr[:, None] * m_arr[None, :])[iu]

    val = G * eps_f * float(np.sum(mprod / r32[iu]))
    return float(val)



pairwise_force = gravitational_force

