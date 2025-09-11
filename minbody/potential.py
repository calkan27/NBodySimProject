from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .forces import dV_d_epsilon as _dVdeps

"""
This module computes gravitational potentials and their derivatives for N-body systems. The softened_potential function calculates the total Plummer-softened gravitational potential energy, while dU_d_eps computes the derivative with respect to the softening parameter. Both functions use efficient numpy operations for pairwise interactions, handle edge cases like single particles or zero gravity, and maintain numerical stability through careful distance calculations. The module supports the energy conservation machinery in adaptive softening schemes. It assumes 2D positions and positive masses.

"""

__all__ = ["softened_potential", "dU_d_eps"]


def softened_potential(
    q: NDArray[np.floating],
    m: NDArray[np.floating],
    G: float,
    eps: float,
) -> float:

    q_arr = np.asarray(q, dtype=float)
    m_arr = np.asarray(m, dtype=float).ravel()

    if not isinstance(q_arr, np.ndarray):
        return 0.0
    if q_arr.ndim != 2:
        return 0.0
    if q_arr.shape[1] != 2:
        return 0.0

    n = int(q_arr.shape[0])
    if n < 2:
        return 0.0
    if float(G) == 0.0:
        return 0.0
    if m_arr.size != n:
        return 0.0

    diff = q_arr[:, None, :] - q_arr[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)

    s2 = float(eps) * float(eps)
    r2_soft = r2 + s2

    iu = np.triu_indices(n, 1)

    r_soft = np.sqrt(r2_soft[iu])
    inv_r = np.zeros_like(r_soft)
    mask = r_soft > 0.0
    if np.any(mask):
        inv_r[mask] = 1.0 / r_soft[mask]

    term = (m_arr[iu[0]] * m_arr[iu[1]]) * inv_r
    U = -float(G) * float(np.sum(term))
    return float(U)


def dU_d_eps(
    q: NDArray[np.floating],
    m: NDArray[np.floating],
    G: float,
    eps: float,
) -> float:

    return float(_dVdeps(q, m, float(eps), float(G)))


