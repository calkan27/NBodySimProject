from __future__ import annotations
import math
from typing import Final, Tuple
from numpy import logaddexp, exp
import numpy as np

from .hamsoft_constants import LAMBDA_SOFTENING as _LAM_SOFT

__all__ = ["grad_eps_target", "eps_target"]

"""
This module computes optimal softening parameters based on particle distributions. Key functions include eps_target which calculates softening using harmonic mean neighbor distances, and grad_eps_target which provides gradients for force calculations. The implementation uses efficient numpy operations for pairwise distance calculations and includes careful handling of numerical edge cases. The module supports both static and gradient-based adaptive softening schemes and assumes 2D particle positions with sufficient neighbor density.

"""



def _pairwise_rms_length_and_grad(q: np.ndarray) -> tuple[float, np.ndarray]:

	q = np.asarray(q, dtype=float)
	n = q.shape[0]
	if n < 2:
		return 0.0, np.zeros_like(q, dtype=float)

	Qsum = np.sum(q, axis=0)                         
	sum_norm2 = float(np.sum(q * q))
	S = n * sum_norm2 - float(np.dot(Qsum, Qsum))

	if S <= 0.0:
		return 0.0, np.zeros_like(q, dtype=float)

	c = 2.0 / (n * (n - 1))
	L = float(np.sqrt(c * S))

	factor = c / L
	gradL = factor * (n * q - Qsum[None, :])         
	return L, gradL



def eps_target(q: np.ndarray, *, alpha: float = 1.0, lam: float = 0.3) -> float:

    q_arr = np.asarray(q, dtype=float)

    if not isinstance(q_arr, np.ndarray):
        return 0.0
    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        return 0.0

    n = int(q_arr.shape[0])
    if n < 2:
        return 0.0

    diff = q_arr[:, None, :] - q_arr[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    r = np.sqrt(r2)

    delta = 1.0e-12
    den = r + delta
    np.fill_diagonal(den, np.inf)

    iu = np.triu_indices(n, 1)

    M = float(n)
    inv_den = 1.0 / den[iu]
    D = float(np.sum(inv_den))

    if not np.isfinite(D) or D <= 0.0:
        return 0.0

    eps_star = float(lam) * (M / D)
    if not np.isfinite(eps_star):
        return 0.0
    return float(eps_star)




def grad_eps_target(q: np.ndarray, *, alpha: float = 1.0, lam: float = 0.3) -> np.ndarray:

    q_arr = np.asarray(q, dtype=float)

    if not isinstance(q_arr, np.ndarray):
        return np.zeros((0, 2), dtype=float)
    if q_arr.ndim != 2:
        n_guess = int(q_arr.shape[0]) if hasattr(q_arr, "shape") and len(q_arr.shape) >= 1 else 0
        return np.zeros((n_guess, 2), dtype=float)
    if q_arr.shape[1] != 2:
        return np.zeros((int(q_arr.shape[0]), 2), dtype=float)

    n = int(q_arr.shape[0])
    if n < 2:
        return np.zeros((n, 2), dtype=float)

    diff = q_arr[:, None, :] - q_arr[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    r = np.sqrt(r2)

    delta = 1.0e-12
    r_safe = np.maximum(r, 1.0e-15)
    den = r_safe + delta
    np.fill_diagonal(r_safe, np.inf)
    np.fill_diagonal(den,    np.inf)

    iu = np.triu_indices(n, 1)
    inv_den_pairs = 1.0 / den[iu]
    D = float(np.sum(inv_den_pairs))

    if not np.isfinite(D) or D <= 0.0:
        return np.zeros((n, 2), dtype=float)

    M = float(n)
    c_pref = float(lam) * (M / (D * D))

    A = 1.0 / (r_safe * den * den)
    np.fill_diagonal(A, 0.0)

    weighted = A[:, :, None] * diff
    grad = -c_pref * np.sum(weighted, axis=1)

    if not np.all(np.isfinite(grad)):
        grad = np.where(np.isfinite(grad), grad, 0.0)

    return np.asarray(grad, dtype=float)








