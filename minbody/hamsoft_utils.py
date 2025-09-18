from __future__ import annotations
from numpy.typing import NDArray
import math
from typing import Tuple

import numpy as np
from .forces import dV_d_epsilon  

"""
This utility module provides helper functions for the Hamiltonian softening implementation. Key functions include symplectic_bounce for exact reflection dynamics at barriers, reflect_if_needed for periodic boundary normalization, dU_depsilon_plummer as a wrapper for potential derivatives, and high-precision summation methods for energy conservation. The module ensures numerical stability through careful handling of boundary cases and provides both exact and approximate solutions for reflection dynamics. It assumes floating-point inputs and maintains compatibility with numpy's extended precision types when available.


"""





if not hasattr(np, "float128"):
	np.float128 = np.longdouble

def symplectic_bounce(
	eps: float,
	pi: float,
	eps_min: float,
	eps_max: float,
	h: float,
	mu: float,
):

	eps = float(eps)
	pi = float(pi)
	h_left = float(h)

	mu_in = float(mu)
	if mu_in == 0.0:
		mu = 1.0
	else:
		mu = mu_in

	a = float(eps_min)
	b = float(eps_max)
	if not np.isfinite(a) or not np.isfinite(b) or b <= a:
		return float(a), float(-pi)

	eps, pi = reflect_if_needed(float(eps), float(pi), a, b)

	tol = 1.0e-18
	if abs(h_left) <= tol:
		return float(eps), float(pi)

	iter_count = 0
	iter_cap = 1000  

	while abs(h_left) > tol:
		v = pi / mu
		if v == 0.0:
			break

		bound = b if v > 0.0 else a
		t_hit = (bound - eps) / v  

		if (not math.isfinite(t_hit)) or (abs(t_hit) <= tol):
			eps = eps + v * h_left
			h_left = 0.0
			break

		if h_left > 0.0:
			if (t_hit > 0.0) and (t_hit <= h_left):
				eps = bound
				pi = -pi
				h_left = h_left - t_hit
			else:
				eps = eps + v * h_left
				h_left = 0.0
		else:
			if (t_hit < 0.0) and (t_hit >= h_left):
				eps = bound
				pi = -pi
				h_left = h_left - t_hit
			else:
				eps = eps + v * h_left
				h_left = 0.0

		iter_count = iter_count + 1
		if iter_count > iter_cap:
			eps = eps + v * h_left
			h_left = 0.0
			break

	eps, pi = reflect_if_needed(float(eps), float(pi), a, b)
	return float(eps), float(pi)


	
def symplectic_reflect_eps(
    eps: float,
    pi: float,
    eps_min: float,
    eps_max: float,
    *legacy_args: float,          
    mu: float | None = None,      
    max_ratio: float = 2.0,      
) -> Tuple[float, float]:

    if len(legacy_args) == 0:
        h = 0.0
    elif len(legacy_args) == 2:
        h, mu_pos = legacy_args
        if mu is None:
            mu = mu_pos
    else:
        print("symplectic_reflect_eps: expected 0 or 2 extra positional args (h, mu)")
        return float(eps), float(pi)

    if mu is None:
        print("symplectic_reflect_eps: missing required argument 'mu'")
        return float(eps), float(pi)

    eps = float(eps)
    pi  = float(pi)
    eps_min = float(eps_min)
    eps_max = float(eps_max)
    h   = float(h)
    if float(mu) == 0.0:
        mu = 1.0
    else:
        mu = float(mu)

    eps, pi = reflect_if_needed(eps, pi, eps_min, eps_max)

    if abs(h) > 0.0 and pi != 0.0:
        eps, pi = symplectic_bounce(eps, pi, eps_min, eps_max, h, mu)

    return float(eps), float(pi)

def reflect_eps_symplectic(
	eps: float,
	pi: float,
	eps_min: float,
	eps_max: float,
	h: float,
	mu: float,
	*,
	max_ratio: float = 2.0,  
) -> tuple[float, float]:
	return symplectic_reflect_eps(eps, pi, eps_min, eps_max, h, mu)


def reflect_if_needed(
    eps: float,
    pi: float,
    eps_min: float,
    eps_max: float,
) -> Tuple[float, float]:
    a = float(eps_min)
    b = float(eps_max)
    e = float(eps)
    p = float(pi)

    R = b - a
    if not np.isfinite(R) or R <= 0.0:
        return float(a), float(-p)

    P = 2.0 * R
    y = (e - a) % P

    if y <= R:
        e_out = a + y
        p_out = p
    else:
        e_out = b - (y - R)
        p_out = -p

    return float(e_out), float(p_out)



def _pairwise_sum(arr: np.ndarray) -> float:
	arr = arr.ravel().astype(np.float64, copy=False)

	if arr.size == 0:
		return 0.0

	while arr.size > 1:
		pair_sum = arr[: (arr.size & ~1)].reshape(-1, 2).sum(axis=1)
		if arr.size % 2:
			arr = np.concatenate([pair_sum, arr[-1:]])
		else:
			arr = pair_sum

	return float(arr[0])



def _pairwise_r2(pos128: NDArray[np.float128]) -> NDArray[np.float128]:
	diff = pos128[:, None, :] - pos128[None, :, :]
	r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True, dtype=np.float128)
	return r2





def _kahan_sum128(arr: NDArray[np.float128]) -> np.float128:
	s = np.float128(0.0)
	err = np.float128(0.0)
	for x in arr.ravel():
		y = x - err
		tmp = s + y
		err = (tmp - s) - y
		s = tmp
	return s


def dU_depsilon_plummer(
	pos: NDArray[np.floating],
	mass: NDArray[np.floating],
	G: float,
	epsilon: float,
) -> float:
	return float(dV_d_epsilon(pos, mass, epsilon, G))


def reflect_and_limit_eps(
	eps: float,
	pi: float,
	eps_min: float,
	eps_max: float,
	h: float,
	mu: float,
	*,
	max_ratio: float = 2.0,
) -> tuple[float, float]:
	if max_ratio < 1.0:
		print("reflect_and_limit_eps: max_ratio must be ≥ 1; returning unmodified (eps, pi).")
		return float(eps), float(pi)

	eps0 = float(eps)
	eps_new, pi_new = symplectic_reflect_eps(
		float(eps), float(pi), float(eps_min), float(eps_max), float(h), float(mu)
	)

	upper = eps0 * max_ratio
	lower = eps0 / max_ratio
	if eps_new > upper:
		eps_new = upper
	elif eps_new < lower:
		eps_new = lower

	eps_new, pi_new = reflect_if_needed(float(eps_new), float(pi_new), float(eps_min), float(eps_max))
	return float(eps_new), float(pi_new)


__all__ = [
	"dU_depsilon_plummer",
	"reflect_if_needed",
	"symplectic_bounce",
	"symplectic_reflect_eps",
	"_pairwise_sum",
	"reflect_eps_symplectic",
	"reflect_and_limit_eps",
]


