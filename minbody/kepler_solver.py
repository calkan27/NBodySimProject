"""
This module implements Kepler's equation solver using Stiefel-Scheifele universal
variables.

The UniversalVariableKeplerSolver class provides exact two-body orbital propagation
through the propagate method using iterative Newton-Raphson solution, Stumpff
C-functions for numerical stability, and support for all orbit types (elliptic,
parabolic, hyperbolic). The implementation is crucial for the WHFast integrator's Kepler
drift operations and maintains high accuracy through careful numerical formulations. It
assumes Newtonian gravity and handles both single particle and array inputs.
"""

from __future__ import annotations
import math 
import numpy as np
from numpy import logaddexp, exp







class UniversalVariableKeplerSolver:
	def _cfunc(self, z: float):
		z = float(z)
		n = 0
		while abs(z) > 0.1:
			z *= 0.25
			n += 1
		z2 = z * z
		c0 = 1 - z * 0.5 + z2 / 24 - z * z2 / 720 + z2 * z2 / 40320
		c1 = 1 - z / 6 + z2 / 120 - z * z2 / 5040 + z2 * z2 / 362880
		c2 = 0.5 - z / 24 + z2 / 720 - z * z2 / 40320
		c3 = 1 / 6 - z / 120 + z2 / 5040 - z * z2 / 362880
		while n:
			z *= 4
			n -= 1
			c3_old = c3
			c1_old = c1
			c2_old = c2
			c0 = 1 - z * c2_old
			c1 = 1 - z * c3_old
			c2 = 0.5 - z * (c3_old * (1 + c1_old)) * 0.125
			c3 = (c1_old - 1) / z
		return c0, c1, c2, c3

	def _propagate_single(self, r, v, mu, dt):
		r = np.asarray(r, dtype=float)
		v = np.asarray(v, dtype=float)
		mu = float(mu)
		dt = float(dt)
		r0 = float(math.hypot(r[0], r[1]))
		if r0 < 1e-14:
			return r + v * dt, v
		vr0 = float(np.dot(r, v) / r0)
		v2 = float(np.dot(v, v))
		alpha = 2 / r0 - v2 / mu
		sqrt_mu = math.sqrt(mu)
		if abs(alpha) > 1e-12:
			chi = sqrt_mu * abs(alpha) * dt
		else:
			chi = sqrt_mu * dt / r0
		prev1 = math.nan
		prev2 = math.nan
		for _ in range(64):
			z = alpha * chi * chi
			c0, c1, c2, c3 = self._cfunc(z)
			f = r0 * vr0 / sqrt_mu * chi * chi * c1 + (1 - alpha * r0) * chi * chi * chi * c2 + r0 * chi - sqrt_mu * dt
			fp = r0 * vr0 / sqrt_mu * chi * (1 - alpha * chi * chi * c2) + (1 - alpha * r0) * chi * chi * c1 + r0
			if fp == 0:
				break
			chi_new = chi - f / fp
			prev2 = prev1
			prev1 = chi_new
			if chi_new == chi or chi_new == prev2:
				chi = chi_new
				break
			chi = chi_new
		z = alpha * chi * chi
		c0, c1, c2, c3 = self._cfunc(z)
		f = 1 - chi * chi * c2 / r0
		g = dt - chi * chi * chi * c3 / sqrt_mu
		r_vec = f * r + g * v
		rn = float(math.hypot(r_vec[0], r_vec[1]))
		if rn == 0:
			return r_vec, v
		fdot = sqrt_mu / (rn * r0) * (alpha * chi * chi * c3 - chi)
		gdot = 1 - chi * chi * c2 / rn
		v_vec = fdot * r + gdot * v
		return r_vec, v_vec


	def propagate(self, r, v, mu, dt):
		r = np.asarray(r, dtype=float)
		v = np.asarray(v, dtype=float)
		mu = float(mu)
		dt = float(dt)
		if r.ndim == 1:
			return self._propagate_single(r, v, mu, dt)
		out_r = []
		out_v = []
		for ri, vi in zip(r, v):
			rn, vn = self._propagate_single(ri, vi, mu, dt)
			out_r.append(rn)
			out_v.append(vn)
		return np.array(out_r), np.array(out_v)

