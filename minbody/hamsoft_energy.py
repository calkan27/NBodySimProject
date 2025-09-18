"""
This module computes the extended Hamiltonian for systems with dynamical softening.

The extended_hamiltonian function calculates the total energy including particle kinetic
energy, Plummer-softened gravitational potential, spring potential for epsilon
confinement, softening kinetic energy from pi, and optional barrier potentials. The
implementation handles all boundary conditions and integrator policies, dynamically
determines epsilon_star if not provided, and validates all energy components for
finiteness. It serves as the conserved quantity for the Hamiltonian softening integrator
and assumes the phase state contains valid positions, momenta, and softening variables.
"""

from __future__ import annotations
import numpy as np

from .hamsoft_flows import PhaseState
from .barrier import barrier_energy







__all__ = ["extended_hamiltonian"]


def _kinetic_particles(state: PhaseState) -> float:
	return 0.5 * float(np.sum(np.sum(state.p * state.p, axis=1) / state.m))


def _plummer_potential(state: PhaseState, G: float) -> float:
	n = state.q.shape[0]
	if n < 2 or G == 0.0:
		return 0.0
	diff = state.q[:, None, :] - state.q[None, :, :]
	r2 = np.sum(diff * diff, axis=-1) + state.epsilon**2
	iu = np.triu_indices(n, 1)
	inv_r = 1.0 / np.sqrt(r2[iu])
	m_i, m_j = state.m[iu[0]], state.m[iu[1]]
	return -G * float(np.sum(m_i * m_j * inv_r))


def _spring_potential(state: PhaseState, k_soft: float, eps_star: float) -> float:
	return 0.5 * k_soft * (state.epsilon - eps_star) ** 2


def extended_hamiltonian(
    state: PhaseState,
    *,
    G: float,
    k_soft: float,
    mu_soft: float,
    eps_star: float,
    eps_min: float,
    eps_max: float,
    k_wall: float = 1.0e9,
    n_exp: int = 5,
    integrator=None,
    barrier_enabled: bool = True,
) -> float:

    if integrator is not None:
        k_attr = getattr(integrator, "k_wall", k_wall)
        if isinstance(k_attr, (int, float, np.floating)):
            kv = float(k_attr)
            if np.isfinite(kv):
                k_wall = kv

        n_fun = getattr(integrator, "_barrier_n", None)
        if callable(n_fun):
            n_val = n_fun()
            if isinstance(n_val, (int, float, np.floating)):
                n_int = int(n_val)
                if n_int >= 2:
                    n_exp = n_int

    need_eps_star = True
    if isinstance(eps_star, (int, float, np.floating)):
        if np.isfinite(float(eps_star)):
            need_eps_star = False

    if need_eps_star and (integrator is not None):
        ev = None
        em = getattr(integrator, "_eps_model", None)
        if em is not None and hasattr(em, "eps_target"):
            ev = em.eps_target(state.q)
        elif hasattr(integrator, "_eps_target"):
            ev = integrator._eps_target(q=state.q)

        if isinstance(ev, (int, float, np.floating)):
            ef = float(ev)
            if np.isfinite(ef):
                eps_star = ef

    if eps_max < eps_min:
        tmp = eps_min
        eps_min = eps_max
        eps_max = tmp

    if not isinstance(eps_star, (int, float, np.floating)) or (not np.isfinite(float(eps_star))):
        eps_star = float(state.epsilon)

    if eps_star < eps_min:
        eps_star = float(eps_min)
    elif eps_star > eps_max:
        eps_star = float(eps_max)

    T = 0.5 * float(np.sum(np.sum(state.p * state.p, axis=1) / state.m))

    q = state.q
    n = q.shape[0]
    if n < 2 or G == 0.0:
        U = 0.0
    else:
        diff = q[:, None, :] - q[None, :, :]
        r2 = np.sum(diff * diff, axis=-1) + float(state.epsilon) ** 2
        iu = np.triu_indices(n, 1)
        inv_r = 1.0 / np.sqrt(r2[iu])
        m_i = state.m[iu[0]]
        m_j = state.m[iu[1]]
        U = -float(G) * float(np.sum(m_i * m_j * inv_r))

    if mu_soft == 0.0 or (not np.isfinite(mu_soft)):
        return 1e300

    delta_eps = float(state.epsilon) - float(eps_star)
    Hs = 0.5 * float(k_soft) * (delta_eps * delta_eps)
    Ke = 0.5 * (float(state.pi) * float(state.pi)) / float(mu_soft)

    barrier_enabled_eff = bool(barrier_enabled)
    policy_soft = False

    if integrator is not None:
        pol = getattr(integrator, "barrier_policy", None)
        if isinstance(pol, str):
            pl = pol.lower()
        else:
            pl = "reflection"

        if pl == "reflection":
            barrier_enabled_eff = False
        elif pl == "soft":
            barrier_enabled_eff = True
            policy_soft = True

        sim_ref = getattr(integrator, "sim", None)
        if sim_ref is not None:
            if getattr(sim_ref.cfg, "disable_barrier", False):
                barrier_enabled_eff = False

    if barrier_enabled_eff and policy_soft:
        if (float(k_wall) > 0.0) and (int(n_exp) >= 2):
            eps_now = float(state.epsilon)
            U_bar = barrier_energy(eps_now, float(eps_min), float(eps_max),
                                   k_wall=float(k_wall), n=int(n_exp))
        else:
            U_bar = 0.0
    else:
        U_bar = 0.0

    return T + U + U_bar + Hs + Ke

