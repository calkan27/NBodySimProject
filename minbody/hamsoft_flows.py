from __future__ import annotations

from dataclasses import dataclass
from typing import Final
import copy
import numpy as np
from .barrier import barrier_force
import math
from math import sqrt, sin, cos
from .hamsoft_utils import reflect_if_needed
from .potential import dU_d_eps as _dU_d_eps
from .forces import dV_d_epsilon as _dVdeps

"""

This module implements the symplectic flow maps for the Hamiltonian softening dynamics. Key components include the PhaseState dataclass encapsulating the extended phase space, spring_oscillation function for the harmonic oscillator evolution of epsilon, strang_softening_step for complete Strang-split timesteps, and auxiliary functions for momentum increments and wall collision detection. The implementation carefully preserves symplectic structure through exact solution of the harmonic oscillator, proper handling of barrier kicks at half-steps, and gradient-based momentum updates. The module includes sophisticated theta-angle calculations for small timesteps to maintain numerical precision. It assumes the integrator provides valid gradient functions and barrier parameters.



"""





__all__ = [
    "PhaseState",
    "spring_oscillation",
    "strang_softening_step",
]

@dataclass(frozen=True, slots=True)
class PhaseState:
    q: np.ndarray         
    p: np.ndarray         
    epsilon: float        
    pi: float             
    m: np.ndarray

def strang_softening_step(
    state: PhaseState,
    dt: float,
    *,
    k_soft: float,
    eps_min: float,
    eps_max: float,
    k_wall: float = 1.0e9,
    n_exp: int | None = None,
    integrator=None,
) -> PhaseState:
    new_state = spring_oscillation(
        state,
        float(dt),
        float(k_soft),
        mu=None,
        cfg=None,
        q_frozen=None,
        integrator=integrator,
        eps_star_override=None,
        grad_override=None,
    )

    do_reflect = False
    if integrator is not None:
        pol_attr = getattr(integrator, "barrier_policy", "reflection")
        if isinstance(pol_attr, str):
            pol = pol_attr.lower()
        else:
            pol = "reflection"

        cfg_ref = getattr(integrator, "sim", None)
        if cfg_ref is not None:
            cfg = getattr(cfg_ref, "cfg", None)
        else:
            cfg = None

        barrier_disabled = False
        if cfg is not None:
            barrier_disabled = bool(getattr(cfg, "disable_barrier", False))

        if pol == "reflection":
            if not barrier_disabled:
                do_reflect = True

    if do_reflect:
        eps_reflect, pi_reflect = reflect_if_needed(
            float(new_state.epsilon),
            float(new_state.pi),
            float(eps_min),
            float(eps_max),
        )
        eps_out = float(eps_reflect)
        pi_out = float(pi_reflect)
    else:
        eps_out = float(new_state.epsilon)
        pi_out = float(new_state.pi)

    return PhaseState(
        q=new_state.q.copy(),
        p=new_state.p.copy(),
        epsilon=float(eps_out),
        pi=float(pi_out),
        m=new_state.m.copy(),
    )


def spring_oscillation(
    state: PhaseState,
    dt: float,
    k_soft: float,
    *,
    mu: float | None = None,
    cfg=None,
    q_frozen: np.ndarray | None = None,
    integrator=None,
    eps_star_override: float | None = None,
    grad_override: np.ndarray | None = None,
) -> PhaseState:
    q = np.asarray(state.q, dtype=float)
    p = np.asarray(state.p, dtype=float)
    m = np.asarray(state.m, dtype=float)

    if q_frozen is not None:
        q_ref = np.asarray(q_frozen, dtype=float)
    else:
        q_ref = q

    if isinstance(mu, (int, float, np.floating)):
        mu_eff = float(mu)
    else:
        if integrator is not None:
            if hasattr(integrator, "mu_soft"):
                mu_eff = float(getattr(integrator, "mu_soft"))
            else:
                mu_eff = float(np.sum(m)) if np.all(np.isfinite(m)) else 1.0
        else:
            mu_eff = float(np.sum(m)) if np.all(np.isfinite(m)) else 1.0

    if not np.isfinite(mu_eff):
        mu_eff = 1.0
    if mu_eff == 0.0:
        mu_eff = 1.0

    k_s = float(k_soft) if isinstance(k_soft, (int, float, np.floating)) else 0.0
    if not np.isfinite(k_s):
        k_s = 0.0

    dt_f = float(dt) if isinstance(dt, (int, float, np.floating)) else 0.0

    eps0 = float(state.epsilon)
    pi0 = float(state.pi)

    if integrator is not None:
        es_call, gg_call = integrator.eps_star_and_grad(q_ref)
    else:
        es_call = None
        gg_call = None

    if isinstance(es_call, (int, float, np.floating)):
        eps_star_from_call = float(es_call)
        if not np.isfinite(eps_star_from_call):
            if (integrator is not None) and hasattr(integrator, "sim") and hasattr(integrator.sim, "manager"):
                eps_star_from_call = float(integrator.sim.manager.s0)
            else:
                eps_star_from_call = float(eps0)
    else:
        if (integrator is not None) and hasattr(integrator, "sim") and hasattr(integrator.sim, "manager"):
            eps_star_from_call = float(integrator.sim.manager.s0)
        else:
            eps_star_from_call = float(eps0)

    if isinstance(gg_call, np.ndarray) and gg_call.shape == q_ref.shape:
        grad_from_call = np.asarray(gg_call, dtype=float)
        if not np.all(np.isfinite(grad_from_call)):
            grad_from_call = np.where(np.isfinite(grad_from_call), grad_from_call, 0.0)
    else:
        grad_from_call = np.zeros_like(q_ref, dtype=float)

    eps_star = float(eps_star_from_call)
    if isinstance(eps_star_override, (int, float, np.floating)):
        eps_star_ov = float(eps_star_override)
        if np.isfinite(eps_star_ov):
            eps_star = float(eps_star_ov)
        else:
            eps_star = float(eps_star_from_call)

    if isinstance(grad_override, np.ndarray) and grad_override.shape == q_ref.shape:
        grad = np.asarray(grad_override, dtype=float)
        if not np.all(np.isfinite(grad)):
            grad = np.where(np.isfinite(grad), grad, 0.0)
    else:
        grad = np.asarray(grad_from_call, dtype=float)

    barrier_enabled = False
    include_curv = False
    k_eff = float(k_s)

    if integrator is not None:
        pol_attr = getattr(integrator, "barrier_policy", "reflection")
        if isinstance(pol_attr, str):
            pol = pol_attr.lower()
        else:
            pol = "reflection"

        sim_ref = getattr(integrator, "sim", None)
        if sim_ref is not None:
            barrier_disabled = bool(getattr(sim_ref.cfg, "disable_barrier", False))
        else:
            barrier_disabled = True

        if pol == "soft":
            if not barrier_disabled:
                barrier_enabled = True

        if sim_ref is not None:
            include_curv = bool(getattr(sim_ref.cfg, "include_barrier_curvature_in_S", False))

    if barrier_enabled:
        if include_curv:
            if integrator is not None:
                sim_ref2 = getattr(integrator, "sim", None)
            else:
                sim_ref2 = None

            if sim_ref2 is not None:
                eps_min = float(getattr(sim_ref2, "_min_softening", 0.0))
                if hasattr(sim_ref2, "manager") and sim_ref2.manager is not None:
                    eps_max = float(getattr(sim_ref2, "_max_softening", 10.0 * sim_ref2.manager.s0))
                else:
                    eps_max = max(1.0, float(eps_min))
            else:
                eps_min = 0.0
                eps_max = 1.0

            k_wall_val = 0.0
            if integrator is not None:
                k_wall_val = float(getattr(integrator, "k_wall", 0.0))

            n_eff = 2
            if integrator is not None:
                if hasattr(integrator, "_barrier_n"):
                    n_tmp = integrator._barrier_n()
                    if isinstance(n_tmp, (int, float, np.floating)):
                        n_eff = int(n_tmp)

            k_eff = float(k_s)
        else:
            k_eff = float(k_s)
    else:
        k_eff = float(k_s)

    if k_eff > 0.0 and mu_eff > 0.0:
        omega = float(np.sqrt(k_eff / mu_eff))
    else:
        omega = 0.0
    theta = float(omega * dt_f)

    if abs(theta) < 1.0e-8:
        th = theta
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th4 * th
        sin_theta = float(th - th3 / 6.0 + th5 / 120.0)
        cos_theta = float(1.0 - th2 / 2.0 + th4 / 24.0)
    else:
        sin_theta = float(np.sin(theta))
        cos_theta = float(np.cos(theta))

    if integrator is not None:
        integrator._last_s_trig = {
            "theta": float(theta),
            "sin": float(sin_theta),
            "cos": float(cos_theta),
            "one_minus_cos": float(1.0 - cos_theta),
        }

    dUbar0 = 0.0
    dUbar_after = 0.0
    pi_kick1 = 0.0
    pi_kick2 = 0.0

    if barrier_enabled:
        if integrator is not None:
            sim_ref3 = getattr(integrator, "sim", None)
        else:
            sim_ref3 = None

        if sim_ref3 is not None:
            eps_min = float(getattr(sim_ref3, "_min_softening", 0.0))
            if hasattr(sim_ref3, "manager") and sim_ref3.manager is not None:
                eps_max = float(getattr(sim_ref3, "_max_softening", 10.0 * sim_ref3.manager.s0))
            else:
                eps_max = max(1.0, float(eps_min))
        else:
            eps_min = 0.0
            eps_max = 1.0

        k_wall_val = 0.0
        if integrator is not None:
            k_wall_val = float(getattr(integrator, "k_wall", 0.0))

        n_eff = 2
        if integrator is not None:
            if hasattr(integrator, "_barrier_n"):
                n_tmp = integrator._barrier_n()
                if isinstance(n_tmp, (int, float, np.floating)):
                    n_eff = int(n_tmp)

        Fbar0 = float(
            barrier_force(
                float(eps0),
                float(eps_min),
                float(eps_max),
                k_wall=float(k_wall_val),
                n=int(max(2, n_eff)),
            )
        )
        dUbar0 = -float(Fbar0)
        pi_kick1 = -0.5 * float(dt_f) * float(dUbar0)
    else:
        dUbar0 = 0.0
        pi_kick1 = 0.0

    Delta0 = float(eps0 - eps_star)
    pi_in = float(pi0 + pi_kick1)

    if (omega != 0.0) and (mu_eff != 0.0):
        mu_omega = float(np.sqrt(mu_eff * max(k_eff, 0.0)))
        one_minus_c = float(1.0 - cos_theta)

        delta_t = float(Delta0 * cos_theta + (pi_in / (mu_eff * omega)) * sin_theta)
        eta_t = float(pi_in * cos_theta - mu_omega * Delta0 * sin_theta)

        denom = float(mu_eff * omega * omega)
        if denom != 0.0:
            I_tau = float((Delta0 / omega) * sin_theta + (pi_in / denom) * one_minus_c)
        else:
            I_tau = 0.0
    else:
        delta_t = float(Delta0)
        eta_t = float(pi_in)
        I_tau = 0.0

    eps_rot = float(eps_star + delta_t)

    if barrier_enabled:
        if integrator is not None:
            sim_ref4 = getattr(integrator, "sim", None)
        else:
            sim_ref4 = None

        if sim_ref4 is not None:
            eps_min = float(getattr(sim_ref4, "_min_softening", 0.0))
            if hasattr(sim_ref4, "manager") and sim_ref4.manager is not None:
                eps_max = float(getattr(sim_ref4, "_max_softening", 10.0 * sim_ref4.manager.s0))
            else:
                eps_max = max(1.0, float(eps_min))
        else:
            eps_min = 0.0
            eps_max = 1.0

        k_wall_val = 0.0
        if integrator is not None:
            k_wall_val = float(getattr(integrator, "k_wall", 0.0))

        n_eff = 2
        if integrator is not None:
            if hasattr(integrator, "_barrier_n"):
                n_tmp = integrator._barrier_n()
                if isinstance(n_tmp, (int, float, np.floating)):
                    n_eff = int(n_tmp)

        Fbar_after = float(
            barrier_force(
                float(eps_rot),
                float(eps_min),
                float(eps_max),
                k_wall=float(k_wall_val),
                n=int(max(2, n_eff)),
            )
        )
        dUbar_after = -float(Fbar_after)
        pi_kick2 = -0.5 * float(dt_f) * float(dUbar_after)
    else:
        dUbar_after = 0.0
        pi_kick2 = 0.0

    pi_out = float(eta_t + pi_kick2)

    J = float(k_s) * float(I_tau)
    p_new = p + float(J) * grad

    if integrator is not None:
        info = {
            "I_tau": float(I_tau),
            "J": float(J),
            "grad_used": grad.copy(),
            "eps_star": float(eps_star),
            "omega": float(omega),
            "theta": float(theta),
            "barrier_kick1": float(pi_kick1),
            "barrier_kick2": float(pi_kick2),
            "k_eff": float(k_eff),
        }
        integrator._last_s_info = info
        integrator._last_s_half = dict(info)

    return PhaseState(
        q=q.copy(),
        p=p_new,
        epsilon=float(eps_rot),
        pi=float(pi_out),
        m=m.copy(),
    )



def spring_oscillation(
    state: PhaseState,
    dt: float,
    k_soft: float,
    *,
    mu: float | None = None,
    cfg=None,
    q_frozen: np.ndarray | None = None,
    integrator=None,
    eps_star_override: float | None = None,
    grad_override: np.ndarray | None = None,
) -> PhaseState:
    q = np.asarray(state.q, dtype=float)
    p = np.asarray(state.p, dtype=float)
    m = np.asarray(state.m, dtype=float)

    if q_frozen is not None:
        q_ref = np.asarray(q_frozen, dtype=float)
    else:
        q_ref = q

    if isinstance(mu, (int, float, np.floating)):
        mu_eff = float(mu)
    else:
        if integrator is not None:
            if hasattr(integrator, "mu_soft"):
                mu_eff = float(getattr(integrator, "mu_soft"))
            else:
                mu_eff = float(np.sum(m)) if np.all(np.isfinite(m)) else 1.0
        else:
            mu_eff = float(np.sum(m)) if np.all(np.isfinite(m)) else 1.0
    if not np.isfinite(mu_eff):
        mu_eff = 1.0
    if mu_eff == 0.0:
        mu_eff = 1.0

    k_s = float(k_soft) if isinstance(k_soft, (int, float, np.floating)) else 0.0
    if not np.isfinite(k_s):
        k_s = 0.0

    dt_f = float(dt) if isinstance(dt, (int, float, np.floating)) else 0.0

    eps0 = float(state.epsilon)
    pi0  = float(state.pi)

    if integrator is not None:
        es_call, gg_call = integrator.eps_star_and_grad(q_ref)
    else:
        es_call = None
        gg_call = None

    if isinstance(es_call, (int, float, np.floating)):
        eps_star_from_call = float(es_call)
        if not np.isfinite(eps_star_from_call):
            if (integrator is not None) and hasattr(integrator, "sim") and hasattr(integrator.sim, "manager"):
                eps_star_from_call = float(integrator.sim.manager.s0)
            else:
                eps_star_from_call = float(eps0)
    else:
        if (integrator is not None) and hasattr(integrator, "sim") and hasattr(integrator.sim, "manager"):
            eps_star_from_call = float(integrator.sim.manager.s0)
        else:
            eps_star_from_call = float(eps0)

    if isinstance(gg_call, np.ndarray) and gg_call.shape == q_ref.shape:
        grad_from_call = np.asarray(gg_call, dtype=float)
        if not np.all(np.isfinite(grad_from_call)):
            grad_from_call = np.where(np.isfinite(grad_from_call), grad_from_call, 0.0)
    else:
        grad_from_call = np.zeros_like(q_ref, dtype=float)

    eps_star = float(eps_star_from_call)
    if isinstance(eps_star_override, (int, float, np.floating)):
        eps_star_ov = float(eps_star_override)
        if np.isfinite(eps_star_ov):
            eps_star = float(eps_star_ov)
        else:
            eps_star = float(eps_star_from_call)

    if isinstance(grad_override, np.ndarray) and grad_override.shape == q_ref.shape:
        grad = np.asarray(grad_override, dtype=float)
        if not np.all(np.isfinite(grad)):
            grad = np.where(np.isfinite(grad), grad, 0.0)
    else:
        grad = np.asarray(grad_from_call, dtype=float)

    barrier_enabled = False
    include_curv = False
    k_eff = float(k_s)

    if integrator is not None:
        pol_attr = getattr(integrator, "barrier_policy", "reflection")
        if isinstance(pol_attr, str):
            pol = pol_attr.lower()
        else:
            pol = "reflection"

        sim_ref = getattr(integrator, "sim", None)
        if sim_ref is not None:
            barrier_disabled = bool(getattr(sim_ref.cfg, "disable_barrier", False))
        else:
            barrier_disabled = True

        if pol == "soft":
            if not barrier_disabled:
                barrier_enabled = True

        if sim_ref is not None:
            include_curv = bool(getattr(sim_ref.cfg, "include_barrier_curvature_in_S", False))

    if barrier_enabled:
        if include_curv:
            if integrator is not None:
                sim_ref2 = getattr(integrator, "sim", None)
            else:
                sim_ref2 = None
            if sim_ref2 is not None:
                eps_min = float(getattr(sim_ref2, "_min_softening", 0.0))
                if hasattr(sim_ref2, "manager") and sim_ref2.manager is not None:
                    eps_max = float(getattr(sim_ref2, "_max_softening", 10.0 * sim_ref2.manager.s0))
                else:
                    eps_max = max(1.0, float(eps_min))
            else:
                eps_min = 0.0
                eps_max = 1.0

            k_wall_val = 0.0
            if integrator is not None:
                k_wall_val = float(getattr(integrator, "k_wall", 0.0))
            n_eff = 2
            if integrator is not None:
                if hasattr(integrator, "_barrier_n"):
                    n_tmp = integrator._barrier_n()
                    if isinstance(n_tmp, (int, float, np.floating)):
                        n_eff = int(n_tmp)

            k_eff = float(k_s)  
        else:
            k_eff = float(k_s)
    else:
        k_eff = float(k_s)

    if k_eff > 0.0 and mu_eff > 0.0:
        omega = float(np.sqrt(k_eff / mu_eff))
    else:
        omega = 0.0
    theta = float(omega * dt_f)

    if abs(theta) < 1.0e-8:
        th  = theta
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th4 * th
        sin_theta = float(th  - th3 / 6.0  + th5 / 120.0)
        cos_theta = float(1.0 - th2 / 2.0  + th4 / 24.0)
    else:
        sin_theta = float(np.sin(theta))
        cos_theta = float(np.cos(theta))

    if integrator is not None:
        integrator._last_s_trig = {
            "theta": float(theta),
            "sin": float(sin_theta),
            "cos": float(cos_theta),
            "one_minus_cos": float(1.0 - cos_theta),
        }

    dUbar0 = 0.0
    dUbar_after = 0.0
    pi_kick1 = 0.0
    pi_kick2 = 0.0

    if barrier_enabled:
        if integrator is not None:
            sim_ref3 = getattr(integrator, "sim", None)
        else:
            sim_ref3 = None

        if sim_ref3 is not None:
            eps_min = float(getattr(sim_ref3, "_min_softening", 0.0))
            if hasattr(sim_ref3, "manager") and sim_ref3.manager is not None:
                eps_max = float(getattr(sim_ref3, "_max_softening", 10.0 * sim_ref3.manager.s0))
            else:
                eps_max = max(1.0, float(eps_min))
        else:
            eps_min = 0.0
            eps_max = 1.0

        k_wall_val = 0.0
        if integrator is not None:
            k_wall_val = float(getattr(integrator, "k_wall", 0.0))
        n_eff = 2
        if integrator is not None:
            if hasattr(integrator, "_barrier_n"):
                n_tmp = integrator._barrier_n()
                if isinstance(n_tmp, (int, float, np.floating)):
                    n_eff = int(n_tmp)

        Fbar0 = float(barrier_force(float(eps0), float(eps_min), float(eps_max),
                                    k_wall=float(k_wall_val), n=int(max(2, n_eff))))
        dUbar0 = -float(Fbar0)
        pi_kick1 = -0.5 * float(dt_f) * float(dUbar0)
    else:
        dUbar0 = 0.0
        pi_kick1 = 0.0

    Delta0 = float(eps0 - eps_star)
    pi_in  = float(pi0 + pi_kick1)

    if (omega != 0.0) and (mu_eff != 0.0):
        mu_omega = float(np.sqrt(mu_eff * max(k_eff, 0.0)))
        one_minus_c = float(1.0 - cos_theta)

        delta_t = float(Delta0 * cos_theta + (pi_in / (mu_eff * omega)) * sin_theta)
        eta_t   = float(pi_in * cos_theta   - mu_omega * Delta0 * sin_theta)

        denom = float(mu_eff * omega * omega)
        if denom != 0.0:
            I_tau = float((Delta0 / omega) * sin_theta + (pi_in / denom) * one_minus_c)
        else:
            I_tau = 0.0
    else:
        delta_t = float(Delta0)
        eta_t   = float(pi_in)
        I_tau   = 0.0

    eps_rot = float(eps_star + delta_t)

    if barrier_enabled:
        if integrator is not None:
            sim_ref4 = getattr(integrator, "sim", None)
        else:
            sim_ref4 = None

        if sim_ref4 is not None:
            eps_min = float(getattr(sim_ref4, "_min_softening", 0.0))
            if hasattr(sim_ref4, "manager") and sim_ref4.manager is not None:
                eps_max = float(getattr(sim_ref4, "_max_softening", 10.0 * sim_ref4.manager.s0))
            else:
                eps_max = max(1.0, float(eps_min))
        else:
            eps_min = 0.0
            eps_max = 1.0

        k_wall_val = 0.0
        if integrator is not None:
            k_wall_val = float(getattr(integrator, "k_wall", 0.0))
        n_eff = 2
        if integrator is not None:
            if hasattr(integrator, "_barrier_n"):
                n_tmp = integrator._barrier_n()
                if isinstance(n_tmp, (int, float, np.floating)):
                    n_eff = int(n_tmp)

        Fbar_after = float(barrier_force(float(eps_rot), float(eps_min), float(eps_max),
                                         k_wall=float(k_wall_val), n=int(max(2, n_eff))))
        dUbar_after = -float(Fbar_after)
        pi_kick2 = -0.5 * float(dt_f) * float(dUbar_after)
    else:
        dUbar_after = 0.0
        pi_kick2 = 0.0

    pi_out = float(eta_t + pi_kick2)

    J = float(k_s) * float(I_tau)

    p_rows = np.sqrt(np.sum(p * p, axis=1))
    if isinstance(p_rows, np.ndarray):
        if p_rows.size > 0:
            p_scale = float(np.max(p_rows))
        else:
            p_scale = 0.0
    else:
        p_scale = 0.0
    if p_scale < 1.0e-12:
        p_scale = 1.0e-12

    dp_trial = float(J) * grad
    dp_rows = np.sqrt(np.sum(dp_trial * dp_trial, axis=1))
    if isinstance(dp_rows, np.ndarray):
        if dp_rows.size > 0:
            dp_inf = float(np.max(dp_rows))
        else:
            dp_inf = 0.0
    else:
        dp_inf = 0.0

    j_max_cap = 0.02
    if integrator is not None:
        stepper_obj = getattr(integrator, "_hs_stepper", None)
        if stepper_obj is not None:
            getter = getattr(stepper_obj, "_get_j_max_cap", None)
            if callable(getter):
                val_cap = getter()
                if isinstance(val_cap, (int, float, np.floating)):
                    if np.isfinite(float(val_cap)):
                        if float(val_cap) > 0.0:
                            j_max_cap = float(val_cap)

    J_applied = float(J)
    threshold = float(j_max_cap) * float(p_scale)
    if dp_inf > threshold:
        if dp_inf > 0.0:
            scale = float(threshold) / float(dp_inf)
        else:
            scale = 1.0
        J_applied = float(J) * float(scale)
    else:
        J_applied = float(J)

    p_new = p + float(J_applied) * grad

    if integrator is not None:
        info = {
            "I_tau": float(I_tau),
            "J": float(J),
            "J_applied": float(J_applied),
            "grad_used": grad.copy(),
            "eps_star": float(eps_star),
            "omega": float(omega),
            "theta": float(theta),
            "barrier_kick1": float(pi_kick1),
            "barrier_kick2": float(pi_kick2),
            "k_eff": float(k_eff),
        }
        integrator._last_s_info = info
        integrator._last_s_half = dict(info)

    return PhaseState(
        q=q.copy(),
        p=p_new,
        epsilon=float(eps_rot),
        pi=float(pi_out),
        m=m.copy(),
    )



def _osc_segment_update(
    delta0: float,
    eta0: float,
    mu: float,
    k_soft: float,
    dt: float,
) -> tuple[float, float, float]:

    if (not np.isfinite(mu)) or (not np.isfinite(k_soft)) or (not np.isfinite(dt)):
        return float(delta0), float(eta0), float(0.0)

    if mu <= 0.0 or k_soft <= 0.0 or dt == 0.0:
        return float(delta0), float(eta0), float(0.0)

    hp   = np.longdouble
    mu_h = hp(mu)
    ks_h = hp(k_soft)
    dt_h = hp(dt)

    omega_h = np.sqrt(ks_h / mu_h)
    theta_h = omega_h * dt_h

    abs_th   = np.abs(theta_h)
    th_small = hp(1.0e-8)

    if abs_th < th_small:
        th  = theta_h
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th4 * th
        sin_th = th  - th3 / hp(6.0)   + th5 / hp(120.0)
        cos_th = hp(1.0) - th2 / hp(2.0) + th4 / hp(24.0)
    else:
        sin_th = np.sin(theta_h)
        cos_th = np.cos(theta_h)

    if theta_h == hp(0.0):
        sinc_h = hp(1.0)
        omc_over_theta_h = hp(0.0)
    else:
        sinc_h = sin_th / theta_h
        omc_over_theta_h = (hp(1.0) - cos_th) / theta_h

    mu_om_h = mu_h * omega_h

    d0_h = hp(delta0)
    e0_h = hp(eta0)

    if mu_om_h != hp(0.0):
        zeta0_h = e0_h / mu_om_h
    else:
        zeta0_h = hp(0.0)

    delta_t_h = d0_h * cos_th + zeta0_h * (theta_h * sinc_h)
    eta_t_h   = e0_h * cos_th - (mu_om_h * d0_h) * (theta_h * sinc_h)
    J_h       = theta_h * (e0_h * omc_over_theta_h + (mu_om_h * d0_h) * sinc_h)

    return float(delta_t_h), float(eta_t_h), float(J_h)





def s_half_momentum_increment(
    p: np.ndarray,
    epsilon: float,
    pi: float,
    k_soft: float,
    mu_soft: float,
    dt_half: float,
    grad_eps_star: np.ndarray,
    eps_star_value: float
) -> np.ndarray:

    p_arr = np.asarray(p, dtype=float)
    grad = np.asarray(grad_eps_star, dtype=float)

    if not isinstance(grad, np.ndarray) or grad.shape != p_arr.shape:
        return np.zeros_like(p_arr, dtype=float)

    if isinstance(k_soft, (int, float, np.floating)):
        ks = float(k_soft)
    else:
        ks = 0.0

    if isinstance(mu_soft, (int, float, np.floating)):
        mu = float(mu_soft)
    else:
        mu = 0.0

    if (not np.isfinite(ks)) or (not np.isfinite(mu)) or ks <= 0.0 or mu <= 0.0:
        return np.zeros_like(p_arr, dtype=float)

    eps0 = float(epsilon)
    pi0 = float(pi)
    eps_star = float(eps_star_value)

    if (not np.isfinite(eps0)) or (not np.isfinite(pi0)) or (not np.isfinite(eps_star)):
        return np.zeros_like(p_arr, dtype=float)

    dt = float(dt_half)
    if (not np.isfinite(dt)) or dt == 0.0:
        return np.zeros_like(p_arr, dtype=float)

    hp = np.longdouble

    ks_hp = hp(ks)
    mu_hp = hp(mu)
    dt_hp = hp(dt)

    omega_hp = np.sqrt(ks_hp / mu_hp)
    theta_hp = omega_hp * dt_hp

    abs_th = np.abs(theta_hp)
    th_small = hp(1.0e-8)

    if abs_th < th_small:
        th = theta_hp
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th4 * th
        sin_theta_hp = th - th3 / hp(6.0) + th5 / hp(120.0)
        cos_theta_hp = hp(1.0) - th2 / hp(2.0) + th4 / hp(24.0)
    else:
        sin_theta_hp = np.sin(theta_hp)
        cos_theta_hp = np.cos(theta_hp)

    if theta_hp == hp(0.0):
        sinc_h = hp(1.0)
        omc_over_theta_h = hp(0.0)
    else:
        sinc_h = sin_theta_hp / theta_hp
        omc_over_theta_h = (hp(1.0) - cos_theta_hp) / theta_hp

    eta0_hp = hp(pi0)
    delta0_hp = hp(eps0) - hp(eps_star)
    mu_omega_hp = np.sqrt(mu_hp * ks_hp)

    J_hp = theta_hp * (eta0_hp * omc_over_theta_h + (mu_omega_hp * delta0_hp) * sinc_h)

    if not np.isfinite(J_hp):
        return np.zeros_like(p_arr, dtype=float)

    dp = float(J_hp) * grad
    return np.asarray(dp, dtype=float)


def _compute_eps_wall_hits(
    eps0: float,
    pi0: float,
    eps_star: float,
    k_soft: float,
    mu: float,
    eps_min: float,
    eps_max: float,
    dt: float,
) -> list[tuple[float, float]]:

    out: list[tuple[float, float]] = []

    if not np.isfinite(k_soft):
        return out
    if not np.isfinite(mu):
        return out
    if not np.isfinite(dt):
        return out
    if k_soft <= 0.0:
        return out
    if mu <= 0.0:
        return out
    if dt == 0.0:
        return out

    delta0 = float(eps0) - float(eps_star)
    omega  = math.sqrt(float(k_soft) / float(mu))
    if not np.isfinite(omega):
        return out
    if omega <= 0.0:
        return out

    if mu * omega != 0.0:
        B = float(pi0) / (float(mu) * float(omega))
    else:
        B = 0.0
    A = float(delta0)
    R = math.sqrt(A * A + B * B)
    if not np.isfinite(R):
        return out
    if R == 0.0:
        return out

    phi0 = math.atan2(B, A)

    def _solutions_for_wall_in_window(wall_eps: float, t_lo: float, t_hi: float) -> list[float]:
        ts: list[float] = []
        target = (float(wall_eps) - float(eps_star)) / R
        if abs(target) <= 1.0:
            alpha = math.acos(target)
            base1 = (phi0 + alpha) / omega
            base2 = (phi0 - alpha) / omega
            period = (2.0 * math.pi) / omega

            def _scan_branch(base_t: float, tlo: float, thi: float) -> None:
                if period != 0.0:
                    k_min_f = (tlo - base_t) / period
                    k_max_f = (thi - base_t) / period
                else:
                    k_min_f = 0.0
                    k_max_f = -1.0

                k_min = int(math.floor(k_min_f)) - 2
                k_max = int(math.ceil(k_max_f)) + 2

                k = k_min
                while k <= k_max:
                    t_k = base_t + k * period
                    if t_k > tlo:
                        if t_k < thi:
                            ts.append(float(t_k))
                    k = k + 1
                return

            _scan_branch(float(base1), float(t_lo), float(t_hi))
            _scan_branch(float(base2), float(t_lo), float(t_hi))
        return ts

    t_lo = float(min(0.0, dt))
    t_hi = float(max(0.0, dt))

    cands: list[tuple[float, float]] = []

    tlist_lo = _solutions_for_wall_in_window(float(eps_min), float(t_lo), float(t_hi))
    i = 0
    n1 = len(tlist_lo)
    while i < n1:
        cands.append((float(tlist_lo[i]), float(eps_min)))
        i = i + 1

    tlist_hi = _solutions_for_wall_in_window(float(eps_max), float(t_lo), float(t_hi))
    j = 0
    n2 = len(tlist_hi)
    while j < n2:
        cands.append((float(tlist_hi[j]), float(eps_max)))
        j = j + 1

    if not cands:
        return out

    cands.sort(key=lambda z: z[0])

    tol = 1.0e-12
    idx = 0
    N = len(cands)
    while idx < N:
        t_i, w_i = cands[idx]
        keep = True
        jdx = idx + 1
        while jdx < N:
            t_j, w_j = cands[jdx]
            if abs(t_j - t_i) <= tol:
                jdx = jdx + 1
            else:
                break
        out.append((float(t_i), float(w_i)))
        idx = jdx

    return out




def s_flow(
    eps: float,
    pi: float,
    eps_star: float,
    mu_soft: float,
    k_soft: float,
    h: float,
) -> tuple[float, float]:

    mu = float(mu_soft)
    ks = float(k_soft)
    dt = float(h)

    if not np.isfinite(mu) or not np.isfinite(ks):
        return float(eps), float(pi)
    if mu <= 0.0 or ks <= 0.0 or dt == 0.0:
        if mu != 0.0:
            return float(eps + (pi / mu) * dt), float(pi)
        return float(eps), float(pi)

    hp = np.float128 if hasattr(np, "float128") else np.longdouble
    mu_hp   = hp(mu)
    ks_hp   = hp(ks)
    dt_hp   = hp(dt)
    eps_hp  = hp(eps)
    pi_hp   = hp(pi)
    epss_hp = hp(eps_star)

    omega_hp = np.sqrt(ks_hp / mu_hp)
    theta_hp = omega_hp * dt_hp
    ath = float(np.abs(theta_hp))


    if ath < 1.0e-6:
        th  = theta_hp
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th4 * th
        th6 = th3 * th3
        th7 = th6 * th
        th8 = th4 * th4
        th9 = th8 * th

        sin_theta_hp = th  - th3 / hp(6.0)   + th5 / hp(120.0) \
                         - th7 / hp(5040.0) + th9 / hp(362880.0)
        cos_theta_hp = hp(1.0) - th2 / hp(2.0) + th4 / hp(24.0) \
                                     - th6 / hp(720.0) + th8 / hp(40320.0)
    else:
        sin_theta_hp = np.sin(theta_hp)
        cos_theta_hp = np.cos(theta_hp)

    delta0_hp    = eps_hp - epss_hp
    mu_omega_hp  = np.sqrt(mu_hp * ks_hp)


    delta_t_hp = delta0_hp * cos_theta_hp + (pi_hp / mu_omega_hp) * sin_theta_hp
    eta_t_hp   = pi_hp    * cos_theta_hp - (mu_omega_hp * delta0_hp) * sin_theta_hp

    eps_new = float(epss_hp + delta_t_hp)
    pi_new  = float(eta_t_hp)
    return float(eps_new), float(pi_new)

def pi_half_kick(
    q: NDArray[np.floating],
    m: NDArray[np.floating],
    G: float,
    eps: float,
    pi: float,
    eps_star: float,
    k_soft: float,
    h_half: float,
    *,
    dB_d_eps: float = 0.0,
) -> tuple[float, float, float]:

    q_arr = np.asarray(q, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        return float(pi), 0.0, 0.0
    if m_arr.size != q_arr.shape[0]:
        return float(pi), 0.0, 0.0

    if q_arr.shape[0] >= 2 and float(G) != 0.0:
        dU = float(_dVdeps(q_arr, m_arr, float(eps), float(G)))
    else:
        dU = 0.0

    bterm = float(dB_d_eps)
    pi_new = float(pi) - (float(dU) + float(bterm)) * float(h_half)

    kterm_ret = 0.0
    return float(pi_new), float(dU), float(kterm_ret)

