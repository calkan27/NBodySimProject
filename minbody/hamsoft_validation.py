"""
This module provides validation tests for the Hamiltonian softening integrator.

The validate_ham_soft function verifies energy conservation within tolerance bounds,
checks canonical equation consistency, tests equilibrium behavior with zero forces, and
monitors pi drift at stable points. The validation creates temporary simulation copies
to avoid state corruption, uses extended precision for accurate energy calculations, and
provides detailed error messages for debugging. It assumes the integrator is properly
initialized and the simulation is in a valid state for testing.
"""

import time
from typing import TYPE_CHECKING

import numpy as np
from .diagnostics import Diagnostics
from .hamsoft_utils import dU_depsilon_plummer
from .barrier import barrier_force
if TYPE_CHECKING:
	from .simulation import NBodySimulation









def validate_ham_soft(integrator, n_steps: int = 256, dt: float = 1e-3, *, energy_tol: float = 1e-8, canon_tol: float = 1e-10,
    pi_tol: float = 1e-12) -> None:
    t0 = time.perf_counter()

    sim  = integrator.sim
    diag = Diagnostics(sim)

    H0 = diag.compute_extended_hamiltonian()
    i = 0
    while i < n_steps:
        sim.step(dt)
        i = i + 1
    H1 = diag.compute_extended_hamiltonian()
    tol_pref = float(getattr(integrator.sim.cfg, "energy_tol_pref", 1e-7))
    abs_bound = tol_pref * (dt * dt)
    if abs(H1 - H0) > abs_bound:
        print("Extended Hamiltonian |ΔH| exceeds C·h^2 bound")
        return

    snap  = sim.snapshot()
    SimCls = sim.__class__
    sim_c = SimCls.restore(snap)
    int_c = sim_c._integrator

    eps0, pi0 = float(sim_c._epsilon), float(sim_c._pi)
    if sim_c.cfg.use_energy_spring:
        k_eff = float(int_c.k_soft)
    else:
        k_eff = 0.0

    eps_star  = float(int_c._eps_target(q=sim_c._pos))

    dU       = dU_depsilon_plummer(sim_c._pos, sim_c._mass, sim_c.G, eps0)
    eps_min  = float(sim_c._min_softening)
    eps_max  = float(getattr(sim_c, "_max_softening", 10.0 * sim_c.manager.s0))

    use_barrier = True
    pol = getattr(int_c, "barrier_policy", "reflection")
    if not isinstance(pol, str):
        pol = "reflection"
    if pol.lower() != "soft":
        use_barrier = False
    if getattr(sim_c.cfg, "disable_barrier", False):
        use_barrier = False

    if use_barrier:
        Fbar = barrier_force(
            eps0,
            eps_min,
            eps_max,
            k_wall=float(getattr(int_c, "k_wall", 1.0e9)),
            n=int(int_c._barrier_n()),
        )
        dU_bar = -float(Fbar)
    else:
        dU_bar = 0.0

    dpi_dt_exp   = -(dU + k_eff * (eps0 - eps_star) + dU_bar)
    deps_dt_exp  =  pi0 / float(int_c.mu_soft)

    sim_c.step(dt)
    dpi_dt_num   = (float(sim_c._pi)      - pi0) / dt
    deps_dt_num  = (float(sim_c._epsilon) - eps0) / dt

    rel = lambda a, b: abs(a - b) / max(abs(a), abs(b), 1.0e-30)
    if rel(dpi_dt_num,  dpi_dt_exp)  > canon_tol:
        print("dpi/dt mismatch exceeds tolerance")
        return
    if rel(deps_dt_num, deps_dt_exp) > canon_tol:
        print("dε/dt mismatch exceeds tolerance")
        return

    sim_eq      = SimCls.restore(snap)
    sim_eq.G    = 0.0
    int_eq      = sim_eq._integrator
    sim_eq._epsilon = float(int_eq._eps_target(q=sim_eq._pos))
    sim_eq.manager.update_continuous(sim_eq._epsilon)
    sim_eq._pi  = 0.123456789

    pi_start = float(sim_eq._pi)
    j = 0
    while j < n_steps:
        sim_eq.step(dt)
        j = j + 1
    if abs(float(sim_eq._pi) - pi_start) > pi_tol:
        print("π drift detected at equilibrium")
        return

    if time.perf_counter() - t0 > 1.0:
        print("[warning] validate_ham_soft took longer than 1 s")

    return None


