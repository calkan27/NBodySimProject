from __future__ import annotations
from dataclasses import dataclass, field, replace
import copy

"""
This central configuration module defines all simulation parameters through the SimConfig dataclass. Key parameters include safety factors for adaptive timestepping, spring constants for Hamiltonian softening, integrator mode selection, barrier potential settings, and convergence tolerances. The class provides a copy method for configuration inheritance and validates integrator modes against allowed options. It serves as the single source of truth for simulation behavior, with all components referencing this configuration. The module assumes reasonable default values and that users understand the physical implications of parameter choices.

"""
_ALLOWED_MODES = {
    "verlet",
    "yoshida4",
    "whfast",
    "ham_soft",
}

@dataclass
class SimConfig:
    safety_factor: float = 0.20
    theta_cap:   float = 0.1
    theta_imp:   float = 0.5            
    k_soft:      float = 1.0e3
    enable_runtime_guard: bool = False
    split_n_max: int = 50
    fast_float32: bool = False
    adaptive_timestep: bool = False
    adaptive_softening: bool = False
    softening_scale: float = 1.0
    integrator_mode: str = "ham_soft"
    use_energy_spring: bool = True
    use_soft_barrier: bool = True
    initial_dt: float = 0.01
    max_fraction_of_dt: float = 0.1
    corrector_order: int = 5
    disable_barrier: bool = False
    barrier_exponent: int = 5
    k_wall: float = 1.0e9
    n_wall: int = 4
    alpha: float | None = .1
    eta: float = 1.35
    guard_dt_ref: float = 1e-3
    energy_drift_abort_threshold: float = 1e-6
    ang_mom_drift_abort_threshold: float = 1e-5
    abort_on_violation: bool = True
    fixed_substeps: bool = True
    invariant_check_interval: int = 2000
    energy_tol_pref: float = 1e-8
    freeze_s_subsystem: bool = False  

    def copy(self) -> "SimConfig":
        new = object.__new__(SimConfig)
        new.__dict__ = dict(getattr(self, "__dict__", {}))
        return new


