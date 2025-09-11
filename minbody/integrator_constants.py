from __future__ import annotations

from .sim_config import SimConfig
from .hamsoft_constants import CHI_EPS
from .softening_manager import _ABS_SOFTENING_FLOOR

"""
This module centralizes numerical constants and configuration defaults for the integration subsystem. The IntegratorConstants class uses a metaclass to provide attribute access with fallback values, sources defaults from SimConfig, and defines critical parameters like SPLIT_N_MAX, INITIAL_DT, SAFETY_FACTOR, and barrier parameters. The module ensures consistent parameter usage across integration schemes while allowing configuration overrides. It assumes SimConfig provides valid default values and that constants are accessed as class attributes.


"""


class _ICMeta(type):
    def __getattr__(cls, name: str):
        return 0.0


class IntegratorConstants(metaclass=_ICMeta):
    _cfg = SimConfig()

    SPLIT_N_MAX        = int(getattr(_cfg, "split_n_max", 16384))
    INITIAL_DT         = float(getattr(_cfg, "initial_dt", 1.0e-2))
    MAX_FRACTION_OF_DT = float(getattr(_cfg, "max_fraction_of_dt", 0.1))

    SAFETY_FACTOR  = float(getattr(_cfg, "safety_factor", 0.20))
    THETA_CAP      = float(getattr(_cfg, "theta_cap", 0.10))
    THETA_IMP      = 0.10
    EPS_P          = 1.0e-12

    K_SOFT_DEFAULT    = float(getattr(_cfg, "k_soft", 0.0))
    USE_ENERGY_SPRING = bool(getattr(_cfg, "use_energy_spring", True))
    BARRIER_EXPONENT  = int(getattr(_cfg, "barrier_exponent", 5))
    ABS_SOFTENING_FLOOR = float(_ABS_SOFTENING_FLOOR)

    CHI_EPS = float(CHI_EPS)

    CORRECTOR_ORDER = int(getattr(_cfg, "corrector_order", 5))


__all__ = ["IntegratorConstants"]

