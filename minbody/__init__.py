"""
This initialization file serves as the main entry point for the N-body simulation
package, exposing all public APIs through a clean namespace.

It imports and re-exports core simulation classes (Body, BodyView, NBodySimulation),
integrators (Integrator, HamiltonianSofteningIntegrator), numerical solvers
(UniversalVariableKeplerSolver), force calculations, softening management utilities,
diagnostic tools, feature extractors for machine learning, stability analyzers, dataset
utilities, and ML model training pipelines. The file establishes the package's public
interface while maintaining internal module organization, allowing users to import any
major component directly from the package root. It assumes all imported modules are
properly implemented and follows Python packaging conventions for namespace management.
"""

from .utils import set_global_seed
from .sim_config import SimConfig
from .simulation_validator import SimulationValidator
from .softening_manager import SofteningManager
from .softening import grad_eps_target                     


from .body import Body
from .body_view import BodyView
from .simulation import NBodySimulation
from .integrator import Integrator
from .hamiltonian_softening_integrator import (
    HamiltonianSofteningIntegrator,
)

from .kepler_solver import UniversalVariableKeplerSolver
from .forces import gravitational_force, dV_d_epsilon
from .geometry_cache import geometry_buffers
from .barrier import barrier_force, barrier_energy, barrier_curvature
from .hamsoft_utils import (
    symplectic_bounce,
    symplectic_reflect_eps,
    reflect_if_needed,
    reflect_eps_symplectic,
    reflect_and_limit_eps,
    dU_depsilon_plummer,
)
from .hamsoft_flows import (
    PhaseState,
    spring_oscillation,
    strang_softening_step,
)
from .hamsoft_energy import extended_hamiltonian
from .hamsoft_constants import LAMBDA_SOFTENING, CHI_EPS
from .tangent_map import TangentMap


from .diagnostics import Diagnostics
from .hamsoft_validation import validate_ham_soft

from .dynamical_features import DynamicalFeatures
from .evolution_features import EvolutionFeatures
from .stability_analyzer import StabilityAnalyzer
from .batch_stability_analyzer import BatchStabilityAnalyzer

from .data_utils import DataUtils
from .scaler_utils import ScalerUtils
from .stability_dataset import StabilityDataset
from .initial_condition_generator import (
    InitialConditionGenerator,
    GeneratorConfig,
)
from .specialized_generators import SpecializedGenerators
from .ml_training_pipeline import MLTrainingPipeline

from .model_zoo import MLP, make_mlp
from .train_mlp import MLPTrainer
from .train_lightgbm import main as train_lightgbm_main








__all__ = [
    "set_global_seed",
    "SimConfig",
    "SimulationValidator",
    "SofteningManager",
    "grad_eps_target",
    "Body",
    "BodyView",
    "NBodySimulation",
    "Integrator",
    "HamiltonianSofteningIntegrator",
    "UniversalVariableKeplerSolver",
    "gravitational_force",
    "dV_d_epsilon",
    "geometry_buffers",
    "barrier_force",
    "barrier_energy",
    "barrier_curvature",   
    "symplectic_bounce",
    "symplectic_reflect_eps",
    "reflect_if_needed",
    "reflect_eps_symplectic",
    "reflect_and_limit_eps",
    "dU_depsilon_plummer",
    "PhaseState",
    "spring_oscillation",
    "strang_softening_step",
    "extended_hamiltonian",
    "LAMBDA_SOFTENING",
    "CHI_EPS",
    "TangentMap",
    "Diagnostics",
    "validate_ham_soft",
    "DynamicalFeatures",
    "EvolutionFeatures",
    "StabilityAnalyzer",
    "BatchStabilityAnalyzer",
    "DataUtils",
    "ScalerUtils",
    "StabilityDataset",
    "InitialConditionGenerator",
    "GeneratorConfig",
    "SpecializedGenerators",
    "MLTrainingPipeline",
    "MLP",
    "make_mlp",
    "MLPTrainer",
    "train_lightgbm_main",
]

