# Adaptive-Softened N-Body Dynamics with Machine Learning

## Project Purpose

This repository couples a production-grade N-body integrator with the formal mathematical framework presented in **Unified Theory of Adaptive-Softened N-Body Dynamics**. The project demonstrates a fixed-step, symmetric Strang-split integrator that **conserves linear and angular momentum exactly and bounds the drift of a modified Hamiltonian by O(h²)**—our main theorem.

## Formal Proof

* **UnifiedTheoryOfAdaptive-SoftenedN-BodyDynamics.txt** – The typeset manuscript containing the full derivation and theorem statement, proving energy conservation properties and symplectic structure preservation under adaptive softening evolution.

## Quick Start

To run this N-body simulation and ML training pipeline, first install dependencies with `pip install -r requirements.txt`. The main workflow involves three steps:

1. **Generate N-body simulation data** by running the ML training pipeline which creates diverse initial conditions (hierarchical systems, equal-mass polygons, close encounters) and analyzes their stability over time using multiple metrics (MEGNO, energy drift, escape detection).

2. **Train machine learning models** on the generated stability data using either:
   - MLP neural network (`train_mlp.py`) with automatic threshold optimization
   - LightGBM gradient boosting (`train_lightgbm.py`) with hyperparameter tuning

3. **Predict stability** of new N-body configurations using trained models, achieving 5+ orders of magnitude speedup over direct integration.

## Key Features

The simulation supports multiple integration schemes:
- **Verlet**: Standard 2nd-order symplectic integrator
- **Yoshida4**: 4th-order symplectic composition method
- **WHFast**: Wisdom-Holman for hierarchical systems
- **ham_soft**: Novel Hamiltonian softening integrator implementing our theoretical framework

## Usage Examples
```python
# Quick test with 10 systems
from ml_training_pipeline import MLTrainingPipeline
pipeline = MLTrainingPipeline(n_systems=10, n_steps=1000, dt=0.01)
df = pipeline.quick_test_pipeline()

# Generate diverse dataset
df = pipeline.generate_diverse_dataset()

# Create individual simulation
from simulation import NBodySimulation
sim = NBodySimulation(
    masses=[1.0, 0.5, 0.1],
    positions=[[0, 0], [1, 0], [2, 0]],
    velocities=[[0, 0], [0, 1], [0, 0.5]],
    integrator_mode='ham_soft'
)
sim.step(0.01)
