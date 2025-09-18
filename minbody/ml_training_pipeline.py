"""
This module orchestrates the complete machine learning workflow for stability
prediction.

The MLTrainingPipeline class generates diverse N-body datasets with multiple system
types (random, hierarchical, polygon, close encounters), runs stability analysis on
generated systems, produces labeled datasets for ML training, and supports both diverse
and focused dataset generation strategies. Methods include generate_diverse_dataset for
comprehensive training data, generate_focused_dataset for specific stability regimes,
and quick_test_pipeline for rapid validation. The pipeline integrates all components
from initial condition generation through stability analysis to dataset creation. It
assumes sufficient computational resources for batch simulation processing.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


from .initial_condition_generator import InitialConditionGenerator, GeneratorConfig
from .specialized_generators import SpecializedGenerators

from .simulation import NBodySimulation
from .stability_analyzer import StabilityAnalyzer
from .batch_stability_analyzer import BatchStabilityAnalyzer
from .utils import set_global_seed



class MLTrainingPipeline:
	def __init__(self, n_systems: int = 1000, n_steps: int = 1000, dt: float = 0.01):
		self.n_systems = n_systems
		self.n_steps = max(500, min(2000, n_steps))
		self.dt = dt
		
		self.ic_generator = InitialConditionGenerator()
		self.batch_analyzer = BatchStabilityAnalyzer(n_steps=self.n_steps, dt=self.dt, mode='full')
		
	def generate_diverse_dataset(self) -> pd.DataFrame:
		print(f"Generating {self.n_systems} diverse N-body systems...")
		
		simulations = []
		
		n_random = int(0.4 * self.n_systems)
		print(f"\n1. Generating {n_random} random systems...")
		
		for i in range(n_random):
			n_bodies = np.random.randint(3, 6) 
			
			config = GeneratorConfig(
				mass_range=(0.1, 10.0),
				use_log_mass=(i % 2 == 0), 
				position_scale=np.random.uniform(0.5, 2.0),
				velocity_virial_fraction=np.random.uniform(0.8, 1.2),
				velocity_perturbation=np.random.uniform(0.05, 0.2),
				softening=np.random.uniform(0.001, 0.1)
			)
			
			generator = InitialConditionGenerator(config)
			sim = generator.create_simulation(n_bodies)
			simulations.append(sim)
			
		n_hierarchical = int(0.3 * self.n_systems)
		print(f"2. Generating {n_hierarchical} hierarchical systems...")
		
		for i in range(n_hierarchical):
			mass_ratio1 = np.random.uniform(0.1, 1.0)
			mass_ratio2 = np.random.uniform(0.1, 2.0)
			separation_ratio = np.random.uniform(3, 50)  
			
			masses, positions, velocities = SpecializedGenerators.generate_hierarchical_triple(
				mass_ratio1, mass_ratio2, separation_ratio
			)
			
			velocities += np.random.randn(*velocities.shape) * 0.05
			
			sim = NBodySimulation(
				masses=masses,
				positions=positions,
				velocities=velocities,
				G=1.0,
				softening=0.01
			)
			simulations.append(sim)
			
		n_polygon = int(0.2 * self.n_systems)
		print(f"3. Generating {n_polygon} polygon configurations...")
		
		for i in range(n_polygon):
			n_bodies = np.random.randint(3, 8)
			radius = np.random.uniform(0.5, 3.0)
			rotation_fraction = np.random.uniform(0, 1.0)
			
			masses, positions, velocities = SpecializedGenerators.generate_equal_mass_polygon(
				n_bodies, radius, rotation_fraction
			)
			
			sim = NBodySimulation(
				masses=masses,
				positions=positions,
				velocities=velocities,
				G=1.0,
				softening=0.05
			)
			simulations.append(sim)
			
		n_close = self.n_systems - n_random - n_hierarchical - n_polygon
		print(f"4. Generating {n_close} close encounter systems...")
		
		for i in range(n_close):
			n_bodies = np.random.randint(3, 5)
			
			config = GeneratorConfig(
				position_scale=0.1,
				velocity_virial_fraction=1.5,  
				velocity_perturbation=0.3,
				softening=0.001
			)
			
			generator = InitialConditionGenerator(config)
			sim = generator.create_simulation(n_bodies)
			simulations.append(sim)
			
		print(f"\nAnalyzing {len(simulations)} systems...")
		results_df = self.batch_analyzer.analyze_batch(simulations, show_progress=True)
		
		system_types = (
			['random'] * n_random +
			['hierarchical'] * n_hierarchical +
			['polygon'] * n_polygon +
			['close_encounter'] * n_close
		)
		results_df['system_type'] = system_types
		
		return results_df
		
	def generate_focused_dataset(self, focus: str = 'boundary') -> pd.DataFrame:
		print(f"Generating {self.n_systems} systems focused on {focus} cases...")
		
		simulations = []
		
		if focus == 'boundary':
			for i in range(self.n_systems):
				if i % 3 == 0:
					separation_ratio = np.random.uniform(5, 15) 
					masses, pos, vel = SpecializedGenerators.generate_hierarchical_triple(
						separation_ratio=separation_ratio
					)
					sim = NBodySimulation(masses=masses, positions=pos, velocities=vel)
					
				elif i % 3 == 1:
					config = GeneratorConfig(
						velocity_virial_fraction=1.0,
						velocity_perturbation=np.random.uniform(0.1, 0.3)
					)
					generator = InitialConditionGenerator(config)
					sim = generator.create_simulation(np.random.randint(3, 5))
					
				else:
					n = np.random.randint(4, 7)
					rotation_fraction = np.random.uniform(0.3, 0.7)  
					masses, pos, vel = SpecializedGenerators.generate_equal_mass_polygon(
						n, rotation_fraction=rotation_fraction
					)
					sim = NBodySimulation(masses=masses, positions=pos, velocities=vel)
					
				simulations.append(sim)
				
		elif focus == 'stable':
			for i in range(self.n_systems):
				separation_ratio = np.random.uniform(20, 100)
				masses, pos, vel = SpecializedGenerators.generate_hierarchical_triple(
					separation_ratio=separation_ratio
				)
				
				vel += np.random.randn(*vel.shape) * 0.01
				
				sim = NBodySimulation(
					masses=masses, positions=pos, velocities=vel,
					softening=0.01
				)
				simulations.append(sim)
				
		else: 
			for i in range(self.n_systems):
				config = GeneratorConfig(
					position_scale=0.1,  
					velocity_virial_fraction=np.random.uniform(1.5, 2.0),
					velocity_perturbation=0.5,
					softening=0.001
				)
				generator = InitialConditionGenerator(config)
				sim = generator.create_simulation(np.random.randint(3, 6))
				simulations.append(sim)
				
		results_df = self.batch_analyzer.analyze_batch(simulations)
		results_df['dataset_focus'] = focus
		
		return results_df
		
	def quick_test_pipeline(self) -> pd.DataFrame:
		set_global_seed(42)
		
		print("Running quick test with 10 systems...")
		
		test_sims = []
		generator = InitialConditionGenerator()
		
		for i in range(10):
			n_bodies = 3 + (i % 3)
			sim = generator.create_simulation(n_bodies)
			test_sims.append(sim)
		
		print("\nTesting unified analyzer in core mode...")
		results = []
		
		for i, sim in enumerate(test_sims):
			analyzer = StabilityAnalyzer(sim, n_steps=100, dt=0.01, mode='core')
			result = analyzer.run_stability_analysis()
			result['system_id'] = i
			results.append(result)
			
			if result['is_stable']:
				status = 'STABLE'
			else:
				status = 'UNSTABLE'
			print(f"System {i}: {status} (E_drift={result['energy_drift']:.2e})")
		
		test_df = pd.DataFrame(results)
		
		n_stable = int(sum(test_df['is_stable']))
		n_unstable = len(test_df) - n_stable
		print(f"\nTest complete. {n_stable} stable, {n_unstable} unstable")
		
		return test_df


