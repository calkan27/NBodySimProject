import numpy as np
import pandas as pd
import datetime
from typing import List, Dict
from .simulation import NBodySimulation
from .stability_analyzer import StabilityAnalyzer

"""
This module provides batch processing capabilities for analyzing the stability of multiple N-body simulations efficiently. The BatchStabilityAnalyzer class wraps StabilityAnalyzer to process lists of simulations, collecting stability metrics into pandas DataFrames for analysis. The analyze_simulation method runs individual stability analysis with energy drift detection and pathological case flagging, analyze_batch processes multiple simulations with progress reporting, save_batch_results exports results to CSV format, and get_feature_matrix extracts features as numpy arrays for ML training. The analyzer tracks softening policies (static/adaptive/adaptive-ham) and detects extreme energy drift cases that indicate numerical instability. It assumes simulations are pre-configured and ready to run, with consistent timestep and integration parameters across the batch.


"""





class BatchStabilityAnalyzer:
	def __init__(self, n_steps: int = 1000, dt: float = 0.01, mode: str = 'core') -> None:
		self.n_steps = n_steps
		self.dt = dt
		self.mode = mode
		self.results = []

	def analyze_simulation(self, sim: NBodySimulation) -> Dict[str, float]:
		analyzer = StabilityAnalyzer(sim, self.n_steps, self.dt, mode=self.mode)
		result = analyzer.run_stability_analysis() or {}

		if 'energy_drift' in result:
			if abs(result['energy_drift']) > 10:
				print(f"[warning] Extreme energy drift detected: {result['energy_drift']}")
				result['is_stable'] = 0.0
				result['pathological_energy'] = True
			else:
				result['pathological_energy'] = False
		else:
			result['pathological_energy'] = False

		if sim._integrator_mode == 'ham_soft':
			result['softening_policy'] = 'adaptive-ham'
		elif sim._adaptive_softening:
			result['softening_policy'] = 'adaptive-classic'
		else:
			result['softening_policy'] = 'static'

		return result



	def analyze_batch(self, simulations: List[NBodySimulation], show_progress: bool = True) -> pd.DataFrame:
		self.results = []
		if show_progress:
			print(f"Analyzing {len(simulations)} simulations...")
		for i, sim in enumerate(simulations):
			if show_progress and i % 10 == 0:
				print(f"  Progress: {i}/{len(simulations)}")
			diag_dict = self.analyze_simulation(sim)
			if diag_dict:
				diag_dict['simulation_id'] = i
				diag_dict['mode'] = self.mode
				self.results.append(diag_dict)
			else:
				print(f"[warning] Simulation {i} failed, skipping")
				failed_result = {'simulation_id': i, 'is_stable': np.nan, 'mode': self.mode, 'pathological_energy': False}
				self.results.append(failed_result)
		if show_progress:
			print(f"Completed: {len(self.results)} simulations analyzed")
		return pd.DataFrame(self.results)

	def save_batch_results(self, filename: str) -> None:
		if not self.results:
			print("[error] No results to save. Run analyze_batch first.")
			return
		df = pd.DataFrame(self.results)
		df.to_csv(filename, index=False)
		print(f"Saved {len(df)} results to {filename}")

	def get_feature_matrix(self) -> np.ndarray:
		if not self.results:
			print("[error] No results available. Run analyze_batch first.")
			return np.array([])
		df = pd.DataFrame(self.results)
		return df.values







