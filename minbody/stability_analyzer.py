import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import math
from .simulation import NBodySimulation
from .diagnostics import Diagnostics
from .dynamical_features import DynamicalFeatures
from .evolution_features import EvolutionFeatures

"""
This comprehensive module analyzes the long-term stability of N-body systems. The StabilityAnalyzer class computes stability metrics through time integration, including energy and angular momentum conservation, trajectory boundedness and escape detection, MEGNO chaos indicators, and Lyapunov time estimates. It supports multiple analysis modes (minimal, core, full) with increasing detail levels, generates feature vectors for ML training, and provides detailed diagnostic output. The analyzer integrates systems forward in time while monitoring conserved quantities and dynamical indicators. It assumes simulations are properly configured with consistent timesteps and sufficient integration time for meaningful statistics.



"""

def _effective_n_steps(dt: float, t_target: float, n_steps_user: int) -> int:
	return max(n_steps_user, int(np.ceil(t_target / dt)))

def _H(sim):
    return sim._integrator.compute_extended_hamiltonian()



class StabilityAnalyzer:
	def __init__(self, sim: NBodySimulation,
				 n_steps: int = 1000,
				 dt: float = 0.01,
				 mode: str = 'core'):
		self.sim = sim
		self.n_steps = max(1, int(n_steps))
		self.dt = float(dt)
		self.mode = mode
		self._initial_mass = sim._mass.copy()
		self._initial_pos = sim._pos.copy()
		self._initial_vel = sim._vel.copy()
		self.diagnostics = Diagnostics(sim)
		if mode == 'full':
			self.features = DynamicalFeatures(sim)

	def _quick_virial_radius(self) -> float:
		m = self.sim._mass
		pos = self.sim._pos
		G = self.sim.G
		U = 0.0
		for i in range(len(m) - 1):
			for j in range(i + 1, len(m)):
				r = np.linalg.norm(pos[j] - pos[i]) + 1e-12
				U -= G * m[i] * m[j] / r
		tot_mass = float(m.sum())
		if U:
			return abs(-G * tot_mass ** 2 / (2 * U))
		return 1.0

	def _energy_drift_tolerance(self) -> float:
		tol_base = 3e-4
		dt_factor = (self.dt / 0.01) ** 1.5
		soft_factor = (self.sim.manager.softening / 0.05) ** 0.5
		return tol_base * dt_factor * soft_factor

	def run_stability_analysis(self) -> Dict[str, float]:
		sim_copy = NBodySimulation.restore(self.sim.snapshot())

		if self.mode == 'minimal':
			E0 = _H(sim_copy)
			if isinstance(self.n_steps, int):
				n_steps = self.n_steps
			else:
				n_steps = 0
			i = 0
			while i < n_steps:
				sim_copy.step(self.dt)
				i += 1
			E1 = _H(sim_copy)

			if isinstance(E0, (int, float, np.floating)):
				E0f = float(E0)
			else:
				E0f = float('nan')
			if isinstance(E1, (int, float, np.floating)):
				E1f = float(E1)
			else:
				E1f = float('nan')
			if np.isfinite(E0f) and abs(E0f) > 0.0 and np.isfinite(E1f):
				energy_drift = abs((E1f - E0f) / E0f)
			elif np.isfinite(E0f) and np.isfinite(E1f):
				energy_drift = abs(E1f - E0f)
			else:
				energy_drift = float('inf')

			return {
				'is_stable': float(energy_drift < 0.01),
				'energy_drift': energy_drift,
				'mode': 'minimal'
			}

		diag = Diagnostics(sim_copy)
		E0 = _H(sim_copy)
		L0 = diag.angular_momentum()

		com_drifts, j_eps_values = [], []
		theta_eps_values, ang_mom_tilts, ang_mom_vars, tidal_traces = [], [], [], []
		if isinstance(self.n_steps, int):
			n_steps = self.n_steps
		else:
			n_steps = 0
		sample_interval = max(1, n_steps // 100)

		i = 0
		while i < n_steps:
			sim_copy.step(self.dt)
			if (i % sample_interval) == 0:
				metrics = diag.step_metrics()
				com_drifts.append(metrics.get('com_drift', float('nan')))
				j_eps_values.append(metrics.get('J_eps', float('nan')))
				theta_eps_values.append(metrics.get('theta_eps', float('nan')))
				ang_mom_tilts.append(metrics.get('cos_theta', float('nan')))
				ang_mom_vars.append(metrics.get('var_L', float('nan')))
				tidal_traces.append(metrics.get('tr_hessian', float('nan')))
			i += 1

		E1 = _H(sim_copy)
		L1 = diag.angular_momentum()

		if self.mode == 'full':
			if isinstance(self.n_steps, int):
				n_samp = self.n_steps // 2
			else:
				n_samp = 0
			n_samp = min(50, n_samp)
			if n_samp > 0:
				evolution_analyzer = EvolutionFeatures(sim_copy, n_samples=n_samp, dt=self.dt)
				megno, lyap_time = evolution_analyzer.compute_megno(min(100, n_samp), self.dt)
			else:
				megno, lyap_time = 2.0, float('inf')
		else:
			megno, lyap_time = 2.0, float('inf')

		if isinstance(E0, (int, float, np.floating)):
			E0f = float(E0)
		else:
			E0f = float('nan')
		if isinstance(E1, (int, float, np.floating)):
			E1f = float(E1)
		else:
			E1f = float('nan')
		if np.isfinite(E0f) and abs(E0f) > 0.0 and np.isfinite(E1f):
			energy_drift = abs((E1f - E0f) / E0f)
		elif np.isfinite(E0f) and np.isfinite(E1f):
			energy_drift = abs(E1f - E0f)
		else:
			energy_drift = float('inf')

		if isinstance(L0, (int, float, np.floating)):
			L0f = float(L0)
		else:
			L0f = float('nan')
		if isinstance(L1, (int, float, np.floating)):
			L1f = float(L1)
		else:
			L1f = float('nan')
		if np.isfinite(L0f) and abs(L0f) > 0.0 and np.isfinite(L1f):
			ang_mom_drift = abs((L1f - L0f) / L0f)
		elif np.isfinite(L0f) and np.isfinite(L1f):
			ang_mom_drift = abs(L1f - L0f)
		else:
			ang_mom_drift = float('inf')

		if len(com_drifts) > 0:
			com_mean = float(np.mean(com_drifts))
		else:
			com_mean = float('nan')
		if len(com_drifts) > 0:
			com_max = float(np.max(com_drifts))
		else:
			com_max = float('nan')
		if len(j_eps_values) > 0:
			j_mean = float(np.mean(j_eps_values))
		else:
			j_mean = float('nan')
		if len(j_eps_values) > 0:
			j_std = float(np.std(j_eps_values))
		else:
			j_std = float('nan')
		if len(theta_eps_values) > 0:
			th_mean = float(np.mean(theta_eps_values))
		else:
			th_mean = float('nan')
		if len(theta_eps_values) > 0:
			th_std = float(np.std(theta_eps_values))
		else:
			th_std = float('nan')
		if len(ang_mom_tilts) > 0:
			ct_mean = float(np.mean(ang_mom_tilts))
		else:
			ct_mean = float('nan')
		if len(ang_mom_tilts) > 0:
			ct_min = float(np.min(ang_mom_tilts))
		else:
			ct_min = float('nan')
		if len(ang_mom_vars) > 0:
			var_mean = float(np.mean(ang_mom_vars))
		else:
			var_mean = float('nan')
		if len(ang_mom_vars) > 0:
			var_max = float(np.max(ang_mom_vars))
		else:
			var_max = float('nan')
		if len(tidal_traces) > 0:
			tr_mean = float(np.mean(tidal_traces))
		else:
			tr_mean = float('nan')
		if len(tidal_traces) > 0:
			tr_max = float(np.max(tidal_traces))
		else:
			tr_max = float('nan')

		is_stable = (
			(energy_drift < 0.01) and
			(ang_mom_drift < 0.01) and
			(com_mean < 1.0) and
			(megno < 10.0)
		)

		result = {
			'is_stable': float(is_stable),
			'energy_drift': energy_drift,
			'angular_momentum_drift': ang_mom_drift,
			'com_drift_mean': com_mean,
			'com_drift_max': com_max,
			'j_eps_mean': j_mean,
			'j_eps_std': j_std,
			'theta_eps_mean': th_mean,
			'theta_eps_std': th_std,
			'cos_theta_mean': ct_mean,
			'cos_theta_min': ct_min,
			'ang_mom_var_mean': var_mean,
			'ang_mom_var_max': var_max,
			'tidal_trace_mean': tr_mean,
			'tidal_trace_max': tr_max,
			'MEGNO': float(megno) if isinstance(megno, (int, float, np.floating)) else float('nan'),
			'lyapunov_time': float(lyap_time) if isinstance(lyap_time, (int, float, np.floating)) else float('nan'),
			'mode': self.mode
		}

		if self.mode == 'full':
			initial = DynamicalFeatures(self.sim).extract_all()
			for k in initial:
				result[f'initial_{k}'] = initial[k]

		return result


	def _run_core_analysis(self) -> Dict[str, float]:
		self._reset_simulation()
		E0 = self.diagnostics.energy()
		R_vir = self._compute_virial_radius()
		v_squared = np.sum(self._initial_vel ** 2, axis=1)
		v_rms = np.sqrt(np.mean(v_squared))
		if v_rms > 0:
			T_cr = R_vir / v_rms
		else:
			T_cr = float("inf")
		if math.isfinite(T_cr) and T_cr > 0:
			t_target = 10.0 * T_cr
		else:
			t_target = self.n_steps * self.dt
		n_iter = _effective_n_steps(self.dt, t_target, self.n_steps)
		old_n = self.n_steps
		self.n_steps = n_iter
		max_radial_distance = 0.0
		for _ in range(n_iter):
			self.sim.step(self.dt)
			radial_distances = np.sqrt(np.sum(self.sim._pos ** 2, axis=1))
			current_max = np.max(radial_distances)
			if current_max > max_radial_distance:
				max_radial_distance = current_max
		E_final = self.diagnostics.energy()
		if E0 != 0:
			energy_drift = abs((E_final - E0) / E0)
		else:
			energy_drift = 0.0
		megno, lyap_time = EvolutionFeatures(self.sim, n_samples=100, dt=self.dt).compute_megno(100, self.dt)
		is_stable = self._determine_stability(
			energy_drift,
			max_radial_distance,
			R_vir,
			lyap_time,
			T_cr,
		)
		self.n_steps = old_n
		return {
			"mode": "core",
			"energy_drift": energy_drift,
			"max_radial_distance": max_radial_distance,
			"virial_radius": R_vir,
			"MEGNO": megno,
			"lyapunov_time": lyap_time,
			"crossing_time": T_cr,
			"is_stable": float(is_stable),
			"n_steps": float(n_iter),
			"dt": self.dt,
			"total_time": n_iter * self.dt,
		}

	def _run_full_analysis(self) -> Dict[str, float]:
		self._reset_simulation()
		initial_diagnostics = self._compute_initial_diagnostics()
		t_target = 10.0 * initial_diagnostics["crossing_time"]
		if not math.isfinite(t_target) or t_target <= 0.0:
			t_target = self.n_steps * self.dt
		n_iter = _effective_n_steps(self.dt, t_target, self.n_steps)
		old_n = self.n_steps
		self.n_steps = n_iter
		evolution_diagnostics = self._run_evolution_with_tracking(n_iter)
		megno, lyap_time = EvolutionFeatures(self.sim, n_samples=200, dt=self.dt).compute_megno(200, self.dt)
		stability_criteria = self._compute_stability_criteria(initial_diagnostics, evolution_diagnostics, lyap_time)
		is_stable = self._determine_stability_from_criteria(stability_criteria)
		if hasattr(self, "features"):
			ml_features = self.features.extract_all()
		else:
			ml_features = {}
		evolution_extra = EvolutionFeatures(self.sim, n_samples=20, dt=self.dt).extract_evolution_features()
		self.n_steps = old_n
		return {
			"mode": "full",
			**initial_diagnostics,
			**evolution_diagnostics,
			"MEGNO": megno,
			"lyapunov_time": lyap_time,
			**stability_criteria,
			**ml_features,
			**evolution_extra,
			"is_stable": float(is_stable),
			"n_steps": float(n_iter),
			"dt": self.dt,
			"total_integration_time": n_iter * self.dt,
		}

	def _reset_simulation(self):
		self.sim._mass = self._initial_mass.copy()
		self.sim._pos = self._initial_pos.copy()
		self.sim._vel = self._initial_vel.copy()
		mgr = self.sim.manager
		mgr.begin_step()
		mgr.s = mgr.s0
		mgr._step_s2 = mgr.s0 ** 2
		mgr.s2 = mgr._step_s2
		self.sim.softening_energy_delta = 0.0
		self.sim._softening_dirty = False
		self.sim._acc_cached = False

	def _compute_virial_radius(self) -> float:
		PE = self.diagnostics.potential_energy()
		total_mass = np.sum(self.sim._mass)
		if PE != 0:
			R_vir = -self.sim.G * total_mass ** 2 / (2 * PE)
		else:
			distances = []
			n = self.sim.n_bodies
			for i in range(n):
				for j in range(i + 1, n):
					dx = self.sim._pos[j, 0] - self.sim._pos[i, 0]
					dy = self.sim._pos[j, 1] - self.sim._pos[i, 1]
					r = np.sqrt(dx * dx + dy * dy)
					distances.append(r)
			if distances:
				R_vir = float(np.mean(distances))
			else:
				R_vir = 1.0
		return abs(R_vir)

	def _get_state_vector(self) -> np.ndarray:
		pos = self.sim._pos.reshape(-1)
		vel = self.sim._vel.reshape(-1)
		return np.concatenate([pos, vel])

	def _determine_stability(self, energy_drift: float, max_radius: float,
						   R_vir: float, lyapunov_time: float, T_cr: float) -> bool:
		energy_drift_rate = energy_drift / (self.n_steps * self.dt)
		good_energy = energy_drift_rate < 1.2 * self._energy_drift_tolerance()
		good_escape = max_radius <= 10.0 * R_vir
		good_chaos = lyapunov_time >= 50.0 * T_cr
		return good_energy and good_escape and good_chaos

	def _determine_stability_from_criteria(self, criteria: Dict[str, float]) -> bool:
		lyapunov_unstable = criteria['lyapunov_to_crossing_ratio'] < 50
		energy_drift_rate = criteria['energy_drift_threshold'] / (self.n_steps * self.dt)
		tol = self._energy_drift_tolerance()
		energy_unstable = energy_drift_rate > tol
		escape_unstable = criteria['escape_radius_ratio'] > 10.0
		return not (lyapunov_unstable or energy_unstable or escape_unstable)

	def _compute_initial_diagnostics(self) -> Dict[str, float]:
		E0 = self.diagnostics.energy()
		L0 = self.diagnostics.angular_momentum()
		px0, py0 = self.diagnostics.linear_momentum()
		com_pos, com_vel = self.diagnostics.center_of_mass()
		PE = self.diagnostics.potential_energy()
		total_mass = np.sum(self.sim._mass)
		if PE != 0:
			R_vir = -self.sim.G * total_mass ** 2 / (2 * PE)
		else:
			distances = []
			n = self.sim.n_bodies
			for i in range(n):
				for j in range(i + 1, n):
					dx = self.sim._pos[j, 0] - self.sim._pos[i, 0]
					dy = self.sim._pos[j, 1] - self.sim._pos[i, 1]
					r = np.sqrt(dx * dx + dy * dy)
					distances.append(r)
			if distances:
				R_vir = float(np.mean(distances))
			else:
				R_vir = 1.0
		v_squared = np.sum(self.sim._vel ** 2, axis=1)
		v_rms = np.sqrt(np.mean(v_squared))
		if v_rms > 0:
			T_cr = R_vir / v_rms
		else:
			T_cr = np.inf
		return {
			'initial_energy': E0,
			'initial_angular_momentum': L0,
			'initial_linear_momentum_x': px0,
			'initial_linear_momentum_y': py0,
			'initial_com_x': com_pos[0],
			'initial_com_y': com_pos[1],
			'initial_com_vx': com_vel[0],
			'initial_com_vy': com_vel[1],
			'virial_radius': abs(R_vir),
			'crossing_time': T_cr,
			'binding_energy': E0,
			'initial_kinetic_energy': self.diagnostics.kinetic_energy(),
			'initial_potential_energy': PE
		}

	def _run_evolution_with_tracking(self, n_iter: int) -> Dict[str, float]:
		E0_ext = _H(self.sim)
		L0 = self.diagnostics.angular_momentum()

		energy_values_ext = [E0_ext]
		angular_momentum_values = [L0]
		max_distances = []
		initial_positions = self.sim._pos.copy()
		R_vir = self._compute_virial_radius()

		for _ in range(n_iter):
			self.sim.step(self.dt)

			E_ext = _H(self.sim)
			L = self.diagnostics.angular_momentum()

			energy_values_ext.append(E_ext)
			angular_momentum_values.append(L)

			distances_from_origin = np.sqrt(np.sum(self.sim._pos ** 2, axis=1))
			max_distances.append(np.max(distances_from_origin))

		energy_array_ext = np.array(energy_values_ext)
		if E0_ext != 0:
			relative_energy_drift_ext = np.abs((energy_array_ext - E0_ext) / E0_ext)
			max_energy_drift = np.max(relative_energy_drift_ext)
			mean_energy_drift = float(np.mean(relative_energy_drift_ext[1:]))
			final_energy_drift = float(relative_energy_drift_ext[-1])
		else:
			max_energy_drift = 0.0
			mean_energy_drift = 0.0
			final_energy_drift = 0.0

		final_energy_phys = self.diagnostics.energy()

		L_array = np.array(angular_momentum_values)
		if L0 != 0:
			relative_L_drift = np.abs((L_array - L0) / L0)
			max_L_drift = np.max(relative_L_drift)
		else:
			max_L_drift = 0.0

		final_positions = self.sim._pos
		position_changes = np.sqrt(np.sum((final_positions - initial_positions) ** 2, axis=1))
		final_distances = np.sqrt(np.sum(final_positions ** 2, axis=1))
		escaped_bodies = np.sum(final_distances > 5 * R_vir)

		return {
			'relative_energy_drift': final_energy_drift,               
			'max_relative_energy_drift': max_energy_drift,             
			'mean_relative_energy_drift': mean_energy_drift,           
			'relative_angular_momentum_drift': max_L_drift,
			'max_distance_from_origin': np.max(max_distances),
			'mean_position_change': float(np.mean(position_changes)),
			'max_position_change': float(np.max(position_changes)),
			'final_energy': float(final_energy_phys),                  
			'final_angular_momentum': float(L_array[-1]),
			'escaped_bodies': float(escaped_bodies),
			'escape_fraction': float(escaped_bodies) / self.sim.n_bodies
		}

	def _compute_stability_criteria(self, initial_diag: Dict, evolution_diag: Dict,
								  lyapunov_time: float) -> Dict[str, float]:
		T_cr = initial_diag['crossing_time']
		R_vir = initial_diag['virial_radius']
		if np.isfinite(lyapunov_time) and np.isfinite(T_cr):
			lyapunov_criterion = lyapunov_time / T_cr
		else:
			lyapunov_criterion = np.inf
		return {
			'lyapunov_to_crossing_ratio': lyapunov_criterion,
			'energy_drift_threshold': evolution_diag['max_relative_energy_drift'],
			'escape_radius_ratio': evolution_diag['max_distance_from_origin'] / R_vir
		}

	def serialize_to_dict(self, diagnostics: Dict[str, float], max_bodies: int = None) -> Dict:
		data = {
			'n_bodies': self.sim.n_bodies,
			'G': self.sim.G,
			'softening': self.sim.manager.softening,
			'min_softening': self.sim._min_softening,
			'adaptive': float(self.sim._adaptive),
			'integrator_mode': self.sim._integrator_mode,
		}
		if max_bodies is not None and self.sim.n_bodies > max_bodies:
			data['mass_min'] = float(np.min(self._initial_mass))
			data['mass_max'] = float(np.max(self._initial_mass))
			data['mass_mean'] = float(np.mean(self._initial_mass))
			data['mass_std'] = float(np.std(self._initial_mass))
			data['x_min'] = float(np.min(self._initial_pos[:, 0]))
			data['x_max'] = float(np.max(self._initial_pos[:, 0]))
			data['x_mean'] = float(np.mean(self._initial_pos[:, 0]))
			data['x_std'] = float(np.std(self._initial_pos[:, 0]))
			data['y_min'] = float(np.min(self._initial_pos[:, 1]))
			data['y_max'] = float(np.max(self._initial_pos[:, 1]))
			data['y_mean'] = float(np.mean(self._initial_pos[:, 1]))
			data['y_std'] = float(np.std(self._initial_pos[:, 1]))
			data['vx_min'] = float(np.min(self._initial_vel[:, 0]))
			data['vx_max'] = float(np.max(self._initial_vel[:, 0]))
			data['vx_mean'] = float(np.mean(self._initial_vel[:, 0]))
			data['vx_std'] = float(np.std(self._initial_vel[:, 0]))
			data['vy_min'] = float(np.min(self._initial_vel[:, 1]))
			data['vy_max'] = float(np.max(self._initial_vel[:, 1]))
			data['vy_mean'] = float(np.mean(self._initial_vel[:, 1]))
			data['vy_std'] = float(np.std(self._initial_vel[:, 1]))
		else:
			for i, mass in enumerate(self._initial_mass):
				data[f'mass_{i}'] = mass
			for i in range(len(self._initial_pos)):
				data[f'x_{i}'] = self._initial_pos[i, 0]
				data[f'y_{i}'] = self._initial_pos[i, 1]
			for i in range(len(self._initial_vel)):
				data[f'vx_{i}'] = self._initial_vel[i, 0]
				data[f'vy_{i}'] = self._initial_vel[i, 1]
		data.update(diagnostics)
		return data

	def save_to_csv(self, filename: str, diagnostics: Dict[str, float] = None):
		if diagnostics is None:
			diagnostics = self.run_stability_analysis()
		data = self.serialize_to_dict(diagnostics)
		df = pd.DataFrame([data])
		df.to_csv(filename, index=False)

