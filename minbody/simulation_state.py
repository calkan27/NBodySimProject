"""
This module manages the internal state representation for N-body simulations.

The SimulationState class maintains numpy arrays for positions, velocities, masses, and
accelerations, provides property accessors with validation, handles state initialization
from various input formats, and supports efficient state serialization/restoration. The
implementation separates state management from simulation logic, enabling clean
separation of concerns. It validates all state modifications for physical consistency
and assumes state arrays maintain compatible dimensions throughout the simulation.
"""

from __future__ import annotations
import numpy as np
import copy
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import NBodySimulation
    from .body import Body




class SimulationState:

	def __init__(self):
		self.n_bodies: int = 0
		self._mass: np.ndarray = np.empty(0, dtype=np.float64)
		self._pos: np.ndarray = np.empty((0, 2), dtype=np.float64)
		self._vel: np.ndarray = np.empty((0, 2), dtype=np.float64)
		self._acc: np.ndarray = np.empty((0, 2), dtype=np.float64)
		self._buf_diff: np.ndarray | None = None
		self._buf_r2: np.ndarray | None = None
		
	@property
	def pos(self) -> np.ndarray:
		return self._pos
	
	@property
	def vel(self) -> np.ndarray:
		return self._vel
	
	@property
	def mass(self) -> np.ndarray:
		return self._mass
	

	@property
	def acc(self) -> np.ndarray:
		return self._acc



	@pos.setter
	def pos(self, value: np.ndarray) -> None:
		arr = np.asarray(value, dtype=self._pos.dtype)
		if arr.ndim == 1:
			arr = arr.reshape(-1, 2)
		elif arr.ndim == 2 and arr.shape[1] >= 2:
			arr = arr[:, :2]
		else:
			print("sim.pos must be shape (N,2) or flat length 2N")
			return
		if arr.shape != self._pos.shape:
			print(f"shape mismatch when assigning to sim.pos: "
							 f"expected {self._pos.shape}, got {arr.shape}")
			return
		self._pos[...] = arr  

	@vel.setter
	def vel(self, value: np.ndarray) -> None:
		arr = np.asarray(value, dtype=self._vel.dtype)
		if arr.ndim == 1:
			arr = arr.reshape(-1, 2)
		elif arr.ndim == 2 and arr.shape[1] >= 2:
			arr = arr[:, :2]
		else:
			print("sim.vel must be shape (N,2) or flat length 2N")
			return
		if arr.shape != self._vel.shape:
			print(f"shape mismatch when assigning to sim.vel: "
							 f"expected {self._vel.shape}, got {arr.shape}")
			return
		self._vel[...] = arr  

	@mass.setter
	def mass(self, value: np.ndarray) -> None:
		arr = np.asarray(value, dtype=np.float64).ravel()
		if arr.shape != self._mass.shape:
			print(f"shape mismatch when assigning to sim.mass: "
							 f"expected {self._mass.shape}, got {arr.shape}")
			return
		if np.any(arr <= 0) or not np.all(np.isfinite(arr)):
			print("all masses must be positive finite numbers")
			return
		self._mass[...] = arr
	
	def build_state(self, bodies: List[Body] | None, masses, positions, velocities) -> bool:
		if bodies is None:
			if masses is None or positions is None:
				return False
				
			masses = list(masses)
			positions = list(positions)
			if velocities is None:
				velocities = []
			else:
				velocities = list(velocities)
			
			if len(velocities) == 0:
				velocities = [(0.0, 0.0)] * len(masses)
			elif len(velocities) == 1 and len(masses) > 1:
				velocities = velocities * len(masses)
				
			if len(velocities) != len(masses):
				return False
				
			self.n_bodies = len(masses)
			self._mass = np.asarray(masses, dtype=np.float64)
			self._pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
			self._vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
			
		else:
			self.n_bodies = len(bodies)
			mass_list = []
			for b in bodies:
				mass_list.append(b.mass)
			self._mass = np.array(mass_list, dtype=np.float64)
			
			pos_list = []
			for b in bodies:
				pos_list.append((b.x, b.y))
			self._pos = np.array(pos_list, dtype=np.float64)
			
			vel_list = []
			for b in bodies:
				vel_list.append((b.vx, b.vy))
			self._vel = np.array(vel_list, dtype=np.float64)
			
		if np.any(self._mass <= 0) or not np.all(np.isfinite(self._mass)):
			return False
			
		self._acc = np.zeros_like(self._pos)
		return True
	
	def disable_simulation(self) -> None:
		self.n_bodies = 0
		self._mass = np.empty(0, dtype=np.float64)
		self._pos = np.empty((0, 2), dtype=np.float64)
		self._vel = np.empty((0, 2), dtype=np.float64)
		self._acc = np.empty((0, 2), dtype=np.float64)
		
		if getattr(self, "_buf_diff", None) is not None:
			if self._buf_diff.shape[0] != self.n_bodies:
				del self._buf_diff
				self._buf_diff = None
				
		if getattr(self, "_buf_r2", None) is not None:
			if self._buf_r2.shape[0] != self.n_bodies:
				del self._buf_r2
				self._buf_r2 = None
	
	def snapshot(self, sim: "NBodySimulation") -> dict:
		sim.commit_state()
		
		soft_state = {
			"s": getattr(sim.manager, "s", 0.0),
			"s2": getattr(sim.manager, "s2", 0.0),
			"_step_s2": getattr(sim.manager, "_step_s2", 0.0),
			"_pending_energy_delta": getattr(sim.manager, "_pending_energy_delta", 0.0),
			"_history": list(getattr(sim.manager, "_history", [])),
			"_step_finished": getattr(sim.manager, "_step_finished", False),
		}
		
		if hasattr(sim, "_integrator"):
			int_state = {
				"dt_prev": sim._integrator._dt_prev,
				"eps_prev": sim._integrator._eps_prev,
				"_top_dt": sim._integrator._top_dt,
				"_last_update_tick": sim._integrator._last_update_tick,
				"_cached_min_sep": sim._integrator._cached_min_sep,
				"k_soft": getattr(sim._integrator, "k_soft", 0.0),
				"mu_soft": getattr(sim._integrator, "mu_soft", 1.0),
			}
		else:
			int_state = {
				"dt_prev": None,
				"eps_prev": None,
				"_top_dt": None,
				"_last_update_tick": 0,
				"_cached_min_sep": None,
				"k_soft": 0.0,
				"mu_soft": 1.0,
			}
		
		sim_flags = {
			"_acc_cached": getattr(sim, "_acc_cached", True),
			"_in_integration": getattr(sim, "_in_integration", False),
			"softening_energy_delta": getattr(sim, "softening_energy_delta", 0.0),
			"_adaptive_timestep": getattr(sim, "_adaptive_timestep", False),
			"_adaptive_softening": getattr(sim, "_adaptive_softening", False),
			"_epsilon": getattr(sim, "_epsilon", 0.0),
			"_pi": getattr(sim, "_pi", 0.0),
		}
		
		snap = {
			"masses": getattr(self, "_mass", np.empty(0)),
			"positions": getattr(self, "_pos", np.empty((0, 2))),
			"velocities": getattr(self, "_vel", np.empty((0, 2))),
			"softening": soft_state["s"],
			"softening_s2": soft_state["s2"],
			"pending_energy": sim_flags["softening_energy_delta"],
			"integrator_state": int_state,
			"softening_mgr_state": soft_state,
			"sim_state": sim_flags,
			"cfg": getattr(sim, "cfg", None).copy() if hasattr(sim, "cfg") else None,
			"has_integrated": bool(getattr(sim, "_has_integrated", False)),
			"sim": {
				"masses": getattr(self, "_mass", np.empty(0)),
				"positions": getattr(self, "_pos", np.empty((0, 2))),
				"velocities": getattr(self, "_vel", np.empty((0, 2))),
				"flags": sim_flags,
			},
			"integrator": int_state,
			"softening_mgr": soft_state,
			"acc": getattr(self, "_acc", np.empty((0, 2))),
		}
		return snap
	

	@staticmethod
	def restore_to_sim(state_dict: dict, sim: "NBodySimulation") -> None:
		sim_data = state_dict.get("sim", state_dict)
		
		sim._state._mass = sim_data["masses"].copy()
		sim._state._pos = sim_data["positions"].copy()
		sim._state._vel = sim_data["velocities"].copy()
		sim._state.n_bodies = len(sim._state._mass)
		
		if "acc" in state_dict:
			sim._state._acc = state_dict["acc"].copy()
		else:
			sim._state._acc = np.zeros_like(sim._state._pos)
		
		soft_data = state_dict.get("softening_mgr_state", state_dict.get("softening_mgr", {}))
		mgr = sim.manager
		mgr.s = soft_data.get("s", mgr.s)
		mgr.s2 = soft_data.get("s2", mgr.s2)
		mgr._step_s2 = soft_data.get("_step_s2", mgr._step_s2)
		mgr._pending_energy_delta = soft_data.get("_pending_energy_delta", 0.0)
		
		hist = soft_data.get("_history")
		if hist is not None:
			mgr._history = list(hist)
			mgr._history_len_at_begin = len(mgr._history)
		
		mgr._step_finished = soft_data.get("_step_finished", False)
		
		sim_state = state_dict.get("sim_state", sim_data.get("flags", {}))
		sim._epsilon = float(sim_state.get("_epsilon", sim.manager.s))
		sim.manager.update_continuous(sim._epsilon)
		sim._pi = float(sim_state.get("_pi", 0.0))
		
		sim.softening_energy_delta = sim_state.get(
			"softening_energy_delta", state_dict.get("pending_energy", 0.0)
		)
		sim._in_integration = sim_state.get("_in_integration", False)
		sim._acc_cached = sim_state.get("_acc_cached", False)
		sim._has_integrated = bool(state_dict.get("has_integrated", False))
		
		int_state = state_dict.get("integrator_state", state_dict.get("integrator", {}))
		integ = getattr(sim, "_integrator", None)
		if integ is not None:
			integ.k_soft = float(int_state.get("k_soft", getattr(integ, "k_soft", 0.0)))
			integ.mu_soft = float(int_state.get("mu_soft", getattr(integ, "mu_soft", 1.0)))
			integ._dt_prev = int_state.get("dt_prev")
			integ._eps_prev = int_state.get("eps_prev")
			integ._top_dt = int_state.get("_top_dt")
			integ._last_update_tick = int_state.get("_last_update_tick", 0)
			integ._cached_min_sep = None
	
	def set_fast_mode(self, float32: bool = True) -> None:
		if float32:
			self._pos = self._pos.astype(np.float32, copy=False)
			self._vel = self._vel.astype(np.float32, copy=False)
			self._acc = self._acc.astype(np.float32, copy=False)
			self._mass = self._mass.astype(np.float32, copy=False)
		else:
			self._pos = self._pos.astype(np.float64, copy=False)
			self._vel = self._vel.astype(np.float64, copy=False)
			self._acc = self._acc.astype(np.float64, copy=False)
			self._mass = self._mass.astype(np.float64, copy=False)
