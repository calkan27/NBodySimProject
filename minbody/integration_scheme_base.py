"""
This abstract base class defines the interface for numerical integration schemes.

The IntegrationScheme class provides common functionality including
flag_positions_changed for cache invalidation, drift for position updates, kick for
velocity updates, ensure_buffer for efficient array allocation, and method stubs for
specific integration algorithms. It serves as the foundation for Verlet, Yoshida4, and
WHFast implementations, managing shared resources like geometric buffers and
acceleration caches. The class assumes the parent integrator maintains valid simulation
references and array allocations.
"""

from __future__ import annotations
import math
from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .integrator import Integrator



class IntegrationScheme:
	def __init__(self, integrator: "Integrator") -> None:
		self.integ = integrator
		self._warned_reentrant = False


	def flag_positions_changed(self) -> None:
		integ = self.integ
		integ._cached_min_sep = None
		integ.sim._acc_cached = False

		integ._pos_hash = np.int64(np.sum(integ.sim._pos, dtype=np.float64))

		if hasattr(integ, "_eps_tick"):
			integ._eps_tick += 1
		if hasattr(integ, "_dV_cache_key"):
			integ._dV_cache_key = None

	def drift(self, h_coeff: float) -> None:
		self.integ.sim._pos += h_coeff * self.integ.sim._vel
		self.flag_positions_changed()

	def kick(self, dt: float) -> None:
		self._apply_V_operator(dt)

	def _apply_V_operator(self, dt: float) -> None:
		sim = self.integ.sim
		dt = float(dt)
		if sim.n_bodies >= 2 and sim.G != 0.0:
			acc = sim._accel()
			sim._vel += dt * acc
		sim._acc_cached = False

	def ensure_buffer(
		self,
		name: str,
		shape: Tuple[int, int],
		*,
		dtype: np.dtype | str = np.float64,
	) -> np.ndarray:
		rows = int(shape[0])
		cols = int(shape[1])

		if rows < 0:
			rows = 0
		if cols < 0:
			cols = 0

		buf = getattr(self.integ, name, None)

		cap_attr_shape = f"{name}_cap_shape"
		cap_shape = getattr(self.integ, cap_attr_shape, None)

		if isinstance(cap_shape, tuple) and len(cap_shape) == 2:
			rows_cap = int(cap_shape[0])
			cols_cap = int(cap_shape[1])
		else:
			legacy_cap = getattr(self.integ, f"{name}_cap", 0)
			if isinstance(legacy_cap, (int, float, np.floating)):
				rows_cap = int(legacy_cap)
				cols_cap = int(legacy_cap)
			else:
				if isinstance(buf, np.ndarray) and buf.ndim == 2:
					rows_cap = int(buf.shape[0])
					cols_cap = int(buf.shape[1])
				else:
					rows_cap = 0
					cols_cap = 0

		req_dtype = np.dtype(dtype)

		need_realloc = False
		if not isinstance(buf, np.ndarray):
			need_realloc = True
		else:
			if buf.ndim != 2:
				need_realloc = True
			else:
				if rows_cap < rows or cols_cap < cols:
					need_realloc = True
				else:
					if buf.dtype != req_dtype:
						need_realloc = True

		if need_realloc:
			new_rows = int(math.ceil(rows * 1.5))
			new_cols = int(math.ceil(cols * 1.5))

			if new_rows < rows:
				new_rows = rows
			if new_cols < cols:
				new_cols = cols

			buf = np.empty((new_rows, new_cols), dtype=req_dtype)
			setattr(self.integ, name, buf)
			setattr(self.integ, cap_attr_shape, (new_rows, new_cols))
			setattr(self.integ, f"{name}_cap", max(new_rows, new_cols))
		else:
			setattr(self.integ, cap_attr_shape, (rows_cap, cols_cap))
			if not hasattr(self.integ, f"{name}_cap"):
				setattr(self.integ, f"{name}_cap", max(rows_cap, cols_cap))

		return buf[:rows, :cols]



	def _verlet_kernel(self, h: float) -> None:
		sim = self.integ.sim
		if sim._in_integration:
			if not getattr(self, "_warned_reentrant", False):
				print("[warning] Integrator._verlet called re-entrantly; call ignored")
				self._warned_reentrant = True
			return

		sim._acc_cached = False
		sim._in_integration = True

		h2 = 0.5 * h
		acc_old = sim._accel()
		sim._vel += h2 * acc_old
		sim._pos += h * sim._vel
		self.flag_positions_changed()
		acc_new = sim._accel()
		sim._vel += h2 * acc_new

		sim._acc_cached = False
		sim._in_integration = False

	def _kepler_propagate(self, r: np.ndarray, v: np.ndarray, mu: float, dt: float):
		return self.integ._uv_solver.propagate(r, v, mu, dt)

	def apply_corrector(self, order: int) -> None:
		sim = self.integ.sim
		if order is None or int(order) <= 0:
			return
		if sim.n_bodies < 1 or sim.G == 0.0:
			return

		h_ref = 0.0

		top_dt = getattr(self.integ, "_top_dt", None)
		if isinstance(top_dt, (int, float, np.floating)):
			val = float(abs(top_dt))
			if np.isfinite(val) and val > 0.0:
				h_ref = val

		if not (np.isfinite(h_ref) and h_ref > 0.0):
			hsr = getattr(self.integ, "h_sub_ref", None)
			if isinstance(hsr, (int, float, np.floating)):
				val = float(abs(hsr))
				if np.isfinite(val) and val > 0.0:
					h_ref = val

		if not (np.isfinite(h_ref) and h_ref > 0.0):
			return

		h2 = 0.5 * h_ref
		self._apply_V_operator(h2)

		extra_refresh = 0
		if int(order) >= 6:
			extra_refresh = 2
		elif int(order) >= 4:
			extra_refresh = 1

		for _ in range(extra_refresh):
			sim._acc_cached = False
			_ = sim._accel()

		sim._acc_cached = False
		

