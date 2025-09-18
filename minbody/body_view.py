"""
This module implements BodyView, a proxy class providing Body-like access to individual
particles stored in the simulation's numpy arrays.

The class uses properties with getters and setters to map attribute access (mass, x, y,
vx, vy) directly to the appropriate array indices in the parent simulation, maintaining
the same interface as Body while operating on the efficient array storage. This design
allows intuitive particle manipulation without data copying, bridging the gap between
user-friendly object notation and performance-critical array operations. The view
assumes the parent simulation maintains valid array structures and that the body index
remains within bounds.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulation import NBodySimulation




class BodyView:
	__slots__ = ("_sim", "_i")

	def __init__(self, sim: "NBodySimulation", idx: int) -> None:
		self._sim = sim
		self._i = int(idx)

	@property
	def mass(self) -> float:
		return float(self._sim._mass[self._i])
	@mass.setter
	def mass(self, v: float) -> None:
		self._sim._mass[self._i] = float(v)

	@property
	def x(self) -> float:
		return float(self._sim._pos[self._i, 0])
	@x.setter
	def x(self, v: float) -> None:
		self._sim._pos[self._i, 0] = float(v)

	@property
	def y(self) -> float:
		return float(self._sim._pos[self._i, 1])
	@y.setter
	def y(self, v: float) -> None:
		self._sim._pos[self._i, 1] = float(v)

	@property
	def vx(self) -> float:
		return float(self._sim._vel[self._i, 0])
	@vx.setter
	def vx(self, v: float) -> None:
		self._sim._vel[self._i, 0] = float(v)

	@property
	def vy(self) -> float:
		return float(self._sim._vel[self._i, 1])
	@vy.setter
	def vy(self, v: float) -> None:
		self._sim._vel[self._i, 1] = float(v)

	def __repr__(self) -> str:
		return (f"Body(mass={self.mass}, x={self.x}, y={self.y}, "
				f"vx={self.vx}, vy={self.vy})")

