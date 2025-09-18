"""
This module implements a Kahan summation accumulator for tracking energy changes with
high numerical precision.

The EnergyAccumulator class uses compensated summation to minimize floating-point
roundoff errors when accumulating many small energy deltas from softening changes,
spring potential updates, and barrier interactions. It maintains separate accumulators
for each energy component while tracking the total accumulated change. The
implementation is critical for maintaining energy conservation in long simulations where
standard floating-point addition would accumulate significant errors. It assumes all
energy deltas are finite floating-point values.
"""

from __future__ import annotations
from dataclasses import dataclass



@dataclass
class _Kahan:
    total: float = 0.0          
    comp:  float = 0.0         

    def add(self, x: float) -> None:
        y = float(x) - self.comp
        t = self.total + y
        self.comp = (t - self.total) - y
        self.total = t


class EnergyAccumulator:
	def __init__(self) -> None:
		self._soft = _Kahan()    
		self._spring = _Kahan()  
		self._bar = _Kahan()     


		self._soft_overall = _Kahan()

	@property
	def softening_delta(self) -> float:
		return self._soft.total

	@property
	def spring_delta(self) -> float:
		return self._spring.total

	@property
	def barrier_delta(self) -> float:
		return self._bar.total

	@property
	def total_delta(self) -> float:
		return float(self._soft.total + self._spring.total + self._bar.total)

	def add_softening(self, dE: float) -> None:
		val = float(dE)
		self._soft.add(val)
		self._soft_overall.add(val)

	def add_spring(self, dE: float) -> None:
		val = float(dE)
		self._spring.add(val)
		self._soft_overall.add(val)

	def add_barrier(self, dE: float) -> None:
		val = float(dE)
		self._bar.add(val)
		self._soft_overall.add(val)

	def reset(self) -> None:
		for k in (self._soft, self._spring, self._bar, self._soft_overall):
			k.total = 0.0
			k.comp = 0.0

	def add(self, dE: float) -> None:
		val = float(dE)
		self._soft.add(val)
		self._soft_overall.add(val)

	def total(self) -> float:
		return float(self._soft.total + self._spring.total + self._bar.total)

