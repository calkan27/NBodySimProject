"""
This module defines the Body class, a simple data container for individual celestial
bodies in the simulation.

The class stores fundamental properties (mass, position x/y, velocity vx/vy) as
floating-point attributes and provides a clean string representation for debugging. It
serves as the basic building block for initial condition specification before conversion
to the optimized numpy array format used during simulation. The class makes no
assumptions about units or coordinate systems, leaving those decisions to the simulation
layer.
"""
class Body:
	def __init__(self, mass: float, x: float, y: float, vx: float, vy: float):
		self.mass = float(mass)
		self.x = float(x)
		self.y = float(y)
		self.vx = float(vx)
		self.vy = float(vy)

	def __repr__(self) -> str:
		return f"Body(mass={self.mass}, x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy})"

