"""
This module computes variational dynamics for chaos analysis in N-body systems.

The TangentMap class calculates variational accelerations for trajectory perturbations
using the Jacobian of gravitational forces, supporting MEGNO and Lyapunov exponent
calculations. The implementation efficiently handles the tensor operations required for
tangent space evolution and properly accounts for softening in force derivatives. It
assumes access to current positions and accelerations from the parent simulation.
"""

import numpy as np




class TangentMap:
    def __init__(self, sim):

        self.sim = sim

    def variational_accel(self, delta_r):
        sim   = self.sim
        integ = sim._integrator            

        n = sim.n_bodies
        G = sim.G
        if n < 2 or G == 0.0:
            return np.zeros_like(delta_r)

        pos  = sim._pos                    
        mass = sim._mass                   
        s2   = sim.manager.step_s2         


        if (
            getattr(sim, "_buf_diff", None) is None
            or sim._buf_diff.shape != (n, n, pos.shape[1])
        ):
            sim._buf_diff = np.empty((n, n, pos.shape[1]), dtype=pos.dtype)
        diff = sim._buf_diff               

        r2 = integ._ensure_buffer("_buf_r2", (n, n), dtype=pos.dtype)

        np.subtract(pos[None, :, :], pos[:, None, :], out=diff)
        np.einsum("ijk,ijk->ij", diff, diff, out=r2)
        r2 += s2
        np.fill_diagonal(r2, np.inf)

        inv_r2 = 1.0 / r2
        inv_r3 = inv_r2 * np.sqrt(inv_r2)          

        d_diff = delta_r[None, :, :] - delta_r[:, None, :]
        dot    = np.einsum("ijk,ijk->ij", diff, d_diff)  

        coeff  = 3.0 * dot * inv_r2 * inv_r3             
        term   = d_diff * inv_r3[..., None] - coeff[..., None] * diff

        delta_a = G * np.sum(mass[None, :, None] * term, axis=1)
        return delta_a

