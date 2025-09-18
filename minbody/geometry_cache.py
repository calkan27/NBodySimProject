from __future__ import annotations
import numpy as np
from typing import Tuple

"""
This module provides optimized geometric calculations for N-body force computations. The geometry_buffers function efficiently computes pairwise position differences, squared distances, and inverse cubed distances in a single pass, using Einstein summation notation for performance. It handles softening by adding epsilon^2 to squared distances and fills diagonal elements with appropriate values to exclude self-interactions. The module serves as a computational kernel for force calculations, minimizing redundant distance computations across the codebase. It assumes 2D position arrays and non-negative softening parameters.

"""




__all__ = ["geometry_buffers"]

def geometry_buffers(
    pos: np.ndarray,
    eps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=float)

    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)

    inv_r3 = np.zeros_like(r2, dtype=float)
    mask = (r2 + eps * eps) > 0.0
    if np.any(mask):
        inv_r3[mask] = np.power(r2[mask] + eps * eps, -1.5)

    np.fill_diagonal(inv_r3, 0.0)
    return diff, r2, inv_r3



