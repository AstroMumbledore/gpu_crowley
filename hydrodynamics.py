import numpy as np
import cupy as cp
from numba import jit
import logging
import sys

# These flags should be set in main.py
USE_NUMBA = False
USE_CUPY = False

# Determine which library to use
if USE_CUPY:
    xp = cp  # Use CuPy for GPU
else:
    xp = np  # Use NumPy for CPU

if USE_NUMBA:
    @jit(nopython=True)
    def compute_G_matrix(positions, mu):
        N = len(positions)
        G_matrix = np.zeros((3 * N, 3 * N))
        I = np.eye(3)
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_diff = positions[i] - positions[j]
                    r_norm = np.linalg.norm(r_diff)
                    if r_norm != 0:
                        G = (1 / (8 * np.pi * mu)) * (I / r_norm + np.outer(r_diff, r_diff) / r_norm**3)
                    else:
                        G = np.zeros((3, 3))
                    G_matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3] = G
        return G_matrix
else:
    def compute_G_matrix(positions, mu):
        N = len(positions)
        G_matrix = xp.zeros((3 * N, 3 * N))
        I = xp.eye(3)
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_diff = positions[i] - positions[j]
                    r_norm = xp.linalg.norm(r_diff)
                    if r_norm != 0:
                        G = (1 / (8 * xp.pi * mu)) * (I / r_norm + xp.outer(r_diff, r_diff) / r_norm**3)
                    else:
                        G = xp.zeros((3, 3))
                    G_matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3] = G
        return G_matrix

def check_G_matrix(G_matrix, threshold):
    logger = logging.getLogger(__name__)
    if xp.any(G_matrix > threshold) or xp.any(G_matrix < -threshold):
        logger.warning("G_matrix contains values exceeding the threshold.")
        logger.warning(f"Max value in G_matrix: {np.max(G_matrix)}")
        logger.warning(f"Min value in G_matrix: {np.min(G_matrix)}")
        sys.exit(1)
    else:
        logger.info("G_matrix values are within the safe range.")

if USE_NUMBA:
    @jit(nopython=True)
    def solve_for_vel_others(spheres, G_matrix, ST):
        N = len(spheres)
        velocities = np.array([sphere.velocity for sphere in spheres]).flatten()
        rhs = np.zeros(3 * N)
        vel_others = np.zeros_like(velocities)

        for i in range(N):
            for j in range(N):
                if i != j:
                    G_ij = G_matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                    rhs[3 * i:3 * i + 3] += G_ij @ (velocities[3 * j:3 * j + 3] - vel_others[3 * j:3 * j + 3]) / ST

        A = np.eye(3 * N) - G_matrix
        vel_others = np.linalg.solve(A, rhs).reshape((N, 3))

        for i, sphere in enumerate(spheres):
            sphere.update_vel_others(vel_others[i])

else:
    def solve_for_vel_others(spheres, G_matrix, ST):
        N = len(spheres)
        velocities = xp.array([sphere.velocity for sphere in spheres]).flatten()
        rhs = xp.zeros(3 * N)
        vel_others = xp.zeros_like(velocities)

        for i in range(N):
            for j in range(N):
                if i != j:
                    G_ij = G_matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                    rhs[3 * i:3 * i + 3] += G_ij @ (velocities[3 * j:3 * j + 3] - vel_others[3 * j:3 * j + 3]) / ST

        A = xp.eye(3 * N) - G_matrix
        vel_others = xp.linalg.solve(A, rhs).reshape((N, 3))

        for i, sphere in enumerate(spheres):
            sphere.update_vel_others(xp.asnumpy(vel_others[i])) if USE_CUPY else sphere.update_vel_others(vel_others[i])
