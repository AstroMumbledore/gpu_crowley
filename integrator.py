from scipy.integrate import solve_ivp
import numpy as np
import cupy as cp
from utils import check_and_scale_blow_up

# Flags set in main.py
USE_NUMBA = False
USE_CUPY = False
USE_ADAPTIVE_TIMESTEP = False  # Flag for adaptive time stepping

# Set the appropriate library alias (numpy or cupy)
if USE_CUPY:
    xp = cp  # CuPy for GPU
else:
    xp = np  # NumPy for CPU

# Define the ODE function using the appropriate library
def velocity_ode(t, v, vel_others, ST, ST_ROT, g, FR_PART):
    Zhat = xp.array([0, 0, 1])
    yhat = xp.array([0, 1, 0])
    dv_dt = (vel_others - v) / ST - 2 * xp.cross(ST_ROT * Zhat, v) - (1 / ST) * yhat
    return dv_dt

# Integrate and update spheres
def integrate_and_update_spheres(spheres, dt, ST, ST_ROT, g, FR_PART, velocity_threshold, acceleration_threshold):
    if USE_ADAPTIVE_TIMESTEP:
        # Adjust dt based on the smallest time scale
        min_time_scale = min(ST, ST_ROT)
        if min_time_scale < 1:
            dt = dt / min_time_scale  # Scale the time step based on ST or ST_ROT

    for sphere in spheres:
        # Convert velocity to the appropriate library (NumPy or CuPy)
        velocity = xp.array(sphere.velocity)
        vel_others = xp.array(sphere.vel_others)

        if USE_ADAPTIVE_TIMESTEP:
            # Adaptive time-stepping using LSODA (SciPy's method with automatic step size control)
            sol = solve_ivp(velocity_ode, [0, dt], velocity, method='LSODA',
                            args=(vel_others, ST, ST_ROT, g, FR_PART), t_eval=[dt])
        else:
            # Fixed time step integration
            sol = solve_ivp(velocity_ode, [0, dt], velocity, method='RK45', 
                            args=(vel_others, ST, ST_ROT, g, FR_PART), t_eval=[dt], max_step=dt)

        # Retrieve new velocity
        new_velocity = sol.y[:, -1]

        # Update position
        new_position = sphere.position + new_velocity * dt

        # Update acceleration
        new_acceleration = (new_velocity - sphere.velocity) / dt

        # Update sphere
        sphere.update(sphere.stokes_no, new_position, xp.asnumpy(new_velocity) if USE_CUPY else new_velocity, sphere.vel_others, new_acceleration, dt)
        
        # Check for blow-ups and scale down if necessary
        check_and_scale_blow_up(sphere, velocity_threshold, acceleration_threshold)
