# main.py
import numpy as np
from sphere import Sphere
from hydrodynamics import compute_G_matrix, solve_for_vel_others
from integrator import velocity_ode, integrate_and_update_spheres
from utils import  check_G_matrix,  check_for_clumping, check_and_scale_blow_up, plot_spheres
from utils import save_sphere_history_npz, save_sphere_history_hdf5, setup_logging, log_start_time_and_parameters, log_sphere_data,log_end_time
import h5py

# main.py
USE_CUPY = True
USE_NUMBA = False
USE_ADAPTIVE_TIMESTEP = True  # Enable adaptive time stepping

import integrator
integrator.USE_CUPY = USE_CUPY
integrator.USE_NUMBA = USE_NUMBA
integrator.USE_ADAPTIVE_TIMESTEP = USE_ADAPTIVE_TIMESTEP

if __name__ == "__main__":
    setup_logging() 


# Parameters
gmatrix_threshold = 100
acceleration_threshold = 150
velocity_threshold = 10

#Time scales
t_flow = 100  #flow time  1 kilosecond = 1000 s
omega = 1e-2    #Rotational frequency
t_response = 10   #Response time of the particle

g = 6.0
#Units for Non-dimensionalisation
v_terminal = g * t_response
time_unit = t_flow
velocity_unit = v_terminal
length_unit =  velocity_unit* time_unit

#Particles

a = 1

#Fluid parameter
mass = np.pi*4/3*4*t_response
mu = mass/(np.pi*6*t_response*a) # viscosity
#mu = 1.1

FR_PART = 1/20.0
N = 26 # Number of particles
A = 0  # Amplitude of the sinusoid
d = 10*a  # Interparticle distance
clumping_threshold = 3*a

#Non-dimensional numbers
ST = t_response / t_flow  # Stokes number wrt flow
ST_ROT = omega *t_flow # Stokes number wrt rotation"

#Perturbation
k = 1  # Wave number
x_offset = 6  # Base x position offset
y_offset = 60  # Base y position offset

dt = 0.1
total_time = 100
snapshots = 1000  # Number of snapshots you want
time_steps = int(total_time / dt)
snapshot_interval = time_steps // snapshots
plot_directory = 'data/plots'
time=[]

# Initial positions and velocities of spheres with sinusoidal perturbation
spheres = []
for i in range(N):
    x_position = x_offset + i * d
    y_position = y_offset  - A * np.sin(k * 10 * (x_position - x_offset) / (N * d))
    initial_position = np.array([x_position, y_position, 0]) 
    initial_velocity = np.array([0, 1.0*v_terminal, 0])/v_terminal 
    initial_drag_acc = -(1 / ST) * (initial_velocity) 

    spheres.append(Sphere(stokes_no=ST,position=initial_position, velocity=initial_velocity))

# Setup logging
setup_logging()
# Simulation parameters
params = {
    "Stokes number": ST,
    "Rotational Stokes number": ST_ROT,
    "Flow time": t_flow,
    "Terminal velocity": g * t_response,
    "Length unit": length_unit,
    "Velocity unit": velocity_unit,
    "Radius of spheres": a,
    "Distance between spheres":d,
    "viscosity": mu,
    "Perturbation Amplitude": A
    
}
log_start_time_and_parameters(params)  # Log start time and parameters

# Ensure the file is created at the beginning of the simulation
hdf5_filename = 'data/spheres_history.h5'
# Initialize the HDF5 file by creating datasets if they don't exist
with h5py.File(hdf5_filename, 'w') as f:
    f.create_group('spheres_data')  # Create a group for sphere data
npz_filename = 'data/sphere_histories.npz'
save_sphere_history_npz(spheres, npz_filename,step= 0, time= 0)  # Initial save

# Time loop for simulation
for step in range(time_steps):
    print(f"Time step {step+1}/{time_steps}") 
    current_time = step * dt

    positions = np.array([sphere.position for sphere in spheres])
    G_matrix = compute_G_matrix(positions, mu)
    check_G_matrix(G_matrix, threshold=gmatrix_threshold) 
    solve_for_vel_others(spheres, G_matrix, ST)
    integrate_and_update_spheres(spheres, dt, ST, ST_ROT, g, FR_PART, velocity_threshold, acceleration_threshold)
    log_sphere_data(spheres, step + 1)  # Log sphere data
    plot_spheres(spheres, step, snapshot_interval, plot_directory, ST, ST_ROT)
    
    if check_for_clumping(spheres, clumping_threshold):
        break  # Skip the rest of the loop if clumping is detected
    
    time.append(current_time)
    
    save_sphere_history_npz(spheres, npz_filename, step, current_time)  # Save histories at every step
    save_sphere_history_hdf5(spheres, hdf5_filename, step, current_time) # Save data in HDF5 format

# Log end time
log_end_time()
     