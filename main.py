import numpy as np
import configparser
import h5py
from sphere import Sphere
from hydrodynamics import compute_G_matrix, solve_for_vel_others
from integrator import integrate_and_update_spheres
from utils import check_G_matrix, check_for_clumping, plot_spheres
from utils import save_sphere_history_npz, save_sphere_history_hdf5, setup_logging, log_start_time_and_parameters, log_sphere_data, log_end_time

# Load configuration from ini file
config = configparser.ConfigParser()
config.read('crowley.ini')

# Extract parameters
gmatrix_threshold = config.getfloat('Thresholds', 'gmatrix_threshold')
acceleration_threshold = config.getfloat('Thresholds', 'acceleration_threshold')
velocity_threshold = config.getfloat('Thresholds', 'velocity_threshold')

t_flow = config.getfloat('Time Scales', 't_flow')
omega = config.getfloat('Time Scales', 'omega')
t_response = config.getfloat('Time Scales', 't_response')

g = config.getfloat('Units', 'g')

a = config.getfloat('Particles', 'a')
N = config.getint('Particles', 'N')
A = config.getfloat('Particles', 'A')
d = config.getfloat('Particles', 'd') * a
clumping_threshold = config.getfloat('Particles', 'clumping_threshold')

FR_PART = config.getfloat('Simulation', 'FR_PART')
dt = config.getfloat('Simulation', 'dt')
total_time = config.getfloat('Simulation', 'total_time')
snapshots = config.getint('Simulation', 'snapshots')
x_offset = config.getfloat('Simulation', 'x_offset')
y_offset = config.getfloat('Simulation', 'y_offset')
k = config.getfloat('Simulation', 'k')

plot_directory = config.get('Paths', 'plot_directory')
hdf5_filename = config.get('Paths', 'hdf5_filename')
npz_filename = config.get('Paths', 'npz_filename')

# Derived quantities
v_terminal = g * t_response  # Terminal velocity
mass = (4 / 3) * np.pi * 4 * t_response  # Mass of the sphere
mu = mass / (np.pi * 6 * t_response * a)  # Viscosity

time_unit = t_flow
velocity_unit = v_terminal
length_unit = velocity_unit * time_unit
ST = t_response / t_flow
ST_ROT = omega * t_flow

time_steps = int(total_time / dt)
snapshot_interval = time_steps // snapshots

# Initialize spheres
spheres = []
for i in range(N):
    x_position = x_offset + i * d
    y_position = y_offset - A * np.sin(k * 10 * (x_position - x_offset) / (N * d))
    initial_position = np.array([x_position, y_position, 0])
    initial_velocity = np.array([0, 1.0 * v_terminal, 0]) / v_terminal

    spheres.append(Sphere(stokes_no=ST, position=initial_position, velocity=initial_velocity))

# Setup logging
setup_logging()

# Log parameters
params = {
    "Stokes number": ST,
    "Rotational Stokes number": ST_ROT,
    "Flow time": t_flow,
    "Terminal velocity": v_terminal,
    "Length unit": length_unit,
    "Velocity unit": velocity_unit,
    "Radius of spheres": a,
    "Distance between spheres": d,
    "viscosity": mu,
    "Perturbation Amplitude": A,
}
log_start_time_and_parameters(params)

# Initialize the HDF5 file by creating datasets if they don't exist
with h5py.File(hdf5_filename, 'w') as f:
    f.create_group('spheres_data')

# Initial save
save_sphere_history_npz(spheres, npz_filename, step=0, time=0)

# Time loop for simulation
time = []
for step in range(time_steps):
    print(f"Time step {step + 1}/{time_steps}")
    current_time = step * dt

    positions = np.array([sphere.position for sphere in spheres])
    G_matrix = compute_G_matrix(positions, mu)
    check_G_matrix(G_matrix, threshold=gmatrix_threshold)
    solve_for_vel_others(spheres, G_matrix, ST)
    integrate_and_update_spheres(spheres, dt, ST, ST_ROT, g, FR_PART, velocity_threshold, acceleration_threshold)
    log_sphere_data(spheres, step + 1)
    plot_spheres(spheres, step, snapshot_interval, plot_directory, ST, ST_ROT)

    if check_for_clumping(spheres, clumping_threshold):
        break  # Exit the loop if clumping is detected

    time.append(current_time)
    save_sphere_history_npz(spheres, npz_filename, step, current_time)
    save_sphere_history_hdf5(spheres, hdf5_filename, step, current_time)

# Log end time
log_end_time()
