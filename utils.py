# utils.py
import numpy as np
import sys
import os
import logging
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

#to check G matrix
def check_G_matrix(G_matrix, threshold):
    large_values = G_matrix > threshold
    small_values = G_matrix < -threshold
    if np.any(large_values) or np.any(small_values):
        print("Warning: G_matrix contains values exceeding the threshold.")
        print("Max value in G_matrix:", np.max(G_matrix))
        print("Min value in G_matrix:", np.min(G_matrix))
        print("Exiting program due to G_matrix overflow.")
        sys.exit(1)
    else:
        print("G_matrix values are within the safe range.")
        
# Helper function to calculate the average of the last five values in a list
def calculate_average_of_last_five(history_list):
    if len(history_list) >= 5:
        return np.mean(history_list[-5:], axis=0)
    else:
        return np.mean(history_list, axis=0)

# Function to check and scale down values if blow-up is detected
def check_and_scale_blow_up(sphere, velocity_threshold, acceleration_threshold):
    current_velocity = sphere.velocity
    current_acceleration = sphere.acceleration
    
    # Check if the velocity exceeds the threshold
    if np.linalg.norm(current_velocity) > velocity_threshold:
        avg_velocity = calculate_average_of_last_five(sphere.velocity_history)
        print(f"Velocity blow-up detected. Scaling down to average of last 5 time steps: {avg_velocity}")
        sphere.velocity = avg_velocity  # Scale down to the average velocity
    
    # Check if the acceleration exceeds the threshold
    if np.linalg.norm(current_acceleration) > acceleration_threshold:
        avg_acceleration = calculate_average_of_last_five(sphere.acceleration_history)
        print(f"Acceleration blow-up detected. Scaling down to average of last 5 time steps: {avg_acceleration}")
        sphere.acceleration = avg_acceleration  # Scale down to the average acceleration
 
#Check for clumping       
def check_for_clumping(spheres, clumping_threshold):
    N = len(spheres)
    clumping_detected = False
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(spheres[i].position - spheres[j].position)
            if distance < clumping_threshold:
                print(f"Clumping detected: Spheres {i+1} and {j+1} are too close with a distance of {distance:.6f}.")
                clumping_detected = True

    if clumping_detected:
        return True  # Indicate that clumping was detected
    else:
        return False  # No clumping detected
    
# Assuming Sphere class has the attribute `position_history` 
# which stores the position of the sphere at each time step.

def calculate_distance_between_spheres(sphere1, sphere2):
    """
    Calculate the distance between two spheres over time.
    """
    # Extract the position histories of both spheres
    pos_hist1 = np.array(sphere1.position_history)
    pos_hist2 = np.array(sphere2.position_history)
    
    # Calculate the Euclidean distance between the two spheres at each time step
    distances = np.linalg.norm(pos_hist1 - pos_hist2, axis=1)
    
    return distances


def calculate_horizontal_distance_between_spheres(sphere1, sphere2):
    """
    Calculate the horizontal distance (x-axis) between two spheres over time.
    """
    # Extract the x-coordinate history of both spheres
    x_hist1 = np.array([pos[0] for pos in sphere1.position_history])
    x_hist2 = np.array([pos[0] for pos in sphere2.position_history])
    
    # Calculate the absolute horizontal distance between the two spheres at each time step
    horizontal_distances = np.abs(x_hist1 - x_hist2)
    
    return horizontal_distances

# Configure logging

# Configure logging to write to a file
def setup_logging(log_file='data/simulation.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 'w' to overwrite the log file each time
        format='%(message)s',  # Only log messages without timestamp
        level=logging.INFO
    )

# Function to log the start time and simulation parameters
def log_start_time_and_parameters(params):
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Simulation Start Time: {start_time}")
    logging.info("Simulation Parameters:")
    for key, value in params.items():
        logging.info(f"{key}: {value}")

# Function to log time step and sphere velocities (without timestamp)
def log_sphere_data(spheres, timestep):
    logging.info(f"Time step {timestep}")
    for i, sphere in enumerate(spheres):
        logging.info(f"Particle {i+1} velocity: {sphere.velocity}")

# Function to log the end time
def log_end_time():
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Simulation End Time: {end_time}")
        
#Plots


def plot_spheres(spheres, step, snapshot_interval, plot_directory, ST, ST_ROT):
    """
    Generate and save a scatter plot of the spheres' positions at specified intervals.

    Args:
        spheres (list): List of Sphere objects.
        step (int): Current time step.
        snapshot_interval (int): Interval for saving the plots.
        plot_directory (str): Directory to save the plots.
    """
    # Create the plots directory if it doesn't exist
    os.makedirs(plot_directory, exist_ok=True)

    # Check if the current step is a snapshot step
    if step % snapshot_interval == 0:
        # Setup the plot
        fig, ax = plt.subplots()
        ax.grid(True)

        # Calculate dynamic x and y limits based on sphere positions
        x_positions = [sphere.position[0] for sphere in spheres]
        y_positions = [sphere.position[1] for sphere in spheres]
        x_min, x_max = min(x_positions) - 5, max(x_positions) + 5
        y_min, y_max = min(y_positions) - 10, max(y_positions) + 10

        # Adjust plot limits dynamically
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x in code units')
        ax.set_ylabel('y in code units')
        ax.set_title(f'Crowley Instability: St = {ST}, St_rot = {ST_ROT}')

        # Plot sphere positions
        ax.scatter(x_positions, y_positions, color='r')

        # Save the plot as an image file
        plot_path = os.path.join(plot_directory, f'plot_{step:04d}.png')
        plt.savefig(plot_path)
        plt.close(fig)  # Close the figure to free up memory
        
#Time Series
def write_timeseries_data(step, spheres, perturbations, log_file="data/time_series.dat"):
    """
    Logs the positions, velocities, and perturbations of each sphere to a .dat file.

    Args:
        step (int): The current simulation step.
        spheres (list): List of Sphere objects.
        perturbations (list): List of (x_perturbation, y_perturbation) tuples for each sphere.
        log_file (str): Path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(f"Step {step}:\n")
        f.write("Sphere\tx\ty\tz\tvx\tvy\tvz\tx_pert\ty_pert\n")
        for i, sphere in enumerate(spheres):
            x_pos, y_pos, z_pos = sphere.position[0], sphere.position[1], sphere.position[2]
            x_vel, y_vel, z_vel = sphere.velocity[0], sphere.velocity[1], sphere.velocity[2]
            x_pert, y_pert = perturbations[i]

            # Write the data to the log file
            f.write(f"{i}\t{x_pos:.4f}\t{y_pos:.4f}\t{z_pos:.4f}\t{x_vel:.4f}\t{y_vel:.4f}\t{z_vel:.4f}\t{x_pert:.4f}\t{y_pert:.4f}\n")
        f.write("\n")  # Add a blank line between steps

#Perturbations
'''
def calculate_perturbations(sphere, previous_velocity, dt):
    # Calculate x and y perturbations based on previous velocity and current position
    x_perturbation = np.abs(sphere.position_history[0][0] - sphere.position[0])
    y_perturbation = np.abs(sphere.position_history[0][1] - (sphere.position[1] - previous_velocity[1] * dt))

    # Update perturbation values
    sphere.perturbation = np.array([x_perturbation, y_perturbation, 0])

    # Append perturbations to history
    sphere.perturbation_history.append(sphere.perturbation.copy())
'''


def calculate_perturbations(sphere, previous_velocity, dt):
    """
    Calculate the perturbations for a given sphere based on its initial position
    and the previous velocity.
    """
    # Assuming initial position is stored as the first entry in position_history
    initial_position = sphere.position_history[0]
    
    # Calculate the current position
    current_position = sphere.position

    # Calculate perturbations
    x_perturbation = np.abs(initial_position[0] - current_position[0])
    y_perturbation = np.abs(initial_position[1] - (current_position[1] - previous_velocity[1] * dt))
    
    sphere.perturbation = np.array([x_perturbation, y_perturbation, 0])

    # Return the perturbations as an array
    return sphere.perturbation 


#Save in hdf5 file

import h5py

def save_sphere_history_hdf5(spheres, filename, step):
    with h5py.File(filename, 'a') as f:  # Open file in append mode
        # Create datasets if they don't exist
        if 'positions' not in f:
            f.create_dataset('positions', data=np.array([sphere.position for sphere in spheres]), maxshape=(None, 3, len(spheres)), chunks=True)
        
        if 'velocities' not in f:
            f.create_dataset('velocities', data=np.array([sphere.velocity for sphere in spheres]), maxshape=(None, 3, len(spheres)), chunks=True)
        
        if 'accelerations' not in f:
            f.create_dataset('accelerations', data=np.array([sphere.acceleration for sphere in spheres]), maxshape=(None, 3, len(spheres)), chunks=True)
        
        if 'vel_others' not in f:
            f.create_dataset('vel_others', data=np.array([sphere.vel_others for sphere in spheres]), maxshape=(None, 3, len(spheres)), chunks=True)

        if 'perturbations' not in f:
            f.create_dataset('perturbations', data=np.array([sphere.perturbation for sphere in spheres]), maxshape=(None, len(spheres)), chunks=True)

        # Resize datasets to accommodate new data
        for key in f.keys():
            f[key].resize(f[key].shape[0] + 1, axis=0)

        # Append the new data at the last index
        f['positions'][-1] = np.array([sphere.position for sphere in spheres])
        f['velocities'][-1] = np.array([sphere.velocity for sphere in spheres])
        f['accelerations'][-1] = np.array([sphere.acceleration for sphere in spheres])
        f['vel_others'][-1] = np.array([sphere.vel_others for sphere in spheres])
        f['perturbations'][-1] = np.array([sphere.perturbation for sphere in spheres])

#npz
'''
def save_sphere_history_npz(spheres, filename):
    # Prepare data for saving
    data = {
        'positions': np.array([sphere.position_history for sphere in spheres]),
        'velocities': np.array([sphere.velocity_history for sphere in spheres]),
        'accelerations': np.array([sphere.acceleration_history for sphere in spheres]),
        'vel_others': np.array([sphere.vel_others_history for sphere in spheres]),
        'perturbations': np.array([sphere.perturbation_history for sphere in spheres]),
    }
    
    # Save data to a .npz file
    np.savez_compressed(filename, **data)
''' 
def save_sphere_history_npz(spheres, filename, step, time):
    """
    Save sphere histories (positions, velocities, etc.) to a .npz file.
    
    Parameters:
    - spheres: List of Sphere objects
    - filename: .npz file name
    - step: Current time step
    - time: Current simulation time
    """
    # Initialize arrays to hold histories
    positions = np.array([sphere.position for sphere in spheres])
    velocities = np.array([sphere.velocity for sphere in spheres])
    vel_others = np.array([sphere.vel_others for sphere in spheres])
    accelerations = np.array([sphere.acceleration for sphere in spheres])
    
    # Save histories in a single .npz file
    np.savez_compressed(filename,
                        time=time,
                        positions=positions,
                        velocities=velocities,
                        vel_others=vel_others,
                        accelerations=accelerations
                        )
    


import h5py
import numpy as np

def save_sphere_history_hdf5(spheres, filename, step, time):
    """
    Save sphere histories (positions, velocities, etc.) to an HDF5 file with time data.

    Parameters:
    - spheres: List of Sphere objects
    - filename: HDF5 file name
    - step: Current time step
    - time: Current time corresponding to the simulation step
    """
    with h5py.File(filename, 'a') as f:  # Open file in append mode
        # Initialize or resize the time dataset
        if 'time' not in f:
            f.create_dataset('time', data=np.array([time]), maxshape=(None,), chunks=True)
        else:
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            f['time'][-1] = time

        # Positions
        if 'positions' not in f:
            initial_positions = np.array([sphere.position for sphere in spheres]).reshape(1, len(spheres), 3)
            f.create_dataset('positions', data=initial_positions, maxshape=(None, len(spheres), 3), chunks=True)
        else:
            f['positions'].resize((f['positions'].shape[0] + 1), axis=0)
            f['positions'][-1] = np.array([sphere.position for sphere in spheres])

        # Velocities
        if 'velocities' not in f:
            initial_velocities = np.array([sphere.velocity for sphere in spheres]).reshape(1, len(spheres), 3)
            f.create_dataset('velocities', data=initial_velocities, maxshape=(None, len(spheres), 3), chunks=True)
        else:
            f['velocities'].resize((f['velocities'].shape[0] + 1), axis=0)
            f['velocities'][-1] = np.array([sphere.velocity for sphere in spheres])

        # Accelerations
        if 'accelerations' not in f:
            initial_accelerations = np.array([sphere.acceleration for sphere in spheres]).reshape(1, len(spheres), 3)
            f.create_dataset('accelerations', data=initial_accelerations, maxshape=(None, len(spheres), 3), chunks=True)
        else:
            f['accelerations'].resize((f['accelerations'].shape[0] + 1), axis=0)
            f['accelerations'][-1] = np.array([sphere.acceleration for sphere in spheres])

        # Perturbations
        if 'perturbations' not in f:
            initial_perturbations = np.array([sphere.perturbation for sphere in spheres]).reshape(1, len(spheres), 3)
            f.create_dataset('perturbations', data=initial_perturbations, maxshape=(None, len(spheres), 3), chunks=True)
        else:
            f['perturbations'].resize((f['perturbations'].shape[0] + 1), axis=0)
            f['perturbations'][-1] = np.array([sphere.perturbation for sphere in spheres])

        # Velocities from other spheres (vel_others)
        if 'vel_others' not in f:
            initial_vel_others = np.array([sphere.vel_others for sphere in spheres]).reshape(1, len(spheres), 3)
            f.create_dataset('vel_others', data=initial_vel_others, maxshape=(None, len(spheres), 3), chunks=True)
        else:
            f['vel_others'].resize((f['vel_others'].shape[0] + 1), axis=0)
            f['vel_others'][-1] = np.array([sphere.vel_others for sphere in spheres])
            
#Plotting script
import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_and_plot_hdf5(hdf5_filename, quantity, sphere_index=0,  log_scale=False, save_path=None):
    """
    Read a quantity from an HDF5 file and plot the x, y, z components on separate subplots against time.
    
    Parameters:
    - hdf5_filename: str, name of the HDF5 file
    - quantity: str, name of the dataset ('positions', 'velocities', 'vel_others', 'accelerations', or 'perturbations')
    - sphere_index: int, index of the sphere to plot (default: 0)
    """
    # Open the HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Read time data
        time = f['/time'][:]
        
        # Check if the requested quantity exists
        if quantity not in f:
            print(f"Quantity '{quantity}' not found in the HDF5 file.")
            return
        
        # Read the requested quantity
        data = f[f'/{quantity}'][:]
        
        # Ensure sphere_index is within bounds
        if sphere_index >= data.shape[1]:
            print(f"Sphere index {sphere_index} is out of bounds. Available spheres: 0 to {data.shape[1] - 1}.")
            return

        # Extract data for the selected sphere
        sphere_data = data[:, sphere_index, :]
        
        # Create subplots for each component
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        # Plot x-component
        axes[0].plot(time, sphere_data[:, 0], label=f'{quantity} (x-component)', color='b')
        axes[0].set_ylabel(f'{quantity.capitalize()} (x)')
        axes[0].legend()
        axes[0].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Plot y-component
        axes[1].plot(time, sphere_data[:, 1], label=f'{quantity} (y-component)', color='g')
        axes[1].set_ylabel(f'{quantity.capitalize()} (y)')
        axes[1].legend()
        axes[1].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Plot z-component
        axes[2].plot(time, sphere_data[:, 2], label=f'{quantity} (z-component)', color='r')
        axes[2].set_ylabel(f'{quantity.capitalize()} (z)')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Customize layout and show the plot
        plt.suptitle(f'{quantity.capitalize()} of Sphere {sphere_index} vs Time')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Determine the save path
        if save_path is None:
            save_path = f'{quantity}_{sphere_index}.png'
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")      
        plt.show()
        
#all spheres


def read_and_plot_all_spheres(hdf5_filename, quantity, log_scale=False,save_path=None):
    """
    Read a quantity from an HDF5 file and plot the x, y, z components of all spheres on separate subplots against time.
    
    Parameters:
    - hdf5_filename: str, name of the HDF5 file
    - quantity: str, name of the dataset ('positions', 'velocities', 'vel_others', 'accelerations', or 'perturbations')
    """
    # Open the HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Read time data
        time = f['/time'][:]
        
        # Check if the requested quantity exists
        if quantity not in f:
            print(f"Quantity '{quantity}' not found in the HDF5 file.")
            return
        
        # Read the requested quantity
        data = f[f'/{quantity}'][:]

        # Get the number of spheres
        num_spheres = data.shape[1]
        
        # Create subplots for each component (x, y, z)
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        # Plot x-component for all spheres
        for i in range(num_spheres):
            axes[0].plot(time, data[:, i, 0], label=f'Sphere {i+1}', linestyle='-')
        axes[0].set_ylabel(f'{quantity.capitalize()} (x)')
        #axes[0].legend(loc='upper right')
        axes[0].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Plot y-component for all spheres
        for i in range(num_spheres):
            axes[1].plot(time, data[:, i, 1], label=f'Sphere {i+1}', linestyle='-')
        axes[1].set_ylabel(f'{quantity.capitalize()} (y)')
        #axes[1].legend(loc='upper right')
        axes[1].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Plot z-component for all spheres
        for i in range(num_spheres):
            axes[2].plot(time, data[:, i, 2], label=f'Sphere {i+1}', linestyle='-')
        axes[2].set_ylabel(f'{quantity.capitalize()} (z)')
        axes[2].set_xlabel('Time')
        #axes[2].legend(loc='upper right')
        axes[2].grid(True)
        if log_scale:
            axes[0].set_yscale('log')

        # Customize layout and show the plot
        plt.suptitle(f'{quantity.capitalize()} of All Spheres vs Time')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Determine the save path
        if save_path is None:
            save_path = f'{quantity}_allspheres.png'
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
        plt.show()

#Perturbations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_perturbations_all_spheres(hdf5_filename, N=20):
    """
    Plot log(perturbations) vs time for all spheres, perform linear fits on the initial N points,
    and print the slope for both horizontal and vertical perturbations. The legend will only show the average slope.
    
    Parameters:
    - hdf5_filename: str, name of the HDF5 file
    - N: int, number of initial points to use for the linear fit (default: 20)
    """
    # Open the HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Read time data
        time = f['/time'][:]

        # Read perturbations data
        perturbations = f['/perturbations'][:]
        
        # Get the number of spheres
        num_spheres = perturbations.shape[1]

        slopes_x = []
        slopes_y = []

        # Plot horizontal (x) and vertical (y) perturbations
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Horizontal perturbations (x-perturbations)
        for i in range(num_spheres):
            perturbation_x = perturbations[:, i, 0]
            log_pert_x = np.log(perturbation_x)

            # Perform a linear fit on the first N points
            slope_x, intercept_x, _, _, _ = linregress(time[:N], log_pert_x[:N])
            slopes_x.append(slope_x)

            # Plot perturbations without showing individual slopes in the legend
            axes[0].plot(time, log_pert_x)

        # Compute and print the average slope for x perturbations
        avg_slope_x = np.mean(slopes_x)
        print(f"Slopes for horizontal (x) perturbations: {slopes_x}")
        print(f"Average slope for horizontal perturbations: {avg_slope_x}")

        axes[0].set_ylabel('log(Perturbation X)')
        axes[0].legend([f'Average Slope: {avg_slope_x:.2f}'], loc='upper right')
        axes[0].grid(True)

        # Vertical perturbations (y-perturbations)
        for i in range(num_spheres):
            perturbation_y = perturbations[:, i, 1]
            log_pert_y = np.log(perturbation_y)

            # Perform a linear fit on the first N points
            slope_y, intercept_y, _, _, _ = linregress(time[:N], log_pert_y[:N])
            slopes_y.append(slope_y)

            # Plot perturbations without showing individual slopes in the legend
            axes[1].plot(time, log_pert_y)

        # Compute and print the average slope for y perturbations
        avg_slope_y = np.mean(slopes_y)
        print(f"Slopes for vertical (y) perturbations: {slopes_y}")
        print(f"Average slope for vertical perturbations: {avg_slope_y}")

        axes[1].set_ylabel('log(Perturbation Y)')
        axes[1].set_xlabel('Time')
        axes[1].legend([f'Average Slope: {avg_slope_y:.2f}'], loc='upper right')
        axes[1].grid(True)

        # Customize layout and show the plot
        plt.suptitle('Log(Perturbations) vs Time for All Spheres')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'perturbationfit_allspheres')
        plt.show()

def plot_perturbations_single_sphere(hdf5_filename, sphere_idx, N=20):
    """
    Plot log(perturbations) vs time for a given sphere, perform linear fits on the initial N points,
    and print the slope for both horizontal and vertical perturbations. The legend will show the average slope.
    
    Parameters:
    - hdf5_filename: str, name of the HDF5 file
    - sphere_idx: int, index of the sphere to plot
    - N: int, number of initial points to use for the linear fit (default: 20)
    """
    # Open the HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Read time data
        time = f['/time'][:]

        # Read perturbations data
        perturbations = f['/perturbations'][:, sphere_idx]

        # Plot horizontal (x) and vertical (y) perturbations
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Horizontal perturbation (x-perturbation)
        perturbation_x = perturbations[:, 0]
        log_pert_x = np.log(perturbation_x)

        # Perform a linear fit on the first N points
        slope_x, intercept_x, _, _, _ = linregress(time[:N], log_pert_x[:N])

        # Print the slope
        print(f"Slope for horizontal (x) perturbation for Sphere {sphere_idx+1}: {slope_x}")

        # Plot perturbations and the fitted line
        axes[0].plot(time, log_pert_x)
        axes[0].plot(time[:N], slope_x * time[:N] + intercept_x, '--')

        axes[0].set_ylabel('log(Perturbation X)')
        axes[0].legend([f'Slope: {slope_x:.2f}'], loc='upper right')
        axes[0].grid(True)

        # Vertical perturbation (y-perturbation)
        perturbation_y = perturbations[:, 1]
        log_pert_y = np.log(perturbation_y)

        # Perform a linear fit on the first N points
        slope_y, intercept_y, _, _, _ = linregress(time[:N], log_pert_y[:N])

        # Print the slope
        print(f"Slope for vertical (y) perturbation for Sphere {sphere_idx+1}: {slope_y}")

        # Plot perturbations and the fitted line
        axes[1].plot(time, log_pert_y)
        axes[1].plot(time[:N], slope_y * time[:N] + intercept_y, '--')

        axes[1].set_ylabel('log(Perturbation Y)')
        axes[1].set_xlabel('Time')
        axes[1].legend([f'Slope: {slope_y:.2f}'], loc='upper right')
        axes[1].grid(True)

        # Customize layout and show the plot
        plt.suptitle(f'Log(Perturbations) vs Time for Sphere {sphere_idx+1}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'perturbationfit_sphere{sphere_idx+1}')
        plt.show()

        





    





