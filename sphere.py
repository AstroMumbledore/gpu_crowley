import numpy as np 
from utils import calculate_perturbations

#SPHERE CLASS
class Sphere:
    def __init__(self, stokes_no, position, velocity):
        self.stokes_no = stokes_no
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        self.vel_others = np.zeros(3, dtype=float)
        self.perturbation = np.zeros(3, dtype=float)
        
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.acceleration_history = [self.velocity.copy()]
        self.vel_others_history = [self.acceleration.copy()]
        self.perturbation_history = [self.perturbation.copy()]

    def update_vel_others(self, new_vel_others):
        self.vel_others = new_vel_others

    def update(self, new_stokes_no, new_position, new_velocity,new_vel_others, new_acceleration, dt):
        self.stokes_no = new_stokes_no
        self.position = new_position
        self.velocity = new_velocity
        self.acceleration = new_acceleration       
        self.vel_others = new_vel_others
        
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.acceleration_history.append(self.acceleration.copy())
        self.vel_others_history.append(self.vel_others.copy())
        
        # Update perturbation
        self.perturbation = calculate_perturbations(self, self.velocity_history[-2], dt)
        self.perturbation_history.append(self.perturbation.copy())
        
        