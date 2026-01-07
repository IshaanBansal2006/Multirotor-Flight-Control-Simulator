"""
Rigid body dynamics for 6-DOF multirotor simulation.

Implements:
- Rigid body dynamics with quaternion attitude representation
- Motor model with first-order lag
- Aerodynamic drag
- Wind disturbances
- RK4 integration
"""

import numpy as np
from src.config import (
    GRAVITY, DRAG_COEFFICIENT, SIM_DT, MOTOR_TIME_CONSTANT
)
from src.utils.math3d import (
    quaternion_derivative, quaternion_normalize, quaternion_rotate_vector,
    rotation_matrix_from_quaternion
)


class MultirotorDynamics:
    """
    6-DOF rigid body dynamics for a multirotor.
    
    State vector:
    - position: [x, y, z] in world frame (ENU)
    - velocity: [vx, vy, vz] in world frame
    - attitude: quaternion [w, x, y, z]
    - angular_rates: [p, q, r] in body frame
    
    Coordinate frame: ENU (East-North-Up)
    - x: forward (East)
    - y: right (North)
    - z: up (Up)
    - Gravity: -z direction
    """
    
    def __init__(self, vehicle, initial_state=None):
        """
        Initialize dynamics model.
        
        Args:
            vehicle: Vehicle object (e.g., Hexacopter)
            initial_state: Dict with initial state, or None for hover
        """
        self.vehicle = vehicle
        self.mass = vehicle.mass
        self.inertia = vehicle.inertia
        self.inertia_inv = vehicle.inertia_inv
        
        # Motor states (for first-order lag model)
        self.motor_commanded_thrusts = np.zeros(vehicle.num_motors)
        self.motor_actual_thrusts = np.zeros(vehicle.num_motors)
        
        # Initialize state
        if initial_state is None:
            # Default: hover at origin
            self.state = {
                'position': np.array([0.0, 0.0, 5.0]),  # Start at 5m altitude
                'velocity': np.array([0.0, 0.0, 0.0]),
                'attitude': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
                'rates': np.array([0.0, 0.0, 0.0])
            }
        else:
            self.state = initial_state.copy()
        
        # Normalize initial quaternion
        self.state['attitude'] = quaternion_normalize(self.state['attitude'])
    
    def set_motor_commands(self, motor_thrusts):
        """
        Set motor thrust commands (will be filtered by motor dynamics).
        
        Args:
            motor_thrusts: Desired motor thrusts [T1, T2, ..., T6] (N)
        """
        # Saturate commands
        self.motor_commanded_thrusts = np.clip(
            motor_thrusts, 0.0, self.vehicle.motor_max_thrust
        )
    
    def step(self, dt, wind_force=np.array([0.0, 0.0, 0.0])):
        """
        Advance simulation by one timestep using RK4 integration.
        
        Args:
            dt: Timestep (seconds)
            wind_force: Wind disturbance force in world frame (N)
        
        Returns:
            Updated state dict
        """
        # Update motor dynamics (first-order lag)
        alpha = dt / (MOTOR_TIME_CONSTANT + dt)
        self.motor_actual_thrusts = (
            alpha * self.motor_commanded_thrusts +
            (1 - alpha) * self.motor_actual_thrusts
        )
        
        # RK4 integration
        k1 = self._compute_derivatives(wind_force)
        k2_state = self._add_state(self.state, k1, dt/2)
        k2 = self._compute_derivatives(wind_force, k2_state)
        k3_state = self._add_state(self.state, k2, dt/2)
        k3 = self._compute_derivatives(wind_force, k3_state)
        k4_state = self._add_state(self.state, k3, dt)
        k4 = self._compute_derivatives(wind_force, k4_state)
        
        # Combine derivatives
        derivative = {
            'position': (k1['position'] + 2*k2['position'] + 2*k3['position'] + k4['position']) / 6,
            'velocity': (k1['velocity'] + 2*k2['velocity'] + 2*k3['velocity'] + k4['velocity']) / 6,
            'attitude': (k1['attitude'] + 2*k2['attitude'] + 2*k3['attitude'] + k4['attitude']) / 6,
            'rates': (k1['rates'] + 2*k2['rates'] + 2*k3['rates'] + k4['rates']) / 6
        }
        
        # Update state
        self.state = self._add_state(self.state, derivative, dt)
        
        # Normalize quaternion
        self.state['attitude'] = quaternion_normalize(self.state['attitude'])
        
        # Ground collision detection and bouncing
        GROUND_LEVEL = 0.0
        BOUNCE_COEFFICIENT = 0.6  # Energy retained after bounce (0.6 = 60% retained)
        GROUND_DAMPING = 0.8  # Damping for angular rates on impact
        
        if self.state['position'][2] < GROUND_LEVEL:
            # Drone hit the ground
            self.state['position'][2] = GROUND_LEVEL
            
            # Bounce: reverse vertical velocity with damping
            if self.state['velocity'][2] < 0:  # Only bounce if moving downward
                self.state['velocity'][2] = -self.state['velocity'][2] * BOUNCE_COEFFICIENT
                
                # Damp horizontal velocities on impact (friction)
                self.state['velocity'][0] *= 0.7
                self.state['velocity'][1] *= 0.7
                
                # Damp angular rates on impact
                self.state['rates'] *= GROUND_DAMPING
        
        return self.state
    
    def _add_state(self, state, derivative, dt):
        """Helper to add derivative * dt to state."""
        new_state = {
            'position': state['position'] + derivative['position'] * dt,
            'velocity': state['velocity'] + derivative['velocity'] * dt,
            'attitude': state['attitude'] + derivative['attitude'] * dt,
            'rates': state['rates'] + derivative['rates'] * dt
        }
        # Normalize quaternion
        new_state['attitude'] = quaternion_normalize(new_state['attitude'])
        return new_state
    
    def _compute_derivatives(self, wind_force, state=None):
        """
        Compute state derivatives.
        
        Args:
            wind_force: Wind disturbance force in world frame (N)
            state: State to compute derivatives for, or None for current state
        
        Returns:
            Dict of derivatives
        """
        if state is None:
            state = self.state
        
        # Position derivative: velocity
        d_position = state['velocity'].copy()
        
        # Velocity derivative: acceleration from forces
        # Forces: gravity, thrust, drag, wind
        total_thrust_body = np.array([0.0, 0.0, np.sum(self.motor_actual_thrusts)])
        
        # Rotate thrust to world frame
        R = rotation_matrix_from_quaternion(state['attitude'])
        total_thrust_world = R @ total_thrust_body
        
        # Gravity (acts in -z direction in ENU frame)
        gravity_force = np.array([0.0, 0.0, -GRAVITY * self.mass])
        
        # Drag (proportional to velocity, acts in opposite direction)
        drag_force = -DRAG_COEFFICIENT * state['velocity']
        
        # Total force
        total_force = gravity_force + total_thrust_world + drag_force + wind_force
        
        # Acceleration
        d_velocity = total_force / self.mass
        
        # Attitude derivative: quaternion derivative from angular rates
        d_attitude = quaternion_derivative(state['attitude'], state['rates'])
        
        # Angular rate derivative: from torques
        # Compute torques from motor thrusts
        torques = self._compute_motor_torques()
        
        # Euler's equation: I * d(omega)/dt = tau - omega x (I * omega)
        I_omega = self.inertia @ state['rates']
        omega_cross_I_omega = np.cross(state['rates'], I_omega)
        d_rates = self.inertia_inv @ (torques - omega_cross_I_omega)
        
        return {
            'position': d_position,
            'velocity': d_velocity,
            'attitude': d_attitude,
            'rates': d_rates
        }
    
    def _compute_motor_torques(self):
        """
        Compute torques from motor thrusts.
        
        Returns:
            Torques [tau_x, tau_y, tau_z] in body frame (N*m)
        """
        torques = np.zeros(3)
        
        motor_positions = self.vehicle.get_motor_positions()
        
        for i in range(self.vehicle.num_motors):
            # Thrust vector in body frame (acts in +z direction)
            thrust_vec = np.array([0.0, 0.0, self.motor_actual_thrusts[i]])
            
            # Torque from thrust: tau = r x F
            r = motor_positions[i]
            torque_from_thrust = np.cross(r, thrust_vec)
            torques[:2] += torque_from_thrust[:2]  # Roll and pitch
            
            # Yaw torque from reaction torque (motor spin direction)
            direction = self.vehicle.get_motor_directions()[i]
            torques[2] += direction * self.vehicle.rotor_drag_coeff * self.motor_actual_thrusts[i]
        
        return torques
    
    def get_state(self):
        """Get current state."""
        return self.state.copy()
    
    def get_motor_thrusts(self):
        """Get current actual motor thrusts."""
        return self.motor_actual_thrusts.copy()

