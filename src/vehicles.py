"""
Vehicle models: hexacopter geometry and parameters.
"""

import numpy as np
from src.config import (
    HEX_RADIUS, HEX_MASS, HEX_INERTIA, HEX_MOTOR_ANGLES,
    HEX_MOTOR_DIRECTIONS, MOTOR_MAX_THRUST, ROTOR_DRAG_COEFFICIENT
)


class Hexacopter:
    """
    Hexacopter vehicle model with 6 motors arranged in a circle.
    
    Motor layout:
    - Motor 0: 0° (forward)
    - Motor 1: 60°
    - Motor 2: 120°
    - Motor 3: 180° (backward)
    - Motor 4: 240°
    - Motor 5: 300°
    
    Spin directions alternate: [1, -1, 1, -1, 1, -1] for yaw control.
    """
    
    def __init__(self):
        """Initialize hexacopter with default parameters."""
        self.mass = HEX_MASS
        self.inertia = HEX_INERTIA.copy()
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        self.num_motors = 6
        self.motor_angles = HEX_MOTOR_ANGLES.copy()
        self.motor_directions = HEX_MOTOR_DIRECTIONS.copy()
        self.motor_max_thrust = MOTOR_MAX_THRUST
        self.rotor_drag_coeff = ROTOR_DRAG_COEFFICIENT
        
        # Compute motor positions in body frame
        self.motor_positions = np.zeros((self.num_motors, 3))
        for i in range(self.num_motors):
            angle = self.motor_angles[i]
            self.motor_positions[i, 0] = HEX_RADIUS * np.cos(angle)
            self.motor_positions[i, 1] = HEX_RADIUS * np.sin(angle)
            self.motor_positions[i, 2] = 0.0  # All motors in same plane
        
        # Compute allocation matrix (mixer)
        self._compute_allocation_matrix()
    
    def _compute_allocation_matrix(self):
        """
        Compute the allocation matrix that maps desired forces/torques to motor thrusts.
        
        For hexacopter:
        - Total thrust: sum of all motor thrusts
        - Roll torque: differential thrust between left and right motors
        - Pitch torque: differential thrust between front and back motors
        - Yaw torque: differential reaction torque from motor spin directions
        
        The allocation matrix M satisfies: [Fz, tau_x, tau_y, tau_z] = M * [T1, T2, ..., T6]
        We need the inverse: [T1, ..., T6] = M_inv * [Fz, tau_x, tau_y, tau_z]
        """
        # Build allocation matrix
        # Each column corresponds to one motor
        M = np.zeros((4, self.num_motors))
        
        for i in range(self.num_motors):
            # Position of motor in body frame
            pos = self.motor_positions[i]
            direction = self.motor_directions[i]
            
            # Total thrust (all motors contribute equally)
            M[0, i] = 1.0
            
            # Roll torque (tau_x): moment about x-axis
            # tau = r x F, where r is position and F is thrust in +z direction
            # For roll: we want torque about x-axis, which comes from y-component of position
            M[1, i] = -pos[1]  # Negative because positive y gives negative roll
            
            # Pitch torque (tau_y): moment about y-axis
            # For pitch: torque comes from x-component of position
            M[2, i] = pos[0]  # Positive x gives positive pitch
            
            # Yaw torque (tau_z): reaction torque from motor spin
            # Motors spinning CCW (+1) produce positive yaw torque
            # Motors spinning CW (-1) produce negative yaw torque
            M[3, i] = direction * self.rotor_drag_coeff
        
        # Use pseudoinverse to get motor commands from desired forces/torques
        # This handles the over-actuated system (6 motors, 4 DOF)
        self.allocation_matrix = M
        self.allocation_matrix_inv = np.linalg.pinv(M)
    
    def compute_motor_thrusts(self, desired_thrust, desired_torques):
        """
        Compute motor thrusts from desired total thrust and torques.
        
        Args:
            desired_thrust: Desired total thrust in body +z direction (N)
            desired_torques: Desired torques [tau_x, tau_y, tau_z] (N*m)
        
        Returns:
            Motor thrusts [T1, T2, ..., T6] (N)
        """
        # Desired forces/torques vector
        desired = np.array([
            desired_thrust,
            desired_torques[0],
            desired_torques[1],
            desired_torques[2]
        ])
        
        # Compute motor thrusts using pseudoinverse
        motor_thrusts = self.allocation_matrix_inv @ desired
        
        # Saturate motor thrusts
        motor_thrusts = np.clip(motor_thrusts, 0.0, self.motor_max_thrust)
        
        return motor_thrusts
    
    def get_motor_positions(self):
        """Get motor positions in body frame."""
        return self.motor_positions.copy()
    
    def get_motor_directions(self):
        """Get motor spin directions."""
        return self.motor_directions.copy()

