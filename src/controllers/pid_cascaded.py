"""
Cascaded PID controller for multirotor flight control.

Structure:
1. Outer loop: Position controller -> desired attitude/thrust
2. Middle loop: Attitude controller -> desired angular rates
3. Inner loop: Rate controller -> desired torques

This cascaded structure is common in multirotor control because:
- It separates fast dynamics (attitude/rates) from slow dynamics (position)
- Allows tuning each loop independently
- Provides good disturbance rejection
"""

import numpy as np
from src.config import (
    PID_POSITION_KP, PID_POSITION_KI, PID_POSITION_KD,
    PID_ATTITUDE_KP, PID_ATTITUDE_KI, PID_ATTITUDE_KD,
    PID_RATE_KP, PID_RATE_KI, PID_RATE_KD,
    DERIVATIVE_FILTER_TAU, GRAVITY
)
from src.utils.math3d import euler_from_quaternion


class CascadedPIDController:
    """
    Cascaded PID controller for 6-DOF multirotor control.
    
    Control structure:
    - Position (x, y, z) -> desired roll/pitch/thrust
    - Attitude (roll, pitch, yaw) -> desired angular rates (p, q, r)
    - Rates (p, q, r) -> desired torques (tau_x, tau_y, tau_z)
    """
    
    def __init__(self, mixer, enable_integral=True, enable_anti_windup=True, integral_limit=10.0):
        """
        Initialize controller.
        
        Args:
            mixer: Control mixer object
            enable_integral: Whether to use integral action
            enable_anti_windup: Whether to enable anti-windup (clamp integrator)
            integral_limit: Maximum integrator value (for anti-windup)
        """
        self.mixer = mixer
        self.enable_integral = enable_integral
        self.enable_anti_windup = enable_anti_windup
        self.integral_limit = integral_limit
        
        # Position controller gains (can be overridden by experiment)
        self.kp_pos = PID_POSITION_KP.copy()
        self.ki_pos = PID_POSITION_KI.copy()
        self.kd_pos = PID_POSITION_KD.copy()
        
        # Attitude controller gains
        self.kp_att = PID_ATTITUDE_KP.copy()
        self.ki_att = PID_ATTITUDE_KI.copy()
        self.kd_att = PID_ATTITUDE_KD.copy()
        
        # Rate controller gains
        # Higher proportional gain = faster response but can cause overshoot
        # Higher derivative gain = more damping, reduces oscillation
        # Lower gains = more robust but slower response
        self.kp_rate = PID_RATE_KP.copy()
        self.ki_rate = PID_RATE_KI.copy()
        self.kd_rate = PID_RATE_KD.copy()
        
        # Integrators
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.integral_rate = np.zeros(3)
        
        # Previous errors for derivative
        self.prev_error_pos = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        self.prev_error_rate = np.zeros(3)
        
        # Filtered derivatives
        self.filtered_derivative_pos = np.zeros(3)
        self.filtered_derivative_att = np.zeros(3)
        self.filtered_derivative_rate = np.zeros(3)
        
        # Target setpoints
        self.target_position = np.array([0.0, 0.0, 5.0])
        self.target_yaw = 0.0
    
    def update(self, estimated_state, dt):
        """
        Compute control outputs.
        
        Args:
            estimated_state: Dict with position, velocity, attitude, rates
            dt: Timestep (seconds)
        
        Returns:
            Dict with 'torques' and 'thrust'
        """
        pos = estimated_state['position']
        vel = estimated_state['velocity']
        att_quat = estimated_state['attitude']
        rates = estimated_state['rates']
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(att_quat)
        attitude = np.array([roll, pitch, yaw])
        
        # ====================================================================
        # Outer Loop: Position Controller
        # ====================================================================
        # Position error
        error_pos = self.target_position - pos
        
        # Update integral (with optional anti-windup)
        if self.enable_integral:
            self.integral_pos += error_pos * dt
            # Anti-windup: clamp integral to prevent windup when actuators saturate
            # Without anti-windup, integrator accumulates error even when control
            # is saturated, causing overshoot and poor recovery
            if self.enable_anti_windup:
                self.integral_pos = np.clip(
                    self.integral_pos,
                    -self.integral_limit,
                    self.integral_limit
                )
        else:
            self.integral_pos = np.zeros(3)
        
        # Derivative (velocity error)
        error_vel = -vel  # Desired velocity is zero for hover
        
        # Low-pass filter on derivative
        alpha = dt / (DERIVATIVE_FILTER_TAU + dt)
        self.filtered_derivative_pos = (
            alpha * error_vel +
            (1 - alpha) * self.filtered_derivative_pos
        )
        
        # PID output: desired acceleration
        desired_accel = (
            self.kp_pos * error_pos +
            self.ki_pos * self.integral_pos +
            self.kd_pos * self.filtered_derivative_pos
        )
        
        # Convert desired acceleration to desired attitude and thrust
        # For small angles: desired roll = -desired_accel_y / g
        #                   desired pitch = desired_accel_x / g
        # Total thrust compensates for gravity + desired z acceleration
        desired_roll = -desired_accel[1] / GRAVITY
        desired_pitch = desired_accel[0] / GRAVITY
        desired_thrust = self.mass * (GRAVITY + desired_accel[2])
        
        # Clamp desired roll/pitch to reasonable limits (±30 degrees)
        max_angle = np.radians(30)
        desired_roll = np.clip(desired_roll, -max_angle, max_angle)
        desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
        
        # ====================================================================
        # Middle Loop: Attitude Controller
        # ====================================================================
        desired_attitude = np.array([desired_roll, desired_pitch, self.target_yaw])
        error_att = desired_attitude - attitude
        
        # Wrap yaw error to [-pi, pi]
        error_att[2] = np.arctan2(np.sin(error_att[2]), np.cos(error_att[2]))
        
        # Update integral
        if self.enable_integral:
            self.integral_att += error_att * dt
            if self.enable_anti_windup:
                self.integral_att = np.clip(
                    self.integral_att,
                    -self.integral_limit,
                    self.integral_limit
                )
        else:
            self.integral_att = np.zeros(3)
        
        # Derivative: use actual angular rates (negative feedback for damping)
        # The derivative term should oppose the current rates to provide damping
        # We use the negative of current rates as the derivative signal
        rate_derivative = -rates
        
        # Filter derivative
        alpha = dt / (DERIVATIVE_FILTER_TAU + dt)
        self.filtered_derivative_att = (
            alpha * rate_derivative +
            (1 - alpha) * self.filtered_derivative_att
        )
        
        # PID output: desired angular rates
        desired_rates = (
            self.kp_att * error_att +
            self.ki_att * self.integral_att +
            self.kd_att * self.filtered_derivative_att
        )
        
        # ====================================================================
        # Inner Loop: Rate Controller
        # ====================================================================
        error_rate = desired_rates - rates
        
        # Update integral
        if self.enable_integral:
            self.integral_rate += error_rate * dt
            if self.enable_anti_windup:
                self.integral_rate = np.clip(
                    self.integral_rate,
                    -self.integral_limit,
                    self.integral_limit
                )
        else:
            self.integral_rate = np.zeros(3)
        
        # Derivative: rate of change of rates (approximate as zero for now)
        # In practice, you'd need rate of change of rates, which requires
        # either differentiation or a model
        error_rate_derivative = np.zeros(3)
        
        # Filter derivative
        alpha = dt / (DERIVATIVE_FILTER_TAU + dt)
        self.filtered_derivative_rate = (
            alpha * error_rate_derivative +
            (1 - alpha) * self.filtered_derivative_rate
        )
        
        # PID output: desired torques
        desired_torques = (
            self.kp_rate * error_rate +
            self.ki_rate * self.integral_rate +
            self.kd_rate * self.filtered_derivative_rate
        )
        
        # Store errors for next iteration
        self.prev_error_pos = error_pos
        self.prev_error_att = error_att
        self.prev_error_rate = error_rate
        
        return {
            'torques': desired_torques,
            'thrust': desired_thrust
        }
    
    def set_target(self, position, yaw=None):
        """
        Set target setpoint.
        
        Args:
            position: Target position [x, y, z]
            yaw: Target yaw angle (radians), or None to keep current
        """
        self.target_position = np.array(position)
        if yaw is not None:
            self.target_yaw = yaw
    
    def reset_integrators(self):
        """Reset all integrators (useful when switching modes)."""
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.integral_rate = np.zeros(3)
    
    @property
    def mass(self):
        """Get vehicle mass (needed for thrust calculation)."""
        # This should be passed in or accessed from vehicle
        # For now, use default
        from src.config import HEX_MASS
        return HEX_MASS

