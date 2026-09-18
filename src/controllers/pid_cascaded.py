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
    HEX_MASS,
    PID_POSITION_KP, PID_POSITION_KI, PID_POSITION_KD,
    PID_ATTITUDE_KP, PID_ATTITUDE_KI, PID_ATTITUDE_KD,
    PID_RATE_KP, PID_RATE_KI, PID_RATE_KD,
    DERIVATIVE_FILTER_TAU, GRAVITY
)
from src.controllers.attitude_setpoint import thrust_vector_setpoint
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

        # Saturation flags from the previous step, used for conditional
        # anti-windup: an axis whose command is already clipped must not keep
        # integrating, or the integrator winds up while the vehicle cannot act
        # on it and then overshoots on the way back.
        self.tilt_saturated = False
        self.thrust_saturated = False
    
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
            gate = np.ones(3)
            if self.enable_anti_windup:
                # Conditional integration: hold the axes that were saturated on
                # the previous step. Clamping alone still lets the integrator
                # fill up while the vehicle cannot follow the command.
                if self.tilt_saturated:
                    gate[0] = gate[1] = 0.0
                if self.thrust_saturated:
                    gate[2] = 0.0
            self.integral_pos += error_pos * dt * gate
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
        
        # Convert the desired acceleration into a thrust-vector setpoint. This
        # is the exact inverse rather than the small-angle one, so the
        # collective carries its own 1/cos(tilt) term and the horizontal
        # command is limited against the vertical component before the angles
        # are computed. See controllers/attitude_setpoint.py.
        setpoint = thrust_vector_setpoint(desired_accel, yaw, self.mass)
        desired_roll = setpoint.roll
        desired_pitch = setpoint.pitch
        desired_thrust = setpoint.thrust
        self.tilt_saturated = setpoint.tilt_saturated
        self.thrust_saturated = desired_thrust >= self.max_total_thrust
        
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
        self.tilt_saturated = False
        self.thrust_saturated = False
    
    @property
    def mass(self):
        """Vehicle mass, from the mixer's vehicle."""
        vehicle = getattr(self.mixer, "vehicle", None)
        return vehicle.mass if vehicle is not None else HEX_MASS

    @property
    def max_total_thrust(self):
        """Largest collective the mixer can actually deliver."""
        vehicle = getattr(self.mixer, "vehicle", None)
        if vehicle is None:
            return float("inf")
        return vehicle.num_motors * vehicle.motor_max_thrust

