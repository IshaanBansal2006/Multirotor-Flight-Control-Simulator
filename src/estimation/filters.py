"""
State estimation filters: complementary filter and EKF.

The estimator fuses sensor measurements to produce an estimate of the
true state, which is then used by the controller.
"""

import numpy as np
from src.config import COMP_FILTER_ALPHA, GRAVITY
from src.utils.math3d import (
    quaternion_from_euler, euler_from_quaternion, quaternion_multiply,
    quaternion_normalize, rotation_matrix_from_quaternion
)


class ComplementaryFilter:
    """
    Complementary filter for attitude estimation.
    
    Fuses gyroscope (high frequency, good short-term) with accelerometer
    (low frequency, good long-term) to estimate attitude.
    
    This is a simple but effective estimator for attitude.
    """
    
    def __init__(self):
        """Initialize complementary filter."""
        self.alpha = COMP_FILTER_ALPHA  # Trust gyro more (0-1)
        self.attitude_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    
    def update(self, gyro, accel, dt):
        """
        Update attitude estimate.
        
        Args:
            gyro: Angular rates [p, q, r] from gyroscope
            accel: Accelerometer reading [ax, ay, az] in body frame
            dt: Timestep (seconds)
        
        Returns:
            Estimated attitude quaternion [w, x, y, z]
        """
        # Predict attitude from gyro (integrate)
        # dq/dt = 0.5 * q * [0, p, q, r]
        from src.utils.math3d import quaternion_derivative
        dq = quaternion_derivative(self.attitude_quat, gyro)
        predicted_quat = self.attitude_quat + dq * dt
        predicted_quat = quaternion_normalize(predicted_quat)
        
        # Correct using accelerometer (measure gravity direction)
        # At hover, accelerometer should read [0, 0, g] in body frame
        # This gives us roll and pitch (but not yaw)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Only use if significant acceleration
            # Normalize accelerometer reading
            accel_normalized = accel / accel_norm
            
            # Expected gravity direction in body frame (from predicted attitude)
            R_pred = rotation_matrix_from_quaternion(predicted_quat)
            gravity_world = np.array([0.0, 0.0, -1.0])  # Downward
            expected_gravity_body = R_pred.T @ gravity_world
            
            # Compute error
            error = np.cross(expected_gravity_body, accel_normalized)
            
            # Convert error to quaternion correction (small angle approximation)
            # Error is the cross product, which gives us the axis and magnitude
            correction_angle = np.linalg.norm(error)
            if correction_angle > 0.01:
                correction_axis = error / correction_angle
                # Small angle correction: create quaternion from axis-angle
                # For small angles: q ≈ [1, axis_x*angle/2, axis_y*angle/2, axis_z*angle/2]
                correction_angle_scaled = correction_angle * (1 - self.alpha) * 0.1
                half_angle = correction_angle_scaled * 0.5
                correction_quat = np.array([
                    1.0,  # w (cos(half_angle) ≈ 1 for small angles)
                    correction_axis[0] * half_angle,
                    correction_axis[1] * half_angle,
                    correction_axis[2] * half_angle
                ])
                correction_quat = quaternion_normalize(correction_quat)
                # Apply correction
                predicted_quat = quaternion_multiply(predicted_quat, correction_quat)
        
        # Fuse: weighted combination
        # For simplicity, we trust the gyro prediction mostly (alpha close to 1)
        # The correction from accelerometer is already applied above
        self.attitude_quat = predicted_quat
        self.attitude_quat = quaternion_normalize(self.attitude_quat)
        
        return self.attitude_quat


class SimpleEstimator:
    """
    Simple state estimator combining complementary filter for attitude
    and alpha-beta filter for position/velocity.
    
    This is a lightweight estimator suitable for demonstration.
    A full EKF would be more accurate but more complex.
    """
    
    def __init__(self):
        """Initialize estimator."""
        self.attitude_filter = ComplementaryFilter()
        
        # Position/velocity estimates
        self.position = np.array([0.0, 0.0, 5.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Alpha-beta filter parameters
        self.alpha_pos = 0.7  # Position gain
        self.beta_pos = 0.3   # Velocity gain
        
        # Last GPS update
        self.last_gps_time = 0.0
    
    def update(self, sensors, dt):
        """
        Update state estimate from sensor measurements.
        
        Args:
            sensors: Dict with gyro, accel, baro, gps_available, gps_position, gps_velocity
            dt: Timestep (seconds)
        
        Returns:
            Estimated state dict
        """
        # Update attitude using complementary filter
        attitude_quat = self.attitude_filter.update(sensors['gyro'], sensors['accel'], dt)
        
        # Altitude and vertical velocity: alpha-beta filter on the barometer.
        # Predict with the current velocity, then correct both states from the
        # same residual. Deriving the velocity from the already-corrected
        # position instead divides the correction by dt and turns the
        # barometer noise into a velocity estimate 1/dt times too large.
        if dt > 1e-6:
            predicted_z = self.position[2] + self.velocity[2] * dt
            residual_z = sensors['baro'] - predicted_z
            self.position[2] = predicted_z + self.alpha_pos * residual_z
            self.velocity[2] += (self.beta_pos / dt) * residual_z
        
        # Use GPS for horizontal position/velocity when available
        if sensors['gps_available'] and sensors['gps_position'] is not None:
            # Alpha-beta filter update
            pos_error = sensors['gps_position'] - self.position
            self.position += self.alpha_pos * pos_error
            
            if sensors['gps_velocity'] is not None:
                vel_error = sensors['gps_velocity'] - self.velocity
                self.velocity += self.beta_pos * vel_error
        
        # Convert quaternion to Euler for rates estimate
        # For rates, we can use the gyro directly (it's already in body frame)
        estimated_rates = sensors['gyro'].copy()
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'attitude': attitude_quat.copy(),
            'rates': estimated_rates.copy()
        }
    
    def reset(self, initial_position=None, initial_attitude=None):
        """Reset estimator to initial state."""
        if initial_position is not None:
            self.position = np.array(initial_position)
        if initial_attitude is not None:
            self.attitude_filter.attitude_quat = quaternion_normalize(np.array(initial_attitude))
        self.velocity = np.array([0.0, 0.0, 0.0])

