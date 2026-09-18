"""
Sensor simulation: IMU, barometer, GPS.

Sensors add noise, bias, and delays to true state measurements.
"""

import numpy as np
from src.config import (
    GYRO_NOISE_STD, GYRO_BIAS_DRIFT, ACCEL_NOISE_STD, ACCEL_BIAS,
    BARO_NOISE_STD, BARO_BIAS,
    GPS_UPDATE_RATE, GPS_POSITION_NOISE_STD, GPS_VELOCITY_NOISE_STD,
    IMU_DELAY_STEPS, BARO_DELAY_STEPS, GPS_DELAY_STEPS, SIM_DT, GRAVITY
)
from src.utils.math3d import rotation_matrix_from_quaternion


class SensorSimulator:
    """
    Simulates sensors with noise, bias, and delays.
    """
    
    def __init__(self, enable_noise=True, seed=None):
        """
        Initialize sensor simulator.
        
        Args:
            enable_noise: Whether to add noise to measurements
            seed: Seed for the sensor noise generator. None draws from the OS
                entropy source, so runs are not reproducible; pass an int to
                make a run repeatable.
        """
        self.enable_noise = enable_noise
        self.rng = np.random.default_rng(seed)
        
        # Gyro bias (drifts over time)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        
        # GPS update counter
        self.gps_counter = 0.0
        self.gps_update_period = 1.0 / GPS_UPDATE_RATE
        
        # Delay buffers
        self.imu_buffer = []
        self.baro_buffer = []
        self.gps_buffer = []
    
    def update(self, true_state, dt):
        """
        Generate sensor measurements from true state.
        
        Args:
            true_state: Dict with position, velocity, attitude, rates
            dt: Timestep (seconds)
        
        Returns:
            Dict with sensor measurements
        """
        # Update gyro bias (drift). Bias drift is part of the noise model, so
        # it must not accumulate when noise is disabled.
        if self.enable_noise:
            self.gyro_bias += self.rng.normal(0, GYRO_BIAS_DRIFT * dt, 3)
        
        # IMU: Gyroscope
        gyro_true = true_state['rates'].copy()
        if self.enable_noise:
            gyro_noise = self.rng.normal(0, GYRO_NOISE_STD, 3)
            gyro = gyro_true + self.gyro_bias + gyro_noise
        else:
            gyro = gyro_true + self.gyro_bias
        
        # IMU: Accelerometer (measures specific force, not acceleration)
        # In body frame: accel = R^T * (a_world - g_world)
        # At hover: a_world ≈ [0, 0, -g], so accel_body ≈ [0, 0, g]
        R = rotation_matrix_from_quaternion(true_state['attitude'])
        gravity_world = np.array([0.0, 0.0, -GRAVITY])
        gravity_body = R.T @ gravity_world
        
        # Add acceleration from motion (simplified)
        accel_body = gravity_body.copy()
        if self.enable_noise:
            accel_noise = self.rng.normal(0, ACCEL_NOISE_STD, 3)
            accel_body += accel_noise
            accel_body += ACCEL_BIAS
        else:
            accel_body += ACCEL_BIAS
        
        # Barometer: Altitude (z position)
        baro_true = true_state['position'][2]
        if self.enable_noise:
            baro = baro_true + self.rng.normal(BARO_BIAS, BARO_NOISE_STD)
        else:
            baro = baro_true + BARO_BIAS
        
        # GPS: Position and velocity (low update rate)
        self.gps_counter += dt
        gps_available = False
        gps_position = None
        gps_velocity = None
        
        if self.gps_counter >= self.gps_update_period:
            self.gps_counter = 0.0
            gps_available = True
            
            if self.enable_noise:
                gps_position = true_state['position'] + self.rng.normal(0, GPS_POSITION_NOISE_STD, 3)
                gps_velocity = true_state['velocity'] + self.rng.normal(0, GPS_VELOCITY_NOISE_STD, 3)
            else:
                gps_position = true_state['position'].copy()
                gps_velocity = true_state['velocity'].copy()
        
        # Apply delays (simple implementation: store in buffer)
        # For simplicity, we'll just return delayed measurements
        # In a real implementation, you'd maintain proper delay buffers
        
        return {
            'gyro': gyro,
            'accel': accel_body,
            'baro': baro,
            'gps_available': gps_available,
            'gps_position': gps_position,
            'gps_velocity': gps_velocity
        }
    
    def set_noise_enabled(self, enabled):
        """Enable or disable sensor noise."""
        self.enable_noise = enabled

