"""
State estimation filters: complementary filter and EKF.

The estimator fuses sensor measurements to produce an estimate of the
true state, which is then used by the controller.
"""

import numpy as np
from src.config import (
    ATTITUDE_ACCEL_GAIN, EST_TAU_XY, EST_TAU_Z, GPS_VELOCITY_GAIN, GRAVITY
)
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
        self.accel_gain = ATTITUDE_ACCEL_GAIN
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
        # Only trust the accelerometer as a gravity reference when it reads
        # close to 1 g. Under sustained acceleration it is measuring something
        # else and would tilt the estimate.
        if abs(accel_norm - GRAVITY) < 0.2 * GRAVITY:
            measured_gravity_body = accel / accel_norm

            # Where the current estimate says gravity should point, in body axes
            R_pred = rotation_matrix_from_quaternion(predicted_quat)
            expected_gravity_body = R_pred.T @ np.array([0.0, 0.0, -1.0])

            # Body-frame rotation that carries the expected direction onto the
            # measured one. cross(measured, expected) is the correct order:
            # cross(expected, measured) turns this correction into positive
            # feedback, which is why the estimate used to wander by a few
            # degrees and would flip outright at any useful gain.
            error = np.cross(measured_gravity_body, expected_gravity_body)

            # Applied as a rate, so the gain means the same thing at any loop
            # frequency: ATTITUDE_ACCEL_GAIN radians of correction per second
            # per radian of error.
            half_angle = 0.5 * self.accel_gain * dt
            correction_quat = quaternion_normalize(np.array([
                1.0,
                error[0] * half_angle,
                error[1] * half_angle,
                error[2] * half_angle,
            ]))
            predicted_quat = quaternion_multiply(predicted_quat, correction_quat)
        
        self.attitude_quat = predicted_quat
        self.attitude_quat = quaternion_normalize(self.attitude_quat)
        
        return self.attitude_quat


class AlphaBetaTracker:
    """Critically damped alpha-beta tracker for one axis of a constant-velocity target.

    Tuned by a time constant rather than by per-update blend fractions, because
    the gains have to be derived from the interval they are applied over. With
    r = exp(-dt / tau):

        alpha = 1 - r^2          position correction
        beta  = (1 - r)^2        velocity correction, applied as beta / dt

    The same tau therefore behaves the same way whether the filter is corrected
    at 200 Hz from the barometer or at 10 Hz from GPS. Writing alpha and beta as
    constants instead makes the velocity gain beta/dt scale with the loop rate,
    which is how a 0.5 m barometer sigma turned into a 33 m/s velocity estimate.

    The accelerometer in this simulator reports gravity only (see
    estimation/sensors.py), so there is no translational acceleration to
    integrate and a constant-velocity model is the most this sensor suite
    supports.
    """

    def __init__(self, tau, position=0.0, velocity=0.0):
        self.tau = float(tau)
        self.position = float(position)
        self.velocity = float(velocity)
        self.time_since_correction = 0.0

    def gains(self, dt):
        """Alpha and beta for a correction interval of ``dt`` seconds."""
        decay = np.exp(-dt / self.tau)
        alpha = 1.0 - decay * decay
        beta = (1.0 - decay) ** 2
        return alpha, beta

    def predict(self, dt):
        """Dead reckon forward. Called every control step, measurement or not."""
        self.position += self.velocity * dt
        self.time_since_correction += dt

    def correct(self, measurement):
        """Fold in one position measurement, using the interval since the last."""
        dt = self.time_since_correction
        if dt <= 1e-9:
            return
        alpha, beta = self.gains(dt)
        residual = measurement - self.position
        self.position += alpha * residual
        self.velocity += (beta / dt) * residual
        self.time_since_correction = 0.0

    def nudge_velocity(self, measurement, gain):
        """Blend in a direct velocity measurement (GPS reports one)."""
        self.velocity += gain * (measurement - self.velocity)

    def reset(self, position=0.0, velocity=0.0):
        self.position = float(position)
        self.velocity = float(velocity)
        self.time_since_correction = 0.0


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

        # One tracker per axis. Altitude is corrected from the barometer at the
        # control rate; x and y are dead reckoned between GPS fixes and
        # corrected when one arrives.
        self.trackers = [
            AlphaBetaTracker(EST_TAU_XY),
            AlphaBetaTracker(EST_TAU_XY),
            AlphaBetaTracker(EST_TAU_Z, position=5.0),
        ]
    
    @property
    def position(self):
        return np.array([t.position for t in self.trackers])
    
    @property
    def velocity(self):
        return np.array([t.velocity for t in self.trackers])
    
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
        
        # Dead reckon every axis forward, then correct the ones that have a
        # measurement this step.
        for tracker in self.trackers:
            tracker.predict(dt)

        # Altitude from the barometer, every step. GPS also reports z with the
        # same sigma but at a twentieth of the rate, so folding it into the
        # altitude estimate would only add noise; it is deliberately unused.
        self.trackers[2].correct(sensors['baro'])

        if sensors['gps_available'] and sensors['gps_position'] is not None:
            for axis in (0, 1):
                self.trackers[axis].correct(sensors['gps_position'][axis])
            if sensors['gps_velocity'] is not None:
                for axis in (0, 1):
                    self.trackers[axis].nudge_velocity(
                        sensors['gps_velocity'][axis], GPS_VELOCITY_GAIN)
        
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
            for tracker, value in zip(self.trackers, np.asarray(initial_position, dtype=float)):
                tracker.reset(position=value)
        else:
            for tracker in self.trackers:
                tracker.reset(position=tracker.position)
        if initial_attitude is not None:
            self.attitude_filter.attitude_quat = quaternion_normalize(np.array(initial_attitude))

