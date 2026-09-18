"""
Configuration constants for the multirotor simulator.

Coordinate Frame: ENU (East-North-Up)
- x: forward (East)
- y: right (North) 
- z: up (Up)
- Gravity: -z direction (g = 9.81 m/s^2)
- Body frame: +z is upward (thrust acts in +z direction)
"""

import numpy as np

# ============================================================================
# Physical Constants
# ============================================================================
GRAVITY = 9.81  # m/s^2 (acts in -z direction in ENU frame)
AIR_DENSITY = 1.225  # kg/m^3

# ============================================================================
# Simulation Parameters
# ============================================================================
SIM_DT = 0.005  # Simulation timestep (seconds) - 200 Hz
UI_UPDATE_RATE = 30  # UI update rate (Hz)
MAX_SIM_TIME = 300.0  # Maximum simulation time (seconds)

# ============================================================================
# Hexacopter Parameters
# ============================================================================
# Hexacopter geometry: motors arranged in a circle
HEX_RADIUS = 0.3  # Distance from center to motor (meters)
HEX_MASS = 2.0  # Total mass (kg)
HEX_INERTIA = np.array([
    [0.05, 0.0, 0.0],   # Ixx, Ixy, Ixz
    [0.0, 0.05, 0.0],   # Iyx, Iyy, Iyz
    [0.0, 0.0, 0.1]     # Izx, Izy, Izz
])  # kg*m^2

# Motor parameters
MOTOR_MAX_THRUST = 15.0  # Maximum thrust per motor (N)
MOTOR_TIME_CONSTANT = 0.05  # Motor response time constant (seconds)
MOTOR_MIN_THRUST = 0.0  # Minimum thrust (cannot go negative)

# Motor positions (in body frame, meters)
# Hexacopter: 6 motors arranged at 60-degree intervals
HEX_MOTOR_ANGLES = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180  # radians
# Motor spin directions: +1 for CCW, -1 for CW (alternating for yaw control)
HEX_MOTOR_DIRECTIONS = np.array([1, -1, 1, -1, 1, -1])

# ============================================================================
# Aerodynamics
# ============================================================================
DRAG_COEFFICIENT = 0.1  # Linear drag coefficient (N/(m/s))
ROTOR_DRAG_COEFFICIENT = 0.001  # Rotor drag coefficient for yaw torque

# ============================================================================
# Wind Disturbances
# ============================================================================
WIND_GUST_DURATION = 0.5  # Duration of gust (seconds)
WIND_GUST_MAGNITUDE = 5.0  # Maximum gust force (N)
CONSTANT_WIND_ENABLED = False
CONSTANT_WIND_VELOCITY = np.array([0.0, 0.0, 0.0])  # m/s

# ============================================================================
# Controller Default Gains
# ============================================================================
# Cascaded PID Controller
# Position gains: Higher Kp for z (altitude), moderate for x/y
# Chosen by a closed-loop sweep over kp against the estimator time constants,
# scored on all four demo scenarios plus both gain variants of the noisy one
# (docs/decisions/002). The shipped kp of 2.0 on x/y asked for more bandwidth
# than 10 Hz GPS with a 0.5 m sigma can support: every candidate at or above
# 1.2 lost control in at least one scenario, every candidate at 0.7 lost none.
PID_POSITION_KP = np.array([0.7, 0.7, 5.0])  # x, y, z (higher z for altitude)
PID_POSITION_KI = np.array([0.1, 0.1, 0.4])  # Moderate integral for steady-state accuracy

# Position damping is DERIVED, not hand-picked. The outer loop closes as
#     a = Kp * e - Kd * v   ->   omega_n = sqrt(Kp),  zeta = Kd / (2 sqrt(Kp))
# so a target damping ratio fixes Kd for any Kp. The shipped Kd of
# [0.5, 0.5, 2.0] was zeta 0.18 on x/y, which is why any lateral command
# overshot far enough to hit the tilt limit and never recover.
POSITION_DAMPING_RATIO = 0.9  # slightly under critical: fast, ~0.2% overshoot
PID_POSITION_KD = 2.0 * POSITION_DAMPING_RATIO * np.sqrt(PID_POSITION_KP)

# Attitude gains: Stronger for roll/pitch, moderate for yaw
PID_ATTITUDE_KP = np.array([8.0, 8.0, 5.0])  # roll, pitch, yaw
PID_ATTITUDE_KI = np.array([0.5, 0.5, 0.3])  # Moderate integral
PID_ATTITUDE_KD = np.array([2.0, 2.0, 1.0])  # Damping

# Rate gains: Inner loop needs to be fast and well-damped
PID_RATE_KP = np.array([0.25, 0.25, 0.2])  # p, q, r
PID_RATE_KI = np.array([0.03, 0.03, 0.02])  # Small integral for rate loop
PID_RATE_KD = np.array([0.1, 0.1, 0.06])  # Rate damping

# Anti-windup
INTEGRAL_SATURATION = 10.0
DERIVATIVE_FILTER_TAU = 0.01  # Low-pass filter time constant for derivative

# Attitude setpoint limits (src/controllers/attitude_setpoint.py)
MAX_TILT_ANGLE = np.radians(30.0)  # largest tilt the position loop may command
MIN_VERTICAL_ACCEL = 0.5 * GRAVITY  # never point the thrust vector below the horizon

# ============================================================================
# LQR Controller Parameters
# ============================================================================
# LQR is used for inner loop (attitude + rates)
# State: [roll, pitch, yaw, p, q, r]
# Input: [tau_x, tau_y, tau_z]
LQR_Q = np.diag([10.0, 10.0, 5.0, 1.0, 1.0, 1.0])  # State penalty
LQR_R = np.diag([0.1, 0.1, 0.1])  # Control penalty

# ============================================================================
# Sensor Parameters
# ============================================================================
# IMU
GYRO_NOISE_STD = 0.01  # rad/s
GYRO_BIAS_DRIFT = 0.0001  # rad/s per second
ACCEL_NOISE_STD = 0.1  # m/s^2
ACCEL_BIAS = np.array([0.0, 0.0, 0.0])  # m/s^2

# Barometer
BARO_NOISE_STD = 0.5  # meters
BARO_BIAS = 0.0  # meters

# GPS
GPS_UPDATE_RATE = 10.0  # Hz
GPS_POSITION_NOISE_STD = 0.5  # meters
GPS_VELOCITY_NOISE_STD = 0.2  # m/s

# Sensor delays
IMU_DELAY_STEPS = 0  # Number of simulation steps
BARO_DELAY_STEPS = 0
GPS_DELAY_STEPS = 0

# ============================================================================
# Estimator Parameters
# ============================================================================
# Complementary filter. The accelerometer correction is applied as a RATE, so
# the gain means the same thing at any loop frequency: radians of correction per
# second per radian of tilt error. Measured flat-optimal between 0.4 and 8;
# 2.0 gives 0.20 deg mean attitude error under full sensor noise against
# 0.32 deg for pure gyro integration over the same 12 s window, and unlike pure
# integration it bounds the drift over a long run.
ATTITUDE_ACCEL_GAIN = 2.0

# Alpha-beta position/velocity trackers (src/estimation/filters.py).
# These are TIME CONSTANTS, not per-update blend fractions: the filter derives
# alpha and beta from the interval each correction covers, so the same value
# behaves the same way whether it is corrected at 200 Hz from the barometer or
# at 10 Hz from GPS. Chosen by sweeping tau against the measured velocity-
# estimate noise and step lag - see docs/decisions/002.
EST_TAU_Z = 0.10   # altitude and vertical velocity (barometer, 200 Hz)
EST_TAU_XY = 0.70  # horizontal position and velocity (GPS, 10 Hz)
GPS_VELOCITY_GAIN = 0.35  # blend weight for the GPS velocity report

# EKF (if implemented)
EKF_PROCESS_NOISE = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001])
EKF_MEASUREMENT_NOISE = np.diag([0.1, 0.1, 0.5, 0.01, 0.01, 0.01])

# ============================================================================
# Default Setpoints
# ============================================================================
DEFAULT_TARGET_POSITION = np.array([0.0, 0.0, 5.0])  # x, y, z (meters)
DEFAULT_TARGET_YAW = 0.0  # radians

# ============================================================================
# UI Parameters
# ============================================================================
PLOT_HISTORY_LENGTH = 1000  # Number of points to show in plots
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000

# ============================================================================
# Logging
# ============================================================================
LOG_DIR = "logs"
ENABLE_LOGGING = True

# ============================================================================
# Experiment/Scenario Configuration Overrides
# ============================================================================
# These can be overridden by experiments
ENABLE_INTEGRAL = True
ENABLE_ANTI_WINDUP = True
INTEGRAL_LIMIT = 10.0

ENABLE_WIND = False
WIND_FORCE_VECTOR = np.array([0.0, 0.0, 0.0])  # Constant wind force (N)
WIND_GUST_TIME = None  # Time to inject gust (seconds), None = no gust
WIND_GUST_DURATION = 0.5  # Duration of gust (seconds)
WIND_GUST_MAGNITUDE = 5.0  # Maximum gust force (N)

ENABLE_SENSOR_NOISE = False

