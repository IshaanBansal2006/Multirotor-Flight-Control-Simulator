"""
Predefined experiment scenarios for flight control demonstration.

Each experiment isolates and demonstrates specific control concepts:
- Controller architecture choices
- Gain tuning effects
- Disturbance rejection
- Actuator saturation and anti-windup
- Sensor noise impact
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from src.config import (
    PID_ATTITUDE_KD, PID_ATTITUDE_KI, PID_ATTITUDE_KP,
    PID_POSITION_KI, PID_POSITION_KP,
    PID_RATE_KD, PID_RATE_KI, PID_RATE_KP,
    POSITION_DAMPING_RATIO,
)


def default_pid_gains(**overrides) -> Dict[str, np.ndarray]:
    """The gain set from config.py, with the position damping derived from Kp.

    Every scenario used to carry its own copy of all nine gain arrays, so a
    change in config.py reached none of them. Deriving kd_pos here also means a
    scenario cannot end up with a proportional gain and a damping gain that
    disagree: pass kp_pos and kd_pos follows for the same damping ratio.
    """
    kp_pos = np.asarray(overrides.pop("kp_pos", PID_POSITION_KP), dtype=float)
    gains = {
        "kp_pos": kp_pos.copy(),
        "ki_pos": PID_POSITION_KI.copy(),
        "kd_pos": 2.0 * POSITION_DAMPING_RATIO * np.sqrt(kp_pos),
        "kp_att": PID_ATTITUDE_KP.copy(),
        "ki_att": PID_ATTITUDE_KI.copy(),
        "kd_att": PID_ATTITUDE_KD.copy(),
        "kp_rate": PID_RATE_KP.copy(),
        "ki_rate": PID_RATE_KI.copy(),
        "kd_rate": PID_RATE_KD.copy(),
    }
    gains.update({key: np.asarray(value, dtype=float).copy()
                  for key, value in overrides.items()})
    return gains


@dataclass
class Experiment:
    """Experiment configuration."""
    name: str
    description: str
    controller_type: str  # "pid" or "lqr"
    
    # Controller gains
    control_gains: Dict[str, np.ndarray]
    
    # Controller settings
    enable_integral: bool = True
    enable_anti_windup: bool = True
    integral_limit: float = 10.0
    
    # Environment
    enable_wind: bool = False
    wind_force_vector: Optional[np.ndarray] = None
    wind_gust_time: Optional[float] = None  # Time to inject gust (seconds)
    wind_gust_duration: float = 0.5
    wind_gust_magnitude: float = 5.0
    
    # Sensors
    enable_sensor_noise: bool = False
    
    # Initial conditions
    initial_position: np.ndarray = None
    initial_velocity: np.ndarray = None
    initial_attitude: np.ndarray = None  # Euler angles [roll, pitch, yaw]
    
    # Target setpoint
    target_position: np.ndarray = None
    target_yaw: float = 0.0
    
    # Simulation duration
    duration: float = 10.0
    
    # Expected behavior description
    expected_behavior: str = ""
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if self.initial_position is None:
            self.initial_position = np.array([0.0, 0.0, 5.0])
        if self.initial_velocity is None:
            self.initial_velocity = np.array([0.0, 0.0, 0.0])
        if self.initial_attitude is None:
            self.initial_attitude = np.array([0.0, 0.0, 0.0])
        if self.target_position is None:
            self.target_position = np.array([0.0, 0.0, 5.0])


def get_experiment(name: str) -> Experiment:
    """
    Get experiment by name.
    
    Args:
        name: Experiment name
        
    Returns:
        Experiment configuration
    """
    experiments = {
        "nominal_hover": create_nominal_hover(),
        "hover_with_sensor_noise": create_hover_with_sensor_noise(),
        "lateral_step_xy": create_lateral_step_xy(),
        "wind_gust_left": create_wind_gust_left(),
        "wind_gust_right": create_wind_gust_right(),
        "wind_gust_front": create_wind_gust_front(),
        "wind_gust_back": create_wind_gust_back(),
        "wind_gust_down": create_wind_gust_down(),
    }
    
    if name not in experiments:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(experiments.keys())}")
    
    return experiments[name]


def list_experiments():
    """List all available experiments."""
    return [
        "nominal_hover",
        "hover_with_sensor_noise",
        "lateral_step_xy",
        "wind_gust_left",
        "wind_gust_right",
        "wind_gust_front",
        "wind_gust_back",
        "wind_gust_down",
    ]


def create_nominal_hover() -> Experiment:
    """
    Nominal Hover Experiment
    
    Demonstrates: Stable, well-tuned controller with conservative gains.
    Expected: Smooth, stable response with minimal overshoot.
    """
    return Experiment(
        name="nominal_hover",
        description="Stable hover with conservative, well-tuned gains",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=False,
        enable_sensor_noise=False,
        duration=10.0,
        expected_behavior="Smooth convergence to hover with minimal overshoot. "
                         "Demonstrates proper gain tuning for stability."
    )


def create_hover_with_sensor_noise() -> Experiment:
    """
    Hover with Sensor Noise Experiment
    
    Demonstrates: Effect of noisy measurements on a well-tuned controller.
    Expected: Small jitter in attitude/position but overall stable hover.
    """
    return Experiment(
        name="hover_with_sensor_noise",
        description="Stable hover with moderate sensor noise enabled",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=False,
        enable_sensor_noise=True,
        duration=10.0,
        expected_behavior="Drone maintains hover near the target but with visible small oscillations "
                         "due to noisy measurements. Demonstrates robustness of the controller to "
                         "sensor noise and the trade-off between filtering and responsiveness."
    )


def create_lateral_step_xy() -> Experiment:
    """
    Lateral Position Step Experiment
    
    Demonstrates: Lateral (X/Y) position control response to a step input.
    Expected: Drone translates to a new X/Y position while holding altitude.
    """
    return Experiment(
        name="lateral_step_xy",
        description="Lateral position step from origin to (3 m, 2 m) at constant altitude",
        controller_type="pid",
        # A quarter more x/y bandwidth than the tuned baseline, expressed as a
        # multiplier so it tracks any future retune instead of pinning a number.
        control_gains=default_pid_gains(
            kp_pos=PID_POSITION_KP * np.array([1.25, 1.25, 1.0])
        ),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=False,
        enable_sensor_noise=False,
        initial_position=np.array([0.0, 0.0, 5.0]),
        target_position=np.array([3.0, 2.0, 5.0]),
        duration=12.0,
        expected_behavior="From a hover at the origin, the drone performs a lateral step to (3 m, 2 m) "
                         "while holding altitude at 5 m. Position plots in X/Y clearly show step response, "
                         "including rise time, overshoot, and settling behavior."
    )


def create_wind_gust_left() -> Experiment:
    """Wind gust from the left side."""
    return Experiment(
        name="wind_gust_left",
        description="Wind gust from left",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=True,
        wind_force_vector=np.array([-10.0, 0.0, 0.0]),  # Left (negative X)
        wind_gust_time=3.0,
        wind_gust_duration=1.0,
        wind_gust_magnitude=10.0,
        enable_sensor_noise=False,
        duration=15.0,
        expected_behavior="Drone pushed to the right, then recovers to hover position."
    )


def create_wind_gust_right() -> Experiment:
    """Wind gust from the right side."""
    return Experiment(
        name="wind_gust_right",
        description="Wind gust from right",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=True,
        wind_force_vector=np.array([10.0, 0.0, 0.0]),  # Right (positive X)
        wind_gust_time=3.0,
        wind_gust_duration=1.0,
        wind_gust_magnitude=10.0,
        enable_sensor_noise=False,
        duration=15.0,
        expected_behavior="Drone pushed to the left, then recovers to hover position."
    )


def create_wind_gust_front() -> Experiment:
    """Wind gust from the front."""
    return Experiment(
        name="wind_gust_front",
        description="Wind gust from front",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=True,
        wind_force_vector=np.array([0.0, 10.0, 0.0]),  # Front (positive Y)
        wind_gust_time=3.0,
        wind_gust_duration=1.0,
        wind_gust_magnitude=10.0,
        enable_sensor_noise=False,
        duration=15.0,
        expected_behavior="Drone pushed backward, then recovers to hover position."
    )


def create_wind_gust_back() -> Experiment:
    """Wind gust from the back."""
    return Experiment(
        name="wind_gust_back",
        description="Wind gust from back",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=True,
        wind_force_vector=np.array([0.0, -10.0, 0.0]),  # Back (negative Y)
        wind_gust_time=3.0,
        wind_gust_duration=1.0,
        wind_gust_magnitude=10.0,
        enable_sensor_noise=False,
        duration=15.0,
        expected_behavior="Drone pushed forward, then recovers to hover position."
    )


def create_wind_gust_down() -> Experiment:
    """Wind gust from above (downward)."""
    return Experiment(
        name="wind_gust_down",
        description="Wind gust downward",
        controller_type="pid",
        control_gains=default_pid_gains(),
        enable_integral=True,
        enable_anti_windup=True,
        integral_limit=10.0,
        enable_wind=True,
        wind_force_vector=np.array([0.0, 0.0, -15.0]),  # Downward (negative Z)
        wind_gust_time=3.0,
        wind_gust_duration=1.0,
        wind_gust_magnitude=15.0,
        enable_sensor_noise=False,
        duration=15.0,
        expected_behavior="Drone pushed downward, then recovers to hover altitude."
    )

