"""
Main simulator class that coordinates dynamics, controllers, estimation, and logging.
"""

import numpy as np
from src.vehicles import Hexacopter
from src.dynamics import MultirotorDynamics
from src.controllers.pid_cascaded import CascadedPIDController
from src.controllers.lqr_inner import LQRInnerController
from src.controllers.mixers import HexacopterMixer
from src.estimation.sensors import SensorSimulator
from src.estimation.filters import SimpleEstimator
from src.utils.logging import SimulationLogger
from src.utils.metrics import (
    compute_settling_time, compute_overshoot, compute_max_attitude_error
)
from src.config import (
    SIM_DT, WIND_GUST_MAGNITUDE, WIND_GUST_DURATION,
    DEFAULT_TARGET_POSITION, DEFAULT_TARGET_YAW
)
from src.utils.math3d import euler_from_quaternion


class Simulator:
    """
    Main simulator that coordinates all components.
    """
    
    def __init__(self, enable_logging=True):
        """Initialize simulator."""
        # Create vehicle
        self.vehicle = Hexacopter()
        
        # Create dynamics
        self.dynamics = MultirotorDynamics(self.vehicle)
        
        # Create mixer
        self.mixer = HexacopterMixer(self.vehicle)
        
        # Create controllers
        self.pid_controller = CascadedPIDController(self.mixer, enable_integral=True)
        self.lqr_controller = LQRInnerController(self.mixer, enable_integral=True)
        self.current_controller = self.pid_controller
        self.controller_mode = "pid"
        
        # Create sensors and estimator
        self.sensors = SensorSimulator(enable_noise=True)
        self.estimator = SimpleEstimator()
        
        # Initialize estimator with true initial state
        initial_state = self.dynamics.get_state()
        self.estimator.reset(
            initial_position=initial_state['position'],
            initial_attitude=initial_state['attitude']
        )
        
        # Create logger
        self.logger = SimulationLogger(enable=enable_logging)
        
        # Simulation state
        self.time = 0.0
        self.running = True
        
        # Wind disturbance
        self.wind_force = np.array([0.0, 0.0, 0.0])
        self.wind_gust_active = False
        self.wind_gust_start_time = 0.0
        self.wind_gust_direction = np.array([0.0, 0.0, 0.0])
        self.wind_gust_duration = WIND_GUST_DURATION
        self.wind_gust_magnitude = WIND_GUST_MAGNITUDE
        
        # Set initial target
        self.current_controller.set_target(DEFAULT_TARGET_POSITION, DEFAULT_TARGET_YAW)
    
    def step(self):
        """Advance simulation by one timestep."""
        if not self.running:
            return
        
        # Update wind gust
        self._update_wind()
        
        # Get true state
        true_state = self.dynamics.get_state()
        
        # Simulate sensors
        sensor_measurements = self.sensors.update(true_state, SIM_DT)
        
        # Update estimator
        estimated_state = self.estimator.update(sensor_measurements, SIM_DT)
        
        # Compute control
        controls = self.current_controller.update(estimated_state, SIM_DT)
        
        # Convert controls to motor commands
        motor_thrusts = self.mixer.compute_motor_thrusts(
            controls['thrust'],
            controls['torques']
        )
        
        # Apply motor commands
        self.dynamics.set_motor_commands(motor_thrusts)
        
        # Step dynamics
        self.dynamics.step(SIM_DT, self.wind_force)
        
        # Log data (get final states after step)
        true_state_final = self.dynamics.get_state()
        sensor_measurements_final = self.sensors.update(true_state_final, SIM_DT)
        estimated_state_final = self.estimator.update(sensor_measurements_final, SIM_DT)
        
        # Convert attitude to Euler for logging
        roll, pitch, yaw = euler_from_quaternion(true_state_final['attitude'])
        true_state_log = {
            'position': true_state_final['position'],
            'velocity': true_state_final['velocity'],
            'attitude': np.array([roll, pitch, yaw]),
            'rates': true_state_final['rates']
        }
        
        roll_est, pitch_est, yaw_est = euler_from_quaternion(estimated_state_final['attitude'])
        estimated_state_log = {
            'position': estimated_state_final['position'],
            'velocity': estimated_state_final['velocity'],
            'attitude': np.array([roll_est, pitch_est, yaw_est]),
            'rates': estimated_state_final['rates']
        }
        
        self.logger.log(
            self.time,
            true_state_log,
            estimated_state_log,
            controls,
            motor_thrusts,
            self.wind_force
        )
        
        # Advance time
        self.time += SIM_DT
    
    def _update_wind(self):
        """Update wind disturbance."""
        if self.wind_gust_active:
            elapsed = self.time - self.wind_gust_start_time
            if elapsed < self.wind_gust_duration:
                # Active gust
                self.wind_force = self.wind_gust_magnitude * self.wind_gust_direction
            else:
                # Gust finished
                self.wind_gust_active = False
                self.wind_force = np.array([0.0, 0.0, 0.0])
        else:
            self.wind_force = np.array([0.0, 0.0, 0.0])
    
    def inject_wind_gust(self, direction, magnitude=None, duration=None):
        """
        Inject a wind gust.
        
        Args:
            direction: Direction vector (will be normalized)
            magnitude: Gust magnitude (N), uses default if None
            duration: Gust duration (s), uses default if None
        """
        direction = np.array(direction)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        
        self.wind_gust_direction = direction
        if magnitude is not None:
            self.wind_gust_magnitude = magnitude
        if duration is not None:
            self.wind_gust_duration = duration
        self.wind_gust_active = True
        self.wind_gust_start_time = self.time
    
    def set_target(self, position, yaw=None):
        """Set target setpoint."""
        self.current_controller.set_target(position, yaw)
    
    def set_controller_mode(self, mode):
        """Switch controller mode."""
        if mode == "pid":
            self.current_controller = self.pid_controller
            self.controller_mode = "pid"
        elif mode == "lqr":
            self.current_controller = self.lqr_controller
            self.controller_mode = "lqr"
        
        # Reset integrators when switching
        self.current_controller.reset_integrators()
    
    def update_controller_gain(self, gain_name, value):
        """Update a controller gain."""
        if not hasattr(self.current_controller, gain_name):
            return
        
        current_gain = getattr(self.current_controller, gain_name)
        
        # For vector gains, update all components
        if isinstance(current_gain, np.ndarray):
            current_gain[:] = value
        else:
            setattr(self.current_controller, gain_name, value)
    
    def set_sensor_noise(self, enabled):
        """Enable or disable sensor noise."""
        self.sensors.set_noise_enabled(enabled)
    
    def set_integral_enabled(self, enabled):
        """Enable or disable integral action."""
        self.current_controller.enable_integral = enabled
        if not enabled:
            self.current_controller.reset_integrators()
    
    def get_state(self):
        """Get current state for UI display."""
        true_state = self.dynamics.get_state()
        
        # Convert quaternion to Euler angles for display
        roll, pitch, yaw = euler_from_quaternion(true_state['attitude'])
        
        return {
            'position': true_state['position'],
            'velocity': true_state['velocity'],
            'attitude': np.array([roll, pitch, yaw]),  # Euler angles in radians
            'attitude_quat': true_state['attitude'],  # Keep quaternion for 3D viz
            'rates': true_state['rates'],
            'motor_thrusts': self.dynamics.get_motor_thrusts(),
            'wind_force': self.wind_force
        }
    
    def get_time(self):
        """Get current simulation time."""
        return self.time
    
    def export_report(self):
        """Export simulation report with plots and summary."""
        # Save CSV and plots
        self.logger.save_csv()
        self.logger.save_plots()
        
        # Compute metrics
        if len(self.logger.data['time']) > 0:
            time_array = np.array(self.logger.data['time'])
            pos_array = np.array(self.logger.data['position'])
            att_array = np.array(self.logger.data['attitude'])
            
            # Final errors
            final_pos = pos_array[-1]
            target_pos = self.current_controller.target_position
            final_pos_error = np.linalg.norm(final_pos - target_pos)
            final_alt_error = abs(final_pos[2] - target_pos[2])
            
            # Overshoot
            max_overshoot = compute_overshoot(time_array, pos_array[:, 2], target_pos[2])
            
            # Settling time
            settling_time = compute_settling_time(time_array, pos_array[:, 2], target_pos[2])
            if settling_time is None:
                settling_time = time_array[-1]
            
            # Max attitude error
            max_att_error = compute_max_attitude_error(att_array)
            
            # Max wind response
            wind_array = np.array(self.logger.data['wind_force'])
            max_wind_response = np.max(np.linalg.norm(wind_array, axis=1)) if len(wind_array) > 0 else 0.0
            
            metrics = {
                'final_pos_error': final_pos_error,
                'final_alt_error': final_alt_error,
                'max_overshoot': max_overshoot,
                'settling_time': settling_time,
                'max_attitude_error': np.degrees(max_att_error),
                'max_wind_response': max_wind_response,
                'gust_responses': ["Wind gust injected during simulation"]
            }
            
            # Generate summary
            controller_name = "Cascaded PID" if self.controller_mode == "pid" else "LQR Inner Loop"
            self.logger.generate_summary(controller_name, metrics)
            
            print(f"Report exported to: {self.logger.session_dir}")

