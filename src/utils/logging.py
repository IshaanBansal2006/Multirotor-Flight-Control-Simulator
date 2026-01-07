"""
Logging utilities for saving simulation data and generating reports.
"""

import os
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class SimulationLogger:
    """
    Logs simulation data to CSV and generates plots and summary reports.
    """
    
    def __init__(self, log_dir="logs", enable=True):
        """
        Initialize logger.
        
        Args:
            log_dir: Base directory for logs
            enable: Whether logging is enabled
        """
        self.enable = enable
        self.log_dir = log_dir
        self.session_dir = None
        self.data = {
            'time': [],
            'position': [],
            'velocity': [],
            'attitude': [],
            'rates': [],
            'estimated_position': [],
            'estimated_velocity': [],
            'estimated_attitude': [],
            'estimated_rates': [],
            'motor_thrusts': [],
            'wind_force': [],
            'control_torques': [],
            'control_thrust': [],
        }
        
        if self.enable:
            self._create_session_dir()
    
    def _create_session_dir(self):
        """Create a timestamped directory for this simulation session."""
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.log_dir, timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
    
    def log(self, time, state, estimated_state, controls, motor_thrusts, wind_force):
        """
        Log one timestep of data.
        
        Args:
            time: Current simulation time
            state: True state dict with position, velocity, attitude, rates
            estimated_state: Estimated state dict
            controls: Control outputs (torques, thrust)
            motor_thrusts: Motor thrust commands
            wind_force: Wind disturbance force
        """
        if not self.enable:
            return
        
        self.data['time'].append(time)
        self.data['position'].append(state['position'].copy())
        self.data['velocity'].append(state['velocity'].copy())
        self.data['attitude'].append(state['attitude'].copy())
        self.data['rates'].append(state['rates'].copy())
        
        self.data['estimated_position'].append(estimated_state['position'].copy())
        self.data['estimated_velocity'].append(estimated_state['velocity'].copy())
        self.data['estimated_attitude'].append(estimated_state['attitude'].copy())
        self.data['estimated_rates'].append(estimated_state['rates'].copy())
        
        self.data['motor_thrusts'].append(motor_thrusts.copy())
        self.data['wind_force'].append(wind_force.copy())
        self.data['control_torques'].append(controls['torques'].copy())
        self.data['control_thrust'].append(controls['thrust'])
    
    def save_csv(self):
        """Save all logged data to CSV files."""
        if not self.enable or not self.session_dir:
            return
        
        # Save main data file
        csv_path = os.path.join(self.session_dir, "data.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['time']
            header.extend([f'pos_{ax}' for ax in ['x', 'y', 'z']])
            header.extend([f'vel_{ax}' for ax in ['x', 'y', 'z']])
            header.extend([f'att_{ax}' for ax in ['roll', 'pitch', 'yaw']])
            header.extend([f'rate_{ax}' for ax in ['p', 'q', 'r']])
            header.extend([f'est_pos_{ax}' for ax in ['x', 'y', 'z']])
            header.extend([f'est_vel_{ax}' for ax in ['x', 'y', 'z']])
            header.extend([f'est_att_{ax}' for ax in ['roll', 'pitch', 'yaw']])
            header.extend([f'est_rate_{ax}' for ax in ['p', 'q', 'r']])
            header.extend([f'motor_{i}' for i in range(len(self.data['motor_thrusts'][0]))])
            header.extend([f'wind_{ax}' for ax in ['x', 'y', 'z']])
            header.extend([f'torque_{ax}' for ax in ['x', 'y', 'z']])
            header.append('thrust')
            
            writer.writerow(header)
            
            # Data rows
            for i in range(len(self.data['time'])):
                row = [self.data['time'][i]]
                row.extend(self.data['position'][i])
                row.extend(self.data['velocity'][i])
                row.extend(self.data['attitude'][i])
                row.extend(self.data['rates'][i])
                row.extend(self.data['estimated_position'][i])
                row.extend(self.data['estimated_velocity'][i])
                row.extend(self.data['estimated_attitude'][i])
                row.extend(self.data['estimated_rates'][i])
                row.extend(self.data['motor_thrusts'][i])
                row.extend(self.data['wind_force'][i])
                row.extend(self.data['control_torques'][i])
                row.append(self.data['control_thrust'][i])
                writer.writerow(row)
    
    def save_plots(self):
        """Generate and save plots."""
        if not self.enable or not self.session_dir or len(self.data['time']) == 0:
            return
        
        time = np.array(self.data['time'])
        
        # Position plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        pos = np.array(self.data['position'])
        est_pos = np.array(self.data['estimated_position'])
        
        for i, ax_label in enumerate(['x', 'y', 'z']):
            axes[i].plot(time, pos[:, i], label='True', linewidth=2)
            axes[i].plot(time, est_pos[:, i], '--', label='Estimated', linewidth=1.5)
            axes[i].set_ylabel(f'{ax_label} (m)')
            axes[i].grid(True)
            axes[i].legend()
        axes[2].set_xlabel('Time (s)')
        axes[0].set_title('Position')
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, 'position.png'), dpi=150)
        plt.close()
        
        # Attitude plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        att = np.array(self.data['attitude'])
        est_att = np.array(self.data['estimated_attitude'])
        
        for i, ax_label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[i].plot(time, np.degrees(att[:, i]), label='True', linewidth=2)
            axes[i].plot(time, np.degrees(est_att[:, i]), '--', label='Estimated', linewidth=1.5)
            axes[i].set_ylabel(f'{ax_label} (deg)')
            axes[i].grid(True)
            axes[i].legend()
        axes[2].set_xlabel('Time (s)')
        axes[0].set_title('Attitude')
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, 'attitude.png'), dpi=150)
        plt.close()
        
        # Motor thrusts
        fig, ax = plt.subplots(figsize=(10, 6))
        motor_thrusts = np.array(self.data['motor_thrusts'])
        for i in range(motor_thrusts.shape[1]):
            ax.plot(time, motor_thrusts[:, i], label=f'Motor {i+1}', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thrust (N)')
        ax.set_title('Motor Thrusts')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, 'motor_thrusts.png'), dpi=150)
        plt.close()
        
        # Wind disturbance
        fig, ax = plt.subplots(figsize=(10, 6))
        wind = np.array(self.data['wind_force'])
        ax.plot(time, wind[:, 0], label='Wind X', linewidth=1.5)
        ax.plot(time, wind[:, 1], label='Wind Y', linewidth=1.5)
        ax.plot(time, wind[:, 2], label='Wind Z', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Wind Disturbance')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, 'wind.png'), dpi=150)
        plt.close()
    
    def generate_summary(self, controller_mode, metrics):
        """
        Generate a markdown summary report.
        
        Args:
            controller_mode: String describing controller mode used
            metrics: Dict with performance metrics
        """
        if not self.enable or not self.session_dir:
            return
        
        summary_path = os.path.join(self.session_dir, "summary.md")
        with open(summary_path, 'w') as f:
            f.write("# Simulation Summary Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Controller Configuration\n\n")
            f.write(f"- **Mode:** {controller_mode}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Final Position Error:** {metrics.get('final_pos_error', 'N/A'):.3f} m\n")
            f.write(f"- **Final Altitude Error:** {metrics.get('final_alt_error', 'N/A'):.3f} m\n")
            f.write(f"- **Max Overshoot (Altitude):** {metrics.get('max_overshoot', 'N/A'):.3f} m\n")
            f.write(f"- **Settling Time (Altitude):** {metrics.get('settling_time', 'N/A'):.3f} s\n")
            f.write(f"- **Max Attitude Error:** {metrics.get('max_attitude_error', 'N/A'):.3f} deg\n")
            f.write(f"- **Max Wind Response:** {metrics.get('max_wind_response', 'N/A'):.3f} m\n\n")
            
            f.write("## Notes\n\n")
            if 'gust_responses' in metrics:
                f.write("### Gust Responses\n")
                for note in metrics['gust_responses']:
                    f.write(f"- {note}\n")
                f.write("\n")
            
            f.write("## Files\n\n")
            f.write("- `data.csv`: Full time series data\n")
            f.write("- `position.png`: Position plots\n")
            f.write("- `attitude.png`: Attitude plots\n")
            f.write("- `motor_thrusts.png`: Motor commands\n")
            f.write("- `wind.png`: Wind disturbances\n")

