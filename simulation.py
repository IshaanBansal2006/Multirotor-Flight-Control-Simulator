"""
Command-line entry point for running flight control experiments.

Usage:
    python simulation.py --experiment nominal_hover
    python simulation.py --experiment wind_gust
    python simulation.py --list  # List available experiments
"""

import argparse
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import get_experiment, list_experiments
from src.simulator import Simulator
from src.config import SIM_DT
from src.utils.math3d import quaternion_from_euler


def run_experiment(experiment_name: str):
    """
    Run a flight control experiment.
    
    Args:
        experiment_name: Name of experiment to run
    """
    print(f"\n{'='*70}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*70}\n")
    
    # Load experiment
    try:
        exp = get_experiment(experiment_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Print experiment description
    print(f"Description: {exp.description}")
    print(f"Expected Behavior: {exp.expected_behavior}\n")
    
    # Create simulator with experiment configuration
    sim = Simulator(enable_logging=True)
    
    # Configure controller
    controller = sim.pid_controller if exp.controller_type == "pid" else sim.lqr_controller
    sim.set_controller_mode(exp.controller_type)
    
    # Set controller gains from experiment
    for gain_name, gain_value in exp.control_gains.items():
        if hasattr(controller, gain_name):
            setattr(controller, gain_name, gain_value.copy())
    
    # Configure controller settings
    controller.enable_integral = exp.enable_integral
    controller.enable_anti_windup = exp.enable_anti_windup
    controller.integral_limit = exp.integral_limit
    
    # Set target
    controller.set_target(exp.target_position, exp.target_yaw)
    
    # Configure sensors
    sim.set_sensor_noise(exp.enable_sensor_noise)
    
    # Set initial state
    initial_attitude_quat = quaternion_from_euler(
        exp.initial_attitude[0],  # roll
        exp.initial_attitude[1],  # pitch
        exp.initial_attitude[2]   # yaw
    )
    sim.dynamics.state['position'] = exp.initial_position.copy()
    sim.dynamics.state['velocity'] = exp.initial_velocity.copy()
    sim.dynamics.state['attitude'] = initial_attitude_quat.copy()
    sim.dynamics.state['rates'] = np.array([0.0, 0.0, 0.0])
    
    # Reset estimator
    sim.estimator.reset(
        initial_position=exp.initial_position,
        initial_attitude=initial_attitude_quat
    )
    
    # Run simulation
    num_steps = int(exp.duration / SIM_DT)
    print(f"Running simulation for {exp.duration:.1f} seconds ({num_steps} steps)...")
    
    for step in range(num_steps):
        # Inject wind gust if specified
        if exp.enable_wind and exp.wind_gust_time is not None:
            if abs(sim.time - exp.wind_gust_time) < SIM_DT / 2:
                # Inject gust now
                sim.inject_wind_gust(
                    exp.wind_force_vector,
                    magnitude=exp.wind_gust_magnitude,
                    duration=exp.wind_gust_duration
                )
                print(f"  Wind gust injected at t={sim.time:.2f}s")
        
        sim.step()
        
        # Progress indicator
        if step % (num_steps // 10) == 0:
            progress = (step / num_steps) * 100
            print(f"  Progress: {progress:.0f}%", end='\r')
    
    print(f"  Progress: 100%")
    
    # Export results
    print("\nGenerating results...")
    sim.export_report()
    
    # Compute and display metrics
    from src.analysis import analyze_experiment
    metrics = analyze_experiment(sim.logger, exp.target_position)
    
    print(f"\n{'='*70}")
    print("Experiment Results:")
    print(f"{'='*70}")
    print(f"Rise Time (10-90%): {metrics['rise_time']:.2f} s")
    print(f"Overshoot: {metrics['overshoot']:.2f} m")
    print(f"Settling Time (2%): {metrics['settling_time']:.2f} s")
    print(f"Max Thrust Command: {metrics['max_thrust']:.2f} N")
    print(f"Max Attitude Deviation: {metrics['max_attitude_dev']:.2f} deg")
    print(f"Final Position Error: {metrics['final_pos_error']:.2f} m")
    print(f"\nResults saved to: {sim.logger.session_dir}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run flight control experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulation.py --experiment nominal_hover
  python simulation.py --experiment wind_gust
  python simulation.py --list
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='nominal_hover',
        help='Experiment name to run (default: nominal_hover)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Experiments:")
        print("=" * 70)
        for exp_name in list_experiments():
            exp = get_experiment(exp_name)
            print(f"\n{exp_name}:")
            print(f"  {exp.description}")
            print(f"  Expected: {exp.expected_behavior}")
        print("\n")
        return
    
    # Run experiment
    run_experiment(args.experiment)


if __name__ == "__main__":
    main()

