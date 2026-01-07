"""
Analysis utilities for flight control experiments.

Computes performance metrics:
- Rise time
- Overshoot
- Settling time
- Max thrust
- Max attitude deviation
"""

import numpy as np
from typing import Dict


def analyze_experiment(logger, target_position: np.ndarray) -> Dict[str, float]:
    """
    Analyze experiment results and compute metrics.
    
    Args:
        logger: SimulationLogger instance with logged data
        target_position: Target position [x, y, z]
        
    Returns:
        Dictionary of metrics
    """
    if len(logger.data['time']) == 0:
        return {
            'rise_time': 0.0,
            'overshoot': 0.0,
            'settling_time': 0.0,
            'max_thrust': 0.0,
            'max_attitude_dev': 0.0,
            'final_pos_error': 0.0
        }
    
    time_array = np.array(logger.data['time'])
    pos_array = np.array(logger.data['position'])
    att_array = np.array(logger.data['attitude'])
    thrust_array = np.array(logger.data['control_thrust'])
    
    # Altitude (z-axis)
    altitude = pos_array[:, 2]
    target_alt = target_position[2]
    
    # Rise time: time from 10% to 90% of final value
    rise_time = compute_rise_time(time_array, altitude, target_alt)
    
    # Overshoot: maximum deviation above target
    overshoot = compute_overshoot(time_array, altitude, target_alt)
    
    # Settling time: time to reach and stay within 2% of target
    settling_time = compute_settling_time(time_array, altitude, target_alt)
    
    # Max thrust command
    max_thrust = np.max(thrust_array) if len(thrust_array) > 0 else 0.0
    
    # Max attitude deviation (in degrees)
    max_attitude_dev = compute_max_attitude_deviation(att_array)
    
    # Final position error
    final_pos = pos_array[-1]
    final_pos_error = np.linalg.norm(final_pos - target_position)
    
    return {
        'rise_time': rise_time,
        'overshoot': overshoot,
        'settling_time': settling_time,
        'max_thrust': max_thrust,
        'max_attitude_dev': max_attitude_dev,
        'final_pos_error': final_pos_error
    }


def compute_rise_time(time: np.ndarray, signal: np.ndarray, target: float) -> float:
    """
    Compute rise time (10% to 90% of target).
    
    Args:
        time: Time array
        signal: Signal array
        target: Target value
        
    Returns:
        Rise time in seconds, or 0 if not found
    """
    if len(signal) == 0:
        return 0.0
    
    initial = signal[0]
    final = target
    range_val = final - initial
    
    if abs(range_val) < 1e-6:
        return 0.0
    
    # Find 10% and 90% points
    threshold_10 = initial + 0.1 * range_val
    threshold_90 = initial + 0.9 * range_val
    
    t_10 = None
    t_90 = None
    
    for i in range(len(signal)):
        if t_10 is None and signal[i] >= threshold_10:
            t_10 = time[i]
        if signal[i] >= threshold_90:
            t_90 = time[i]
            break
    
    if t_10 is None or t_90 is None:
        return 0.0
    
    return t_90 - t_10


def compute_overshoot(time: np.ndarray, signal: np.ndarray, target: float) -> float:
    """
    Compute maximum overshoot above target.
    
    Args:
        time: Time array
        signal: Signal array
        target: Target value
        
    Returns:
        Maximum overshoot (positive value)
    """
    if len(signal) == 0:
        return 0.0
    
    overshoots = signal - target
    max_overshoot = np.max(overshoots) if len(overshoots) > 0 else 0.0
    
    return max(0.0, max_overshoot)


def compute_settling_time(time: np.ndarray, signal: np.ndarray, target: float, 
                          tolerance: float = 0.02) -> float:
    """
    Compute settling time (time to reach and stay within tolerance of target).
    
    Args:
        time: Time array
        signal: Signal array
        target: Target value
        tolerance: Fractional tolerance (default 0.02 = 2%)
        
    Returns:
        Settling time in seconds, or total time if never settles
    """
    if len(signal) == 0:
        return 0.0
    
    threshold = abs(target) * tolerance
    settled_indices = []
    
    # Find all points within tolerance
    for i in range(len(signal)):
        if abs(signal[i] - target) <= threshold:
            settled_indices.append(i)
    
    if len(settled_indices) == 0:
        return time[-1]  # Never settled
    
    # Find longest continuous period within tolerance
    # Start from end and work backwards
    for i in range(len(signal) - 1, -1, -1):
        if abs(signal[i] - target) <= threshold:
            # Check if we stay within tolerance for rest of signal
            all_within = True
            for j in range(i, len(signal)):
                if abs(signal[j] - target) > threshold:
                    all_within = False
                    break
            if all_within:
                return time[i]
    
    return time[-1]


def compute_max_attitude_deviation(attitude_array: np.ndarray) -> float:
    """
    Compute maximum attitude deviation (roll, pitch, or yaw).
    
    Args:
        attitude_array: Array of [roll, pitch, yaw] in radians
        
    Returns:
        Maximum deviation in degrees
    """
    if len(attitude_array) == 0:
        return 0.0
    
    # Convert to degrees
    att_deg = np.degrees(attitude_array)
    
    # Find maximum absolute deviation from zero
    max_dev = np.max(np.abs(att_deg))
    
    return max_dev

