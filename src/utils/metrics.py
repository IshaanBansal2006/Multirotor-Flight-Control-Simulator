"""
Performance metrics computation: overshoot, settling time, etc.
"""

import numpy as np


def compute_settling_time(time, signal, setpoint, threshold=0.02, window_size=50):
    """
    Compute settling time: time to reach and stay within threshold of setpoint.
    
    Args:
        time: Time array
        signal: Signal array
        setpoint: Target value
        threshold: Fraction of setpoint for threshold (e.g., 0.02 = 2%)
        window_size: Number of consecutive samples that must be within threshold
    
    Returns:
        Settling time in seconds, or None if never settled
    """
    error = np.abs(signal - setpoint)
    threshold_value = abs(setpoint) * threshold if abs(setpoint) > 0.01 else threshold
    
    # Find where error is below threshold
    below_threshold = error < threshold_value
    
    # Find consecutive windows
    for i in range(len(below_threshold) - window_size):
        if np.all(below_threshold[i:i+window_size]):
            return time[i]
    
    return None


def compute_overshoot(time, signal, setpoint):
    """
    Compute maximum overshoot as percentage of setpoint.
    
    Args:
        time: Time array
        signal: Signal array
        setpoint: Target value
    
    Returns:
        Maximum overshoot as percentage, or 0 if no overshoot
    """
    if len(signal) == 0:
        return 0.0
    
    # Find maximum deviation above setpoint
    overshoot = np.max(signal - setpoint)
    
    if overshoot > 0 and abs(setpoint) > 0.01:
        return (overshoot / abs(setpoint)) * 100.0
    elif overshoot > 0:
        return overshoot
    
    return 0.0


def compute_steady_state_error(signal, setpoint, last_n_samples=100):
    """
    Compute steady-state error (mean of last N samples).
    
    Args:
        signal: Signal array
        setpoint: Target value
        last_n_samples: Number of final samples to average
    
    Returns:
        Steady-state error
    """
    if len(signal) < last_n_samples:
        return np.mean(signal - setpoint)
    
    return np.mean(signal[-last_n_samples:] - setpoint)


def compute_rmse(signal, setpoint):
    """
    Compute root mean square error.
    
    Args:
        signal: Signal array
        setpoint: Target value
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((signal - setpoint) ** 2))


def compute_max_attitude_error(attitude_history, target_attitude=None):
    """
    Compute maximum attitude error.
    
    Args:
        attitude_history: Array of [roll, pitch, yaw] over time
        target_attitude: Target [roll, pitch, yaw], defaults to [0, 0, 0]
    
    Returns:
        Maximum attitude error in radians
    """
    if target_attitude is None:
        target_attitude = np.array([0.0, 0.0, 0.0])
    
    attitude_array = np.array(attitude_history)
    errors = attitude_array - target_attitude
    
    # Compute magnitude of attitude error (Euler angle distance)
    error_magnitudes = np.linalg.norm(errors, axis=1)
    
    return np.max(error_magnitudes)

