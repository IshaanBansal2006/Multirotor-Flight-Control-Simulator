"""
Control allocation mixers for converting desired forces/torques to motor commands.
"""

import numpy as np


class ControlMixer:
    """Base class for control mixers."""
    
    def compute_motor_thrusts(self, desired_thrust, desired_torques):
        """
        Compute motor thrusts from desired forces/torques.
        
        Args:
            desired_thrust: Total thrust (N)
            desired_torques: [tau_x, tau_y, tau_z] (N*m)
        
        Returns:
            Motor thrusts array
        """
        raise NotImplementedError


class HexacopterMixer(ControlMixer):
    """Mixer for hexacopter using vehicle's allocation matrix."""
    
    def __init__(self, vehicle):
        """
        Initialize mixer.
        
        Args:
            vehicle: Hexacopter vehicle object
        """
        self.vehicle = vehicle
    
    def compute_motor_thrusts(self, desired_thrust, desired_torques):
        """Compute motor thrusts using vehicle's allocation matrix."""
        return self.vehicle.compute_motor_thrusts(desired_thrust, desired_torques)

