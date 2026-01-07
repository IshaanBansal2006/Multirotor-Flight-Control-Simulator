"""
LQR (Linear Quadratic Regulator) controller for inner loop (attitude + rates).

LQR is used for the inner loop to provide better damping and robustness
compared to PID. The outer loop (position) remains PID.

State: [roll, pitch, yaw, p, q, r]
Input: [tau_x, tau_y, tau_z]

The system is linearized around hover (small angles assumption).
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from src.config import HEX_INERTIA, LQR_Q, LQR_R
from src.utils.math3d import euler_from_quaternion


class LQRInnerController:
    """
    LQR controller for attitude and rate stabilization.
    
    This controller linearizes the system around hover and uses LQR
    to compute optimal gains. The outer loop (position) should use PID.
    """
    
    def __init__(self, mixer, enable_integral=True):
        """
        Initialize LQR controller.
        
        Args:
            mixer: Control mixer object
            enable_integral: Whether to use integral action (for outer loop)
        """
        self.mixer = mixer
        self.enable_integral = enable_integral
        
        # Compute LQR gains
        self.K = self._compute_lqr_gains()
        
        # For outer loop (position), we still use PID
        # This is a simplified version - in practice, you'd have a full PID
        from src.config import PID_POSITION_KP, PID_POSITION_KI, PID_POSITION_KD
        self.kp_pos = PID_POSITION_KP.copy()
        self.ki_pos = PID_POSITION_KI.copy()
        self.kd_pos = PID_POSITION_KD.copy()
        
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.filtered_derivative_pos = np.zeros(3)
        
        self.target_position = np.array([0.0, 0.0, 5.0])
        self.target_yaw = 0.0
    
    def _compute_lqr_gains(self):
        """
        Compute LQR gains by solving the Riccati equation.
        
        Linearized system around hover:
        - State: [roll, pitch, yaw, p, q, r]
        - Input: [tau_x, tau_y, tau_z]
        
        Dynamics:
        - d(roll)/dt ≈ p
        - d(pitch)/dt ≈ q
        - d(yaw)/dt ≈ r
        - d(p)/dt = I_inv * tau_x (simplified, ignoring gyroscopic effects for linearization)
        - d(q)/dt = I_inv * tau_y
        - d(r)/dt = I_inv * tau_z
        
        A = [0 0 0 1 0 0]   B = [0 0 0]
            [0 0 0 0 1 0]       [0 0 0]
            [0 0 0 0 0 1]       [0 0 0]
            [0 0 0 0 0 0]       [I_inv[0,0] 0 0]
            [0 0 0 0 0 0]       [0 I_inv[1,1] 0]
            [0 0 0 0 0 0]       [0 0 I_inv[2,2]]
        """
        I = HEX_INERTIA
        I_inv = np.linalg.inv(I)
        
        # Build A matrix (6x6)
        A = np.zeros((6, 6))
        A[0, 3] = 1.0  # d(roll)/dt = p
        A[1, 4] = 1.0  # d(pitch)/dt = q
        A[2, 5] = 1.0  # d(yaw)/dt = r
        # Rates derivatives are zero in linearized model (gyroscopic terms ignored)
        
        # Build B matrix (6x3)
        B = np.zeros((6, 3))
        B[3, 0] = I_inv[0, 0]  # d(p)/dt from tau_x
        B[4, 1] = I_inv[1, 1]  # d(q)/dt from tau_y
        B[5, 2] = I_inv[2, 2]  # d(r)/dt from tau_z
        
        # Solve continuous-time algebraic Riccati equation
        # A^T P + P A - P B R^-1 B^T P + Q = 0
        P = solve_continuous_are(A, B, LQR_Q, LQR_R)
        
        # Compute gain matrix: K = R^-1 B^T P
        K = np.linalg.inv(LQR_R) @ B.T @ P
        
        return K
    
    def update(self, estimated_state, dt):
        """
        Compute control outputs using LQR for inner loop.
        
        Args:
            estimated_state: Dict with position, velocity, attitude, rates
            dt: Timestep (seconds)
        
        Returns:
            Dict with 'torques' and 'thrust'
        """
        pos = estimated_state['position']
        vel = estimated_state['velocity']
        att_quat = estimated_state['attitude']
        rates = estimated_state['rates']
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(att_quat)
        
        # ====================================================================
        # Outer Loop: Position Controller (PID)
        # ====================================================================
        error_pos = self.target_position - pos
        
        if self.enable_integral:
            self.integral_pos += error_pos * dt
            self.integral_pos = np.clip(self.integral_pos, -10.0, 10.0)
        else:
            self.integral_pos = np.zeros(3)
        
        error_vel = -vel
        alpha = dt / (0.01 + dt)
        self.filtered_derivative_pos = (
            alpha * error_vel + (1 - alpha) * self.filtered_derivative_pos
        )
        
        desired_accel = (
            self.kp_pos * error_pos +
            self.ki_pos * self.integral_pos +
            self.kd_pos * self.filtered_derivative_pos
        )
        
        # Convert to desired attitude
        from src.config import GRAVITY
        desired_roll = -desired_accel[1] / GRAVITY
        desired_pitch = desired_accel[0] / GRAVITY
        max_angle = np.radians(30)
        desired_roll = np.clip(desired_roll, -max_angle, max_angle)
        desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
        
        desired_thrust = self.mass * (GRAVITY + desired_accel[2])
        
        # ====================================================================
        # Inner Loop: LQR Controller
        # ====================================================================
        # State vector: [roll, pitch, yaw, p, q, r]
        state = np.array([roll, pitch, yaw, rates[0], rates[1], rates[2]])
        
        # Desired state: [desired_roll, desired_pitch, target_yaw, 0, 0, 0]
        desired_state = np.array([desired_roll, desired_pitch, self.target_yaw, 0.0, 0.0, 0.0])
        
        # Error
        state_error = state - desired_state
        # Wrap yaw error
        state_error[2] = np.arctan2(np.sin(state_error[2]), np.cos(state_error[2]))
        
        # LQR control: u = -K * (x - x_desired)
        desired_torques = -self.K @ state_error
        
        self.prev_error_pos = error_pos
        
        return {
            'torques': desired_torques,
            'thrust': desired_thrust
        }
    
    def set_target(self, position, yaw=None):
        """Set target setpoint."""
        self.target_position = np.array(position)
        if yaw is not None:
            self.target_yaw = yaw
    
    def reset_integrators(self):
        """Reset integrators."""
        self.integral_pos = np.zeros(3)
    
    @property
    def mass(self):
        """Get vehicle mass."""
        from src.config import HEX_MASS
        return HEX_MASS

