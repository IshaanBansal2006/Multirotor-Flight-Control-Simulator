"""
3D Math utilities: quaternions, rotation matrices, and coordinate transformations.

Coordinate Frame: ENU (East-North-Up)
- World frame: x forward (East), y right (North), z up (Up)
- Body frame: x forward, y right, z up
- Quaternions: [w, x, y, z] format (scalar first)
"""

import numpy as np


def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles (ZYX convention) to quaternion.
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
    
    Returns:
        Quaternion as [w, x, y, z]
    """
    # ZYX convention: first yaw (z), then pitch (y), then roll (x)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def euler_from_quaternion(q):
    """
    Convert quaternion to Euler angles (ZYX convention).
    
    Args:
        q: Quaternion as [w, x, y, z]
    
    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_normalize(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    return q / norm


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q_result = q1 * q2
    
    Args:
        q1, q2: Quaternions as [w, x, y, z]
    
    Returns:
        Product quaternion as [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """Return conjugate of quaternion (inverse rotation)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate_vector(q, v):
    """
    Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion as [w, x, y, z]
        v: Vector as [x, y, z]
    
    Returns:
        Rotated vector as [x, y, z]
    """
    # Convert vector to quaternion
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    # Rotate: q * v * q_conj
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return result[1:4]  # Return vector part


def rotation_matrix_from_quaternion(q):
    """
    Convert quaternion to rotation matrix (world to body).
    
    Args:
        q: Quaternion as [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix (ZYX convention).
    
    Args:
        roll, pitch, yaw: Euler angles in radians
    
    Returns:
        3x3 rotation matrix
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    
    return R


def quaternion_derivative(q, omega):
    """
    Compute quaternion derivative given angular velocity.
    
    The quaternion derivative is: dq/dt = 0.5 * q * [0, omega_x, omega_y, omega_z]
    
    Args:
        q: Current quaternion [w, x, y, z]
        omega: Angular velocity in body frame [p, q, r] (rad/s)
    
    Returns:
        Quaternion derivative [dw, dx, dy, dz]
    """
    w, x, y, z = q
    p, q_val, r = omega
    
    # Quaternion representation of angular velocity
    omega_quat = np.array([0.0, p, q_val, r])
    
    # dq/dt = 0.5 * q * omega_quat
    dq = 0.5 * quaternion_multiply(q, omega_quat)
    
    return dq


def skew_symmetric(v):
    """
    Create skew-symmetric matrix from vector.
    
    For vector v = [vx, vy, vz], returns:
    [  0  -vz   vy ]
    [ vz    0  -vx ]
    [-vy   vx    0 ]
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

