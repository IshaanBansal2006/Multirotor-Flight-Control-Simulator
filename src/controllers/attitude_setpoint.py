"""Turn an acceleration command into a thrust-vector setpoint.

Both the cascaded PID and the LQR controller need the same step: the outer loop
produces a desired world-frame acceleration, and the attitude loop needs a
roll, a pitch and a collective thrust that would produce it. Doing it in one
place keeps the two controllers from drifting apart.

The conversion is the exact inverse of the thrust direction rather than the
small-angle approximation, which buys two things:

* the collective carries its own ``1 / cos(tilt)`` term, so the vehicle stops
  losing lift while it is tilted;
* the horizontal command is limited against the vertical component before the
  angles are computed, so the loop stays linear right up to the tilt limit
  instead of hitting an angle clamp while the damping term still asks for more.

Convention (ENU world, body z up, ZYX Euler), matching ``utils/math3d`` and
``dynamics._compute_derivatives``: the body z axis expressed in the world frame
is ``[sin(pitch), -sin(roll) cos(pitch), cos(roll) cos(pitch)]``, so for a
thrust acceleration of magnitude ``c``

    a_x = c sin(pitch)
    a_y = -c sin(roll) cos(pitch)
    a_z = c cos(roll) cos(pitch) - g
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import GRAVITY, MAX_TILT_ANGLE, MIN_VERTICAL_ACCEL


@dataclass
class AttitudeSetpoint:
    """What the attitude loop should track, and whether the command was clipped."""

    roll: float
    pitch: float
    thrust: float
    tilt_saturated: bool


def thrust_vector_setpoint(desired_accel, yaw, mass,
                           max_tilt=MAX_TILT_ANGLE) -> AttitudeSetpoint:
    """Convert a world-frame acceleration command into roll, pitch and thrust.

    Args:
        desired_accel: Desired world-frame acceleration [ax, ay, az] (m/s^2),
            not including gravity.
        yaw: Current yaw angle (radians). The horizontal command is rotated
            into the yaw-aligned frame the roll and pitch angles live in.
        mass: Vehicle mass (kg).
        max_tilt: Largest tilt angle the attitude loop may be asked for (rad).

    Returns:
        An ``AttitudeSetpoint``. ``tilt_saturated`` is True when the horizontal
        command had to be shortened, which the caller should use to stop
        integrating the saturated axes.
    """
    desired_accel = np.asarray(desired_accel, dtype=float)

    # Fixed-pitch rotors cannot pull, so never ask the thrust vector to point
    # below the horizon; keep a floor under the vertical component.
    vertical = max(GRAVITY + desired_accel[2], MIN_VERTICAL_ACCEL)

    # The most horizontal acceleration the tilt limit can deliver at that
    # vertical component.
    max_horizontal = vertical * np.tan(max_tilt)
    horizontal = desired_accel[:2].copy()
    horizontal_norm = float(np.linalg.norm(horizontal))
    tilt_saturated = horizontal_norm > max_horizontal
    if tilt_saturated and horizontal_norm > 1e-9:
        horizontal *= max_horizontal / horizontal_norm

    # Rotate the horizontal command out of the world frame and into the
    # yaw-aligned frame.
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    forward = horizontal[0] * cos_yaw + horizontal[1] * sin_yaw
    right = -horizontal[0] * sin_yaw + horizontal[1] * cos_yaw

    magnitude = float(np.linalg.norm([forward, right, vertical]))
    pitch = float(np.arcsin(np.clip(forward / magnitude, -1.0, 1.0)))
    cos_pitch = max(float(np.cos(pitch)), 1e-3)
    roll = float(np.arcsin(np.clip(-right / (magnitude * cos_pitch), -1.0, 1.0)))

    return AttitudeSetpoint(roll=roll, pitch=pitch, thrust=mass * magnitude,
                            tilt_saturated=tilt_saturated)
