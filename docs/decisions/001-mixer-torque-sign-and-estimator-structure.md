# Decision 001: Mixer Torque Sign and Estimator Structure

## Context
- Shooting real footage for the launch video meant running every scenario headless and reading
  the logs. Every one of the eight experiments ended inverted (max attitude 180 deg) on the
  ground, including `nominal_hover` with sensor noise disabled.
- `vehicles._compute_allocation_matrix` built `M[1,i] = -pos[1]`, `M[2,i] = +pos[0]`, while
  `dynamics._compute_motor_torques` computes `tau = r x F = [ry*T, -rx*T, 0]`. Requesting
  `tau_x = +0.2 N.m` made the dynamics produce `-0.2 N.m`: the attitude loop was positive
  feedback.
- `Simulator.step` sampled the sensors and the estimator twice per timestep — once for control,
  once again for logging — advancing the estimator at twice the control rate.
- `SimpleEstimator.update` derived vertical velocity as `(baro - position_z) / dt` *after*
  `position_z` had already been corrected toward that same barometer reading, so the correction
  was divided by `dt` instead of being a time derivative.
- Sensor noise and gyro bias drift drew from the global `np.random`, unseeded, and the drift
  accumulated even with noise disabled, so no run was reproducible.
- `ComparisonUI.setup_ui` selects the first scenario, which calls `reset_simulation`, which
  touches `self.sim_timer` before `__init__` creates it. The resulting `AttributeError` was
  caught and printed, leaving the initial simulators uncreated.

## Decision
Fix the seven correctness defects above in `src/` and leave every gain and filter bandwidth
untouched, so the remaining scenario failures stay visible as tuning work rather than being
masked by a controller redesign.

## Reason
- Each fix resolves code that contradicts other code in the repo or contradicts physics
  (negative collective thrust from fixed-pitch rotors); none of them is a tuning preference.
- `MOTOR_MIN_THRUST` was already declared in `config.py` and unused — the collective-thrust
  guard was intended and missing, and without it an unachievable thrust demand was spread
  across the motors and then clipped per motor, silently destroying roll/pitch authority.
- Re-tuning the position loop, adding tilt-compensated thrust, or choosing an estimator
  bandwidth are design decisions with several defensible answers; they belong in their own
  decisions with their own before/after numbers.

## Consequences
Files changed: `src/vehicles.py`, `src/simulator.py`, `src/estimation/filters.py`,
`src/estimation/sensors.py`, `src/ui_comparison.py`.

`Simulator(enable_logging=True, seed=None)` and `SensorSimulator(enable_noise=True, seed=None)`
now take a seed; `seed=None` keeps the previous non-reproducible behaviour.

Measured at seed 0, base gains, via `Simulator.step`:

| Scenario | max attitude before | after | final position error before | after |
|---|---|---|---|---|
| `nominal_hover` | 180.0 deg | 0.0 deg | 22.07 m | 0.00 m |
| `wind_gust_down` | 180.0 deg | 0.0 deg | 205.69 m | 0.05 m |
| `wind_gust_left` | 180.0 deg | 30.7 deg | 28.95 m | 9.97 m |
| `hover_with_sensor_noise` | 179.9 deg | 12.0 deg | 120.46 m | 3.31 m |
| `lateral_step_xy` | 180.0 deg | 179.5 deg | 30.12 m | 458.31 m |

Still open, deliberately not addressed here:

- Lateral position loop is under-damped: `kd_pos` 0.5 against `kp_pos` 2.0-2.5 gives zeta ~= 0.18.
  `lateral_step_xy` diverges at base and aggressive gains; at conservative gains it stays upright
  with a 21.5 m excursion and a 15.5 m final error.
- Collective thrust has no `1/cos(tilt)` term, so the vehicle loses about 13% of its lift at the
  30 deg attitude clamp.
- Lateral gusts leave the vehicle airborne but about 10 m off the setpoint; `experiments.py`
  says "recovers to hover position".
- The alpha-beta gains (`alpha_pos` 0.7, `beta_pos` 0.3) are per-update fractions, not
  dt-aware. At 200 Hz with a 0.5 m barometer sigma the vertical velocity estimate carries
  sigma ~= 33 m/s; feeding the controller a clean velocity instead holds hover to +/-0.1 m
  under full sensor noise.
- Horizontal position and velocity are not propagated between 10 Hz GPS updates.
- `analysis.compute_rise_time` returns 0.00 s whenever a run starts at its setpoint, which is
  every hover scenario, and `compute_settling_time` returns the run length when the signal
  never settles — the GUI displays that as if it had settled.
