# Decision 002: Position Loop and Estimator Tuning

## Context
- Decision 001 made the vehicle controllable but left three known gaps: the alpha-beta gains
  were per-update fractions rather than dt-aware, the lateral position loop was under-damped
  at zeta ~= 0.19, and the collective carried no tilt compensation. With those in place
  `lateral_step_xy` still diverged to 458 m and `hover_with_sensor_noise` drifted +7.02 m.
- Measuring the attitude estimator in isolation turned up a second sign error of the same
  family as the mixer one: `ComplementaryFilter` computed its correction as
  `cross(expected_gravity, measured_gravity)`, which is positive feedback. At the shipped
  gain of 0.002 it only cost a few degrees, so it never announced itself; raised to any
  useful gain it flipped the estimate outright (141 deg mean error at a gain of 2.0).
- Damping the position loop alone made things worse, not better: the damping term acts on an
  estimated velocity, so an estimator slow enough to be quiet adds phase lag to exactly the
  term that is supposed to provide damping. `nominal_hover` overshoot went from 0.04 m to
  5.61 m when kd was raised without re-tuning the estimator.

## Decision
Fix the attitude correction sign, make every estimator gain a time constant, derive the
position damping from the proportional gain, convert the acceleration command through an
exact tilt-limited thrust-vector inversion, and pick the remaining four numbers
(`PID_POSITION_KP` x/y and z, `EST_TAU_Z`, `EST_TAU_XY`) from a closed-loop sweep scored on
all four demo scenarios plus both gain variants of the noisy one.

## Reason
- The loop and the estimator cannot be tuned separately, so they were swept together rather
  than one at a time. The sweep is in the decision record below and is reproducible.
- Expressing gains as time constants and deriving kd from kp removes the two classes of
  error that caused this: a gain that silently depends on the loop rate, and a proportional
  and damping gain that disagree with each other.
- The exact thrust-vector inversion replaces a small-angle approximation, an angle clamp and
  a missing `1/cos(tilt)` term with one conversion that cannot disagree with itself. Both
  controllers now share it.
- x/y bandwidth had to come down, not up: with 10 Hz GPS at a 0.5 m sigma, every swept
  candidate with `kp_xy >= 1.2` lost control in at least one scenario and every candidate at
  0.7 lost none. The shipped 2.0 was asking for bandwidth the sensors cannot support.

## Consequences
Files changed: `src/config.py`, `src/experiments.py`, `src/estimation/filters.py`,
`src/controllers/pid_cascaded.py`, `src/controllers/lqr_inner.py`, and a new
`src/controllers/attitude_setpoint.py`.

Values, all measured rather than chosen by hand:

| Constant | Was | Now | Picked by |
|---|---|---|---|
| `PID_POSITION_KP` | `[2.0, 2.0, 5.0]` | `[0.7, 0.7, 5.0]` | closed-loop sweep, 72 candidates |
| `PID_POSITION_KD` | `[0.5, 0.5, 2.0]` | derived, `[1.51, 1.51, 4.02]` | `2 * 0.9 * sqrt(Kp)` |
| `EST_TAU_Z` | n/a (alpha 0.7 / beta 0.3 per update) | `0.10 s` | most robust of 4 finalists over 4 seeds |
| `EST_TAU_XY` | n/a | `0.70 s` | closed-loop sweep |
| `ATTITUDE_ACCEL_GAIN` | n/a (`(1 - 0.98) * 0.1`, wrong sign) | `2.0 1/s` | flat-optimal between 0.4 and 8 |

Attitude estimate error under full sensor noise, open loop over 12 s, against a trajectory
generated from body rates so the gyro and the true attitude are consistent:

| Motion | Before | After | Pure gyro, no correction |
|---|---|---|---|
| Still | 3.13 deg mean | 0.14 deg | 0.32 deg |
| 0.35 rad/s at 0.3 Hz | 2.70 deg | 0.20 deg | 0.32 deg |
| 1.20 rad/s at 0.8 Hz | 2.48 deg | 0.21 deg | 0.32 deg |

Vertical velocity estimate noise at a steady hover under full sensor noise, measured
directly from the estimator: sigma 32.8 m/s before, **0.589 m/s** at `EST_TAU_Z = 0.10`.
That is a factor of 56, and it is what the position loop's damping term now acts on.

All eight scenarios, base gains, seed 0, via `Simulator.step`:

| Scenario | Peak attitude | Max position error | Final position error |
|---|---|---|---|
| `nominal_hover` | 0.0 deg | 0.14 m | 0.00 m |
| `hover_with_sensor_noise` | 4.1 deg | 0.33 m | 0.14 m |
| `lateral_step_xy` | 10.8 deg | 3.61 m (the step itself) | 0.33 m |
| `wind_gust_left` / `right` / `front` / `back` | 29.6 deg | 3.74 m | 0.38 m |
| `wind_gust_down` | 0.0 deg | 1.26 m | 0.05 m |

Two `expected_behavior` strings in `src/experiments.py` that decision 001 recorded as
unmet are now met: the sensor-noise scenario holds the setpoint to 0.33 m, and the lateral
gusts recover to 0.38 m rather than settling about 10 m away.

`src/experiments.py` no longer carries nine gain arrays per scenario. `default_pid_gains()`
reads them from `config.py` and derives `kd_pos` from whatever `kp_pos` it is given, so a
scenario cannot hold a proportional gain and a damping gain that disagree, and a retune
reaches every scenario.

Still open:

- The accelerometer model in `src/estimation/sensors.py` reports gravity only, with no
  translational specific force, so the estimator can only run a constant-velocity model and
  cannot use accelerometer data to raise the position-loop bandwidth. This is now the single
  largest limit on how tight the position loop can be.
- GPS reports z with the same sigma as the barometer at a twentieth of the rate; it is
  measured and deliberately discarded rather than fused.
- There is no gust or wind-force estimator; disturbances are rejected reactively through the
  position integrator, which is why a lateral gust still costs 3.74 m and 29.6 deg.
- `analysis.compute_rise_time` still returns 0.00 s whenever a run starts at its setpoint,
  and `compute_settling_time` still returns the run length when the signal never settles.
- `LQRInnerController` is implemented and shares the new attitude-setpoint conversion, but no
  scenario selects it, so it has never been compared against the PID cascade.
