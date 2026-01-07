# Scenario-Driven Flight Control Experiment Framework

A **Software-In-The-Loop (SIL) flight control experiment platform** designed for flight-controls engineering interviews (Reliable Robotics–style). This framework demonstrates control concepts through curated experiments rather than manual tuning.

## Philosophy

**This is NOT a tuning sandbox.** Instead, it's an **experiment playground** where:

- Predefined scenarios isolate and demonstrate specific control concepts
- Each experiment has fixed controller gains, disturbances, and configurations
- Users observe behavior and understand WHY control choices matter
- Results are automatically analyzed and visualized

This mirrors real control validation and flight-test workflows where scenarios are predefined and behavior is observed.

## Features

- **5 Predefined Experiments** demonstrating key control concepts
- **Cascaded PID Controller** with configurable anti-windup
- **6-DOF Rigid Body Dynamics** with ground collision
- **Automatic Analysis** (rise time, overshoot, settling time, etc.)
- **Headless Operation** - no GUI required
- **Automatic Plotting** with descriptive titles

## Installation

### Prerequisites

- Python 3.10 or higher
- NumPy, SciPy, Matplotlib

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### List Available Experiments

```bash
python simulation.py --list
```

### Run an Experiment

```bash
# Nominal hover (default)
python simulation.py --experiment nominal_hover

# Wind gust disturbance
python simulation.py --experiment wind_gust

# Aggressive controller (shows overshoot)
python simulation.py --experiment aggressive_controller

# Saturation without anti-windup (shows integrator windup)
python simulation.py --experiment saturation_no_antiwindup

# Saturation with anti-windup (shows proper recovery)
python simulation.py --experiment saturation_with_antiwindup
```

Results are automatically saved to `logs/<timestamp>/` with:
- `data.csv` - Full time series data
- `position.png` - Position plots
- `attitude.png` - Attitude plots
- `motor_thrusts.png` - Motor command history
- `summary.md` - Performance metrics

## Experiment Descriptions

### 1. Nominal Hover

**Purpose:** Demonstrates stable, well-tuned controller with conservative gains.

**Configuration:**
- Conservative PID gains
- Integral action enabled with anti-windup
- No disturbances
- No sensor noise

**Expected Behavior:** Smooth convergence to hover with minimal overshoot. Demonstrates proper gain tuning for stability.

**Key Takeaway:** Well-tuned controllers provide smooth, stable response.

---

### 2. Aggressive Controller

**Purpose:** Shows the trade-off between performance and robustness.

**Configuration:**
- High proportional gains (3x nominal)
- Low derivative gains (reduced damping)
- No disturbances

**Expected Behavior:** Fast initial response but with significant overshoot and oscillation. Shows why aggressive gains reduce robustness and stability margins.

**Key Takeaway:** Higher gains = faster response but less robust. There's a fundamental trade-off.

---

### 3. Wind Gust Disturbance

**Purpose:** Demonstrates disturbance rejection capability.

**Configuration:**
- Moderate, well-tuned gains
- Wind gust injected at t=3s in +X direction
- Gust magnitude: 8.0 N

**Expected Behavior:** Drone deviates from setpoint when gust hits, then recovers. Shows controller's disturbance rejection capability and settling time.

**Key Takeaway:** Controllers must balance tracking performance with disturbance rejection.

---

### 4. Saturation WITHOUT Anti-Windup

**Purpose:** Demonstrates integrator windup when actuators saturate.

**Configuration:**
- Same gains as Nominal Hover
- Strong wind gust (12.0 N) to cause saturation
- Integral enabled
- **Anti-windup DISABLED**
- High integral limit (allows windup)

**Expected Behavior:** Poor recovery after gust. Integrator accumulates error while actuators are saturated, causing overshoot and slow return. Demonstrates why anti-windup is essential.

**Key Takeaway:** Without anti-windup, integrators can cause instability under saturation.

---

### 5. Saturation WITH Anti-Windup

**Purpose:** Shows proper recovery when anti-windup prevents integrator windup.

**Configuration:**
- Same as experiment 4
- **Anti-windup ENABLED**
- Reasonable integral limit

**Expected Behavior:** Controlled recovery after gust. Anti-windup prevents integrator from accumulating error during saturation, enabling faster, smoother return to setpoint.

**Key Takeaway:** Anti-windup is critical for handling actuator saturation gracefully.

**Compare with:** `saturation_no_antiwindup` to see the difference.

## Architecture

### Control Structure

**Cascaded PID Controller:**
1. **Outer Loop (Position):** Position error → desired attitude/thrust
2. **Middle Loop (Attitude):** Attitude error → desired angular rates
3. **Inner Loop (Rates):** Rate error → desired torques

**Why Cascaded?**
- Separates fast dynamics (attitude/rates) from slow dynamics (position)
- Allows independent tuning of each loop
- Provides good disturbance rejection
- Industry standard for multirotor control

### Key Concepts Demonstrated

#### 1. Proportional Gain (Kp)
- **Higher Kp:** Faster response, but can cause overshoot and oscillation
- **Lower Kp:** More stable, but slower response
- **Trade-off:** Performance vs. robustness

#### 2. Derivative Gain (Kd)
- Provides damping
- Reduces overshoot and oscillation
- **Higher Kd:** More damping, smoother response
- **Too high:** Can cause noise sensitivity

#### 3. Integral Gain (Ki)
- Eliminates steady-state error
- **Problem:** Can cause "windup" when actuators saturate
- **Solution:** Anti-windup (clamp integrator)

#### 4. Anti-Windup
- Prevents integrator from accumulating error during saturation
- **Without it:** Poor recovery, overshoot, possible instability
- **With it:** Controlled recovery, faster return to setpoint

#### 5. Actuator Saturation
- Real actuators have limits (max thrust)
- When saturated, control authority is lost
- Integrators continue accumulating error → windup
- Anti-windup prevents this

## Interview Talking Points

### Cascaded Control Rationale

**Why cascaded?**
- Position loop is slow (seconds)
- Attitude loop is fast (milliseconds)
- Rate loop is fastest (microseconds)
- Cascading allows each loop to be tuned independently
- Inner loops provide disturbance rejection for outer loops

**Why not single-loop PID?**
- Single loop would need very high gains to track fast dynamics
- High gains → instability, poor robustness
- Cascading provides better performance with lower gains

### Robustness vs. Performance

**Performance:** Fast response, low overshoot, good tracking
**Robustness:** Stability under disturbances, model uncertainty, saturation

**Trade-off:**
- Aggressive gains → better performance, less robust
- Conservative gains → more robust, slower performance
- Real controllers balance both

**How to assess:**
- Test with disturbances (wind gusts)
- Test with model uncertainty
- Test with actuator saturation
- Look at stability margins (phase/gain margins)

### Saturation and Anti-Windup

**Problem:**
- Actuators saturate (hit max thrust)
- Controller commands more than available
- Integrator keeps accumulating error
- When saturation ends, integrator is "wound up"
- Causes overshoot and poor recovery

**Solution:**
- Clamp integrator when actuators are saturated
- Or: Stop integrating when error and control have opposite signs
- Prevents windup accumulation

**Why it matters:**
- Real systems always have saturation
- Without anti-windup, saturation can cause instability
- Essential for safe flight control

### Disturbance Rejection

**What is it?**
- Controller's ability to maintain setpoint despite disturbances
- Measured by: deviation magnitude, recovery time, settling time

**How to improve:**
- Higher gains → faster recovery, but less robust
- Better disturbance models → feedforward compensation
- Active disturbance rejection → estimate and cancel disturbances

**Trade-offs:**
- Too aggressive → instability
- Too conservative → poor rejection
- Balance via gain tuning

## Simplifications

This simulator makes several simplifications for clarity:

1. **Aerodynamics:** Linear drag model (not full aerodynamic model)
2. **Motor Dynamics:** First-order lag (not detailed motor physics)
3. **Estimation:** Complementary filter (not full EKF)
4. **Wind Model:** Simple impulse gusts (not realistic wind field)
5. **Small Angles:** Assumes small attitude angles for linearization
6. **Ideal Sensors:** No noise unless explicitly enabled

**Why?**
- Focus on control concepts, not modeling complexity
- Makes code readable and maintainable
- Sufficient for demonstrating control principles

## Future Extensions

Potential enhancements for a more complete system:

1. **LQR Inner Loop:** Already implemented, can be used in experiments
2. **MPC (Model Predictive Control):** Handle constraints explicitly
3. **System Identification:** Tune parameters from real flight data
4. **Hardware-in-the-Loop (HITL):** Test with real flight controller
5. **Full EKF:** Complete state estimation with all biases
6. **Trajectory Tracking:** Waypoint following and trajectory generation
7. **Fault Detection:** Detect and handle motor failures
8. **Adaptive Control:** Adjust gains based on flight conditions
9. **Full Aerodynamics:** Include rotor wash, ground effect, etc.

## Project Structure

```
.
├── simulation.py          # CLI entry point for experiments
├── src/
│   ├── experiments.py     # Predefined experiment scenarios
│   ├── config.py          # Configuration constants
│   ├── simulator.py       # Main simulator coordinator
│   ├── dynamics.py        # 6-DOF rigid body dynamics
│   ├── controllers/
│   │   ├── pid_cascaded.py  # Cascaded PID controller
│   │   └── mixers.py        # Control allocation
│   ├── estimation/
│   │   ├── sensors.py       # Sensor simulation
│   │   └── filters.py       # State estimation
│   ├── utils/
│   │   ├── logging.py       # Data logging and plots
│   │   ├── analysis.py       # Performance metrics
│   │   └── math3d.py        # Quaternion utilities
│   └── vehicles.py          # Vehicle geometry
├── logs/                   # Experiment results
└── README.md
```

## Coordinate Frame

**ENU (East-North-Up) Convention:**
- **x-axis:** Forward (East)
- **y-axis:** Right (North)
- **z-axis:** Up (Up)
- **Gravity:** Acts in -z direction (9.81 m/s²)
- **Thrust:** Acts in body +z direction

## Performance Metrics

Each experiment automatically computes:

- **Rise Time:** Time from 10% to 90% of target
- **Overshoot:** Maximum deviation above target
- **Settling Time:** Time to reach and stay within 2% of target
- **Max Thrust:** Maximum thrust command
- **Max Attitude Deviation:** Maximum roll/pitch/yaw deviation
- **Final Position Error:** Final distance from target

## License

This is a portfolio project for demonstration purposes.

## Acknowledgments

This simulator demonstrates concepts commonly used in:
- Drone flight control systems
- Aerospace control systems
- Robotics applications
- Control theory education
