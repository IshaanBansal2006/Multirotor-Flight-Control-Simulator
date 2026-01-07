# Multirotor Flight Control Simulator

This project is a small **software‑in‑the‑loop (SIL) multirotor simulator** I built to explore flight control concepts in a hands‑on way.

It focuses on:
- How cascaded PID controllers behave in 6‑DOF flight
- How tuning choices show up in real trajectories and plots
- How disturbances (like wind gusts) and sensor noise affect stability
- How to structure a simple but clear simulation + visualization stack in Python

I wanted something that feels “physical” and visual, but is still small enough to read in one sitting.

---

## What it does

At a high level:

- Simulates a **6‑DOF multirotor** (position, velocity, attitude, angular rates)
- Runs a **cascaded PID controller**:
  - Outer loop: position → desired attitude + thrust
  - Middle loop: attitude → desired body rates
  - Inner loop: body rates → torques / motor commands
- Provides a set of **predefined scenarios** (“experiments”) that highlight specific behaviors:
  - Stable hover
  - Hover with sensor noise
  - Lateral position step in X/Y
  - Wind gusts from different directions
- Visualizes everything in a **PySide6 GUI**:
  - Two simulators side‑by‑side: “conservative” vs “aggressive” gains
  - A 3D view with a world‑fixed ground grid, drone model, and wind arrows
  - Live plots of position, attitude, and motor thrusts
  - Simple metrics (rise time, overshoot, settling) for altitude

There is also a **CLI mode** (no GUI) that runs scenarios and saves logs, plots, and summary metrics.

---

## Why I built this

A few things I wanted to practice and understand better:

- **Control design in context**  
  I’ve seen PID and LQR in textbooks, but I wanted to wire up a real‑ish stack end‑to‑end:
  state update → controller → mixer → dynamics → plots.  
  The goal was to see how “paper” tuning choices actually look in trajectories, not just in equations.

- **Cascaded controllers for multirotors**  
  Most serious multirotor controllers use a cascade (position → attitude → rates).  
  I wanted to feel the difference between changing gains in each loop and see how that propagates.

- **Disturbances and noise**  
  It’s easy to get something stable in a clean environment. I wanted to:
  - Inject wind gusts from different directions and see disturbance rejection
  - Turn on sensor noise and see how much jitter shows up in attitude and position

- **Small, clean Python architecture**  
  I wanted a project where:
  - Dynamics, controllers, experiments, analysis, and UI are separated reasonably well
  - It’s still approachable enough that I can walk through it quickly

---

## Main components

### Dynamics and simulator

- `dynamics.py`  
  Implements a simple 6‑DOF rigid body model for the multirotor:
  - Position and velocity in a world frame
  - Attitude and body rates
  - Gravity and thrust
  - A basic ground collision model with a small “bounce” and friction

- `simulator.py`  
  Wraps everything together:
  - Holds the dynamics, controller, mixer, and estimator
  - Steps the simulation at a fixed time step
  - Handles wind gust injection and sensor noise
  - Optionally logs data for analysis and plotting

### Controllers

- `controllers/pid_cascaded.py`  
  Implements the cascaded PID structure:
  - Outer position loop:
    - Takes position and velocity
    - Outputs desired attitude and thrust
  - Middle attitude loop:
    - Takes attitude and desired attitude
    - Outputs desired body rates
  - Inner rate loop:
    - Takes body rates and desired body rates
    - Outputs torques

  It includes:
  - Integral action with a configurable **integral limit**
  - Simple **anti‑windup** via integrator clamping
  - Basic derivative filtering

### Experiments (scenarios)

- `experiments.py`  
  Defines a small set of scenarios as data:

  - `nominal_hover`  
    Well‑tuned conservative gains, no wind, no noise.  
    Shows smooth convergence to a hover at a fixed altitude.

  - `hover_with_sensor_noise`  
    Same gains as nominal hover, but with sensor noise enabled.  
    Drone should stay near the setpoint with some visible jitter.

  - `lateral_step_xy`  
    Start at (0, 0, 5) m, step to (3, 2, 5) m.  
    Highlights lateral position control and step‑response behavior in X/Y.

  - `wind_gust_left`, `wind_gust_right`, `wind_gust_front`, `wind_gust_back`, `wind_gust_down`  
    Apply a wind gust from a specific direction at a set time:
    - The drone gets pushed off the setpoint
    - The controller works to bring it back
    - You can compare how conservative vs aggressive tuning handles the disturbance

Each scenario specifies:
- Gains
- Anti‑windup settings
- Whether sensor noise is on
- Wind direction, magnitude, and timing
- Initial conditions and target position/yaw
- Duration and a short “expected behavior” description

---

## Analysis and metrics

- `analysis.py`  
  Contains functions that take logged data and compute standard step‑response style metrics:

  - **Rise time** (10–90% of target)
  - **Overshoot** (max altitude above target)
  - **Settling time** (time to stay within a tolerance band)
  - **Max thrust** command
  - **Max attitude deviation**
  - **Final position error**

The CLI mode uses these functions to print a text summary and write a small report alongside plots.

The GUI reuses the rise/overshoot/settling logic to display a small metrics block for altitude on each side of the comparison.

---

## GUI

- `src/app.py` – entry point for the GUI: `python -m src.app`
- `src/ui_comparison.py` – main window

The GUI does a few things:

- **Two simulators side‑by‑side**  
  For any selected scenario, it spawns:
  - A “conservative” controller (gains scaled down, more damping)
  - An “aggressive” controller (gains scaled up, less damping)

- **3D visualization**  
  - World‑fixed ground grid at z = 0
  - Drone body represented as a small mesh + six motor spheres
  - Camera is fully free (mouse drag + wheel) with:
    - Zoom buttons
    - A **Reset Camera** button that returns to the default view
  - For wind scenarios, a **3D vector field** of arrows shows the wind direction throughout space

- **Live plots**  
  For each side (conservative/aggressive):
  - Position vs time (X, Y, Z)
  - Attitude vs time (Roll, Pitch, Yaw)
  - Motor thrust history

- **Simple metrics + scenario text**  
  - A small “Metrics (Z)” block per side with rise time, overshoot, and settling time
  - A scenario description block at the top that shows:
    - Short description of the scenario
    - Expected behavior

The goal of the GUI is to make it easy to just switch scenarios and visually see what changed, without touching any code or tuning live.

---

## CLI usage

### Installation

pip install -r requirements.txt### List available experiments

python simulation.py --list### Run an experiment (headless)

# Nominal hover (default)
python simulation.py --experiment nominal_hover

# Hover with sensor noise
python simulation.py --experiment hover_with_sensor_noise

# Lateral X/Y position step
python simulation.py --experiment lateral_step_xy

# Wind gust scenarios
python simulation.py --experiment wind_gust_left
python simulation.py --experiment wind_gust_right
python simulation.py --experiment wind_gust_front
python simulation.py --experiment wind_gust_back
python simulation.py --experiment wind_gust_downEach run writes results to `logs/<timestamp>/`:

- `data.csv` – time series
- `position.png`, `attitude.png`, `motor_thrusts.png` – plots
- `summary.md` – key metrics from `analysis.py`

---

## GUI usage

Run:

python -m src.appIn the window:

- Choose a scenario from the dropdown
- Click **Start Simulation**
- Watch the two sides evolve:
  - 3D motion vs grid
  - Plots
  - Metrics block for altitude
- Use the mouse and zoom buttons to navigate the 3D view
- Click **Reset Camera** if the view gets messy

---

## Coordinate frame and simplifications

- World frame is ENU‑style:
  - x: forward
  - y: right
  - z: up
- Gravity acts along negative z
- The model is intentionally simplified:
  - Linear drag, simple ground contact, no detailed rotor aerodynamics
  - Basic sensor noise model (when enabled)
  - Simple complementary‑filter style estimation

The idea is to keep the code and behavior understandable, rather than to be a perfect physical model.

---

## Project structure (high level)

.
├── simulation.py          # CLI entry point for experiments
├── src/
│   ├── app.py             # GUI entry point
│   ├── ui_comparison.py   # Side-by-side comparison GUI
│   ├── experiments.py     # Scenario definitions
│   ├── config.py          # Configuration constants
│   ├── simulator.py       # Main simulator coordinator
│   ├── dynamics.py        # 6-DOF rigid body dynamics
│   ├── controllers/
│   │   ├── pid_cascaded.py  # Cascaded PID controller
│   │   └── mixers.py        # Control allocation
│   ├── estimation/
│   │   ├── sensors.py       # Sensor simulation
│   │   └── filters.py       # Simple estimation
│   ├── utils/
│   │   ├── logging.py       # Data logging + plotting
│   │   ├── analysis.py      # Performance metrics
│   │   └── math3d.py        # Quaternion / 3D math
│   └── vehicles.py          # Vehicle geometry
└── logs/                    # Saved runs---

This is not meant to be a full autopilot, just a compact playground where I can see flight‑control ideas come to life in 3D and in plots.