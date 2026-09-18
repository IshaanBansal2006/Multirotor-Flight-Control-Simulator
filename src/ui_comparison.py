"""
Comparison UI for flight control experiments.

Shows two simulations side-by-side:
- Conservative gains (loose/stable)
- Aggressive gains (harsh/fast)

User can select experiment scenario and compare behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Try to import OpenGL for 3D visualization
try:
    import pyqtgraph.opengl as gl
    OPENGL_AVAILABLE = True
except (ImportError, AttributeError, OSError, Exception):
    OPENGL_AVAILABLE = False
    gl = None

from src.config import SIM_DT, UI_UPDATE_RATE, PLOT_HISTORY_LENGTH, HEX_RADIUS, HEX_MOTOR_ANGLES
from src.experiments import get_experiment, list_experiments
from src.simulator import Simulator
from src.utils.math3d import quaternion_from_euler, rotation_matrix_from_quaternion
from src.analysis import compute_rise_time, compute_overshoot, compute_settling_time
from PySide6.QtGui import QWheelEvent


class CustomGLViewWidget(gl.GLViewWidget):
    """Custom GLViewWidget that zooms towards mouse cursor position."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_speed = 0.1  # Zoom speed factor
        self._camera_distance = 40.0
        self._camera_elevation = 10.0
        self._camera_azimuth = 45.0
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events to zoom towards cursor position."""
        if not OPENGL_AVAILABLE:
            return
        
        # Get mouse position in widget coordinates
        mouse_pos = event.position()
        widget_size = self.size()
        
        # Normalize mouse position to [-1, 1] range
        x_norm = 2.0 * (mouse_pos.x() / widget_size.width()) - 1.0
        y_norm = 1.0 - 2.0 * (mouse_pos.y() / widget_size.height())  # Flip Y
        
        # Get wheel delta: positive = scroll up, negative = scroll down
        # We want scroll up to zoom IN (decrease distance), scroll down to zoom OUT (increase distance)
        delta = event.angleDelta().y()
        # Invert: positive delta should decrease distance (zoom in), negative should increase (zoom out)
        zoom_factor = 1.0 - (delta / 1200.0) * self.zoom_speed
        
        # Calculate the 3D point under the mouse cursor
        # pyqtgraph uses spherical coordinates orbiting around origin
        # We need to calculate where the mouse ray intersects a plane at the current view distance
        import math
        
        # Convert camera angles to radians
        az_rad = math.radians(self._camera_azimuth)
        el_rad = math.radians(self._camera_elevation)
        
        # Calculate camera's view direction (from camera towards origin)
        # Camera is at distance from origin, looking at origin
        view_dir = np.array([
            -math.sin(az_rad) * math.cos(el_rad),
            -math.cos(az_rad) * math.cos(el_rad),
            math.sin(el_rad)
        ])
        
        # Calculate right and up vectors for the camera view plane
        right = np.array([math.cos(az_rad), -math.sin(az_rad), 0])
        up = np.cross(view_dir, right)
        up = up / (np.linalg.norm(up) + 1e-6)
        
        # Estimate field of view (pyqtgraph default is around 60 degrees)
        fov_deg = 60.0
        fov_rad = math.radians(fov_deg)
        aspect = widget_size.width() / widget_size.height()
        
        # Calculate the point on a plane at the current view distance
        # The plane is perpendicular to view_dir
        plane_distance = self._camera_distance * 0.5  # Use a plane at half the distance
        plane_center = -view_dir * plane_distance
        
        # Calculate plane dimensions based on FOV
        plane_width = 2.0 * plane_distance * math.tan(fov_rad / 2.0)
        plane_height = plane_width / aspect
        
        # Calculate the 3D point under the mouse cursor on this plane
        offset = right * (x_norm * plane_width / 2.0) + up * (y_norm * plane_height / 2.0)
        target_point_3d = plane_center + offset
        
        # Calculate the direction from origin to this target point
        target_dir = target_point_3d / (np.linalg.norm(target_point_3d) + 1e-6)
        
        # Calculate desired azimuth and elevation to look at this point
        target_az = math.degrees(math.atan2(-target_dir[0], -target_dir[1]))
        target_el = math.degrees(math.asin(np.clip(target_dir[2], -1, 1)))
        
        # Calculate angle differences (handle wrapping for azimuth)
        az_diff = (target_az - self._camera_azimuth + 180) % 360 - 180
        el_diff = target_el - self._camera_elevation
        
        # Update camera distance
        old_distance = self._camera_distance
        self._camera_distance *= zoom_factor
        self._camera_distance = max(2.0, min(100.0, self._camera_distance))
        
        # Adjust azimuth and elevation to zoom towards mouse position
        # The shift strength depends on how far mouse is from center and how much we're zooming
        mouse_distance_from_center = math.sqrt(x_norm**2 + y_norm**2)
        zoom_amount = abs(1.0 - zoom_factor)  # How much we're zooming
        shift_strength = min(1.0, mouse_distance_from_center) * zoom_amount * 0.8  # Scale the shift
        
        # Interpolate towards the target angles
        new_azimuth = self._camera_azimuth + az_diff * shift_strength
        new_elevation = self._camera_elevation + el_diff * shift_strength
        
        # Store updated values
        self._camera_azimuth = new_azimuth
        self._camera_elevation = new_elevation
        
        # Set new camera position
        self.setCameraPosition(
            distance=self._camera_distance,
            elevation=self._camera_elevation,
            azimuth=self._camera_azimuth
        )
        
        event.accept()
    
    def setCameraPosition(self, **kwargs):
        """Override to track camera state."""
        super().setCameraPosition(**kwargs)
        if 'distance' in kwargs:
            self._camera_distance = kwargs['distance']
        if 'elevation' in kwargs:
            self._camera_elevation = kwargs['elevation']
        if 'azimuth' in kwargs:
            self._camera_azimuth = kwargs['azimuth']


class ComparisonUI(QMainWindow):
    """Comparison UI showing two simulations side-by-side."""
    
    def __init__(self):
        """Initialize comparison UI."""
        super().__init__()
        
        self.setWindowTitle("Flight Control Experiment Comparison")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Two simulators: conservative and aggressive
        self.sim_conservative = None
        self.sim_aggressive = None
        self.current_experiment = None
        
        # Data buffers for plotting (one for each simulator)
        self.data_conservative = {
            'time': [],
            'position': [[], [], []],
            'attitude': [[], [], []],
            'motor': [[] for _ in range(6)],
        }
        self.data_aggressive = {
            'time': [],
            'position': [[], [], []],
            'attitude': [[], [], []],
            'motor': [[] for _ in range(6)],
        }
        
        # Simulation state
        self.running = False
        self.sim_time = 0.0
        
        # Timers must exist before setup_ui(): building the UI selects the first
        # scenario, which resets the simulation and stops the sim timer.
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        
        self.setup_ui()
        
        self.ui_timer.start(int(1000 / UI_UPDATE_RATE))
    
    def setup_ui(self):
        """Set up the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top: Experiment selection
        top_group = QGroupBox("Experiment Selection")
        top_layout = QHBoxLayout()
        
        top_layout.addWidget(QLabel("Scenario:"))
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(list_experiments())
        self.experiment_combo.currentTextChanged.connect(self.on_experiment_changed)
        top_layout.addWidget(self.experiment_combo)
        
        top_layout.addStretch()
        
        # Reset camera button (resets both 3D views to default)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera_views)
        top_layout.addWidget(self.reset_camera_button)
        
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        top_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        top_layout.addWidget(self.reset_button)
        
        top_group.setLayout(top_layout)
        main_layout.addWidget(top_group)
        
        # Scenario description / expected behavior text
        self.scenario_desc_label = QLabel("")
        self.scenario_desc_label.setWordWrap(True)
        self.scenario_desc_label.setStyleSheet("font-size: 11px; color: #dddddd;")
        main_layout.addWidget(self.scenario_desc_label)
        
        # Main comparison area: Side-by-side
        comparison_splitter = QSplitter(Qt.Horizontal)
        comparison_splitter.setChildrenCollapsible(False)
        
        # Left: Conservative gains
        self.left_panel = self.create_simulator_panel("Conservative Gains", "#6495ED")  # Lighter blue (cornflower blue)
        comparison_splitter.addWidget(self.left_panel)
        
        # Right: Aggressive gains
        self.right_panel = self.create_simulator_panel("Aggressive Gains", "red")
        comparison_splitter.addWidget(self.right_panel)
        
        comparison_splitter.setStretchFactor(0, 1)
        comparison_splitter.setStretchFactor(1, 1)
        comparison_splitter.setSizes([800, 800])
        
        main_layout.addWidget(comparison_splitter)
        
        # Load initial experiment
        self.on_experiment_changed(list_experiments()[0])
        
        # Initialize wind vector field for both panels
        self.update_wind_vector_field(self.left_panel)
        self.update_wind_vector_field(self.right_panel)
    
    def create_simulator_panel(self, title, color):
        """Create a panel for one simulator."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title and gain display
        title_group = QGroupBox(title)
        title_layout = QVBoxLayout()

        # Gain values display
        gain_label = QLabel("Gains: Not loaded")
        gain_label.setWordWrap(True)
        gain_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        title_layout.addWidget(gain_label)
        
        # Metrics display
        metrics_label = QLabel(
            "Metrics:\n"
            "  Rise: -- s\n"
            "  Overshoot: -- m\n"
            "  Settling: -- s"
        )
        metrics_label.setWordWrap(True)
        metrics_label.setStyleSheet("font-size: 11px; color: #cccccc;")
        title_layout.addWidget(metrics_label)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_out_btn = QPushButton("Zoom Out")
        # Zoom In decreases distance (factor < 1), Zoom Out increases distance (factor > 1)
        zoom_in_btn.clicked.connect(lambda: self.zoom_camera(panel, 0.8))   # Decrease distance = zoom in
        zoom_out_btn.clicked.connect(lambda: self.zoom_camera(panel, 1.2))  # Increase distance = zoom out
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        title_layout.addLayout(zoom_layout)

        title_group.setLayout(title_layout)
        layout.addWidget(title_group)
        
        # 3D Visualization
        if OPENGL_AVAILABLE:
            view3d = CustomGLViewWidget()
            # Set dark background so white grid is visible
            try:
                view3d.setBackgroundColor('k')  # Black background
            except:
                pass
            # Start zoomed out more so drone is clearly visible
            # Lower elevation = steeper angle (more overhead view)
            view3d.setCameraPosition(distance=40, elevation=10, azimuth=45)
            view3d.setMinimumSize(400, 200)  # Smaller height so plots fit better
            layout.addWidget(view3d)
            panel.view3d = view3d
            # Setup scene after widget is added to layout (important for OpenGL)
            self.setup_3d_scene(view3d, panel)
        else:
            no_gl_label = QLabel("3D visualization unavailable\n(OpenGL not available)")
            no_gl_label.setAlignment(Qt.AlignCenter)
            no_gl_label.setMinimumSize(400, 300)
            layout.addWidget(no_gl_label)
            panel.view3d = None
        
        # Plots
        plots_group = QGroupBox("Live Plots")
        plots_layout = QVBoxLayout()
        
        # Position plot
        pos_plot = pg.PlotWidget(title="Position")
        # Position legend at top-right with smaller font and reduced spacing, moved up 5px
        pos_plot.addLegend(offset=(10, 5), labelTextSize='7pt', colCount=1, rowCount=3, spacing=2)
        pos_plot.setLabel('left', 'Position (m)')
        pos_plot.setLabel('bottom', 'Time (s)')
        pos_plot.setMinimumHeight(150)
        pos_plot.setBackground('w')
        plots_layout.addWidget(pos_plot)
        
        # Attitude plot
        att_plot = pg.PlotWidget(title="Attitude")
        # Position legend at top-right with smaller font and reduced spacing, moved up 5px
        att_plot.addLegend(offset=(10, 5), labelTextSize='7pt', colCount=1, rowCount=3, spacing=2)
        att_plot.setLabel('left', 'Angle (deg)')
        att_plot.setLabel('bottom', 'Time (s)')
        att_plot.setMinimumHeight(150)
        att_plot.setBackground('w')
        plots_layout.addWidget(att_plot)
        
        # Motor thrusts plot
        motor_plot = pg.PlotWidget(title="Motor Thrusts")
        motor_plot.setLabel('left', 'Thrust (N)')
        motor_plot.setLabel('bottom', 'Time (s)')
        motor_plot.setMinimumHeight(150)
        motor_plot.setBackground('w')
        plots_layout.addWidget(motor_plot)
        
        plots_group.setLayout(plots_layout)
        layout.addWidget(plots_group)
        
        # Store plot references
        panel.pos_plot = pos_plot
        panel.att_plot = att_plot
        panel.motor_plot = motor_plot
        panel.gain_label = gain_label
        panel.metrics_label = metrics_label
        
        # Initialize 3D visualization state
        panel.vehicle_mesh = None
        panel.motor_meshes = []
        panel.grid = None
        panel.ground_plane = None
        panel.axes = None
        panel.wind_arrows = []  # Store wind vector field arrows
        # Always in free camera mode - no follow mode
        panel.follow_mode = False
        panel.camera_distance = 40.0  # Track camera distance for zooming (start more zoomed out)
        
        return panel
    
    def setup_3d_scene(self, view3d, panel):
        """Set up 3D visualization scene for a panel."""
        if not OPENGL_AVAILABLE or view3d is None:
            return
        
        # Clear existing items
        view3d.clear()
        
        # Add grid at ground level (z=0) - white lines only, smaller squares
        # Grid is PERMANENTLY fixed at z=0 in world frame - never moves
        grid = gl.GLGridItem()
        grid.scale(5, 5, 1)  # 50x50 unit grid with smaller squares
        # Position exactly at z=0 (ground level)
        grid.translate(0, 0, 0)
        view3d.addItem(grid)
        panel.grid = grid
        
        # No ground plane - just grid for cleaner look
        panel.ground_plane = None
        
        # Add axes
        axes = gl.GLAxisItem()
        axes.setSize(2, 2, 2)
        view3d.addItem(axes)
        panel.axes = axes
        
        # Force update to ensure grid is rendered
        view3d.update()
    
    def on_experiment_changed(self, experiment_name):
        """Handle experiment selection change."""
        if experiment_name:
            try:
                self.current_experiment = get_experiment(experiment_name)
                
                # Update scenario description text
                desc = self.current_experiment.description
                expected = self.current_experiment.expected_behavior
                self.scenario_desc_label.setText(
                    f"<b>Description:</b> {desc}<br>"
                    f"<b>Expected:</b> {expected}"
                )
                
                self.reset_simulation()
                # Update wind vector field visualization
                self.update_wind_vector_field(self.left_panel)
                self.update_wind_vector_field(self.right_panel)
            except Exception as e:
                print(f"Error loading experiment: {e}")
    
    def start_simulation(self):
        """Start both simulations."""
        if self.current_experiment is None:
            return
        
        if self.running:
            # Pause
            self.running = False
            self.sim_timer.stop()
            self.start_button.setText("Start Simulation")
        else:
            # Start
            self.running = True
            self.sim_timer.start(int(1000 * SIM_DT))
            self.start_button.setText("Pause")
            
            # Create simulators if needed
            if self.sim_conservative is None or self.sim_aggressive is None:
                self.create_simulators()
    
    def reset_simulation(self):
        """Reset both simulations."""
        self.running = False
        self.sim_timer.stop()
        self.start_button.setText("Start Simulation")
        self.sim_time = 0.0
        
        # Clear data
        self.data_conservative = {
            'time': [],
            'position': [[], [], []],
            'attitude': [[], [], []],
            'motor': [[] for _ in range(6)],
        }
        self.data_aggressive = {
            'time': [],
            'position': [[], [], []],
            'attitude': [[], [], []],
            'motor': [[] for _ in range(6)],
        }
        
        # Create new simulators
        self.create_simulators()
        
        # Update plots
        self.update_ui()
    
    def create_simulators(self):
        """Create two simulators with conservative and aggressive gains."""
        if self.current_experiment is None:
            return
        
        exp = self.current_experiment
        
        # Create conservative simulator
        self.sim_conservative = Simulator(enable_logging=False)
        self.configure_simulator(self.sim_conservative, exp, conservative=True)
        
        # Create aggressive simulator
        self.sim_aggressive = Simulator(enable_logging=False)
        self.configure_simulator(self.sim_aggressive, exp, conservative=False)
        
        # Update gain labels
        self.update_gain_labels()
    
    def configure_simulator(self, sim, exp, conservative=True):
        """Configure a simulator with experiment settings and gain variant."""
        controller = sim.pid_controller if exp.controller_type == "pid" else sim.lqr_controller
        sim.set_controller_mode(exp.controller_type)
        
        # Apply base gains from experiment
        for gain_name, gain_value in exp.control_gains.items():
            if hasattr(controller, gain_name):
                base_gain = gain_value.copy()
                
                # Modify gains based on variant
                if conservative:
                    # Conservative: reduce gains by 30%, increase damping
                    if 'kp' in gain_name:
                        modified_gain = base_gain * 0.7
                    elif 'kd' in gain_name:
                        modified_gain = base_gain * 1.3  # More damping
                    else:
                        modified_gain = base_gain * 0.8
                else:
                    # Aggressive: increase gains by 50%, reduce damping
                    if 'kp' in gain_name:
                        modified_gain = base_gain * 1.5
                    elif 'kd' in gain_name:
                        modified_gain = base_gain * 0.7  # Less damping
                    else:
                        modified_gain = base_gain * 1.2
                
                setattr(controller, gain_name, modified_gain)
        
        # Configure controller settings
        controller.enable_integral = exp.enable_integral
        controller.enable_anti_windup = exp.enable_anti_windup
        controller.integral_limit = exp.integral_limit
        
        # Set target
        controller.set_target(exp.target_position, exp.target_yaw)
        
        # Configure sensors
        sim.set_sensor_noise(exp.enable_sensor_noise)
        
        # Set initial state
        initial_attitude_quat = quaternion_from_euler(
            exp.initial_attitude[0],
            exp.initial_attitude[1],
            exp.initial_attitude[2]
        )
        # Ensure drone starts at z=5m (clearly in the air)
        initial_pos = exp.initial_position.copy()
        initial_pos[2] = 5.0  # Always start at 5m altitude
        
        sim.dynamics.state['position'] = initial_pos
        sim.dynamics.state['velocity'] = exp.initial_velocity.copy()
        sim.dynamics.state['attitude'] = initial_attitude_quat.copy()
        sim.dynamics.state['rates'] = np.array([0.0, 0.0, 0.0])
        
        # Reset estimator
        sim.estimator.reset(
            initial_position=initial_pos,
            initial_attitude=initial_attitude_quat
        )
        
        # Reset time
        sim.time = 0.0
    
    def update_gain_labels(self):
        """Update gain display labels."""
        if self.sim_conservative is None or self.sim_aggressive is None:
            return
        
        # Conservative gains
        ctrl_cons = self.sim_conservative.pid_controller
        cons_text = f"Position Kp: {ctrl_cons.kp_pos[2]:.2f}, Ki: {ctrl_cons.ki_pos[2]:.2f}, Kd: {ctrl_cons.kd_pos[2]:.2f}\n"
        cons_text += f"Attitude Kp: {ctrl_cons.kp_att[0]:.2f}, Ki: {ctrl_cons.ki_att[0]:.2f}, Kd: {ctrl_cons.kd_att[0]:.2f}"
        self.left_panel.gain_label.setText(cons_text)
        
        # Aggressive gains
        ctrl_agg = self.sim_aggressive.pid_controller
        agg_text = f"Position Kp: {ctrl_agg.kp_pos[2]:.2f}, Ki: {ctrl_agg.ki_pos[2]:.2f}, Kd: {ctrl_agg.kd_pos[2]:.2f}\n"
        agg_text += f"Attitude Kp: {ctrl_agg.kp_att[0]:.2f}, Ki: {ctrl_agg.ki_att[0]:.2f}, Kd: {ctrl_agg.kd_att[0]:.2f}"
        self.right_panel.gain_label.setText(agg_text)
    
    def update_simulation(self):
        """Update both simulations by one timestep."""
        if not self.running:
            return
        
        if self.sim_conservative is None or self.sim_aggressive is None:
            return
        
        # Handle wind gust injection for both
        if self.current_experiment.enable_wind and self.current_experiment.wind_gust_time is not None:
            for sim in [self.sim_conservative, self.sim_aggressive]:
                if abs(sim.time - self.current_experiment.wind_gust_time) < SIM_DT / 2:
                    sim.inject_wind_gust(
                        self.current_experiment.wind_force_vector,
                        magnitude=self.current_experiment.wind_gust_magnitude,
                        duration=self.current_experiment.wind_gust_duration
                    )
        
        # Step both simulators
        self.sim_conservative.step()
        self.sim_aggressive.step()
        
        self.sim_time += SIM_DT
    
    def update_ui(self):
        """Update UI plots and displays."""
        if self.sim_conservative is None or self.sim_aggressive is None:
            return
        
        # Get states
        state_cons = self.sim_conservative.get_state()
        state_agg = self.sim_aggressive.get_state()
        
        # Update data buffers
        self.update_data_buffer(self.data_conservative, state_cons, self.sim_conservative.time)
        self.update_data_buffer(self.data_aggressive, state_agg, self.sim_aggressive.time)
        
        # Update plots
        self.update_plots(self.left_panel, self.data_conservative, "blue")
        self.update_plots(self.right_panel, self.data_aggressive, "red")
        
        # Update simple metrics overlay (based on altitude response)
        self.update_metrics_overlay(self.left_panel, self.data_conservative)
        self.update_metrics_overlay(self.right_panel, self.data_aggressive)
        
        # Update 3D visualizations
        self.update_3d_visualization(self.left_panel, state_cons)
        self.update_3d_visualization(self.right_panel, state_agg)
    
    def reset_camera_views(self):
        """Reset both 3D cameras to the default position."""
        for panel in [self.left_panel, self.right_panel]:
            if not hasattr(panel, "view3d") or panel.view3d is None:
                continue
            view = panel.view3d
            # Reset tracked camera state if using CustomGLViewWidget
            if isinstance(view, CustomGLViewWidget):
                view._camera_distance = 40.0
                view._camera_elevation = 10.0
                view._camera_azimuth = 45.0
            # Reset panel camera distance used by zoom buttons
            panel.camera_distance = 40.0
            view.setCameraPosition(distance=40, elevation=10, azimuth=45)
    
    def create_arrow_mesh(self, direction, length=1.0):
        """Create a 3D arrow mesh pointing in the given direction."""
        # Normalize direction
        dir_norm = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Arrow parameters
        shaft_length = length * 0.7
        shaft_radius = 0.03
        head_length = length * 0.3
        head_radius = 0.08
        n_segments = 8
        
        # Create shaft (cylinder)
        shaft_md = gl.MeshData.cylinder(rows=1, cols=n_segments, radius=[shaft_radius, shaft_radius], length=shaft_length)
        shaft_mesh = gl.GLMeshItem(meshdata=shaft_md, color=(0.3, 0.5, 1.0, 0.7), smooth=False)
        
        # Create head (cone)
        head_md = gl.MeshData.cylinder(rows=1, cols=n_segments, radius=[head_radius, 0.0], length=head_length)
        head_mesh = gl.GLMeshItem(meshdata=head_md, color=(0.3, 0.5, 1.0, 0.7), smooth=False)
        
        # Calculate rotation to align with direction
        z_axis = np.array([0, 0, 1])
        if np.allclose(np.abs(np.dot(dir_norm, z_axis)), 1.0):
            # Special case: direction is along z-axis
            if dir_norm[2] < 0:
                rotation_axis = [1, 0, 0]
                rotation_angle = np.pi
            else:
                rotation_axis = [0, 0, 1]
                rotation_angle = 0
        else:
            rotation_axis = np.cross(z_axis, dir_norm)
            rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-6)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, dir_norm), -1, 1))
        
        # Position and rotate shaft
        shaft_mesh.translate(0, 0, shaft_length / 2)
        if not np.allclose(rotation_angle, 0):
            shaft_mesh.rotate(rotation_angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2], local=False)
        
        # Position and rotate head
        head_mesh.translate(0, 0, shaft_length + head_length / 2)
        if not np.allclose(rotation_angle, 0):
            head_mesh.rotate(rotation_angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2], local=False)
        
        return [shaft_mesh, head_mesh]
    
    def update_wind_vector_field(self, panel):
        """Update wind vector field visualization based on current experiment."""
        if not OPENGL_AVAILABLE or panel.view3d is None:
            return
        
        # Remove old wind arrows
        if hasattr(panel, 'wind_arrows'):
            for arrow_group in panel.wind_arrows:
                for arrow_part in arrow_group:
                    panel.view3d.removeItem(arrow_part)
        panel.wind_arrows = []
        
        # Check if wind is enabled in current experiment
        if self.current_experiment is None or not self.current_experiment.enable_wind:
            return
        
        # Get wind direction
        wind_direction = self.current_experiment.wind_force_vector
        if wind_direction is None:
            return
        
        # Normalize to get direction
        wind_mag = np.linalg.norm(wind_direction)
        if wind_mag < 1e-6:
            return
        
        wind_dir = wind_direction / wind_mag
        
        # Create vector field: arrows in a wider grid pattern
        grid_size = 5  # 5x5 grid in X and Y (wider coverage)
        z_layers = 4  # 4 layers in Z
        spacing = 4.0  # Increased spacing between arrows
        arrow_length = 1.0  # Slightly longer arrows
        
        # Position arrows in positive Z (above ground)
        # Center the grid around z=4, positioned in the lower portion of positive Z space
        z_center = 4.0
        z_range = 3.0  # Spread arrows over 3 units in Z
        
        for x in range(-grid_size, grid_size + 1):
            for y in range(-grid_size, grid_size + 1):
                for z_idx in range(z_layers):
                    # Calculate z position: centered around z_center, spread over z_range
                    z_pos = z_center + (z_idx - z_layers/2 + 0.5) * (z_range / z_layers)
                    arrow_pos = np.array([x * spacing, y * spacing, z_pos])
                    
                    # Create arrow at this position
                    arrow_parts = self.create_arrow_mesh(wind_dir, arrow_length)
                    for arrow_part in arrow_parts:
                        arrow_part.translate(*arrow_pos)
                        panel.view3d.addItem(arrow_part)
                    
                    panel.wind_arrows.append(arrow_parts)

    def update_metrics_overlay(self, panel, data):
        """Compute and display simple step-response metrics for altitude."""
        # Ensure label exists
        if not hasattr(panel, "metrics_label"):
            return
        if self.current_experiment is None:
            panel.metrics_label.setText(
                "Metrics:\n  Rise: -- s\n  Overshoot: -- m\n  Settling: -- s"
            )
            return
        if len(data["time"]) < 5:
            # Not enough data yet
            panel.metrics_label.setText(
                "Metrics:\n  Rise: -- s\n  Overshoot: -- m\n  Settling: -- s"
            )
            return
        
        # Build numpy arrays
        t = np.array(data["time"])
        z = np.array(data["position"][2])
        target_z = float(self.current_experiment.target_position[2])
        
        # Compute metrics using shared analysis utilities
        rise = compute_rise_time(t, z, target_z)
        overshoot = compute_overshoot(t, z, target_z)
        settling = compute_settling_time(t, z, target_z)
        
        panel.metrics_label.setText(
            f"Metrics (Z):\n"
            f"  Rise: {rise:.2f} s\n"
            f"  Overshoot: {overshoot:.2f} m\n"
            f"  Settling: {settling:.2f} s"
        )
    
    def zoom_camera(self, panel, factor):
        """Zoom camera in or out by a factor."""
        if not OPENGL_AVAILABLE or panel.view3d is None:
            return
        
        # Get current camera position
        # pyqtgraph doesn't have a direct way to get current distance,
        # so we'll track it ourselves
        if not hasattr(panel, 'camera_distance'):
            panel.camera_distance = 40.0  # Default distance (zoomed out)
        
        # Update distance
        panel.camera_distance *= factor
        # Clamp distance to reasonable range
        panel.camera_distance = max(2.0, min(100.0, panel.camera_distance))
        
        # Set new camera position with updated distance
        # Keep using the current elevation/azimuth from the view
        current_elevation = getattr(panel.view3d, "_camera_elevation", 10.0)
        current_azimuth = getattr(panel.view3d, "_camera_azimuth", 45.0)
        panel.view3d.setCameraPosition(
            distance=panel.camera_distance,
            elevation=current_elevation,
            azimuth=current_azimuth
        )
    
    def update_data_buffer(self, data, state, time):
        """Update data buffer for one simulator."""
        data['time'].append(time)
        for i in range(3):
            data['position'][i].append(state['position'][i])
            data['attitude'][i].append(np.degrees(state['attitude'][i]))
        
        motor_thrusts = state.get('motor_thrusts', np.zeros(6))
        for i in range(6):
            data['motor'][i].append(motor_thrusts[i])
        
        # Limit history
        if len(data['time']) > PLOT_HISTORY_LENGTH:
            data['time'].pop(0)
            for hist in data['position'] + data['attitude'] + data['motor']:
                if hist:
                    hist.pop(0)
    
    def update_plots(self, panel, data, color):
        """Update plots for one simulator."""
        if len(data['time']) < 2:
            return
        
        # Position plot - recreate legend after clear to show all items
        panel.pos_plot.clear()
        panel.pos_plot.addLegend(offset=(10, 5), labelTextSize='7pt', colCount=1, rowCount=3, spacing=2)
        for i, (label, c) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
            panel.pos_plot.plot(
                data['time'], data['position'][i],
                pen=pg.mkPen(c, width=2), name=label
            )
        
        # Attitude plot - recreate legend after clear to show all items
        panel.att_plot.clear()
        panel.att_plot.addLegend(offset=(10, 5), labelTextSize='7pt', colCount=1, rowCount=3, spacing=2)
        for i, (label, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
            panel.att_plot.plot(
                data['time'], data['attitude'][i],
                pen=pg.mkPen(c, width=2), name=label
            )
        
        # Motor plot
        panel.motor_plot.clear()
        if data['motor'][0]:
            for i in range(6):
                panel.motor_plot.plot(
                    data['time'], data['motor'][i],
                    pen=pg.intColor(i), width=1.5
                )
    
    def update_3d_visualization(self, panel, state):
        """Update 3D visualization of vehicle for one simulator."""
        if not OPENGL_AVAILABLE or panel.view3d is None:
            return
        
        # Ensure grid exists - recreate if missing
        if not hasattr(panel, 'grid') or panel.grid is None:
            grid = gl.GLGridItem()
            grid.scale(5, 5, 1)  # Smaller squares
            grid.translate(0, 0, 0)  # Exactly at z=0
            panel.view3d.addItem(grid)
            panel.grid = grid
        
        pos = np.array(state['position'])
        
        # Everything always at world coordinates - free camera mode only
        # Grid fixed at world z=0
        if panel.grid:
            panel.grid.resetTransform()
            panel.grid.scale(5, 5, 1)
            panel.grid.translate(0, 0, 0)  # Grid at world z=0
        
        if panel.axes:
            panel.axes.resetTransform()
        
        # Vehicle at actual world position
        vehicle_pos = pos
        
        # Remove old vehicle
        if panel.vehicle_mesh is not None:
            panel.view3d.removeItem(panel.vehicle_mesh)
        
        # Create vehicle representation (red sphere)
        md = gl.MeshData.sphere(rows=10, cols=10, radius=0.3)
        panel.vehicle_mesh = gl.GLMeshItem(meshdata=md, color=(1, 0, 0, 0.8), smooth=True)
        # In follow mode, vehicle_pos is already [0,0,0] (centered)
        # In free mode, vehicle_pos is the actual world position
        panel.vehicle_mesh.translate(*vehicle_pos)
        panel.view3d.addItem(panel.vehicle_mesh)
        
        # Remove old motor meshes
        if hasattr(panel, 'motor_meshes'):
            for mesh in panel.motor_meshes:
                panel.view3d.removeItem(mesh)

        panel.motor_meshes = []
        
        # Add motor positions as small green spheres
        if 'attitude_quat' in state:
            R = rotation_matrix_from_quaternion(state['attitude_quat'])
            
            for i, angle in enumerate(HEX_MOTOR_ANGLES):
                motor_pos_body = np.array([
                    HEX_RADIUS * np.cos(angle),
                    HEX_RADIUS * np.sin(angle),
                    0.0
                ])
                # Motors at actual world position
                motor_pos_world = R @ motor_pos_body + pos
                
                md_motor = gl.MeshData.sphere(rows=5, cols=5, radius=0.05)
                motor_mesh = gl.GLMeshItem(meshdata=md_motor, color=(0, 1, 0, 1), smooth=True)
                motor_mesh.translate(*motor_pos_world)
                panel.view3d.addItem(motor_mesh)
                panel.motor_meshes.append(motor_mesh)

