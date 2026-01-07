"""
Main UI window for the 6-DOF multirotor simulator.

Provides:
- 3D visualization of the vehicle
- Live plots of states
- Interactive controls (sliders, buttons)
- Real-time parameter adjustment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QSlider, QLabel, QComboBox, QCheckBox, QLineEdit,
    QGroupBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Try to import OpenGL, but handle gracefully if it fails
try:
    import pyqtgraph.opengl as gl
    OPENGL_AVAILABLE = True
except (ImportError, AttributeError, OSError, Exception) as e:
    # This is normal in WSL2 without proper OpenGL setup - not a critical error
    print("=" * 70)
    print("NOTE: OpenGL 3D visualization is not available.")
    print("This is normal in WSL2 without X11/OpenGL setup.")
    print("")
    print("The simulator will work perfectly with:")
    print("  ✓ Live plots (position, attitude, motor thrusts)")
    print("  ✓ All controls and interactive features")
    print("  ✓ Full simulation functionality")
    print("  ✓ Real-time parameter adjustment")
    print("")
    print("Only the 3D visualization is disabled - everything else works!")
    print("=" * 70)
    OPENGL_AVAILABLE = False
    gl = None

from src.config import SIM_DT, UI_UPDATE_RATE, PLOT_HISTORY_LENGTH


class SimulatorUI(QMainWindow):
    """Main UI window for the simulator."""
    
    def __init__(self, simulator):
        """
        Initialize UI.
        
        Args:
            simulator: Simulator object with step() method
        """
        super().__init__()
        self.simulator = simulator
        
        self.setWindowTitle("6-DOF Multirotor Flight Simulator")
        self.setGeometry(100, 100, 1200, 800)  # Smaller initial size
        
        # Data buffers for plotting
        self.time_history = []
        self.position_history = [[], [], []]  # x, y, z
        self.attitude_history = [[], [], []]  # roll, pitch, yaw
        self.motor_history = [[] for _ in range(6)]
        self.wind_history = [[], [], []]
        
        # Camera follow mode (default: enabled)
        self.follow_mode = True
        
        self.setup_ui()
        
        # Timer for simulation and UI updates
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.start(int(1000 * SIM_DT))  # Run at simulation rate
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(int(1000 / UI_UPDATE_RATE))  # Update UI at 30 Hz
    
    def setup_ui(self):
        """Set up the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Main horizontal splitter: Left (3D + Plots) | Right (Controls)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # Left side: Vertical splitter for 3D view (top) and plots (bottom)
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        
        # 3D visualization (top of left side)
        if OPENGL_AVAILABLE:
            self.view3d = gl.GLViewWidget()
            self.view3d.setCameraPosition(distance=10, elevation=20, azimuth=45)
            self.setup_3d_scene()
            self.view3d.setMinimumSize(400, 300)  # Minimum size
            left_splitter.addWidget(self.view3d)
        else:
            # Fallback: show a label instead of 3D view
            no_gl_label = QLabel("3D visualization unavailable\n(OpenGL not available)\n\nPlots and controls are still functional.")
            no_gl_label.setAlignment(Qt.AlignCenter)
            no_gl_label.setStyleSheet("font-size: 14px; padding: 20px;")
            no_gl_label.setMinimumSize(400, 300)
            left_splitter.addWidget(no_gl_label)
            self.view3d = None
        
        # Plots section (bottom of left side)
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(5, 5, 5, 5)
        
        plots_group = QGroupBox("Live Plots")
        plots_group_layout = QVBoxLayout()
        
        # Position plot
        self.pos_plot = pg.PlotWidget(title="Position")
        self.pos_plot.addLegend()
        self.pos_plot.setLabel('left', 'Position (m)')
        self.pos_plot.setLabel('bottom', 'Time (s)')
        self.pos_plot.setMinimumHeight(150)  # Minimum height
        self.pos_curves = [
            self.pos_plot.plot(pen='r', name='X'),
            self.pos_plot.plot(pen='g', name='Y'),
            self.pos_plot.plot(pen='b', name='Z')
        ]
        plots_group_layout.addWidget(self.pos_plot)
        
        # Attitude plot
        self.att_plot = pg.PlotWidget(title="Attitude")
        self.att_plot.addLegend()
        self.att_plot.setLabel('left', 'Angle (deg)')
        self.att_plot.setLabel('bottom', 'Time (s)')
        self.att_plot.setMinimumHeight(150)
        self.att_curves = [
            self.att_plot.plot(pen='r', name='Roll'),
            self.att_plot.plot(pen='g', name='Pitch'),
            self.att_plot.plot(pen='b', name='Yaw')
        ]
        plots_group_layout.addWidget(self.att_plot)
        
        # Motor thrusts plot
        self.motor_plot = pg.PlotWidget(title="Motor Thrusts")
        self.motor_plot.setLabel('left', 'Thrust (N)')
        self.motor_plot.setLabel('bottom', 'Time (s)')
        self.motor_plot.setMinimumHeight(150)
        self.motor_curves = [self.motor_plot.plot(pen=pg.intColor(i)) for i in range(6)]
        plots_group_layout.addWidget(self.motor_plot)
        
        plots_group.setLayout(plots_group_layout)
        plots_layout.addWidget(plots_group)
        
        left_splitter.addWidget(plots_widget)
        left_splitter.setStretchFactor(0, 1)  # 3D view
        left_splitter.setStretchFactor(1, 1)  # Plots
        left_splitter.setSizes([400, 400])  # Initial sizes
        
        main_splitter.addWidget(left_splitter)
        
        # Right panel: All controls/inputs
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(450)  # Minimum width for controls
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Controls section
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        controls_group = QGroupBox("Controls")
        controls_group_layout = QVBoxLayout()
        controls_group_layout.setSpacing(10)  # Add spacing between sections
        
        # Controller mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Controller:"))
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(["Cascaded PID", "LQR Inner Loop"])
        self.controller_combo.currentIndexChanged.connect(self.on_controller_changed)
        mode_layout.addWidget(self.controller_combo)
        controls_group_layout.addLayout(mode_layout)
        
        # PID gains sliders
        self.setup_pid_sliders(controls_group_layout)
        
        # Setpoint controls
        setpoint_group = QGroupBox("Setpoint")
        setpoint_group.setMinimumHeight(120)  # Make it taller
        setpoint_layout = QGridLayout()
        setpoint_layout.setSpacing(8)  # Add spacing
        setpoint_layout.setContentsMargins(10, 15, 10, 15)  # Add padding
        
        setpoint_layout.addWidget(QLabel("Altitude (m):"), 0, 0)
        self.altitude_input = QLineEdit("5.0")
        self.altitude_input.setMinimumHeight(25)  # Make input taller
        self.altitude_input.returnPressed.connect(self.on_setpoint_changed)
        setpoint_layout.addWidget(self.altitude_input, 0, 1)
        
        setpoint_layout.addWidget(QLabel("X (m):"), 1, 0)
        self.x_input = QLineEdit("0.0")
        self.x_input.setMinimumHeight(25)
        self.x_input.returnPressed.connect(self.on_setpoint_changed)
        setpoint_layout.addWidget(self.x_input, 1, 1)
        
        setpoint_layout.addWidget(QLabel("Y (m):"), 2, 0)
        self.y_input = QLineEdit("0.0")
        self.y_input.setMinimumHeight(25)
        self.y_input.returnPressed.connect(self.on_setpoint_changed)
        setpoint_layout.addWidget(self.y_input, 2, 1)
        
        setpoint_group.setLayout(setpoint_layout)
        controls_group_layout.addWidget(setpoint_group)
        
        # Wind gust buttons
        wind_group = QGroupBox("Wind Gusts")
        wind_group.setMinimumHeight(180)  # Make it taller
        wind_layout = QGridLayout()
        wind_layout.setSpacing(8)  # Add spacing between buttons
        wind_layout.setContentsMargins(10, 15, 10, 15)  # Add padding
        
        self.gust_buttons = {}
        for i, (name, direction) in enumerate([
            ("+X", [1, 0, 0]), ("-X", [-1, 0, 0]),
            ("+Y", [0, 1, 0]), ("-Y", [0, -1, 0]),
            ("+Z", [0, 0, 1]), ("-Z", [0, 0, -1]),
            ("Random", None)
        ]):
            btn = QPushButton(name)
            btn.setMinimumHeight(35)  # Make buttons taller
            btn.setMinimumWidth(80)   # Make buttons wider
            btn.clicked.connect(lambda checked, d=direction: self.inject_gust(d))
            wind_layout.addWidget(btn, i // 2, i % 2)
            self.gust_buttons[name] = btn
        
        wind_group.setLayout(wind_layout)
        controls_group_layout.addWidget(wind_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        self.noise_checkbox = QCheckBox("Sensor Noise")
        self.noise_checkbox.setChecked(True)
        self.noise_checkbox.stateChanged.connect(self.on_noise_changed)
        options_layout.addWidget(self.noise_checkbox)
        
        self.integral_checkbox = QCheckBox("Integral Action")
        self.integral_checkbox.setChecked(True)
        self.integral_checkbox.stateChanged.connect(self.on_integral_changed)
        options_layout.addWidget(self.integral_checkbox)
        
        self.follow_mode_checkbox = QCheckBox("Follow Mode (Camera follows drone)")
        self.follow_mode_checkbox.setChecked(True)
        self.follow_mode_checkbox.stateChanged.connect(self.on_follow_mode_changed)
        options_layout.addWidget(self.follow_mode_checkbox)
        
        options_group.setLayout(options_layout)
        controls_group_layout.addWidget(options_group)
        
        # Export button
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.on_export)
        controls_group_layout.addWidget(export_btn)
        
        controls_group.setLayout(controls_group_layout)
        controls_layout.addWidget(controls_group)
        
        controls_widget.setLayout(controls_layout)
        controls_scroll.setWidget(controls_widget)
        
        main_splitter.addWidget(controls_scroll)
        main_splitter.setStretchFactor(0, 2)  # Left side (3D + plots)
        main_splitter.setStretchFactor(1, 1)  # Right side (controls)
        main_splitter.setSizes([800, 500])  # Initial sizes
        
        main_layout.addWidget(main_splitter)
    
    def setup_3d_scene(self):
        """Set up 3D visualization scene."""
        if not OPENGL_AVAILABLE or self.view3d is None:
            return
        
        # Clear existing items
        self.view3d.clear()
        
        # Add grid (store reference for follow mode)
        self.grid = gl.GLGridItem()
        self.grid.scale(1, 1, 1)
        self.view3d.addItem(self.grid)
        
        # Add ground plane (solid surface at z=0)
        # Create a large flat surface to represent the ground
        ground_size = 20.0
        ground_verts = np.array([
            [-ground_size, -ground_size, 0],
            [ground_size, -ground_size, 0],
            [ground_size, ground_size, 0],
            [-ground_size, ground_size, 0]
        ])
        ground_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        ground_colors = np.array([
            [0.3, 0.5, 0.3, 0.5],  # Semi-transparent green
            [0.3, 0.5, 0.3, 0.5]
        ])
        self.ground_plane = gl.GLMeshItem(
            vertexes=ground_verts,
            faces=ground_faces,
            faceColors=ground_colors,
            smooth=False,
            drawEdges=False,
            drawFaces=True
        )
        self.view3d.addItem(self.ground_plane)
        
        # Add axes (store reference for follow mode)
        self.axes = gl.GLAxisItem()
        self.axes.setSize(2, 2, 2)
        self.view3d.addItem(self.axes)
        
        # Vehicle visualization (will be updated)
        self.vehicle_mesh = None
        self.last_drone_pos = np.array([0.0, 0.0, 0.0])  # Track last position for smooth following
    
    def setup_pid_sliders(self, layout):
        """Set up PID gain sliders."""
        pid_group = QGroupBox("PID Gains")
        pid_group.setMinimumHeight(280)  # Make it significantly taller
        pid_layout = QGridLayout()
        pid_layout.setSpacing(10)  # Add more spacing between rows
        pid_layout.setContentsMargins(10, 15, 10, 15)  # Add padding
        pid_layout.setColumnStretch(0, 1)  # Label column - minimal space
        pid_layout.setColumnStretch(1, 3)  # Slider column - most space
        pid_layout.setColumnStretch(2, 1)  # Value label - minimal space
        
        # Position gains (using z-axis values as defaults since they're most important)
        pid_layout.addWidget(QLabel("Position Kp:"), 0, 0)
        self.kp_pos_slider = self.create_slider(0, 10, 5, 0.1)  # Default: 5.0 (z-axis value)
        self.kp_pos_slider.setMinimumWidth(150)  # Minimum width for slider
        self.kp_pos_label = QLabel("5.0")
        self.kp_pos_label.setMinimumWidth(40)  # Fixed width for value
        pid_layout.addWidget(self.kp_pos_slider, 0, 1)
        pid_layout.addWidget(self.kp_pos_label, 0, 2)
        
        pid_layout.addWidget(QLabel("Position Ki:"), 1, 0)
        self.ki_pos_slider = self.create_slider(0, 2, 0.4, 0.01)  # Default: 0.4 (z-axis value)
        self.ki_pos_slider.setMinimumWidth(150)
        self.ki_pos_label = QLabel("0.4")
        self.ki_pos_label.setMinimumWidth(40)
        pid_layout.addWidget(self.ki_pos_slider, 1, 1)
        pid_layout.addWidget(self.ki_pos_label, 1, 2)
        
        pid_layout.addWidget(QLabel("Position Kd:"), 2, 0)
        self.kd_pos_slider = self.create_slider(0, 5, 2.0, 0.1)  # Default: 2.0 (z-axis value)
        self.kd_pos_slider.setMinimumWidth(150)
        self.kd_pos_label = QLabel("2.0")
        self.kd_pos_label.setMinimumWidth(40)
        pid_layout.addWidget(self.kd_pos_slider, 2, 1)
        pid_layout.addWidget(self.kd_pos_label, 2, 2)
        
        # Attitude gains (using roll/pitch values as defaults)
        pid_layout.addWidget(QLabel("Attitude Kp:"), 3, 0)
        self.kp_att_slider = self.create_slider(0, 20, 8, 0.5)  # Default: 8.0 (roll/pitch value)
        self.kp_att_slider.setMinimumWidth(150)
        self.kp_att_label = QLabel("8.0")
        self.kp_att_label.setMinimumWidth(40)
        pid_layout.addWidget(self.kp_att_slider, 3, 1)
        pid_layout.addWidget(self.kp_att_label, 3, 2)
        
        pid_layout.addWidget(QLabel("Attitude Ki:"), 4, 0)
        self.ki_att_slider = self.create_slider(0, 5, 0.5, 0.1)  # Default: 0.5 (roll/pitch value)
        self.ki_att_slider.setMinimumWidth(150)
        self.ki_att_label = QLabel("0.5")
        self.ki_att_label.setMinimumWidth(40)
        pid_layout.addWidget(self.ki_att_slider, 4, 1)
        pid_layout.addWidget(self.ki_att_label, 4, 2)
        
        pid_layout.addWidget(QLabel("Attitude Kd:"), 5, 0)
        self.kd_att_slider = self.create_slider(0, 10, 2.0, 0.1)  # Default: 2.0 (roll/pitch value)
        self.kd_att_slider.setMinimumWidth(150)
        self.kd_att_label = QLabel("2.0")
        self.kd_att_label.setMinimumWidth(40)
        pid_layout.addWidget(self.kd_att_slider, 5, 1)
        pid_layout.addWidget(self.kd_att_label, 5, 2)
        
        # Connect sliders
        self.kp_pos_slider.valueChanged.connect(lambda v: self.update_gain('kp_pos', v))
        self.ki_pos_slider.valueChanged.connect(lambda v: self.update_gain('ki_pos', v))
        self.kd_pos_slider.valueChanged.connect(lambda v: self.update_gain('kd_pos', v))
        self.kp_att_slider.valueChanged.connect(lambda v: self.update_gain('kp_att', v))
        self.ki_att_slider.valueChanged.connect(lambda v: self.update_gain('ki_att', v))
        self.kd_att_slider.valueChanged.connect(lambda v: self.update_gain('kd_att', v))
        
        pid_group.setLayout(pid_layout)
        layout.addWidget(pid_group)
    
    def create_slider(self, min_val, max_val, default, step):
        """Create a slider with specified range."""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(default / step))
        slider.setMinimumHeight(30)  # Make slider taller for easier interaction
        return slider
    
    def update_gain(self, gain_name, value):
        """Update controller gain from slider."""
        step = 0.1
        if 'ki' in gain_name:
            step = 0.01
        gain_value = value * step
        
        # Update label
        label_map = {
            'kp_pos': self.kp_pos_label,
            'ki_pos': self.ki_pos_label,
            'kd_pos': self.kd_pos_label,
            'kp_att': self.kp_att_label,
            'ki_att': self.ki_att_label,
            'kd_att': self.kd_att_label
        }
        if gain_name in label_map:
            label_map[gain_name].setText(f"{gain_value:.2f}")
        
        # Update controller
        self.simulator.update_controller_gain(gain_name, gain_value)
    
    def on_controller_changed(self, index):
        """Handle controller mode change."""
        mode = ["pid", "lqr"][index]
        self.simulator.set_controller_mode(mode)
    
    def on_setpoint_changed(self):
        """Handle setpoint change."""
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.altitude_input.text())
            self.simulator.set_target([x, y, z])
        except ValueError:
            pass
    
    def inject_gust(self, direction):
        """Inject wind gust."""
        if direction is None:
            # Random gust
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0, 0])
        self.simulator.inject_wind_gust(direction)
    
    def on_noise_changed(self, state):
        """Handle sensor noise toggle."""
        enabled = state == Qt.Checked
        self.simulator.set_sensor_noise(enabled)
    
    def on_integral_changed(self, state):
        """Handle integral action toggle."""
        enabled = state == Qt.Checked
        self.simulator.set_integral_enabled(enabled)
    
    def on_follow_mode_changed(self, state):
        """Handle follow mode toggle."""
        self.follow_mode = state == Qt.Checked
        # Reset scene when switching modes
        if OPENGL_AVAILABLE and self.view3d is not None:
            self.grid.resetTransform()
            if hasattr(self, 'ground_plane'):
                self.ground_plane.resetTransform()
            self.axes.resetTransform()
            # Reset camera to default position
            self.view3d.setCameraPosition(distance=10, elevation=20, azimuth=45)
    
    def on_export(self):
        """Export simulation report."""
        self.simulator.export_report()
    
    def update_simulation(self):
        """Update simulation (called at simulation rate)."""
        self.simulator.step()
    
    def update_ui(self):
        """Update UI (called at UI update rate)."""
        # Get current state
        state = self.simulator.get_state()
        time = self.simulator.get_time()
        
        # Update data buffers
        self.time_history.append(time)
        for i in range(3):
            self.position_history[i].append(state['position'][i])
            self.attitude_history[i].append(np.degrees(state['attitude'][i]))
            self.wind_history[i].append(state.get('wind_force', [0, 0, 0])[i])
        
        motor_thrusts = state.get('motor_thrusts', np.zeros(6))
        for i in range(6):
            self.motor_history[i].append(motor_thrusts[i])
        
        # Limit history length
        if len(self.time_history) > PLOT_HISTORY_LENGTH:
            self.time_history.pop(0)
            for hist in self.position_history + self.attitude_history + self.motor_history + self.wind_history:
                if hist:
                    hist.pop(0)
        
        # Update plots (only if we have enough data and window is visible)
        if len(self.time_history) > 1 and self.isVisible():
            try:
                for i in range(3):
                    self.pos_curves[i].setData(self.time_history, self.position_history[i])
                    self.att_curves[i].setData(self.time_history, self.attitude_history[i])
                
                if self.motor_history[0]:
                    for i in range(6):
                        self.motor_curves[i].setData(self.time_history, self.motor_history[i])
            except Exception as e:
                # Silently handle plot update errors (can happen during resize)
                pass
        
        # Update 3D visualization
        self.update_3d_visualization(state)
    
    def update_3d_visualization(self, state):
        """Update 3D visualization of vehicle."""
        if not OPENGL_AVAILABLE or self.view3d is None:
            return
        
        # Remove old vehicle
        if self.vehicle_mesh is not None:
            self.view3d.removeItem(self.vehicle_mesh)
        
        # Create simple vehicle representation
        pos = np.array(state['position'])
        
        # Update scene translation for follow mode
        if self.follow_mode:
            # Translate grid, ground, and axes so drone appears at origin
            # This makes the camera follow the drone naturally
            translation = -pos
            self.grid.resetTransform()
            self.grid.translate(*translation)
            self.ground_plane.resetTransform()
            self.ground_plane.translate(*translation)
            self.axes.resetTransform()
            self.axes.translate(*translation)
            
            # Reset camera to default position (looking at origin, where drone now appears)
            self.view3d.setCameraPosition(distance=10, elevation=20, azimuth=45)
            
            # Vehicle appears at origin in follow mode
            vehicle_pos = np.array([0.0, 0.0, 0.0])
        else:
            # Free mode: reset grid, ground, and axes to origin, vehicle at actual position
            self.grid.resetTransform()
            self.ground_plane.resetTransform()
            self.axes.resetTransform()
            vehicle_pos = pos
        
        # For simplicity, use a sphere to represent the vehicle
        # Position it at the vehicle location
        md = gl.MeshData.sphere(rows=10, cols=10, radius=0.3)
        self.vehicle_mesh = gl.GLMeshItem(meshdata=md, color=(1, 0, 0, 0.8), smooth=True)
        self.vehicle_mesh.translate(*vehicle_pos)
        self.view3d.addItem(self.vehicle_mesh)
        
        # Optionally add motor positions as small spheres
        if hasattr(self, 'motor_meshes'):
            for mesh in self.motor_meshes:
                self.view3d.removeItem(mesh)
        
        self.motor_meshes = []
        if 'attitude_quat' in state:
            from src.utils.math3d import rotation_matrix_from_quaternion
            from src.config import HEX_RADIUS, HEX_MOTOR_ANGLES
            R = rotation_matrix_from_quaternion(state['attitude_quat'])
            
            for i, angle in enumerate(HEX_MOTOR_ANGLES):
                motor_pos_body = np.array([
                    HEX_RADIUS * np.cos(angle),
                    HEX_RADIUS * np.sin(angle),
                    0.0
                ])
                if self.follow_mode:
                    # In follow mode, motors relative to origin (where drone is)
                    motor_pos_world = R @ motor_pos_body
                else:
                    # In free mode, motors at actual world position
                    motor_pos_world = R @ motor_pos_body + pos
                
                md_motor = gl.MeshData.sphere(rows=5, cols=5, radius=0.05)
                motor_mesh = gl.GLMeshItem(meshdata=md_motor, color=(0, 1, 0, 1), smooth=True)
                motor_mesh.translate(*motor_pos_world)
                self.view3d.addItem(motor_mesh)
                self.motor_meshes.append(motor_mesh)
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Force layout update after resize
        if hasattr(self, 'view3d') and self.view3d is not None:
            self.view3d.update()

