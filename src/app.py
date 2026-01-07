"""
Main application entry point for the 6-DOF multirotor simulator.

Run with: python -m src.app
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from src.ui_comparison import ComparisonUI

# Check if running headless
def is_headless():
    """Check if running in headless mode (no display)."""
    # On Windows, DISPLAY is not set, so only check QT_QPA_PLATFORM
    if sys.platform == 'win32':
        return os.environ.get('QT_QPA_PLATFORM') == 'offscreen'
    # On Unix/Linux, check both DISPLAY and QT_QPA_PLATFORM
    return os.environ.get('DISPLAY') is None or os.environ.get('QT_QPA_PLATFORM') == 'offscreen'

if __name__ == "__main__":
    # Set environment variable to help with OpenGL initialization issues
    # This prevents PyOpenGL from trying to create a context at import time
    os.environ.setdefault('QT_OPENGL', 'desktop')
    
    app = QApplication(sys.argv)
    
    if is_headless():
        print("Running in headless mode.")
        print("For headless experiments, use: python simulation.py --experiment <name>")
        print("GUI mode requires a display.")
        sys.exit(0)
    
    # Create comparison UI
    ui = ComparisonUI()
    ui.show()
    
    print("Flight Control Experiment Comparison started!")
    print("Controls:")
    print("  - Select experiment scenario from dropdown")
    print("  - Click 'Start Simulation' to run comparison")
    print("  - Compare conservative vs aggressive gains side-by-side")
    print("  - Switch scenarios anytime to see different behaviors")
    
    sys.exit(app.exec_())

