import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.ndimage import laplace
import matplotlib
import time
import os
import sys
from pathlib import Path
import json
import warnings

# Import GPU acceleration libraries with fallbacks
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch not available, using CPU only")

# Import PyQt6 for professional UI
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSplitter, QFrame, QLabel, QComboBox, QSlider, QPushButton,
                               QGroupBox, QFormLayout, QFileDialog, QStatusBar, QCheckBox,
                               QSizePolicy, QProgressBar, QSpacerItem, QDialog, 
                               QProgressDialog, QMessageBox)
    from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QLinearGradient, QBrush, QAction, QKeySequence
    PYQT_AVAILABLE = True
    print("PyQt6 available - using professional UI")
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Falling back to matplotlib-only UI.")

matplotlib.rcParams['toolbar'] = 'None'

class GPURainVisualizer:
    """
    Professional rain simulation with PyQt6 UI and GPU acceleration.
    Features a modern interface, high-quality rendering, and optimized performance.
    """
    
    def __init__(self, parent=None):
        self.parent = parent
        self.use_gpu = TORCH_AVAILABLE and DEVICE.type == "cuda"
        print(f"GPU acceleration {'enabled' if self.use_gpu else 'disabled'}")
        
        # Default settings
        self.current_resolution = 512
        self.size = self.current_resolution
        self.layers = 2
        self.style = 'rain'
        self.frame_count = 0
        self.recording = False
        self.recorded_frames = []
        self.max_record_frames = 200
        self.fps = 60  # Target FPS
        self.last_frame_time = time.time()
        self.fps_counter = 0
        self.display_fps = 0
        self.paused = False
        
        # Style Presets
        self.style_params = {
            'rain': {
                'diff_rates': np.array([0.14, 0.14]),
                'damping': 0.998, 'velocity_decay': 0.98,
                'base_color': 0.05, 'droplet_interval': 10,
                'drop_velocity_range': (1.0, 1.5),
                'visualization': 'water'
            },
            'storm': {
                'diff_rates': np.array([0.12, 0.12]),
                'damping': 0.995, 'velocity_decay': 0.95,
                'base_color': 0.02, 'droplet_interval': 5,
                'drop_velocity_range': (2.0, 3.5),
                'visualization': 'water'
            },
            'ink_rain': {
                'diff_rates': np.array([0.08, 0.08]),
                'damping': 0.999, 'velocity_decay': 1.0,
                'base_color': 0.95, 'droplet_interval': 15,
                'drop_velocity_range': (0, 0),
                'visualization': 'ink'
            },
            'mist': {
                'diff_rates': np.array([0.18, 0.18]),
                'damping': 0.992, 'velocity_decay': 0.99,
                'base_color': 0.15, 'droplet_interval': 3,
                'drop_velocity_range': (0.5, 0.8),
                'visualization': 'water'
            }
        }
        self.params = self.style_params[self.style]
        
        # Simulation Parameters
        self.F = 0.035
        self.damping = self.params['damping']
        self.droplet_size = 8
        self.rain_intensity = 50
        self.show_velocity = False
        self.anti_aliasing = True
        
        # Initialize state
        self.initialize_state()
        
        # Professional theme colors
        self.theme = {
            'primary': '#1e88e5',
            'secondary': '#5e35b1',
            'success': '#43a047',
            'warning': '#fb8c00',
            'danger': '#e53935',
            'background': '#f5f5f5',
            'surface': '#ffffff',
            'text': '#212121',
            'border': '#e0e0e0',
            'dark_primary': '#1565c0',
            'dark_secondary': '#4527a0',
            'dark_success': '#2e7d32',
            'dark_warning': '#f57c00',
            'dark_danger': '#c62828'
        }
        
        # Setup UI
        if PYQT_AVAILABLE:
            self.setup_professional_ui()
        else:
            self.setup_matplotlib_ui()

    def initialize_state(self):
        """Initialize simulation state using GPU if available"""
        if self.use_gpu:
            self.state = {
                'R': torch.full((self.layers, self.size, self.size), 
                               self.params['base_color'], device=DEVICE, dtype=torch.float32),
                'G': torch.full((self.layers, self.size, self.size), 
                               self.params['base_color'], device=DEVICE, dtype=torch.float32),
                'B': torch.full((self.layers, self.size, self.size), 
                               self.params['base_color'], device=DEVICE, dtype=torch.float32),
                'wave': torch.zeros((self.layers, self.size, self.size), 
                                   device=DEVICE, dtype=torch.float32),
                'wave_prev': torch.zeros((self.layers, self.size, self.size), 
                                        device=DEVICE, dtype=torch.float32),
                'vx': torch.zeros((self.layers, self.size, self.size), 
                                 device=DEVICE, dtype=torch.float32),
                'vy': torch.zeros((self.layers, self.size, self.size), 
                                 device=DEVICE, dtype=torch.float32),
            }
        else:
            self.state = {
                'R': np.full((self.layers, self.size, self.size), self.params['base_color']),
                'G': np.full((self.layers, self.size, self.size), self.params['base_color']),
                'B': np.full((self.layers, self.size, self.size), self.params['base_color']),
                'wave': np.zeros((self.layers, self.size, self.size)),
                'wave_prev': np.zeros((self.layers, self.size, self.size)),
                'vx': np.zeros((self.layers, self.size, self.size)),
                'vy': np.zeros((self.layers, self.size, self.size)),
            }

    def setup_professional_ui(self):
        """Create professional PyQt6-based UI"""
        self.app = QApplication(sys.argv) if self.parent is None else None
        
        # Set application style
        self.app.setStyle("Fusion")
        
        # Set palette for consistent theming
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self.theme['background']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self.theme['text']))
        palette.setColor(QPalette.ColorRole.Base, QColor(self.theme['surface']))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(self.theme['background']))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(self.theme['surface']))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(self.theme['text']))
        palette.setColor(QPalette.ColorRole.Text, QColor(self.theme['text']))
        palette.setColor(QPalette.ColorRole.Button, QColor(self.theme['surface']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(self.theme['text']))
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(self.theme['primary']))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.theme['primary']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.theme['surface']))
        self.app.setPalette(palette)
        
        # Set font
        font = QFont("Segoe UI", 9)
        self.app.setFont(font)
        
        self.main_window = QMainWindow() if self.parent is None else QWidget()
        
        if self.parent is None:
            self.main_window.setWindowTitle("AQUAFLUX - Professional Rain Simulator")
            self.main_window.setGeometry(100, 100, 1500, 950)
            self.main_window.setWindowIcon(QIcon(':/icons/rain.png'))  # Placeholder for icon
            
            # Create menu bar
            menubar = self.main_window.menuBar()
            
            # File menu
            file_menu = menubar.addMenu('&File')
            
            save_frame_action = QAction('&Save Frame', self.main_window)
            save_frame_action.setShortcut(QKeySequence.StandardKey.Save)
            save_frame_action.triggered.connect(self.save_current_frame)
            file_menu.addAction(save_frame_action)
            
            save_gif_action = QAction('Save &GIF Animation', self.main_window)
            save_gif_action.triggered.connect(lambda: self.toggle_recording())
            file_menu.addAction(save_gif_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction('E&xit', self.main_window)
            exit_action.setShortcut(QKeySequence.StandardKey.Quit)
            exit_action.triggered.connect(self.main_window.close)
            file_menu.addAction(exit_action)
            
            # View menu
            view_menu = menubar.addMenu('&View')
            
            toggle_pause_action = QAction('&Pause/Resume Simulation', self.main_window)
            toggle_pause_action.setShortcut(Qt.Key.Key_Space)
            toggle_pause_action.triggered.connect(self.toggle_pause)
            view_menu.addAction(toggle_pause_action)
            
            view_menu.addSeparator()
            
            reset_action = QAction('&Reset Simulation', self.main_window)
            reset_action.setShortcut(Qt.Key.Key_R)
            reset_action.triggered.connect(self.reset_simulation)
            view_menu.addAction(reset_action)
            
            # Help menu
            help_menu = menubar.addMenu('&Help')
            
            about_action = QAction('&About AQUAFLUX', self.main_window)
            about_action.triggered.connect(self.show_about_dialog)
            help_menu.addAction(about_action)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitters for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Controls
        left_panel = QFrame()
        left_panel.setFixedWidth(350)
        left_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme['surface']};
                border-right: 1px solid {self.theme['border']};
            }}
            QGroupBox {{
                border: 1px solid {self.theme['border']};
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                font-size: 12px;
            }}
        """)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)
        
        # Header with logo
        header_layout = QHBoxLayout()
        
        # Logo placeholder
        logo_label = QLabel()
        logo_label.setFixedSize(40, 40)
        logo_label.setStyleSheet(f"""
            background-color: {self.theme['primary']};
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 16px;
            qproperty-alignment: AlignCenter;
        """)
        logo_label.setText("💧")
        header_layout.addWidget(logo_label)
        
        header_text = QLabel("AQUAFLUX\nSIMULATOR")
        header_text.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #1e88e5;
            }
            QLabel:last-child {
                font-size: 12px;
                font-weight: normal;
                color: #757575;
            }
        """)
        header_layout.addWidget(header_text)
        header_layout.addStretch()
        
        left_layout.addLayout(header_layout)
        
        # System info panel
        system_group = QGroupBox("System Information")
        system_layout = QVBoxLayout()
        
        gpu_status = "GPU Enabled" if self.use_gpu else "GPU Disabled"
        gpu_color = self.theme['success'] if self.use_gpu else self.theme['warning']
        
        self.gpu_label = QLabel(f"<b>GPU:</b> <span style='color:{gpu_color}'>{gpu_status}</span>")
        system_layout.addWidget(self.gpu_label)
        
        self.memory_label = QLabel("<b>Memory:</b> Calculating...")
        system_layout.addWidget(self.memory_label)
        
        self.fps_label = QLabel(f"<b>FPS:</b> {self.display_fps:.1f}")
        system_layout.addWidget(self.fps_label)
        
        system_group.setLayout(system_layout)
        left_layout.addWidget(system_group)
        
        # Style selection
        style_group = QGroupBox("Simulation Style")
        style_layout = QVBoxLayout()
        
        self.style_combo = QComboBox()
        self.style_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 5px;
                border: 1px solid {self.theme['border']};
                border-radius: 4px;
            }}
        """)
        for style_name in self.style_params.keys():
            display_name = style_name.replace('_', ' ').title()
            self.style_combo.addItem(display_name, style_name)
        self.style_combo.setCurrentIndex(0)
        self.style_combo.currentIndexChanged.connect(self.change_style_combo)
        style_layout.addWidget(self.style_combo)
        
        # Style description
        self.style_description = QLabel("Gentle rain with soft ripples and natural water coloration.")
        self.style_description.setWordWrap(True)
        self.style_description.setStyleSheet("color: #757575; font-size: 11px;")
        style_layout.addWidget(self.style_description)
        
        style_group.setLayout(style_layout)
        left_layout.addWidget(style_group)
        
        # Resolution selection
        res_group = QGroupBox("Resolution")
        res_layout = QVBoxLayout()
        
        self.res_combo = QComboBox()
        self.res_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 5px;
                border: 1px solid {self.theme['border']};
                border-radius: 4px;
            }}
        """)
        self.res_combo.addItem("512 × 512 (Standard)", 512)
        self.res_combo.addItem("1024 × 1024 (High Quality)", 1024)
        self.res_combo.addItem("2048 × 2048 (Ultra HD)", 2048)
        if self.use_gpu:
            self.res_combo.addItem("4096 × 4096 (Maximum Detail)", 4096)
        self.res_combo.setCurrentIndex(0)
        self.res_combo.currentIndexChanged.connect(self.change_resolution_combo)
        res_layout.addWidget(self.res_combo)
        
        res_info = QLabel("Higher resolutions require more GPU memory and processing power.")
        res_info.setWordWrap(True)
        res_info.setStyleSheet("color: #757575; font-size: 11px;")
        res_layout.addWidget(res_info)
        
        res_group.setLayout(res_layout)
        left_layout.addWidget(res_group)
        
        # Control buttons
        btn_group = QGroupBox("Simulation Controls")
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        
        # Play/Pause button
        self.play_btn = QPushButton("⏸ Pause Simulation")
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['warning']};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.theme['dark_warning']};
            }}
        """)
        self.play_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.play_btn)
        
        # Record button
        self.record_btn = QPushButton("⏺ Record GIF")
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['primary']};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.theme['dark_primary']};
            }}
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        btn_layout.addWidget(self.record_btn)
        
        # Save frame button
        self.save_btn = QPushButton("💾 Save Frame")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['success']};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.theme['dark_success']};
            }}
        """)
        self.save_btn.clicked.connect(self.save_current_frame)
        btn_layout.addWidget(self.save_btn)
        
        # Reset button
        self.reset_btn = QPushButton("🔄 Reset Simulation")
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['secondary']};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.theme['dark_secondary']};
            }}
        """)
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.reset_btn)
        
        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group, 2)
        
        # Parameters
        param_group = QGroupBox("Simulation Parameters")
        param_layout = QFormLayout()
        param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Rain intensity
        intensity_layout = QHBoxLayout()
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setMinimum(10)
        self.intensity_slider.setMaximum(200)
        self.intensity_slider.setValue(self.rain_intensity)
        self.intensity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.intensity_slider.setTickInterval(20)
        self.intensity_slider.valueChanged.connect(self.update_intensity)
        self.intensity_label = QLabel(f"Intensity: {self.rain_intensity}")
        intensity_layout.addWidget(self.intensity_label)
        intensity_layout.addWidget(self.intensity_slider)
        param_layout.addRow(intensity_layout)
        
        # Damping
        damping_layout = QHBoxLayout()
        self.damping_slider = QSlider(Qt.Orientation.Horizontal)
        self.damping_slider.setMinimum(990)
        self.damping_slider.setMaximum(999)
        self.damping_slider.setValue(int(self.damping * 1000))
        self.damping_slider.valueChanged.connect(self.update_damping)
        self.damping_label = QLabel(f"Damping: {self.damping:.3f}")
        damping_layout.addWidget(self.damping_label)
        damping_layout.addWidget(self.damping_slider)
        param_layout.addRow(damping_layout)
        
        # Droplet size
        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(3)
        self.size_slider.setMaximum(30)
        self.size_slider.setValue(self.droplet_size)
        self.size_slider.valueChanged.connect(self.update_droplet_size)
        self.size_label = QLabel(f"Droplet Size: {self.droplet_size}")
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_slider)
        param_layout.addRow(size_layout)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group, 3)
        
        # Spacer to push everything up
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Footer
        footer = QLabel("AQUAFLUX v2.0 • Professional Fluid Simulation Engine")
        footer.setStyleSheet("color: #757575; font-size: 10px; padding: 5px 0;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(footer)
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: white;")
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        
        # Create toolbar
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        self.toolbar.setStyleSheet(f"background-color: {self.theme['surface']};")
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {self.theme['surface']};
                border-top: 1px solid {self.theme['border']};
            }}
        """)
        self.status_bar.showMessage("Ready • GPU Acceleration: " + ("Enabled" if self.use_gpu else "Disabled"))
        
        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1150])  # Default split ratio
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.status_bar)
        
        if self.parent is None:
            self.main_window.setCentralWidget(main_widget)
            self.main_window.show()
        else:
            self.parent.setLayout(main_layout)
        
        # Initialize visualization
        initial_rgb = self.composite()
        self.mat = self.ax.imshow(initial_rgb, interpolation='bicubic', vmin=0, vmax=1)
        self.canvas.draw()
        
        # Setup animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_step)
        self.timer.start(1000 // self.fps)
        
        # Mouse interaction
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Periodic memory update
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds

    def setup_matplotlib_ui(self):
        """Fallback UI using only matplotlib widgets"""
        self.fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 4)
        
        # Main visualization
        self.ax = self.fig.add_subplot(gs[0:2, 0:3])
        self.ax.axis('off')
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, "Ready", ha='center', va='center', fontsize=10)
        
        # Buttons
        btn_width = 0.12
        btn_height = 0.04
        y_pos = 0.15
        
        # Play/pause button
        play_ax = plt.axes([0.05, y_pos, btn_width, btn_height])
        self.play_btn = plt.Button(play_ax, 'Pause', color=self.theme['warning'])
        self.play_btn.on_clicked(self.toggle_pause)
        
        # Record button
        record_ax = plt.axes([0.2, y_pos, btn_width, btn_height])
        self.record_btn = plt.Button(record_ax, 'Record GIF', color=self.theme['primary'])
        self.record_btn.on_clicked(self.toggle_recording)
        
        # Save button
        save_ax = plt.axes([0.35, y_pos, btn_width, btn_height])
        self.save_btn = plt.Button(save_ax, 'Save Frame', color=self.theme['success'])
        self.save_btn.on_clicked(self.save_current_frame)
        
        # Reset button
        reset_ax = plt.axes([0.5, y_pos, btn_width, btn_height])
        self.reset_btn = plt.Button(reset_ax, 'Reset', color=self.theme['secondary'])
        self.reset_btn.on_clicked(self.reset_simulation)
        
        # Style selection
        style_ax = plt.axes([0.65, y_pos + 0.01, 0.15, 0.08], facecolor='lightgoldenrodyellow')
        self.style_radio = RadioButtons(style_ax, list(self.style_params.keys()), active=0)
        self.style_radio.on_clicked(self.change_style_radio)
        
        # Resolution selection
        res_ax = plt.axes([0.82, y_pos + 0.01, 0.15, 0.08], facecolor='lightgoldenrodyellow')
        res_labels = ['512×512', '1024×1024']
        self.res_radio = RadioButtons(res_ax, res_labels, active=0)
        self.res_radio.on_clicked(self.change_resolution_radio)
        
        # Sliders
        slider_width = 0.35
        slider_height = 0.02
        slider_y = 0.08
        
        intensity_ax = plt.axes([0.05, slider_y, slider_width, slider_height])
        self.intensity_slider = Slider(intensity_ax, 'Rain Intensity', 10, 200, 
                                     valinit=self.rain_intensity, valstep=5,
                                     color=self.theme['primary'])
        self.intensity_slider.on_changed(self.update_intensity)
        
        damping_ax = plt.axes([0.05, slider_y - 0.03, slider_width, slider_height])
        self.damping_slider = Slider(damping_ax, 'Damping', 0.99, 0.999, 
                                   valinit=self.damping, valstep=0.001,
                                   color=self.theme['secondary'])
        self.damping_slider.on_changed(self.update_damping)
        
        size_ax = plt.axes([0.05, slider_y - 0.06, slider_width, slider_height])
        self.size_slider = Slider(size_ax, 'Droplet Size', 3, 30, 
                                valinit=self.droplet_size, valstep=1,
                                color=self.theme['success'])
        self.size_slider.on_changed(self.update_droplet_size)
        
        # GPU status
        gpu_text = "GPU: Enabled" if self.use_gpu else "GPU: Disabled"
        self.gpu_status = self.fig.text(0.95, 0.02, gpu_text, ha='right', va='center', 
                                      fontsize=9, color='green' if self.use_gpu else 'red')
        
        # Initial visualization
        initial_rgb = self.composite()
        self.mat = self.ax.imshow(initial_rgb, interpolation='bicubic', vmin=0, vmax=1)
        
        # Mouse interaction
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update, frames=100000, interval=30, blit=True)
        
        plt.show()

    def show_about_dialog(self):
        """Show about dialog with application information"""
        about = QDialog(self.main_window)
        about.setWindowTitle("About AQUAFLUX")
        about.setFixedSize(400, 300)
        
        layout = QVBoxLayout(about)
        
        # Logo and title
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setFixedSize(60, 60)
        logo_label.setStyleSheet(f"""
            background-color: {self.theme['primary']};
            border-radius: 30px;
            color: white;
            font-weight: bold;
            font-size: 24px;
            qproperty-alignment: AlignCenter;
        """)
        logo_label.setText("💧")
        header_layout.addWidget(logo_label)
        
        title_layout = QVBoxLayout()
        title = QLabel("AQUAFLUX")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e88e5;")
        version = QLabel("Professional Rain Simulator v2.0")
        version.setStyleSheet("color: #757575;")
        title_layout.addWidget(title)
        title_layout.addWidget(version)
        title_layout.addStretch()
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Description
        desc = QLabel(
            "AQUAFLUX is a professional fluid dynamics simulator that creates "
            "realistic rain and water simulations using advanced GPU-accelerated "
            "algorithms. Designed for researchers, digital artists, and educators."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; margin: 10px 0;")
        layout.addWidget(desc)
        
        # System info
        sys_info = QLabel(
            f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n"
            f"PyQt Version: {PYQT_AVAILABLE}\n"
            f"PyTorch Version: {TORCH_AVAILABLE}"
        )
        sys_info.setStyleSheet("font-size: 11px; color: #757575; margin: 10px 0;")
        layout.addWidget(sys_info)
        
        # Credits
        credits = QLabel("© 2023 AQUAFLUX Technologies\nAll rights reserved")
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits.setStyleSheet("font-size: 10px; color: #757575; margin-top: 10px;")
        layout.addWidget(credits)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(about.accept)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['primary']};
                color: white;
                border-radius: 4px;
                padding: 5px 15px;
            }}
        """)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        about.exec()

    def update_memory_usage(self):
        """Update memory usage information in the UI"""
        if self.use_gpu:
            # Get GPU memory usage
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_percent = (gpu_mem / gpu_total) * 100
            
            # Format memory usage string
            if gpu_percent > 80:
                color = self.theme['danger']
            elif gpu_percent > 60:
                color = self.theme['warning']
            else:
                color = self.theme['success']
                
            self.memory_label.setText(
                f"<b>GPU Memory:</b> <span style='color:{color}'>{gpu_mem:.1f}MB / {gpu_total:.0f}MB ({gpu_percent:.1f}%)</span>"
            )
        else:
            # Get system memory usage (simplified)
            import psutil
            try:
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                
                if mem_percent > 80:
                    color = self.theme['danger']
                elif mem_percent > 60:
                    color = self.theme['warning']
                else:
                    color = self.theme['success']
                    
                self.memory_label.setText(
                    f"<b>System Memory:</b> <span style='color:{color}'>{mem_percent:.1f}% used</span>"
                )
            except ImportError:
                self.memory_label.setText("<b>Memory:</b> Install psutil for monitoring")

    def change_style_combo(self, index):
        style_name = self.style_combo.itemData(index)
        self.change_style(style_name)
        
        # Update style description
        descriptions = {
            'rain': 'Gentle rain with soft ripples and natural water coloration.',
            'storm': 'Intense rainfall with stronger waves and faster droplet movement.',
            'ink_rain': 'Artistic ink droplets on paper with high contrast and no flow.',
            'mist': 'Light mist with many small droplets creating subtle atmospheric effects.'
        }
        self.style_description.setText(descriptions.get(style_name, ''))
        
        self.status_bar.showMessage(f"Style changed to: {style_name.replace('_', ' ').title()}")
    
    def change_style_radio(self, label):
        self.change_style(label)
        self.status_text.set_text(f"Style changed to: {label}")
        self.fig.canvas.draw_idle()
    
    def change_style(self, new_style):
        if new_style not in self.style_params:
            return
            
        self.style = new_style
        self.params = self.style_params[self.style]
        base = self.params['base_color']
        
        # Update state with new base color
        if self.use_gpu:
            self.state['R'].fill_(base)
            self.state['G'].fill_(base)
            self.state['B'].fill_(base)
            self.state['wave'].zero_()
            self.state['wave_prev'].zero_()
            self.state['vx'].zero_()
            self.state['vy'].zero_()
        else:
            self.state['R'].fill(base)
            self.state['G'].fill(base)
            self.state['B'].fill(base)
            self.state['wave'].fill(0)
            self.state['wave_prev'].fill(0)
            self.state['vx'].fill(0)
            self.state['vy'].fill(0)
            
        # Update UI elements
        self.damping = self.params['damping']
        if hasattr(self, 'damping_slider'):
            self.damping_slider.setValue(int(self.damping * 1000))
            self.damping_label.setText(f"Damping: {self.damping:.3f}")
        
        # Update visualization
        rgb = self.composite()
        self.mat.set_data(rgb)
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

    def change_resolution_combo(self, index):
        new_size = self.res_combo.itemData(index)
        self.change_resolution(new_size)
        self.status_bar.showMessage(f"Resolution changed to: {new_size}×{new_size}")
    
    def change_resolution_radio(self, label):
        size = 512 if label == '512×512' else 1024
        self.change_resolution(size)
        self.status_text.set_text(f"Resolution changed to {size}×{size}")
        self.fig.canvas.draw_idle()
    
    def change_resolution(self, new_size):
        if new_size == self.size:
            return
            
        # Show progress in status bar
        self.status_bar.showMessage(f"Changing resolution to {new_size}×{new_size}... Please wait.")
        QApplication.processEvents()  # Update UI immediately
        
        # Create progress dialog for large resolutions
        if new_size > 1024:
            progress = QProgressDialog("Changing resolution...", "Cancel", 0, 100, self.main_window)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()
        
        # Reinitialize state
        self.size = new_size
        self.current_resolution = new_size
        self.initialize_state()
        
        if new_size > 1024 and hasattr(progress, 'setValue'):
            progress.setValue(50)
            QApplication.processEvents()
        
        # Update visualization
        rgb = self.composite()
        self.mat.set_data(rgb)
        self.mat.set_extent([0, self.size, self.size, 0])
        
        if new_size > 1024 and hasattr(progress, 'setValue'):
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()
        
        # Refresh display
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        
        self.status_bar.showMessage(f"Resolution changed to {new_size}×{new_size} • Memory usage updated", 3000)
        
        # Reset animation if needed
        if hasattr(self, 'timer') and not self.timer.isActive():
            self.timer.start(1000 // self.fps)

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        if hasattr(self, 'play_btn'):
            if self.paused:
                self.play_btn.setText("▶ Resume Simulation")
                self.play_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.theme['success']};
                        color: white;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 6px;
                        font-size: 14px;
                    }}
                    QPushButton:hover {{
                        background-color: {self.theme['dark_success']};
                    }}
                """)
            else:
                self.play_btn.setText("⏸ Pause Simulation")
                self.play_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.theme['warning']};
                        color: white;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 6px;
                        font-size: 14px;
                    }}
                    QPushButton:hover {{
                        background-color: {self.theme['dark_warning']};
                    }}
                """)
        
        status = "Paused" if self.paused else "Running"
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Simulation {status} • Resolution: {self.size}×{self.size}", 2000)
        else:
            self.status_text.set_text(f"Simulation {status}")
            self.fig.canvas.draw_idle()

    def toggle_recording(self, event=None):
        if not self.recording:
            # Start recording
            self.recording = True
            self.recorded_frames = []
            self.record_btn.setText("⏹ Stop Recording")
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.theme['danger']};
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {self.theme['dark_danger']};
                }}
            """)
            
            message = f"Recording started (max {self.max_record_frames} frames)"
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(message)
            else:
                self.status_text.set_text(message)
                self.fig.canvas.draw_idle()
        else:
            # Stop recording and save GIF
            self.recording = False
            self.record_btn.setText("⏺ Record GIF")
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.theme['primary']};
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {self.theme['dark_primary']};
                }}
            """)
            
            if len(self.recorded_frames) > 0:
                message = "Saving GIF... Please wait."
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(message)
                else:
                    self.status_text.set_text(message)
                    self.fig.canvas.draw_idle()
                
                # Save in separate thread to avoid UI freeze
                QTimer.singleShot(100, self.save_gif)
            else:
                message = "Recording stopped (no frames captured)"
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(message)
                else:
                    self.status_text.set_text(message)
                    self.fig.canvas.draw_idle()
        
        if hasattr(self, 'canvas'):
            self.canvas.draw()

    def save_gif(self):
        try:
            if not self.recorded_frames:
                return
                
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rain_simulation_{timestamp}_{len(self.recorded_frames)}frames_{self.current_resolution}x{self.current_resolution}.gif"
            
            # Get save directory
            if PYQT_AVAILABLE:
                save_dir = QFileDialog.getExistingDirectory(
                    self.main_window, "Select Save Directory", str(Path.home() / "Documents")
                )
                if not save_dir:
                    self.status_bar.showMessage("GIF saving cancelled")
                    return
                filename = str(Path(save_dir) / filename)
            
            # Create progress dialog
            progress = None
            if PYQT_AVAILABLE:
                progress = QProgressDialog("Saving GIF...", "Cancel", 0, 100, self.main_window)
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
            
            # Create animation
            gif_fig, gif_ax = plt.subplots(figsize=(8, 8))
            gif_ax.axis('off')
            ims = []
            
            total_frames = len(self.recorded_frames)
            for i, frame in enumerate(self.recorded_frames):
                im = gif_ax.imshow(frame, interpolation='bicubic', vmin=0, vmax=1, animated=True)
                ims.append([im])
                
                # Update progress
                if progress is not None:
                    progress.setValue(int((i / total_frames) * 100))
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        plt.close(gif_fig)
                        self.status_bar.showMessage("GIF saving cancelled")
                        return
            
            ani = FuncAnimation(gif_fig, lambda i: ims[i], frames=len(ims), interval=50, blit=True)
            
            # Save GIF
            writer = PillowWriter(fps=25)
            ani.save(filename, writer=writer)
            plt.close(gif_fig)
            
            message = f"GIF saved successfully: {filename}"
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(message, 5000)
            else:
                self.status_text.set_text(message)
                
        except Exception as e:
            error_msg = f"Error saving GIF: {str(e)}"
            print(f"Detailed error: {e}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(error_msg, 5000)
            else:
                self.status_text.set_text(error_msg)
        
        self.recorded_frames = []
        if hasattr(self, 'canvas'):
            self.canvas.draw()

    def save_current_frame(self, event=None):
        try:
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rain_frame_{timestamp}_{self.frame_count}_{self.current_resolution}x{self.current_resolution}.png"
            
            # Get save directory
            if PYQT_AVAILABLE:
                home_dir = str(Path.home() / "Documents")
                filename, _ = QFileDialog.getSaveFileName(
                    self.main_window, 
                    "Save Frame", 
                    str(Path(home_dir) / filename),
                    "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
                )
                if not filename:
                    self.status_bar.showMessage("Frame saving cancelled")
                    return
                
                # Ensure proper file extension
                if not filename.endswith(('.png', '.jpg', '.jpeg')):
                    filename += '.png'
            
            # Save with high quality
            if hasattr(self, 'fig'):
                self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
            
            message = f"Frame saved: {filename}"
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(message, 5000)
            else:
                self.status_text.set_text(message)
            
            # Show confirmation dialog
            if PYQT_AVAILABLE:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Frame Saved")
                msg.setText("Frame saved successfully!")
                msg.setInformativeText(f"Saved to: {filename}")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
                
        except Exception as e:
            error_msg = f"Error saving frame: {str(e)}"
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(error_msg, 5000)
            else:
                self.status_text.set_text(error_msg)
        
        if hasattr(self, 'canvas'):
            self.canvas.draw()

    def reset_simulation(self, event=None):
        self.frame_count = 0
        self.initialize_state()
        
        message = "Simulation reset successfully"
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message, 2000)
        else:
            self.status_text.set_text(message)
        
        # Update visualization
        rgb = self.composite()
        self.mat.set_data(rgb)
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

    def update_intensity(self, val):
        self.rain_intensity = val
        self.intensity_label.setText(f"Intensity: {self.rain_intensity}")
    
    def update_damping(self, val):
        self.damping = val / 1000.0
        self.damping_label.setText(f"Damping: {self.damping:.3f}")
    
    def update_droplet_size(self, val):
        self.droplet_size = val
        self.size_label.setText(f"Droplet Size: {self.droplet_size}")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        x = int(event.ydata)
        y = int(event.xdata)
        
        if 0 <= x < self.size and 0 <= y < self.size:
            self.add_splash(x, y)
            message = f"Splash added at ({x}, {y})"
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(message, 1500)
            else:
                self.status_text.set_text(message)
            
            # Update visualization immediately
            rgb = self.composite()
            self.mat.set_data(rgb)
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

    def add_raindrop(self):
        """GPU-accelerated raindrop addition"""
        if self.use_gpu:
            # Use PyTorch to generate random positions on GPU
            layer_idx = torch.randint(0, self.layers, (1,), device=DEVICE).item()
            r = self.droplet_size
            
            if self.size <= 2 * r:
                return
                
            x = torch.randint(r, self.size - r, (1,), device=DEVICE).item()
            y = r
            
            # Create circular mask on CPU (more efficient for small masks)
            Y, X = np.ogrid[-r:r, -r:r]
            mask = X**2 + Y**2 <= r**2
            
            # Handle boundaries
            x_start = max(0, x - r)
            x_end = min(self.size, x + r)
            y_start = max(0, y - r)
            y_end = min(self.size, y + r)
            
            # Calculate valid portion of mask
            mask_x_start = max(0, r - (x - x_start))
            mask_x_end = min(2*r, r + (x_end - x))
            mask_y_start = max(0, r - (y - y_start))
            mask_y_end = min(2*r, r + (y_end - y))
            
            valid_mask = mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end]
            if not np.any(valid_mask):
                return
                
            # Convert mask to tensor
            mask_tensor = torch.tensor(valid_mask, device=DEVICE, dtype=torch.bool)
            
            # Create slices
            sl = (layer_idx, slice(x_start, x_end), slice(y_start, y_end))
            
            # Apply raindrop properties
            if self.params['visualization'] == 'water':
                self.state['R'][sl][mask_tensor] = 0.1
                self.state['G'][sl][mask_tensor] = 0.5
                self.state['B'][sl][mask_tensor] = 0.9
            else:
                self.state['R'][sl][mask_tensor] = 0.05
                self.state['G'][sl][mask_tensor] = 0.05
                self.state['B'][sl][mask_tensor] = 0.05
                
            # Add ripple and velocity
            self.state['wave'][sl][mask_tensor] += 0.8
            
            v_min, v_max = self.params['drop_velocity_range']
            velocity = torch.empty(mask_tensor.sum(), device=DEVICE).uniform_(v_min, v_max)
            self.state['vy'][sl][mask_tensor] += velocity
            
        else:
            # CPU version (original implementation)
            layer_idx = np.random.randint(0, self.layers)
            r = self.droplet_size
            
            if self.size <= 2 * r:
                return
                
            x = np.random.randint(r, self.size - r)
            y = r
            
            Y, X = np.ogrid[-r:r, -r:r]
            mask = X**2 + Y**2 <= r**2
            
            x_start = max(0, x - r)
            x_end = min(self.size, x + r)
            y_start = max(0, y - r)
            y_end = min(self.size, y + r)
            
            mask_x_start = max(0, r - (x - x_start))
            mask_x_end = min(2*r, r + (x_end - x))
            mask_y_start = max(0, r - (y - y_start))
            mask_y_end = min(2*r, r + (y_end - y))
            
            valid_mask = mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end]
            if not np.any(valid_mask):
                return
                
            sl = (slice(layer_idx, layer_idx + 1), slice(x_start, x_end), slice(y_start, y_end))
            
            if self.params['visualization'] == 'water':
                self.state['R'][sl][:, valid_mask] = 0.1
                self.state['G'][sl][:, valid_mask] = 0.5
                self.state['B'][sl][:, valid_mask] = 0.9
            else:
                self.state['R'][sl][:, valid_mask] = 0.05
                self.state['G'][sl][:, valid_mask] = 0.05
                self.state['B'][sl][:, valid_mask] = 0.05
                
            self.state['wave'][sl][:, valid_mask] += 0.8
            
            v_min, v_max = self.params['drop_velocity_range']
            self.state['vy'][sl][:, valid_mask] += np.random.uniform(v_min, v_max, valid_mask.sum())

    def add_splash(self, x, y):
        """GPU-accelerated splash effect"""
        r = self.droplet_size * 2
        
        if self.use_gpu:
            layer_idx = torch.randint(0, self.layers, (1,), device=DEVICE).item()
            
            # Handle boundaries
            x_start = max(0, x - r)
            x_end = min(self.size, x + r)
            y_start = max(0, y - r)
            y_end = min(self.size, y + r)
            
            if x_start >= x_end or y_start >= y_end:
                return
                
            # Create mask on CPU
            Y, X = np.ogrid[0:(x_end-x_start), 0:(y_end-y_start)]
            center_x = x - x_start
            center_y = y - y_start
            mask = X**2 + Y**2 <= r**2
            
            if not np.any(mask):
                return
                
            # Convert mask to tensor
            mask_tensor = torch.tensor(mask, device=DEVICE, dtype=torch.bool)
            
            sl = (layer_idx, slice(x_start, x_end), slice(y_start, y_end))
            
            # Apply splash effects
            self.state['wave'][sl][mask_tensor] += 1.0
            
            vx = torch.empty(mask_tensor.sum(), device=DEVICE).uniform_(-1.0, 1.0)
            vy = torch.empty(mask_tensor.sum(), device=DEVICE).uniform_(-1.0, 1.0)
            
            self.state['vx'][sl][mask_tensor] += vx
            self.state['vy'][sl][mask_tensor] += vy
            
        else:
            layer_idx = np.random.randint(0, self.layers)
            
            x_start = max(0, x - r)
            x_end = min(self.size, x + r)
            y_start = max(0, y - r)
            y_end = min(self.size, y + r)
            
            if x_start >= x_end or y_start >= y_end:
                return
                
            Y, X = np.ogrid[0:(x_end-x_start), 0:(y_end-y_start)]
            center_x = x - x_start
            center_y = y - y_start
            mask = X**2 + Y**2 <= r**2
            
            if not np.any(mask):
                return
                
            sl = (slice(layer_idx, layer_idx + 1), slice(x_start, x_end), slice(y_start, y_end))
            
            self.state['wave'][sl][:, mask] += 1.0
            self.state['vx'][sl][:, mask] += np.random.uniform(-1.0, 1.0, mask.sum())
            self.state['vy'][sl][:, mask] += np.random.uniform(-1.0, 1.0, mask.sum())

    def gpu_laplace(self, field):
        """GPU-accelerated Laplacian using PyTorch convolution"""
        if not self.use_gpu:
            return laplace(field, mode='constant', cval=0.0)
            
        # Ensure input is a tensor
        if not torch.is_tensor(field):
            field = torch.tensor(field, device=DEVICE, dtype=torch.float32)
            
        # Create Laplacian kernel
        kernel = torch.tensor([[0.25, 0.5, 0.25],
                              [0.5, -3.0, 0.5],
                              [0.25, 0.5, 0.25]], device=DEVICE).view(1, 1, 3, 3)
        
        # Process each layer separately
        output = torch.zeros_like(field)
        for i in range(field.shape[0]):
            layer = field[i:i+1].unsqueeze(0)  # Add batch and channel dimensions
            # Apply convolution with reflection padding for better boundaries
            padded = F.pad(layer, (1, 1, 1, 1), mode='reflect')
            output[i] = F.conv2d(padded, kernel, padding=0).squeeze()
            
        return output

    def advect_gpu(self, field, vx, vy):
        """GPU-accelerated advection using PyTorch grid sampling"""
        if not self.use_gpu:
            return self.advect_cpu(field, vx, vy)
            
        # Ensure inputs are tensors
        if not torch.is_tensor(field):
            field = torch.tensor(field, device=DEVICE, dtype=torch.float32)
        if not torch.is_tensor(vx):
            vx = torch.tensor(vx, device=DEVICE, dtype=torch.float32)
        if not torch.is_tensor(vy):
            vy = torch.tensor(vy, device=DEVICE, dtype=torch.float32)
            
        output = torch.empty_like(field)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.size, device=DEVICE),
            torch.linspace(-1, 1, self.size, device=DEVICE),
            indexing='ij'
        )
        
        for i in range(self.layers):
            # Get flow field for this layer
            flow_x = vx[i] / self.size
            flow_y = vy[i] / self.size
            
            # Create sampling grid with flow
            sample_x = (grid_x - flow_x).clamp(-1, 1)
            sample_y = (grid_y - flow_y).clamp(-1, 1)
            grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
            
            # Sample field using grid
            field_layer = field[i:i+1].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            advected = F.grid_sample(field_layer, grid, mode='bilinear', padding_mode='border', align_corners=True)
            output[i] = advected.squeeze()
            
        return output

    def advect_cpu(self, field, vx, vy):
        """CPU version of advection using scipy.ndimage.map_coordinates"""
        from scipy.ndimage import map_coordinates
        
        output = np.empty_like(field)
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        
        # Determine if vx and vy are 3D (per-layer) or 2D (shared)
        vx_is_3d = vx.ndim == 3
        vy_is_3d = vy.ndim == 3
        
        for i in range(self.layers):
            layer_vx = vx[i] if vx_is_3d else vx
            layer_vy = vy[i] if vy_is_3d else vy
            
            # Calculate previous positions
            x_prev = x_coords - layer_vx
            y_prev = y_coords - layer_vy
            
            # Clip coordinates to stay within bounds
            x_prev = np.clip(x_prev, 0, self.size-1)
            y_prev = np.clip(y_prev, 0, self.size-1)
            
            # Prepare coordinates for map_coordinates
            coords = np.vstack((y_prev.ravel(), x_prev.ravel()))
            
            # Apply advection
            advected_layer = map_coordinates(
                field[i], coords, order=1, mode='nearest', prefilter=False
            )
            
            # Reshape to proper dimensions
            output[i] = advected_layer.reshape(self.size, self.size)
        
        return output

    def composite(self):
        """Composite layers into final RGB image"""
        if self.use_gpu:
            # Move data to CPU for matplotlib
            R_avg = self.state['R'].mean(dim=0).cpu().numpy()
            G_avg = self.state['G'].mean(dim=0).cpu().numpy()
            B_avg = self.state['B'].mean(dim=0).cpu().numpy()
            wave_avg = self.state['wave'].mean(dim=0).cpu().numpy()
        else:
            R_avg = np.mean(self.state['R'], axis=0)
            G_avg = np.mean(self.state['G'], axis=0)
            B_avg = np.mean(self.state['B'], axis=0)
            wave_avg = np.mean(self.state['wave'], axis=0)
        
        if self.params['visualization'] == 'water':
            # Specular highlight effect for water
            specular = np.clip(wave_avg * 1.5, 0, 1)
            R_vis = np.clip(R_avg + specular * 0.5, 0, 1)
            G_vis = np.clip(G_avg + specular * 0.7, 0, 1)
            B_vis = np.clip(B_avg + specular * 1.0, 0, 1)
        else:  # ink_rain
            # High contrast ink on paper effect
            ink = 1.0 - np.clip(R_avg + G_avg + B_avg, 0, 1)
            R_vis = np.full((self.size, self.size), 0.95)  # Paper color
            G_vis = np.full((self.size, self.size), 0.95)
            B_vis = np.full((self.size, self.size), 0.95)
            # Apply ink
            R_vis -= ink
            G_vis -= ink
            B_vis -= ink
            R_vis = np.clip(R_vis, 0, 1)
            G_vis = np.clip(G_vis, 0, 1)
            B_vis = np.clip(B_vis, 0, 1)
        
        return np.dstack((R_vis, G_vis, B_vis))

    def animation_step(self):
        """PyQt timer callback for animation"""
        if not self.paused:
            self.update_frame()
            self.fps_counter += 1
            
            # Update FPS every second
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.display_fps = self.fps_counter
                self.fps_counter = 0
                self.last_frame_time = current_time
                
                # Update FPS display
                self.fps_label.setText(f"<b>FPS:</b> {self.display_fps:.1f}")
                self.status_bar.showMessage(
                    f"Running at {self.display_fps:.1f} FPS • Resolution: {self.size}×{self.size} • "
                    f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}"
                )
        
        # Always update display
        rgb = self.composite()
        self.mat.set_data(rgb)
        self.canvas.draw()

    def update_frame(self):
        """Update simulation state for a single frame"""
        self.frame_count += 1
        
        # Add raindrops
        interval = max(1, int(self.params['droplet_interval'] * (100 / self.rain_intensity)))
        if self.frame_count % interval == 0:
            self.add_raindrop()
        
        # Wave propagation - GPU accelerated
        if self.use_gpu:
            with torch.no_grad():  # Disable gradient calculation for inference
                wave_lap = self.gpu_laplace(self.state['wave'])
                wave_new = (2 * self.state['wave'] - self.state['wave_prev'] + wave_lap * 0.5) * self.damping
                self.state['wave_prev'] = self.state['wave'].clone()
                self.state['wave'] = wave_new
        else:
            wave_lap = laplace(self.state['wave'], mode='constant', cval=0.0)
            wave_new = (2 * self.state['wave'] - self.state['wave_prev'] + wave_lap * 0.5) * self.damping
            self.state['wave_prev'] = self.state['wave'].copy()
            self.state['wave'] = wave_new
        
        # Reaction-diffusion
        if self.use_gpu:
            with torch.no_grad():
                LR = self.gpu_laplace(self.state['R'])
                LG = self.gpu_laplace(self.state['G'])
                LB = self.gpu_laplace(self.state['B'])
                
                diff_rates = torch.tensor(self.params['diff_rates'], device=DEVICE).view(-1, 1, 1)
                
                dR = diff_rates * LR - self.state['R'] * self.state['G'] * self.state['B'] + self.F * (1 - self.state['R'])
                dG = diff_rates * LG - self.state['G'] * self.state['B'] * self.state['R'] + self.F * (1 - self.state['G'])
                dB = diff_rates * LB - self.state['B'] * self.state['R'] * self.state['G'] + self.F * (1 - self.state['B'])
                
                self.state['R'] += dR
                self.state['G'] += dG
                self.state['B'] += dB
        else:
            LR = laplace(self.state['R'], mode='constant', cval=0.0)
            LG = laplace(self.state['G'], mode='constant', cval=0.0)
            LB = laplace(self.state['B'], mode='constant', cval=0.0)
            
            diff_rates_reshaped = self.params['diff_rates'][:, np.newaxis, np.newaxis]
            dR = diff_rates_reshaped * LR - self.state['R'] * self.state['G'] * self.state['B'] + self.F * (1 - self.state['R'])
            dG = diff_rates_reshaped * LG - self.state['G'] * self.state['B'] * self.state['R'] + self.F * (1 - self.state['G'])
            dB = diff_rates_reshaped * LB - self.state['B'] * self.state['R'] * self.state['G'] + self.F * (1 - self.state['B'])
            
            self.state['R'] += dR
            self.state['G'] += dG
            self.state['B'] += dB
        
        # Advection
        if self.params['velocity_decay'] < 1.0:
            if self.use_gpu:
                with torch.no_grad():
                    self.state['R'] = self.advect_gpu(self.state['R'], self.state['vx'], self.state['vy'])
                    self.state['G'] = self.advect_gpu(self.state['G'], self.state['vx'], self.state['vy'])
                    self.state['B'] = self.advect_gpu(self.state['B'], self.state['vx'], self.state['vy'])
                    self.state['vx'] *= self.params['velocity_decay']
                    self.state['vy'] *= self.params['velocity_decay']
            else:
                self.state['R'] = self.advect_cpu(self.state['R'], self.state['vx'], self.state['vy'])
                self.state['G'] = self.advect_cpu(self.state['G'], self.state['vx'], self.state['vy'])
                self.state['B'] = self.advect_cpu(self.state['B'], self.state['vx'], self.state['vy'])
                self.state['vx'] *= self.params['velocity_decay']
                self.state['vy'] *= self.params['velocity_decay']
        
        # Clamp values
        if self.use_gpu:
            with torch.no_grad():
                self.state['R'] = torch.clamp(self.state['R'], 0, 1)
                self.state['G'] = torch.clamp(self.state['G'], 0, 1)
                self.state['B'] = torch.clamp(self.state['B'], 0, 1)
        else:
            np.clip(self.state['R'], 0, 1, out=self.state['R'])
            np.clip(self.state['G'], 0, 1, out=self.state['G'])
            np.clip(self.state['B'], 0, 1, out=self.state['B'])
        
        # Record frames for GIF
        if self.recording and len(self.recorded_frames) < self.max_record_frames:
            rgb = self.composite()
            self.recorded_frames.append(rgb.copy())
            self.status_bar.showMessage(
                f"Recording: {len(self.recorded_frames)}/{self.max_record_frames} frames captured"
            )
        
        return self.mat

    def run(self):
        """Start the application"""
        if PYQT_AVAILABLE and self.parent is None:
            sys.exit(self.app.exec())
        else:
            plt.show()

if __name__ == '__main__':
    # Suppress matplotlib warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create and run the visualizer
    visualizer = GPURainVisualizer()
    visualizer.run()