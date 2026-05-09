"""
Main application window with controls and simulation display.
"""

import time
import cv2
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QPushButton, QGroupBox, QFormLayout,
    QFileDialog, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage

from config import Config
from color_engine import ColorEngine
from simulation import RainSimulation
from renderer import composite
from image_widget import ImageWidget
from styles import APP_STYLESHEET, RECORDING_ACTIVE_STYLE


class MainWindow(QMainWindow):
    """Main application window containing controls and simulation view."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.setGeometry(*Config.WINDOW_GEOMETRY)

        self.paused = False
        self.recording = False
        self.video_writer = None
        self.fps = Config.DEFAULT_FPS
        self.sim_size = Config.SIM_SIZE

        self.simulation = RainSimulation(size=self.sim_size)

        self._setup_ui()
        self._apply_theme()
        self._setup_timer()

        self.last_time = time.time()
        self.frame_counter = 0

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(int(1000 / self.fps))

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        control_panel = self._create_control_panel()
        self.image_widget = self._create_display_widget()

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.image_widget, 1)

        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Ready. Click on water to create ripples.")

    def _create_control_panel(self):
        panel = QFrame()
        panel.setObjectName("ControlPanel")
        panel.setFixedWidth(Config.CONTROL_PANEL_WIDTH)

        layout = QVBoxLayout(panel)
        layout.setSpacing(Config.PANEL_SPACING)
        layout.setContentsMargins(
            Config.PANEL_MARGIN, Config.PANEL_MARGIN,
            Config.PANEL_MARGIN, Config.PANEL_MARGIN
        )

        title = QLabel("AQUAFLUX")
        title.setObjectName("Title")
        subtitle = QLabel("Interactive Pond Engine")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._create_visuals_group())
        layout.addWidget(self._create_physics_group())
        layout.addWidget(self._create_actions_group())

        layout.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("Footer")
        layout.addWidget(self.fps_label)

        return panel

    def _create_visuals_group(self):
        group = QGroupBox("Visual Settings")
        form_layout = QFormLayout()

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(ColorEngine.get_palette_names())
        form_layout.addRow("Palette:", self.palette_combo)

        self.interactive_check = QCheckBox("Interactive (Click to Drop)")
        self.interactive_check.setChecked(True)
        self.interactive_check.setToolTip("Checked: Random rain + click to add drops.\nUnchecked: Ambient mode (Random rain only).")
        form_layout.addRow("Input Mode:", self.interactive_check)

        group.setLayout(form_layout)
        return group

    def _create_physics_group(self):
        group = QGroupBox("Physics Parameters")
        layout = QVBoxLayout()

        self.intensity_label = QLabel(f"Rain Intensity: {Config.DEFAULT_RAIN_INTENSITY}")
        self.intensity_slider = self._make_slider(
            Config.INTENSITY_MIN, Config.INTENSITY_MAX, Config.DEFAULT_RAIN_INTENSITY
        )
        self.intensity_slider.valueChanged.connect(self._update_param_labels)
        layout.addWidget(self.intensity_label)
        layout.addWidget(self.intensity_slider)

        self.damping_label = QLabel(f"Ripple Persistence: {Config.DEFAULT_DAMPING:.3f}")
        self.damping_slider = self._make_slider(
            Config.DAMPING_MIN, Config.DAMPING_MAX, int(Config.DEFAULT_DAMPING * 1000)
        )
        self.damping_slider.valueChanged.connect(self._update_param_labels)
        layout.addWidget(self.damping_label)
        layout.addWidget(self.damping_slider)

        self.size_label = QLabel(f"Droplet Size: {Config.DEFAULT_DROPLET_SIZE}")
        self.size_slider = self._make_slider(
            Config.DROPLET_SIZE_MIN, Config.DROPLET_SIZE_MAX, Config.DEFAULT_DROPLET_SIZE
        )
        self.size_slider.valueChanged.connect(self._update_param_labels)
        layout.addWidget(self.size_label)
        layout.addWidget(self.size_slider)

        group.setLayout(layout)
        return group

    def _create_actions_group(self):
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self._toggle_pause)
        layout.addWidget(self.btn_pause)

        self.btn_record = QPushButton("⏺  Record Video")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.clicked.connect(self._toggle_recording)
        layout.addWidget(self.btn_record)

        self.btn_save = QPushButton("💾  Save Snapshot")
        self.btn_save.clicked.connect(self._save_frame)
        layout.addWidget(self.btn_save)

        self.btn_reset = QPushButton("↺  Clear Water")
        self.btn_reset.clicked.connect(self._reset_simulation)
        layout.addWidget(self.btn_reset)

        group.setLayout(layout)
        return group

    def _create_display_widget(self):
        widget = ImageWidget()
        widget.setObjectName("Display")
        widget.setStyleSheet("background-color: #000000;")
        widget.mousePressEvent = self._on_image_click
        return widget

    @staticmethod
    def _make_slider(min_val, max_val, default):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _apply_theme(self):
        self.setStyleSheet(APP_STYLESHEET)

    def _update_param_labels(self):
        val_i = self.intensity_slider.value()
        val_d = self.damping_slider.value() / 1000.0
        val_s = self.size_slider.value()

        self.intensity_label.setText(f"Rain Intensity: {val_i}")
        self.damping_label.setText(f"Ripple Persistence: {val_d:.3f}")
        self.size_label.setText(f"Droplet Size: {val_s}")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("▶  Resume" if self.paused else "⏸  Pause")

    def _toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "simulation.mp4", "MP4 Video (*.mp4)"
        )
        if not filename:
            return
        if not filename.endswith(Config.VIDEO_EXTENSION):
            filename += Config.VIDEO_EXTENSION

        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, Config.RECORD_FPS, (self.sim_size, self.sim_size)
        )
        self.recording = True
        self.btn_record.setText("⏹  Stop Recording")
        self.btn_record.setStyleSheet(RECORDING_ACTIVE_STYLE)

    def _stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.btn_record.setText("⏺  Record Video")
        self.btn_record.setStyleSheet("")

    def _save_frame(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "frame.png", "PNG Image (*.png)"
        )
        if filename:
            rgb_array = composite(self.simulation, self.palette_combo.currentText())
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_array)

    def _reset_simulation(self):
        self.simulation.reset_state()

    def _on_image_click(self, event):
        if not self.interactive_check.isChecked():
            return
        if not self.image_widget.original_pixmap:
            return

        pixmap_rect = self.image_widget.pixmap().rect()
        x = int(event.position().x() * (self.sim_size / pixmap_rect.width()))
        y = int(event.position().y() * (self.sim_size / pixmap_rect.height()))

        self.simulation.add_droplet_impact(y, x, self.size_slider.value())

    def _update_animation(self):
        self.frame_counter += 1
        if time.time() - self.last_time >= 1.0:
            self.fps_label.setText(f"FPS: {self.frame_counter}")
            self.frame_counter = 0
            self.last_time = time.time()

        if self.paused:
            return

        intensity = self.intensity_slider.value()
        damping = self.damping_slider.value() / 1000.0
        size = self.size_slider.value()
        palette = self.palette_combo.currentText()

        self.simulation.step(intensity, size, damping)

        rgb_array = composite(self.simulation, palette)
        h, w, ch = rgb_array.shape
        qimg = QImage(rgb_array.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.image_widget.set_image(qimg)

        if self.recording and self.video_writer:
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_array)