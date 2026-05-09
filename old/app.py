import sys
import time
import numpy as np
from scipy.ndimage import laplace, map_coordinates, gaussian_filter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton, 
                               QGroupBox, QFormLayout, QFileDialog, QFrame, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

import cv2

# =============================================================================
# COLOR PALETTE ENGINE
# =============================================================================

class ColorEngine:
    """Handles palette definitions and background rendering."""
    
    PALETTES = {
        'Deep Pond': {
            'surface': (15, 25, 50),    # Dark Blue surface
            'ripple': (80, 140, 200),   # Lighter Blue highlights
            'foam': (200, 220, 240)
        },
        'Rainy Asphalt': {
            'surface': (40, 40, 45),
            'ripple': (90, 95, 110),
            'foam': (180, 190, 200)
        },
        'Zen Garden': {
            'surface': (50, 45, 35),
            'ripple': (90, 85, 60),
            'foam': (140, 130, 100)
        },
        'Cyberpunk': {
            'surface': (20, 10, 35),
            'ripple': (255, 0, 128),
            'foam': (0, 255, 255)
        },
        'Arctic': {
            'surface': (180, 210, 230),
            'ripple': (50, 100, 150),
            'foam': (255, 255, 255)
        },
        'Lava': {
            'surface': (30, 5, 0),
            'ripple': (255, 50, 0),
            'foam': (255, 200, 0)
        }
    }

    @staticmethod
    def get_colors(palette_name):
        if palette_name not in ColorEngine.PALETTES:
            palette_name = 'Deep Pond'
        return ColorEngine.PALETTES[palette_name]

# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class RainSimulation:
    def __init__(self, size=400, layers=3):
        self.size = size
        self.layers = layers
        self.frame_count = 0
        self.reset_state()

    def reset_state(self):
        # Wave height maps
        self.height = np.zeros((self.layers, self.size, self.size), dtype=np.float32)
        self.height_prev = np.zeros((self.layers, self.size, self.size), dtype=np.float32)
        
        # Velocity fields (for fluid movement)
        self.vx = np.zeros((self.layers, self.size, self.size), dtype=np.float32)
        self.vy = np.zeros((self.layers, self.size, self.size), dtype=np.float32)
        
        # Density (for foam/color intensity)
        self.density = np.zeros((self.layers, self.size, self.size), dtype=np.float32)

    def step(self, rain_intensity, droplet_size, damping, ambient_strength=0.01):
        self.frame_count += 1
        
        # 1. Ambient Motion (Constant Rippling)
        # Small random noise to simulate surface tension and wind
        if ambient_strength > 0:
            noise = np.random.uniform(-0.02, 0.02, (self.size, self.size)).astype(np.float32) * ambient_strength
            self.height[0] += noise

        # 2. Add Rain Droplets (Auto mode)
        interval = max(1, int(20 * (100 / rain_intensity)))
        if self.frame_count % interval == 0:
            self.add_raindrop(droplet_size)

        # 3. Fluid Velocity Advection
        self.vx = self.advect(self.vx, self.vx, self.vy)
        self.vy = self.advect(self.vy, self.vx, self.vy)

        # 4. Wave Equation Physics
        lap_h = laplace(self.height, mode='constant', cval=0.0)
        lap_h = np.clip(lap_h, -1.0, 1.0) # Stability clamp
        
        new_height = (2 * self.height - self.height_prev + lap_h * 0.5) * damping
        new_height += self.vx * 0.1 # Velocity influence
        
        self.height_prev = self.height.copy()
        self.height = new_height

        # 5. Density Advection
        self.density = self.advect(self.density, self.vx, self.vy)
        self.density *= 0.995

        # 6. Safety Clamping
        self.height = np.clip(self.height, -5.0, 5.0)
        self.vx = np.clip(self.vx, -2.0, 2.0) * 0.99
        self.vy = np.clip(self.vy, -2.0, 2.0) * 0.99
        
        # Clean NaNs
        self.height = np.nan_to_num(self.height, nan=0.0)
        self.density = np.nan_to_num(self.density, nan=0.0)

    def add_raindrop(self, radius):
        """Automatic small raindrops"""
        layer_idx = np.random.randint(0, self.layers)
        r = int(radius)
        if r < 2: return
        
        x = np.random.randint(r, self.size - r)
        y = np.random.randint(r, self.size - r)
        
        Y, X = np.ogrid[-r:r, -r:r]
        dist = np.sqrt(X**2 + Y**2)
        mask = dist <= r
        
        intensity = np.exp(-(dist**2) / (r/2)**2)
        intensity[~mask] = 0
        
        sl = (slice(layer_idx, layer_idx + 1), slice(x-r, x+r), slice(y-r, y+r))
        
        self.height[sl] += intensity * 0.5
        self.density[sl] += intensity * 0.8
        # Raindrops get a little random velocity
        angle = np.random.random() * 2 * np.pi
        self.vx[sl] += np.cos(angle) * intensity * 0.5
        self.vy[sl] += np.sin(angle) * intensity * 0.5

    def add_droplet_impact(self, x, y, radius):
        """
        Creates a perfect symmetric droplet impact (Pond Ripple).
        No chaotic velocity, just pure vertical displacement.
        """
        r = int(radius * 4) # Large influence radius for clean circles
        if r < 4: r = 4
        
        # Coordinate Grid centered at impact
        Y, X = np.ogrid[-r:r, -r:r]
        dist = np.sqrt(X**2 + Y**2)
        
        # Create a smooth Gaussian pulse
        # This simulates the initial depression and rise of a water drop
        sigma = radius * 1.5
        intensity = np.exp(-(dist**2) / (2 * sigma**2))
        
        # Boundary Checks
        x, y = int(x), int(y)
        x_start, x_end = max(0, x-r), min(self.size, x+r)
        y_start, y_end = max(0, y-r), min(self.size, y+r)
        
        if x_start >= x_end or y_start >= y_end: return

        # Determine slice of the intensity map to use
        ix_start = max(0, r - x)
        ix_end = ix_start + (x_end - x_start)
        iy_start = max(0, r - y)
        iy_end = iy_start + (y_end - y_start)
        
        valid_mask = intensity[ix_start:ix_end, iy_start:iy_end]
        
        # Apply to Layer 0
        sl = (slice(0, 1), slice(x_start, x_end), slice(y_start, y_end))
        
        # Add Height (The Wave)
        # We add a positive peak; the wave equation naturally turns it into rings
        self.height[sl] += valid_mask * 2.0
        
        # Add Density (Visual Foam)
        self.density[sl] += valid_mask * 1.5

        # NO Velocity added. This is the key to "Pond" behavior.
        # The wave propagates purely through height differences (Laplace),
        # creating perfect concentric circles.

    def advect(self, field, vx, vy):
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        vx_avg = np.mean(vx, axis=0)
        vy_avg = np.mean(vy, axis=0)
        
        x_prev = np.clip(x_coords - vx_avg, 0, self.size - 1)
        y_prev = np.clip(y_coords - vy_avg, 0, self.size - 1)
        
        output = np.empty_like(field)
        for i in range(self.layers):
            coords = np.array([y_prev.ravel(), x_prev.ravel()])
            output[i] = map_coordinates(field[i], coords, order=1, mode='nearest', prefilter=False).reshape(self.size, self.size)
        return output

    def composite(self, palette_name):
        """
        Rendering: Refractive Background with Specular Highlights.
        """
        colors = ColorEngine.get_colors(palette_name)
        bg_color = np.array(colors['surface'], dtype=np.float32) / 255.0
        ripple_color = np.array(colors['ripple'], dtype=np.float32) / 255.0
        foam_color = np.array(colors['foam'], dtype=np.float32) / 255.0
        
        # 1. Prepare Maps
        height_avg = np.mean(self.height, axis=0)
        density_avg = np.mean(self.density, axis=0)
        
        height_smooth = gaussian_filter(height_avg, sigma=1.2)
        
        # 2. Refraction (Water as a lens)
        grad_y, grad_x = np.gradient(height_smooth)
        
        # Stronger refraction for clearer "water" look
        strength = 12.0
        warp_x = np.clip(np.arange(self.size)[np.newaxis, :] - grad_x * strength, 0, self.size - 1)
        warp_y = np.clip(np.arange(self.size)[:, np.newaxis] - grad_y * strength, 0, self.size - 1)
        
        # Background Canvas
        bg_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        bg_image[:, :] = bg_color
        
        # Sample Distorted Background
        coords = np.array([warp_y.ravel(), warp_x.ravel()])
        r = map_coordinates(bg_image[:,:,0], coords, order=1, mode='reflect').reshape(self.size, self.size)
        g = map_coordinates(bg_image[:,:,1], coords, order=1, mode='reflect').reshape(self.size, self.size)
        b = map_coordinates(bg_image[:,:,2], coords, order=1, mode='reflect').reshape(self.size, self.size)
        
        refr_img = np.dstack((r, g, b))
        
        # 3. Lighting (Specular Highlights)
        # Fresnel-like effect: ripples facing the light turn brighter
        spec = np.clip(-grad_y * 2.0, 0, 1)
        spec = np.power(spec, 4) * 2.0
        
        # 4. Color Blending
        # Mix background with ripple color based on wave height
        depth_factor = np.clip(np.abs(height_smooth) * 2.5, 0, 1)
        
        final_rgb = (refr_img * (1 - depth_factor[:,:,np.newaxis])) + (ripple_color * depth_factor[:,:,np.newaxis])
        
        # Add Foam
        density_mask = np.clip(density_avg, 0, 1)[:,:,np.newaxis]
        final_rgb = final_rgb * (1 - density_mask) + foam_color * density_mask
        
        # Add Specular Shine
        final_rgb = np.clip(final_rgb + spec[:,:,np.newaxis], 0, 1)
        
        final_rgb = np.nan_to_num(final_rgb, nan=0.0)
        return (final_rgb * 255).astype(np.uint8)

# =============================================================================
# GUI
# =============================================================================

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scaled_pixmap = None
        self.original_pixmap = None

    def set_image(self, qimage):
        self.original_pixmap = QPixmap.fromImage(qimage)
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_pixmap:
            self.scaled_pixmap = self.original_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(self.scaled_pixmap)

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AQUAFLUX - Pond Simulator")
        self.setGeometry(100, 100, 1300, 850)
        
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.fps = 30
        self.sim_size = 400 
        
        self.simulation = RainSimulation(size=self.sim_size)
        
        self.setup_ui()
        self.apply_theme()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(int(1000 / self.fps))
        
        self.last_time = time.time()
        self.frame_counter = 0

    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        control_panel.setFixedWidth(340)
        
        layout = QVBoxLayout(control_panel)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Header
        title = QLabel("AQUAFLUX")
        title.setObjectName("Title")
        subtitle = QLabel("Interactive Pond Engine")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Visuals Group
        visuals_group = QGroupBox("Visual Settings")
        form_layout = QFormLayout()
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(ColorEngine.PALETTES.keys()))
        form_layout.addRow("Palette:", self.palette_combo)
        
        # Interaction Mode
        self.interactive_check = QCheckBox("Interactive Mode")
        self.interactive_check.setChecked(True)
        self.interactive_check.setToolTip("Uncheck to let the simulation run automatically (Zen Mode)")
        form_layout.addRow("Input:", self.interactive_check)
        
        visuals_group.setLayout(form_layout)
        layout.addWidget(visuals_group)

        # Physics Group
        physics_group = QGroupBox("Physics Parameters")
        physics_layout = QVBoxLayout()
        
        # Intensity
        self.intensity_label = QLabel("Rain Intensity: 40")
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 200) # Allow 0 for no rain
        self.intensity_slider.setValue(40)
        self.intensity_slider.valueChanged.connect(self.update_params)
        physics_layout.addWidget(self.intensity_label)
        physics_layout.addWidget(self.intensity_slider)

        # Damping
        self.damping_label = QLabel("Ripple Persistence: 0.996")
        self.damping_slider = QSlider(Qt.Orientation.Horizontal)
        self.damping_slider.setRange(950, 999)
        self.damping_slider.setValue(996)
        self.damping_slider.valueChanged.connect(self.update_params)
        physics_layout.addWidget(self.damping_label)
        physics_layout.addWidget(self.damping_slider)

        # Size
        self.size_label = QLabel("Droplet Size: 8")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(3, 25)
        self.size_slider.setValue(8)
        self.size_slider.valueChanged.connect(self.update_params)
        physics_layout.addWidget(self.size_label)
        physics_layout.addWidget(self.size_slider)
        
        physics_group.setLayout(physics_layout)
        layout.addWidget(physics_group)

        # Actions
        actions_group = QGroupBox("Controls")
        actions_layout = QVBoxLayout()
        
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        actions_layout.addWidget(self.btn_pause)

        self.btn_record = QPushButton("⏺  Record Video")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.clicked.connect(self.toggle_recording)
        actions_layout.addWidget(self.btn_record)

        self.btn_save = QPushButton("💾  Save Snapshot")
        self.btn_save.clicked.connect(self.save_frame)
        actions_layout.addWidget(self.btn_save)
        
        self.btn_reset = QPushButton("↺  Clear Water")
        self.btn_reset.clicked.connect(self.reset_sim)
        actions_layout.addWidget(self.btn_reset)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("Footer")
        layout.addWidget(self.fps_label)

        # --- Display ---
        self.image_widget = ImageWidget()
        self.image_widget.setObjectName("Display")
        self.image_widget.setStyleSheet("background-color: #000000;")
        self.image_widget.mousePressEvent = self.on_image_click
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.image_widget, 1)
        
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Ready. Click on water to create ripples.")

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1A1D23; }
            #ControlPanel { background-color: #20242B; border-right: 1px solid #2E333D; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
            #Title { font-size: 28px; font-weight: bold; color: #00BCD4; }
            #Subtitle { font-size: 12px; color: #757575; margin-bottom: 10px; }
            QGroupBox {
                font-weight: bold; font-size: 13px; color: #B0BEC5;
                border: 1px solid #2E333D; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QSlider::groove:horizontal {
                border: 1px solid #2E333D; height: 6px;
                background: #12151A; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00BCD4; border: 1px solid #00BCD4;
                width: 16px; margin: -5px 0; border-radius: 9px;
            }
            QPushButton {
                background-color: #2E333D; color: #FFFFFF;
                border: none; padding: 12px; border-radius: 6px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3A3F4B; }
            QPushButton:pressed { background-color: #12151A; }
            #RecordBtn { background-color: #D32F2F; }
            #RecordBtn:hover { background-color: #F44336; }
            QComboBox, QCheckBox {
                background-color: #12151A; color: white;
                border: 1px solid #2E333D; border-radius: 4px; padding: 5px;
            }
            QCheckBox::indicator { width: 13px; height: 13px; }
            #Footer { color: #546E7A; font-size: 11px; }
        """)

    def update_params(self):
        val_i = self.intensity_slider.value()
        val_d = self.damping_slider.value() / 1000.0
        val_s = self.size_slider.value()
        
        self.intensity_label.setText(f"Rain Intensity: {val_i}")
        self.damping_label.setText(f"Ripple Persistence: {val_d:.3f}")
        self.size_label.setText(f"Droplet Size: {val_s}")

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("▶  Resume" if self.paused else "⏸  Pause")

    def toggle_recording(self):
        if not self.recording:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Video", "simulation.mp4", "MP4 Video (*.mp4)")
            if filename:
                if not filename.endswith('.mp4'): filename += '.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.sim_size, self.sim_size))
                self.recording = True
                self.btn_record.setText("⏹  Stop Recording")
                self.btn_record.setStyleSheet("background-color: #C62828;")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.setText("⏺  Record Video")
            self.btn_record.setStyleSheet("")

    def save_frame(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "frame.png", "PNG Image (*.png)")
        if filename:
            rgb_array = self.simulation.composite(self.palette_combo.currentText())
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_array)

    def reset_sim(self):
        self.simulation.reset_state()

    def on_image_click(self, event):
        if not self.interactive_check.isChecked():
            return
            
        if not self.image_widget.original_pixmap: return
        pixmap_rect = self.image_widget.pixmap().rect()
        x = int(event.position().x() * (self.sim_size / pixmap_rect.width()))
        y = int(event.position().y() * (self.sim_size / pixmap_rect.height()))
        
        # Call the new symmetric droplet impact function
        self.simulation.add_droplet_impact(y, x, self.size_slider.value())

    def update_animation(self):
        # FPS
        self.frame_counter += 1
        if time.time() - self.last_time >= 1.0:
            self.fps_label.setText(f"FPS: {self.frame_counter}")
            self.frame_counter = 0
            self.last_time = time.time()

        if not self.paused:
            intensity = self.intensity_slider.value()
            damping = self.damping_slider.value() / 1000.0
            size = self.size_slider.value()
            palette = self.palette_combo.current_text()
            
            # Ambient strength kept low to not disturb user-created perfect ripples
            self.simulation.step(intensity, size, damping, ambient_strength=0.02)
            
            rgb_array = self.simulation.composite(palette)
            h, w, ch = rgb_array.shape
            qimg = QImage(rgb_array.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.image_widget.set_image(qimg)
            
            if self.recording and self.video_writer:
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                self.video_writer.write(bgr_array)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())