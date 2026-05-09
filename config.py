"""
Application configuration constants.
Mathematically tuned for smooth, stable fluid dynamics.
"""


class Config:
    """Centralized configuration for the simulation and GUI."""

    # Window settings
    WINDOW_TITLE = "AQUAFLUX - Pond Simulator"
    WINDOW_GEOMETRY = (100, 100, 1300, 850)

    # Simulation defaults
    SIM_SIZE = 400
    SIM_LAYERS = 3
    DEFAULT_FPS = 30
    RECORD_FPS = 20

    # Physics defaults (Tuned for smoothness)
    DEFAULT_RAIN_INTENSITY = 40
    DEFAULT_DAMPING = 0.995       # Smooth fade out (0.5% energy loss per frame)
    DEFAULT_DROPLET_SIZE = 8
    WAVE_SPEED = 0.2              # Drastically lowered to guarantee stable waves

    # Slider ranges
    INTENSITY_MIN = 0
    INTENSITY_MAX = 200
    DAMPING_MIN = 950             # 0.950 
    DAMPING_MAX = 999             # 0.999
    DROPLET_SIZE_MIN = 3
    DROPLET_SIZE_MAX = 25

    # Physics limits
    HEIGHT_CLAMP = 5.0            # Lower clamp prevents extreme spikes
    LAPLACE_CLAMP = 1.0
    DENSITY_DECAY = 0.992         # Foam fades smoothly

    # Rendering (Tuned for optical smoothness)
    REFRACTION_STRENGTH = 8.0     # Lowered to prevent broken-mirror warping
    GAUSSIAN_SIGMA = 2.0          # Increased blur to simulate water surface tension
    HEIGHT_BLEND_FACTOR = 2.5     # Smooth blending of the ripple color
    SPECULAR_POWER = 3
    SPECULAR_STRENGTH = 1.5       # Subtle glint, not blown-out white noise

    # Control panel
    CONTROL_PANEL_WIDTH = 340
    PANEL_MARGIN = 25
    PANEL_SPACING = 15

    # Video
    VIDEO_CODEC = 'mp4v'
    VIDEO_EXTENSION = '.mp4'