"""
Core physics simulation using stable 2D Wave Equation.
"""

import numpy as np
from scipy.ndimage import laplace

from config import Config


class RainSimulation:
    """
    Pure 2D Wave Equation Simulator.
    Tuned for smooth propagation and stable craters.
    """

    def __init__(self, size=None, layers=None):
        self.size = size or Config.SIM_SIZE
        self.layers = layers or Config.SIM_LAYERS
        self.frame_count = 0
        self.reset_state()

    def reset_state(self):
        """Reset all simulation fields to initial state."""
        shape = (self.layers, self.size, self.size)
        self.height = np.zeros(shape, dtype=np.float32)
        self.height_prev = np.zeros(shape, dtype=np.float32)
        self.density = np.zeros(shape, dtype=np.float32)

    def step(self, rain_intensity, droplet_size, damping):
        """Advance simulation by one timestep."""
        self.frame_count += 1

        # 1. Auto Rain Generation
        if rain_intensity > 0:
            interval = max(1, int(20 * (100 / rain_intensity)))
            if self.frame_count % interval == 0:
                self.add_raindrop(droplet_size)

        # 2. Stable Wave Equation Propagation
        lap_h = laplace(self.height, mode='constant', cval=0.0)
        lap_h = np.clip(lap_h, -Config.LAPLACE_CLAMP, Config.LAPLACE_CLAMP)

        new_height = (2 * self.height - self.height_prev + lap_h * Config.WAVE_SPEED) * damping

        self.height_prev = self.height.copy()
        self.height = new_height

        # 3. Density fade (foam)
        self.density *= Config.DENSITY_DECAY

        # 4. Safety clamping
        self.height = np.clip(self.height, -Config.HEIGHT_CLAMP, Config.HEIGHT_CLAMP)
        self.height = np.nan_to_num(self.height, nan=0.0)
        self.density = np.nan_to_num(self.density, nan=0.0)

    def add_raindrop(self, radius):
        """Add automatic raindrop crater."""
        layer_idx = np.random.randint(0, self.layers)
        r = max(2, int(radius * 0.8))
        
        x = np.random.randint(r, self.size - r)
        y = np.random.randint(r, self.size - r)

        Y, X = np.ogrid[-r:r, -r:r]
        dist = np.sqrt(X ** 2 + Y ** 2)
        
        sigma = max(1, r / 1.5)  # Slightly wider drop for smoothness
        intensity = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        x_s, x_e = max(0, x-r), min(self.size, x+r)
        y_s, y_e = max(0, y-r), min(self.size, y+r)
        if x_s >= x_e or y_s >= y_e: return

        ix_s = max(0, r - x)
        ix_e = ix_s + (x_e - x_s)
        iy_s = max(0, r - y)
        iy_e = iy_s + (y_e - y_s)

        sl = (slice(layer_idx, layer_idx+1), slice(x_s, x_e), slice(y_s, y_e))
        
        # Gentle negative push to start the ring smoothly
        self.height[sl] -= intensity[ix_s:ix_e, iy_s:iy_e] * 1.5
        self.density[sl] += intensity[ix_s:ix_e, iy_s:iy_e] * 1.0

    def add_droplet_impact(self, x, y, radius):
        """User Click: Smooth large crater impact."""
        r = int(radius * 3) # Reduced radius for tighter, smoother rings
        if r < 4: r = 4

        Y, X = np.ogrid[-r:r, -r:r]
        dist = np.sqrt(X ** 2 + Y ** 2)

        sigma = radius * 1.5
        intensity = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        x, y = int(x), int(y)
        x_start, x_end = max(0, x - r), min(self.size, x + r)
        y_start, y_end = max(0, y - r), min(self.size, y + r)

        if x_start >= x_end or y_start >= y_end: return

        ix_start = max(0, r - x)
        ix_end = ix_start + (x_end - x_start)
        iy_start = max(0, r - y)
        iy_end = iy_start + (y_end - y_start)

        valid_mask = intensity[ix_start:ix_end, iy_start:iy_end]
        sl = (slice(0, 1), slice(x_start, x_end), slice(y_start, y_end))

        # Smooth, stable impact depth
        self.height[sl] -= valid_mask * 3.0
        self.density[sl] += valid_mask * 2.0