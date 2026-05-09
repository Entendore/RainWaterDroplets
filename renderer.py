"""
Rendering engine: Converts wave state to smooth, realistic water optics.
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from color_engine import ColorEngine
from config import Config


def composite(simulation, palette_name):
    """
    Render simulation state to an RGB image array.
    Returns:
        numpy array of shape (size, size, 3) with uint8 RGB values
    """
    colors = ColorEngine.get_colors_normalized(palette_name)
    size = simulation.size

    # 1. Prepare maps (High sigma blur is key to "water surface tension" look)
    height_avg = np.mean(simulation.height, axis=0)
    density_avg = np.mean(simulation.density, axis=0)
    height_smooth = gaussian_filter(height_avg, sigma=Config.GAUSSIAN_SIGMA)

    # 2. Smooth Refraction (Water acts as a lens)
    grad_y, grad_x = np.gradient(height_smooth)

    warp_x = np.clip(
        np.arange(size)[np.newaxis, :] - grad_x * Config.REFRACTION_STRENGTH,
        0, size - 1
    )
    warp_y = np.clip(
        np.arange(size)[:, np.newaxis] - grad_y * Config.REFRACTION_STRENGTH,
        0, size - 1
    )

    # Background canvas
    bg_image = np.zeros((size, size, 3), dtype=np.float32)
    bg_image[:, :] = colors['surface']

    # Sample distorted background
    coords = np.array([warp_y.ravel(), warp_x.ravel()])
    r = map_coordinates(bg_image[:, :, 0], coords, order=1, mode='reflect').reshape(size, size)
    g = map_coordinates(bg_image[:, :, 1], coords, order=1, mode='reflect').reshape(size, size)
    b = map_coordinates(bg_image[:, :, 2], coords, order=1, mode='reflect').reshape(size, size)
    refr_img = np.dstack((r, g, b))

    # 3. Smooth Color Blending based on wave height
    # Using absolute height creates a smooth transition between calm and rippled
    depth_factor = np.clip(np.abs(height_smooth) * Config.HEIGHT_BLEND_FACTOR, 0, 1)[:, :, np.newaxis]
    
    final_rgb = refr_img * (1 - depth_factor) + colors['ripple'] * depth_factor

    # 4. Smooth Omnidirectional Specular highlights (Glint on the ring edge)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    spec = np.clip(grad_mag * 2.0, 0, 1) 
    spec = np.power(spec, Config.SPECULAR_POWER) * Config.SPECULAR_STRENGTH

    # 5. Add Foam (from initial impact splash)
    density_mask = np.clip(density_avg, 0, 1)[:, :, np.newaxis]
    final_rgb = final_rgb * (1 - density_mask) + colors['foam'] * density_mask

    # 6. Add specular shine
    final_rgb = np.clip(final_rgb + spec[:, :, np.newaxis], 0, 1)

    final_rgb = np.nan_to_num(final_rgb, nan=0.0)
    return (final_rgb * 255).astype(np.uint8)