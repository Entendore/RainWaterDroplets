"""
Color palette engine for water surface rendering.
"""

import numpy as np


class ColorEngine:
    """Handles palette definitions and color retrieval."""

    PALETTES = {
        'Deep Pond': {
            'surface': (15, 25, 50),
            'ripple': (80, 140, 200),
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

    DEFAULT_PALETTE = 'Deep Pond'

    @classmethod
    def get_palette_names(cls):
        return list(cls.PALETTES.keys())

    @classmethod
    def get_colors(cls, palette_name):
        if palette_name not in cls.PALETTES:
            palette_name = cls.DEFAULT_PALETTE
        return cls.PALETTES[palette_name]

    @classmethod
    def get_colors_normalized(cls, palette_name):
        colors = cls.get_colors(palette_name)
        return {
            'surface': np.array(colors['surface'], dtype=np.float32) / 255.0,
            'ripple': np.array(colors['ripple'], dtype=np.float32) / 255.0,
            'foam': np.array(colors['foam'], dtype=np.float32) / 255.0,
        }