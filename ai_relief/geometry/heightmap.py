import numpy as np
import cv2
from ai_relief.config.settings import settings
import logging

class HeightmapGenerator:
    def __init__(self):
        pass

    def generate(self, optimized_depth: np.ndarray) -> np.ndarray:
        """
        Converts the normalized 0-1 optimized depth map into a physical heightmap 
        scaled to the maximum relief depth (in mm). Applies anti-aliasing.
        """
        logging.info(f"Generating heightmap with max depth of {settings.relief_max_depth_mm}mm.")
        
        # Apply a very slight Gaussian blur to act as anti-aliasing for the 3D mesh.
        # This prevents pixel-level "stair-stepping" artifacts on high-resolution 3D prints.
        smoothed = cv2.GaussianBlur(optimized_depth, (3, 3), 0.5)
        
        # Scale the normalized 0.0 - 1.0 range directly into physical millimeters
        heightmap_mm = smoothed * settings.relief_max_depth_mm
        
        return heightmap_mm
