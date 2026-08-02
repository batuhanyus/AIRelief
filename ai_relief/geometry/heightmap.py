import numpy as np
import cv2
from ai_relief.config.settings import settings
import logging

class HeightmapGenerator:
    def __init__(self):
        pass

    def generate(self, optimized_depth: np.ndarray, smoothing_mode: str = "Sharp & Crisp (Bilateral)") -> np.ndarray:
        """
        Converts the normalized 0-1 depth map into a physical heightmap scaled 
        to the maximum relief depth (in mm). Applies selectable edge-preserving smoothing.
        """
        logging.info(f"Generating heightmap (Max Depth: {settings.relief_max_depth_mm}mm, Mode: '{smoothing_mode}').")
        
        depth_float = optimized_depth.astype(np.float32)

        if "bilateral" in smoothing_mode.lower() or "sharp" in smoothing_mode.lower():
            # Bilateral filter preserves sharp structural edges while smoothing out noise
            smoothed = cv2.bilateralFilter(depth_float, d=5, sigmaColor=0.08, sigmaSpace=3.0)
        elif "smooth" in smoothing_mode.lower():
            # Soft anti-aliased Gaussian smoothing
            smoothed = cv2.GaussianBlur(depth_float, (3, 3), 0.5)
        else:
            # Raw / Unfiltered
            smoothed = depth_float

        # Scale normalized [0.0, 1.0] range into physical millimeters
        heightmap_mm = smoothed * settings.relief_max_depth_mm
        
        return heightmap_mm
