import numpy as np
import cv2
from ai_relief.config.settings import settings
import logging

class HeightmapGenerator:
    def __init__(self):
        pass

    def generate(self, optimized_depth: np.ndarray, smoothing_mode: str = "Sharp & Crisp (Bilateral)", image_np: np.ndarray = None, apply_unsharp_mask: bool = False) -> np.ndarray:
        """
        Converts the normalized 0-1 depth map into a physical heightmap scaled 
        to the maximum relief depth (in mm). Applies selectable edge-preserving smoothing.
        """
        logging.info(f"Generating heightmap (Max Depth: {settings.relief_max_depth_mm}mm, Mode: '{smoothing_mode}', Unsharp: {apply_unsharp_mask}).")
        
        depth_float = optimized_depth.astype(np.float32)

        if "guided" in smoothing_mode.lower() and image_np is not None:
            gray_guide = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            if depth_float.shape[:2] != gray_guide.shape[:2]:
                gray_guide = cv2.resize(gray_guide, (depth_float.shape[1], depth_float.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            try:
                guided_filter = cv2.ximgproc.createGuidedFilter(guide=gray_guide, radius=5, eps=0.01)
                smoothed = guided_filter.filter(depth_float)
            except AttributeError:
                logging.warning("cv2.ximgproc not found (opencv-contrib-python missing). Falling back to edgePreservingFilter.")
                depth_uint8 = (np.clip(depth_float, 0, 1) * 255).astype(np.uint8)
                smoothed_uint8 = cv2.edgePreservingFilter(depth_uint8, flags=1, sigma_s=10, sigma_r=0.1)
                smoothed = smoothed_uint8.astype(np.float32) / 255.0
        elif "bilateral" in smoothing_mode.lower() or "sharp" in smoothing_mode.lower():
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
        
        if apply_unsharp_mask:
            logging.info("Applying Unsharp Mask to physical heightmap.")
            blur_pass = cv2.GaussianBlur(heightmap_mm, (0, 0), 1.5)
            # USM formula: original + (original - blurred) * amount
            # Using amount = 0.5
            unsharp_amount = 0.5
            heightmap_mm = heightmap_mm + (heightmap_mm - blur_pass) * unsharp_amount
            # Clamp to prevent going below base or creating extreme spikes
            heightmap_mm = np.clip(heightmap_mm, 0.0, settings.relief_max_depth_mm * 1.5)
            
        return heightmap_mm
