import cv2
import numpy as np
import logging
from typing import Optional

class MultiScaleDetailEnhancer:
    """
    Multi-Scale Laplacian & Edge Frequency Decomposition Enhancer.
    Extracts micro-textures (skin, fabric, hair, patterns), mid-frequency contours,
    and crisp structural edges from RGB photos and injects them into depth maps.
    """

    def __init__(self):
        pass

    def enhance_details(
        self,
        image_np: np.ndarray,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
        micro_detail_strength: float = 0.5,
        edge_sharpness: float = 0.5
    ) -> np.ndarray:
        """
        Enhances the depth map with fine micro-textures and sharp structural edges.

        Args:
            image_np: (H, W, 3) RGB uint8 image.
            depth_map: (H, W) float depth map normalized to [0, 1].
            mask: Optional (H, W) foreground mask (0.0 - 1.0 or uint8).
            micro_detail_strength: Scaling factor for micro-texture details [0.0, 1.0].
            edge_sharpness: Scaling factor for sharp structural contours [0.0, 1.0].

        Returns:
            Enhanced depth map in range [0.0, 1.0].
        """
        img_h, img_w = image_np.shape[:2]

        if depth_map.shape[:2] != (img_h, img_w):
            depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        enhanced_depth = depth_map.copy()

        # Convert image to grayscale float [0, 1]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # 1. Micro-texture extraction using high-pass filtering (Laplacian / Unsharp Masking)
        blur_fine = cv2.GaussianBlur(gray, (0, 0), 1.2)
        micro_high_pass = gray - blur_fine

        # 2. Mid-frequency structure extraction
        blur_mid = cv2.GaussianBlur(gray, (0, 0), 4.0)
        mid_high_pass = blur_fine - blur_mid

        # 3. Structural Edge Contours using Sobel/Scharr gradients
        sobel_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize edge magnitude to [0, 1] safely
        max_edge = edge_mag.max()
        if max_edge > 1e-5:
            edge_mag_norm = edge_mag / max_edge
        else:
            edge_mag_norm = np.zeros_like(edge_mag)

        # Directional edge sign alignment (makes depth pop outward at object borders)
        edge_detail = micro_high_pass * edge_mag_norm

        # Prepare foreground mask constraint
        if mask is not None:
            if mask.shape[:2] != (img_h, img_w):
                mask_res = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_res = mask
            mask_float = mask_res.astype(np.float32) if mask_res.dtype != np.float32 else mask_res
        else:
            mask_float = np.ones((img_h, img_w), dtype=np.float32)

        # Combine micro-texture and edge sharpness enhancements with high physical depth gain
        texture_injection = micro_high_pass * 0.35 * micro_detail_strength
        mid_injection = mid_high_pass * 0.25 * micro_detail_strength
        edge_injection = edge_detail * 0.45 * edge_sharpness

        total_detail = (texture_injection + mid_injection + edge_injection) * mask_float

        # Add total detail into depth map and keep strictly bounded
        enhanced_depth = np.clip(enhanced_depth + total_detail, 0.0, 1.0)

        logging.info(f"Applied multi-scale detail enhancement (Micro: {micro_detail_strength:.2f}, Edges: {edge_sharpness:.2f}).")
        return enhanced_depth
