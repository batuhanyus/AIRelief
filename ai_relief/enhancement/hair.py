import cv2
import numpy as np
import logging

class HairEnhancer:
    def __init__(self):
        pass

    def enhance_hair_details(self, image_np: np.ndarray, depth_map: np.ndarray, mask: np.ndarray, enhancement_strength: float = 0.3) -> np.ndarray:
        """
        Extracts high-frequency details from the image (often representing hair or fine clothing textures) 
        and injects them into the depth map, constrained by the foreground mask.
        """
        img_h, img_w = image_np.shape[:2]
        if depth_map.shape[:2] != (img_h, img_w):
            depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        enhanced_depth = depth_map.copy()
        
        # Convert original image to grayscale for detail extraction
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Use a high-pass filter (Original - Blurred) to get fine details
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        high_pass = cv2.subtract(gray, blurred)
        
        # Normalize high frequencies around 0
        high_pass_norm = (high_pass.astype(np.float32) / 255.0) - 0.5
        
        # We only want to apply this strongly to areas with high texture (like hair).
        # We can use Canny edge detection to create a "texture activity" map.
        edges = cv2.Canny(gray, 100, 200)
        # Blur the edges so the enhancement applies smoothly around textured areas, not just strictly on the edge pixel
        texture_mask = cv2.GaussianBlur(edges, (5, 5), 0).astype(np.float32) / 255.0
        
        # Ensure mask is 0-1 float
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32)
            
        # Apply the details:
        # Details * Texture Mask (only applied where it's actually textured) * Subject Mask (only on foreground) * Strength
        final_detail = high_pass_norm * texture_mask * mask * enhancement_strength
        
        enhanced_depth += final_detail
        
        # Keep depth map properly bounded
        enhanced_depth = np.clip(enhanced_depth, 0.0, 1.0)
        
        return enhanced_depth
