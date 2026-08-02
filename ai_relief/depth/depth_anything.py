import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from ai_relief.config.settings import settings
import logging

class DepthAnythingV2Wrapper:
    def __init__(self):
        self.device = "cpu"
        if settings.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif settings.device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            
        logging.info(f"Loading Depth Anything V2 from {settings.depth_model_path} onto {self.device}")
        
        try:
            # We use the transformers pipeline but pass the local path.
            # The user must have downloaded the HuggingFace format weights to this folder.
            self.pipe = pipeline(
                task="depth-estimation",
                model=settings.depth_model_path,
                device=self.device
            )
        except Exception as e:
            logging.error(f"Failed to load Depth Anything V2 model from {settings.depth_model_path}.")
            logging.error("Ensure you have manually downloaded the HuggingFace weights into this directory.")
            raise e

    def estimate_depth(self, image_path: str) -> np.ndarray:
        """
        Estimates depth from an image.
        Returns a normalized numpy array where values are between 0 and 1.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to open image at {image_path}: {e}")
            raise

        # Run inference
        result = self.pipe(image)
        
        # 'predicted_depth' is the raw 2D tensor. 'depth' is the PIL Image representation.
        if "predicted_depth" in result:
            depth_tensor = result["predicted_depth"]
            depth_array = depth_tensor.squeeze().cpu().numpy()
        else:
            depth_map_pil = result["depth"]
            depth_array = np.array(depth_map_pil, dtype=np.float32)
        
        # Ensure depth map spatial shape matches original image size (height, width)
        target_height, target_width = image.height, image.width
        if depth_array.shape[:2] != (target_height, target_width):
            depth_array = cv2.resize(depth_array, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max > depth_min:
            depth_array = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_array = np.zeros_like(depth_array)
            
        # Apply strength multiplier
        depth_array *= settings.depth_strength
        
        return depth_array
