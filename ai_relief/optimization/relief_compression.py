import numpy as np
import cv2
import logging

class ReliefCompressor:
    def __init__(self, alpha: float = 10.0):
        """
        alpha controls the strength of the logarithmic compression.
        Higher values compress the global depth more while preserving local details.
        """
        self.alpha = alpha

    def compress(self, depth_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Compresses linear depth into bas-relief depth using logarithmic scaling.
        If a mask is provided, the background is pushed flat to the lowest level.
        """
        logging.info("Applying non-linear depth compression for bas-relief.")
        
        # Ensure depth is strictly bounded between 0 and 1
        depth_map = np.clip(depth_map, 0.0, 1.0)
        
        # Logarithmic compression formula:
        # C(x) = log(1 + alpha * x) / log(1 + alpha)
        # This curve is steep at 0 (enhancing fine near details) and flattens out towards 1 (compressing far distances)
        compressed = np.log(1.0 + self.alpha * depth_map) / np.log(1.0 + self.alpha)
        
        if mask is not None:
            # Ensure mask spatial shape matches depth_map (height, width)
            dh, dw = depth_map.shape[:2]
            if mask.shape[:2] != (dh, dw):
                mask = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

            # Flatten background to ensure it doesn't waste precious physical thickness
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
            
            # The background becomes flat (0.0), foreground retains its compressed depth.
            compressed = compressed * mask
            
        return compressed
