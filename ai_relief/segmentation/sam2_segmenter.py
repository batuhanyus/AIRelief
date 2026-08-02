import torch
import numpy as np
from PIL import Image
import logging
from ai_relief.config.settings import settings

class SAM2Segmenter:
    def __init__(self):
        self.device = "cpu"
        if settings.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif settings.device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            
        logging.info(f"Loading SAM 2 from {settings.sam2_model_path} onto {self.device}")
        
        try:
            # NOTE: To use SAM 2, you need to install the `sam2` package from Meta's GitHub repository.
            # Once installed, you can uncomment and use the following lines:
            # from sam2.build_sam import build_sam2
            # from sam2.sam2_image_predictor import SAM2ImagePredictor
            # import os
            # 
            # model_cfg = "sam2.1_hiera_l.yaml" 
            # model_ckpt = os.path.join(settings.sam2_model_path, "sam2.1_hiera_large.pt")
            # sam2_model = build_sam2(model_cfg, model_ckpt, device=self.device)
            # self.predictor = SAM2ImagePredictor(sam2_model)
            
            self.predictor = None # Placeholder until SAM2 is installed
            logging.warning("SAM 2 library not wired up. Using a dummy segmentation mask for testing.")
        except Exception as e:
            logging.error(f"Failed to load SAM 2 model from {settings.sam2_model_path}.")
            raise e

    def segment_foreground(self, image_path: str) -> np.ndarray:
        """
        Segments the foreground object from the background.
        Returns a binary mask (numpy array) where foreground is 1 and background is 0.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to open image at {image_path}: {e}")
            raise
            
        image_np = np.array(image)
        
        # If SAM2 is loaded, run it. Otherwise, return a dummy mask.
        if self.predictor is not None:
            self.predictor.set_image(image_np)
            # A common approach for automatic foreground extraction without points is using a center point
            # or generating an automatic mask. For a photo, the center point usually hits the subject.
            height, width = image_np.shape[:2]
            center_point = np.array([[width // 2, height // 2]])
            point_labels = np.array([1]) # 1 indicates a foreground point
            
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=point_labels,
                multimask_output=False
            )
            return masks[0].astype(np.uint8)
        else:
            # Dummy Mask: Extract the middle 50% for testing purposes
            height, width = image_np.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            h_margin = int(height * 0.25)
            w_margin = int(width * 0.25)
            mask[h_margin:-h_margin, w_margin:-w_margin] = 1
            return mask
