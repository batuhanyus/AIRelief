import cv2
import numpy as np
import mediapipe as mp
import logging

class FaceEnhancer:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, # 1 is suited for faces farther from the camera
            min_detection_confidence=0.5
        )

    def enhance_faces(self, image_np: np.ndarray, depth_map: np.ndarray, enhancement_strength: float = 0.5) -> np.ndarray:
        """
        Detects faces in the image and applies local contrast/sharpness enhancement 
        to those specific regions in the depth map. This ensures facial features 
        are preserved when printed as a bas-relief.
        """
        img_h, img_w = image_np.shape[:2]
        if depth_map.shape[:2] != (img_h, img_w):
            depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        enhanced_depth = depth_map.copy()
        
        # Convert image to RGB for MediaPipe (Assuming image_np is already RGB from PIL)
        results = self.face_detection.process(image_np)
        
        if not results.detections:
            return enhanced_depth
            
        height, width = image_np.shape[:2]
        
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute pixels
            x_min = max(0, int(bboxC.xmin * width))
            y_min = max(0, int(bboxC.ymin * height))
            box_width = int(bboxC.width * width)
            box_height = int(bboxC.height * height)
            
            x_max = min(width, x_min + box_width)
            y_max = min(height, y_min + box_height)
            
            # Skip if bounding box is invalid
            if box_width <= 0 or box_height <= 0:
                continue
                
            # Extract the face region from the original image and depth map
            face_img_region = image_np[y_min:y_max, x_min:x_max]
            face_depth_region = enhanced_depth[y_min:y_max, x_min:x_max]
            
            # Extract high frequency details from the luminance of the original image
            gray_face = cv2.cvtColor(face_img_region, cv2.COLOR_RGB2GRAY)
            
            # Apply unsharp masking to extract details
            blurred_face = cv2.GaussianBlur(gray_face, (0, 0), 3)
            high_freq = cv2.subtract(gray_face, blurred_face)
            
            # Normalize high frequency to a small range (e.g., -0.05 to 0.05) to add to normalized depth map
            high_freq_norm = (high_freq.astype(np.float32) / 255.0) - 0.5
            high_freq_norm *= enhancement_strength
            
            # Add details to depth map
            face_depth_region = face_depth_region + high_freq_norm
            
            # Clip back to 0-1 range
            face_depth_region = np.clip(face_depth_region, 0.0, 1.0)
            
            # Re-inject the enhanced region back into the main depth map
            enhanced_depth[y_min:y_max, x_min:x_max] = face_depth_region
            
        return enhanced_depth
