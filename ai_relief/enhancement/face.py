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
        Detects faces in the image and applies anatomical depth profiling 
        (nose elevation, eye socket recessing) and local contrast/sharpness enhancement 
        to ensure facial features pop with rich 3D relief depth.
        """
        img_h, img_w = image_np.shape[:2]
        if depth_map.shape[:2] != (img_h, img_w):
            depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        enhanced_depth = depth_map.copy()
        
        # Convert image to RGB for MediaPipe
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
                
            # Extract face region
            face_img_region = image_np[y_min:y_max, x_min:x_max]
            face_depth_region = enhanced_depth[y_min:y_max, x_min:x_max]
            bh, bw = y_max - y_min, x_max - x_min

            # 1. Anatomical Face Dome Profile (Nose/Cheek Elevation)
            gy, gx = np.ogrid[:bh, :bw]
            cy, cx = bh * 0.45, bw * 0.5 # Nose tip area estimate
            dist_sq = ((gx - cx) / (bw * 0.5))**2 + ((gy - cy) / (bh * 0.5))**2
            face_dome = np.clip(1.0 - dist_sq, 0.0, 1.0)
            face_dome = np.sin(face_dome * np.pi / 2.0)
            
            # 2. Extract high frequency facial details (eyes, lips, mustache, eyebrows)
            gray_face = cv2.cvtColor(face_img_region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            blurred_face = cv2.GaussianBlur(gray_face, (0, 0), 2.5)
            high_freq = gray_face - blurred_face

            # 3. Extract sharp facial contours via Sobel
            sobel_x = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
            edge_face = np.sqrt(sobel_x**2 + sobel_y**2)
            if edge_face.max() > 1e-5:
                edge_face /= edge_face.max()

            # Feather mask to blend bounding box edges smoothly
            win_y = np.hanning(bh) if bh > 1 else np.ones(bh)
            win_x = np.hanning(bw) if bw > 1 else np.ones(bw)
            feather_mask = np.outer(win_y, win_x).astype(np.float32)

            # Combine anatomical dome with facial feature depth pop
            dome_boost = face_dome * 0.08 * enhancement_strength
            detail_to_add = (high_freq * 0.35 + high_freq * edge_face * 0.25) * feather_mask * enhancement_strength
            
            face_depth_region = face_depth_region + dome_boost + detail_to_add
            face_depth_region = np.clip(face_depth_region, 0.0, 1.0)

            enhanced_depth[y_min:y_max, x_min:x_max] = face_depth_region
            
        return enhanced_depth
