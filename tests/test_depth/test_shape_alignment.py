import numpy as np
from ai_relief.enhancement.hair import HairEnhancer
from ai_relief.enhancement.face import FaceEnhancer
from ai_relief.optimization.relief_compression import ReliefCompressor

def test_hair_enhancer_shape_alignment():
    hair_enhancer = HairEnhancer()
    image_np = np.zeros((579, 693, 3), dtype=np.uint8)
    depth_map = np.zeros((518, 616), dtype=np.float32)
    mask = np.ones((579, 693), dtype=np.uint8)
    
    enhanced = hair_enhancer.enhance_hair_details(image_np, depth_map, mask, enhancement_strength=0.3)
    assert enhanced.shape == (579, 693)

def test_face_enhancer_shape_alignment():
    face_enhancer = FaceEnhancer()
    image_np = np.zeros((579, 693, 3), dtype=np.uint8)
    depth_map = np.zeros((518, 616), dtype=np.float32)
    
    enhanced = face_enhancer.enhance_faces(image_np, depth_map, enhancement_strength=0.5)
    assert enhanced.shape == (579, 693)

def test_relief_compressor_shape_alignment():
    compressor = ReliefCompressor()
    depth_map = np.zeros((579, 693), dtype=np.float32)
    mask = np.ones((518, 616), dtype=np.uint8)
    
    compressed = compressor.compress(depth_map, mask)
    assert compressed.shape == (579, 693)

if __name__ == '__main__':
    test_hair_enhancer_shape_alignment()
    test_face_enhancer_shape_alignment()
    test_relief_compressor_shape_alignment()
    print("ALL TESTS PASSED SUCCESSFULLY!")

