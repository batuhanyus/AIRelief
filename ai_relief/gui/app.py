import gradio as gr
import numpy as np
import tempfile
import os
from PIL import Image
import logging

from ai_relief.config.settings import settings
from ai_relief.depth.depth_anything import DepthAnythingV2Wrapper
from ai_relief.segmentation.sam2_segmenter import SAM2Segmenter
from ai_relief.enhancement.face import FaceEnhancer
from ai_relief.enhancement.hair import HairEnhancer
from ai_relief.optimization.relief_compression import ReliefCompressor
from ai_relief.geometry.heightmap import HeightmapGenerator
from ai_relief.exporters.mesh_writer import MeshExporter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_models():
    """Lazily load models to avoid crashing on startup if weights are missing."""
    global depth_estimator, segmenter, face_enhancer, hair_enhancer, compressor, heightmap_gen, exporter
    try:
        depth_estimator = DepthAnythingV2Wrapper()
        segmenter = SAM2Segmenter()
        face_enhancer = FaceEnhancer()
        hair_enhancer = HairEnhancer()
        compressor = ReliefCompressor(alpha=10.0)
        heightmap_gen = HeightmapGenerator()
        exporter = MeshExporter()
        return True
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return False

# Attempt initial load
models_loaded = load_models()

def process_image(image_path, max_depth, base_thickness, detail_strength):
    if not models_loaded:
        return None, "Error: Models failed to load. Check your weights directory."
        
    if image_path is None:
        return None, "Please upload an image."

    logging.info("Starting pipeline...")
    # Update settings
    settings.relief_max_depth_mm = max_depth
    settings.base_thickness_mm = base_thickness
    
    # 1. Depth Estimation
    raw_depth = depth_estimator.estimate_depth(image_path)
    
    # 2. Segmentation
    mask = segmenter.segment_foreground(image_path)
    
    # 3. Enhancement
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    enhanced_depth = face_enhancer.enhance_faces(image_np, raw_depth, enhancement_strength=detail_strength)
    enhanced_depth = hair_enhancer.enhance_hair_details(image_np, enhanced_depth, mask, enhancement_strength=detail_strength)
    
    # 4. Compression
    compressed_depth = compressor.compress(enhanced_depth, mask)
    
    # 5. Geometry
    heightmap = heightmap_gen.generate(compressed_depth)
    
    # 6. Export to a temporary OBJ file for Gradio
    # Gradio's Model3D supports OBJ natively for browser viewing
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.obj")
    
    try:
        exporter.export(heightmap, output_path)
        return output_path, "Success! 3D Model Generated."
    except Exception as e:
        return None, f"Error generating mesh: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(title="AI Relief") as demo:
    gr.Markdown("# AI Relief")
    gr.Markdown("Convert 2D photos into high-quality, printable 3D bas-reliefs using Depth Anything V2 and SAM 2.")
    
    if not models_loaded:
        gr.Markdown("## ⚠️ Warning: AI Models not loaded. Please ensure you have downloaded the weights to `weights/` as per the settings.")
        
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Photo")
            
            with gr.Accordion("Geometry Settings", open=True):
                max_depth = gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=0.5, label="Max Relief Depth (mm)")
                base_thickness = gr.Slider(minimum=0.5, maximum=10.0, value=2.0, step=0.5, label="Base Thickness (mm)")
                
            with gr.Accordion("Enhancement Settings", open=True):
                detail_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Facial & Hair Detail Strength")
                
            submit_btn = gr.Button("Generate Bas-Relief", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            output_model = gr.Model3D(label="3D Preview (Interactable)", clear_color=(0.9, 0.9, 0.9, 1.0))
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, max_depth, base_thickness, detail_strength],
        outputs=[output_model, status_text]
    )

if __name__ == "__main__":
    demo.launch()
