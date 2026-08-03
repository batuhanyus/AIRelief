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
from ai_relief.enhancement.detail_enhancer import MultiScaleDetailEnhancer
from ai_relief.optimization.relief_compression import ReliefCompressor
from ai_relief.geometry.heightmap import HeightmapGenerator
from ai_relief.exporters.mesh_writer import MeshExporter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_models():
    """Lazily load models to avoid crashing on startup if weights are missing."""
    global depth_estimator, segmenter, face_enhancer, hair_enhancer, detail_enhancer, compressor, heightmap_gen, exporter
    try:
        depth_estimator = DepthAnythingV2Wrapper()
        segmenter = SAM2Segmenter()
        face_enhancer = FaceEnhancer()
        hair_enhancer = HairEnhancer()
        detail_enhancer = MultiScaleDetailEnhancer()
        compressor = ReliefCompressor(alpha=10.0)
        heightmap_gen = HeightmapGenerator()
        exporter = MeshExporter()
        return True
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return False

# Attempt initial load
models_loaded = load_models()

def process_image(
    image_path,
    max_depth,
    base_thickness,
    detail_strength,
    micro_detail_strength,
    edge_sharpness,
    smoothing_mode,
    mesh_resolution,
    remove_background,
    preview_style,
    device_setting
):
    if not models_loaded:
        return None, "Error: Models failed to load. Check your weights directory."
        
    if image_path is None:
        return None, "Please upload an image."

    dev_map = {"Auto (GPU if available)": "auto", "CUDA (GPU)": "cuda", "CPU": "cpu"}
    settings.device = dev_map.get(device_setting, "auto")
    
    logging.info(f"Starting detail-enhanced pipeline on device: {settings.get_device_info()}...")
    # Update settings
    settings.relief_max_depth_mm = max_depth
    settings.base_thickness_mm = base_thickness
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 1. Depth Estimation
    raw_depth = depth_estimator.estimate_depth(image)
    
    # 2. Segmentation / Masking
    if remove_background:
        mask = segmenter.segment_foreground(image)
    else:
        mask = np.ones(image_np.shape[:2], dtype=np.uint8)
    
    # 3. Compression (compress the macro global structure first)
    compressed_depth = compressor.compress(raw_depth, mask)

    # 4. Multi-Stage Detail Enhancement (layer details on top of the compressed base)
    # Face & Hair local enhancement
    enhanced_depth = face_enhancer.enhance_faces(image_np, compressed_depth, enhancement_strength=detail_strength)
    enhanced_depth = hair_enhancer.enhance_hair_details(image_np, enhanced_depth, mask, enhancement_strength=detail_strength)
    
    # Multi-Scale Laplacian & Structural Edge enhancement
    enhanced_depth = detail_enhancer.enhance_details(
        image_np=image_np,
        depth_map=enhanced_depth,
        mask=mask,
        micro_detail_strength=micro_detail_strength,
        edge_sharpness=edge_sharpness
    )
    
    # The enhanced depth is now our final compressed depth with details preserved
    compressed_depth = enhanced_depth
    
    # 5. Heightmap Generation with Edge-Preserving Filter
    heightmap = heightmap_gen.generate(compressed_depth, smoothing_mode=smoothing_mode)
    
    # Parse mesh grid resolution choice
    res_map = {
        "Standard (512px)": 512,
        "High Detail (768px)": 768,
        "Ultra Detail / 3D Print (1024px)": 1024,
        "Native Image Resolution": None
    }
    target_grid_dim = res_map.get(mesh_resolution, 512)

    # 6. Export to temporary GLB (for 3D viewer) and STL (for 3D printing download) files
    temp_dir = tempfile.mkdtemp()
    output_glb_path = os.path.join(temp_dir, "bas_relief.glb")
    output_stl_path = os.path.join(temp_dir, "bas_relief.stl")
    
    try:
        # Export GLB with preview material finish
        exporter.export(
            heightmap=heightmap, 
            output_path=output_glb_path, 
            image_np=image_np, 
            preview_style=preview_style,
            max_grid_dim=target_grid_dim
        )
        # Export STL file for 3D printing
        exporter.export(
            heightmap=heightmap,
            output_path=output_stl_path,
            image_np=image_np,
            preview_style=preview_style,
            max_grid_dim=target_grid_dim
        )
        grid_desc = f"{target_grid_dim}px grid" if target_grid_dim else "Native image resolution grid"
        return output_glb_path, [output_stl_path, output_glb_path], f"Success! 3D Model & STL Export Generated at {grid_desc} using '{preview_style}' material on {settings.get_device_info()}."
    except Exception as e:
        logging.error(f"Error generating mesh: {e}", exc_info=True)
        return None, None, f"Error generating mesh: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(title="AI Relief") as demo:
    gr.Markdown("# AI Relief")
    gr.Markdown("Convert 2D photos into high-quality, printable 3D bas-reliefs using Depth Anything V2 and SAM 2.")
    
    if not models_loaded:
        gr.Markdown("## ⚠️ Warning: AI Models not loaded. Please ensure you have downloaded the weights to `weights/` as per the settings.")
        
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Photo")
            
            with gr.Accordion("Hardware & Device Settings", open=False):
                device_setting = gr.Dropdown(
                    choices=["Auto (GPU if available)", "CUDA (GPU)", "CPU"],
                    value="Auto (GPU if available)",
                    label="Compute Hardware Device"
                )
                device_info_box = gr.Markdown(f"**Active Hardware:** {settings.get_device_info()}")

            with gr.Accordion("Geometry & Resolution Settings", open=True):
                max_depth = gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=0.5, label="Max Relief Depth (mm)")
                base_thickness = gr.Slider(minimum=0.5, maximum=10.0, value=2.0, step=0.5, label="Base Thickness (mm)")
                mesh_resolution = gr.Dropdown(
                    choices=[
                        "Standard (512px)",
                        "High Detail (768px)",
                        "Ultra Detail / 3D Print (1024px)",
                        "Native Image Resolution"
                    ],
                    value="High Detail (768px)",
                    label="3D Mesh Grid Resolution"
                )
                remove_background = gr.Checkbox(label="Remove Background (Isolate Subject)", value=False)
                
            with gr.Accordion("Multi-Scale Detail Enhancement", open=True):
                micro_detail_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="Micro-Texture Detail (Skin, Fabric, Patterns)")
                edge_sharpness = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="Edge & Contour Sharpness")
                detail_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Facial & Hair Feature Boost")
                smoothing_mode = gr.Dropdown(
                    choices=[
                        "Sharp & Crisp (Bilateral)",
                        "Smooth Anti-Aliased",
                        "Raw (Unfiltered)"
                    ],
                    value="Sharp & Crisp (Bilateral)",
                    label="Surface Edge Smoothing Mode"
                )
                preview_style = gr.Dropdown(
                    choices=[
                        "Clay Sculpture",
                        "Original Photo",
                        "Antique Bronze",
                        "White Marble",
                        "Gold Medallion",
                        "Detail Inspector"
                    ],
                    value="Clay Sculpture",
                    label="3D Preview Material / Finish"
                )
                
            submit_btn = gr.Button("Generate High-Detail Bas-Relief", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            output_model = gr.Model3D(
                label="3D Preview (Interactable)", 
                clear_color=(0.14, 0.15, 0.18, 1.0)
            )
            download_file = gr.File(
                label="Download 3D Model (.stl / .glb)",
                file_count="multiple"
            )
            
    submit_btn.click(
        fn=process_image,
        inputs=[
            input_image, 
            max_depth, 
            base_thickness, 
            detail_strength, 
            micro_detail_strength,
            edge_sharpness,
            smoothing_mode,
            mesh_resolution,
            remove_background, 
            preview_style,
            device_setting
        ],
        outputs=[output_model, download_file, status_text]
    )

if __name__ == "__main__":
    demo.launch()
