import numpy as np
import trimesh
import cv2
import logging
from typing import Optional

from ai_relief.config.settings import settings
from ai_relief.visualization.renderer import ReliefRenderer

class MeshExporter:
    def __init__(self):
        self.renderer = ReliefRenderer()

    def _generate_grid_mesh(
        self, 
        heightmap: np.ndarray, 
        vertex_colors: Optional[np.ndarray] = None
    ) -> trimesh.Trimesh:
        """
        Converts a 2D physical heightmap into a watertight 3D trimesh object.
        It generates a top surface, a flat bottom base, and connects them with walls.
        """
        h, w = heightmap.shape
        
        # Scale X/Y to physical dimensions. We assume the longest side is 150mm for a standard print.
        max_dim_mm = 150.0 
        scale = max_dim_mm / max(h, w)
        
        # Create X, Y grid
        x = np.linspace(0, w * scale, w)
        y = np.linspace(0, h * scale, h)
        X, Y = np.meshgrid(x, y)
        
        # Flip heightmap vertically so row 0 (top of 2D image) aligns with maximum Y (top of 3D frame)
        heightmap_flipped = np.flipud(heightmap)
        
        # Z coordinates include the base thickness
        Z = heightmap_flipped + settings.base_thickness_mm
        
        # Create vertices
        top_vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        bottom_vertices = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(Z).flatten()))
        vertices = np.vstack((top_vertices, bottom_vertices))
        
        # Create faces for the top surface (triangulating the grid)
        i, j = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing='ij')
        
        tl = i * w + j
        tr = i * w + (j + 1)
        bl = (i + 1) * w + j
        br = (i + 1) * w + (j + 1)
        
        top_faces_1 = np.column_stack((tl.flatten(), tr.flatten(), bl.flatten()))
        top_faces_2 = np.column_stack((tr.flatten(), br.flatten(), bl.flatten()))
        top_faces = np.vstack((top_faces_1, top_faces_2))
        
        # Bottom faces (reversed winding order to point downwards)
        offset = h * w
        bottom_faces_1 = np.column_stack((tl.flatten() + offset, bl.flatten() + offset, tr.flatten() + offset))
        bottom_faces_2 = np.column_stack((tr.flatten() + offset, bl.flatten() + offset, br.flatten() + offset))
        bottom_faces = np.vstack((bottom_faces_1, bottom_faces_2))
        
        # Wall faces to make it watertight
        walls = []
        
        # Top edge
        for j_idx in range(w - 1):
            walls.extend([[j_idx, j_idx + offset, j_idx + 1], [j_idx + 1, j_idx + offset, j_idx + 1 + offset]])
            
        # Bottom edge
        for j_idx in range(w - 1):
            t1 = (h - 1) * w + j_idx
            t2 = (h - 1) * w + j_idx + 1
            walls.extend([[t1, t2, t1 + offset], [t2, t2 + offset, t1 + offset]])
            
        # Left edge
        for i_idx in range(h - 1):
            t1 = i_idx * w
            t2 = (i_idx + 1) * w
            walls.extend([[t1, t2, t1 + offset], [t2, t2 + offset, t1 + offset]])
            
        # Right edge
        for i_idx in range(h - 1):
            t1 = i_idx * w + (w - 1)
            t2 = (i_idx + 1) * w + (w - 1)
            walls.extend([[t1, t1 + offset, t2], [t2, t1 + offset, t2 + offset]])
            
        wall_faces = np.array(walls)
        
        # Combine all geometry
        all_faces = np.vstack((top_faces, bottom_faces, wall_faces))
        
        logging.info(f"Generated {len(vertices)} vertices and {len(all_faces)} faces.")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(
            vertices=vertices, 
            faces=all_faces, 
            vertex_colors=vertex_colors, 
            process=False
        )
        return mesh

    def export(
        self, 
        heightmap: np.ndarray, 
        output_path: str,
        image_np: Optional[np.ndarray] = None,
        preview_style: str = "clay",
        max_grid_dim: Optional[int] = 512
    ):
        """
        Generates the mesh and saves it to the specified format (GLB, OBJ, STL, etc.)
        Applies material shading and vertex colors when requested.
        """
        h, w = heightmap.shape
        
        # Optional grid downsampling for fast interactive 3D WebGL preview
        if max_grid_dim is not None and max(h, w) > max_grid_dim:
            scale_factor = max_grid_dim / float(max(h, w))
            new_w = max(16, int(w * scale_factor))
            new_h = max(16, int(h * scale_factor))
            logging.info(f"Resizing heightmap grid from ({w}x{h}) to ({new_w}x{new_h}) for preview rendering.")
            heightmap_processed = cv2.resize(heightmap, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if image_np is not None:
                image_processed = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                image_processed = None
        else:
            heightmap_processed = heightmap
            image_processed = image_np

        logging.info(f"Rendering material finish: '{preview_style}'...")
        top_vertex_colors, base_wall_color = self.renderer.render_material_colors(
            heightmap=heightmap_processed,
            image_np=image_processed,
            style=preview_style
        )

        bottom_count = len(top_vertex_colors)
        bottom_vertex_colors = np.tile(base_wall_color, (bottom_count, 1))
        all_vertex_colors = np.vstack((top_vertex_colors, bottom_vertex_colors))

        logging.info("Triangulating heightmap into watertight mesh...")
        mesh = self._generate_grid_mesh(heightmap_processed, vertex_colors=all_vertex_colors)
        
        logging.info(f"Saving mesh to {output_path}...")
        file_ext = output_path.split('.')[-1].lower()
        
        if file_ext == "glb":
            mesh.export(output_path, file_type="glb")
        elif file_ext == "stl":
            mesh.export(output_path, file_type="stl")
        else:
            mesh.export(output_path)
            
        logging.info("Export complete.")
