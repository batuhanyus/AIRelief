import numpy as np
import trimesh
import logging
from ai_relief.config.settings import settings

class MeshExporter:
    def __init__(self):
        pass

    def _generate_grid_mesh(self, heightmap: np.ndarray) -> trimesh.Trimesh:
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
        
        # Z coordinates include the base thickness
        Z = heightmap + settings.base_thickness_mm
        
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
        for j in range(w - 1):
            walls.extend([[j, j + offset, j + 1], [j + 1, j + offset, j + 1 + offset]])
            
        # Bottom edge
        for j in range(w - 1):
            t1 = (h - 1) * w + j
            t2 = (h - 1) * w + j + 1
            walls.extend([[t1, t2, t1 + offset], [t2, t2 + offset, t1 + offset]])
            
        # Left edge
        for i in range(h - 1):
            t1 = i * w
            t2 = (i + 1) * w
            walls.extend([[t1, t2, t1 + offset], [t2, t2 + offset, t1 + offset]])
            
        # Right edge
        for i in range(h - 1):
            t1 = i * w + (w - 1)
            t2 = (i + 1) * w + (w - 1)
            walls.extend([[t1, t1 + offset, t2], [t2, t1 + offset, t2 + offset]])
            
        wall_faces = np.array(walls)
        
        # Combine all geometry
        all_faces = np.vstack((top_faces, bottom_faces, wall_faces))
        
        logging.info(f"Generated {len(vertices)} vertices and {len(all_faces)} faces.")
        
        # Create trimesh object (process=True removes duplicate vertices and fixes normals)
        mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)
        return mesh

    def export(self, heightmap: np.ndarray, output_path: str):
        """
        Generates the mesh and saves it to the specified format (STL, OBJ, etc.) based on the extension.
        """
        logging.info("Triangulating heightmap into watertight mesh...")
        mesh = self._generate_grid_mesh(heightmap)
        
        logging.info(f"Saving mesh to {output_path}...")
        mesh.export(output_path)
        logging.info("Export complete.")
