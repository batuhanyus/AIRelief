import os
import unittest
import tempfile
import numpy as np
import trimesh

from ai_relief.exporters.mesh_writer import MeshExporter

class TestMeshExporter(unittest.TestCase):
    def test_mesh_exporter_stl_export(self):
        """Test exporting a synthetic heightmap to STL format and verifying mesh properties."""
        exporter = MeshExporter()
        
        # Create synthetic 32x32 heightmap
        h, w = 32, 32
        heightmap = np.random.uniform(0.0, 5.0, (h, w)).astype(np.float32)
        image_np = np.zeros((h, w, 3), dtype=np.uint8) + 128
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = os.path.join(temp_dir, "test_output.stl")
            
            # Perform export
            exporter.export(
                heightmap=heightmap,
                output_path=stl_path,
                image_np=image_np,
                preview_style="clay",
                max_grid_dim=64
            )
            
            self.assertTrue(os.path.exists(stl_path), "STL file was not created")
            self.assertGreater(os.path.getsize(stl_path), 0, "STL file is empty")
            
            # Load mesh with trimesh and verify geometry
            mesh = trimesh.load(stl_path)
            self.assertIsInstance(mesh, trimesh.Trimesh, "Exported object is not a valid Trimesh")
            self.assertGreater(len(mesh.vertices), 0, "Mesh has no vertices")
            self.assertGreater(len(mesh.faces), 0, "Mesh has no faces")
            self.assertTrue(mesh.is_watertight, "Exported STL mesh is not watertight")

    def test_mesh_exporter_glb_export(self):
        """Test exporting a synthetic heightmap to GLB format."""
        exporter = MeshExporter()
        
        h, w = 32, 32
        heightmap = np.ones((h, w), dtype=np.float32) * 2.0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            glb_path = os.path.join(temp_dir, "test_output.glb")
            
            exporter.export(
                heightmap=heightmap,
                output_path=glb_path,
                preview_style="antique bronze",
                max_grid_dim=64
            )
            
            self.assertTrue(os.path.exists(glb_path), "GLB file was not created")
            self.assertGreater(os.path.getsize(glb_path), 0, "GLB file is empty")

if __name__ == "__main__":
    unittest.main()
