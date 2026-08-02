import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from PIL import Image

class ReliefRenderer:
    """
    Renders 3D bas-relief material finishes, surface normal shading,
    directional lighting, specular highlights, and texture overlays.
    """

    PREVIEW_STYLES = {
        "Clay Sculpture": "clay",
        "Original Photo": "photo",
        "Antique Bronze": "bronze",
        "White Marble": "marble",
        "Gold Medallion": "gold",
        "Detail Inspector": "detail"
    }

    def __init__(self):
        pass

    def compute_normals_and_shading(
        self, 
        heightmap: np.ndarray, 
        scale_mm: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes unit surface normals (Nx, Ny, Nz) and directional diffuse + curvature shading.
        """
        h, w = heightmap.shape
        
        # Spatial gradients scaled to millimeter dimensions
        dz_dy, dz_dx = np.gradient(heightmap, scale_mm, scale_mm)
        
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(heightmap)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        
        nx /= norm
        ny /= norm
        nz /= norm

        # Key directional light vector (from top-left front)
        lx, ly, lz = 0.4, 0.6, 0.7
        l_len = np.sqrt(lx**2 + ly**2 + lz**2)
        lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

        # Fill light vector (from bottom-right front)
        fx, fy, fz = -0.3, -0.4, 0.6
        f_len = np.sqrt(fx**2 + fy**2 + fz**2)
        fx, fy, fz = fx / f_len, fy / f_len, fz / f_len

        key_diffuse = np.maximum(0.0, nx * lx + ny * ly + nz * lz)
        fill_diffuse = np.maximum(0.0, nx * fx + ny * fy + nz * fz)

        # Combined directional light
        diffuse = 0.75 * key_diffuse + 0.25 * fill_diffuse
        diffuse = np.clip(diffuse, 0.15, 1.0)

        # Compute curvature / Laplacian for crevice ambient occlusion darkening
        lap = cv2.Laplacian(heightmap.astype(np.float64), cv2.CV_64F)
        max_lap = np.max(np.abs(lap)) + 1e-6
        curvature = np.clip(lap / max_lap, -1.0, 1.0)

        return nx, ny, nz, diffuse, curvature

    def render_material_colors(
        self,
        heightmap: np.ndarray,
        image_np: Optional[np.ndarray] = None,
        style: str = "clay"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates RGBA vertex color arrays for the top surface and base/walls of the mesh.
        
        Returns:
            top_vertex_colors: (H*W, 4) uint8 RGBA array aligned with flipped mesh coordinates.
            base_color_rgba: (4,) uint8 RGBA array for solid base/walls.
        """
        h, w = heightmap.shape
        max_dim_mm = 150.0
        scale_mm = max_dim_mm / max(h, w)

        nx, ny, nz, diffuse, curvature = self.compute_normals_and_shading(heightmap, scale_mm)
        style_clean = style.lower().strip()

        # Map display names if necessary
        for label, val in self.PREVIEW_STYLES.items():
            if style.lower() == label.lower():
                style_clean = val
                break

        base_wall_color = np.array([60, 60, 65, 255], dtype=np.uint8)

        if style_clean == "photo" and image_np is not None:
            # Original Photo overlay
            img_rgb = Image.fromarray(image_np).resize((w, h), Image.Resampling.LANCZOS)
            img_rgb_np = np.array(img_rgb, dtype=np.float32)
            
            # Modulate photo colors with subtle 3D lighting (70% photo, 30% directional light)
            shaded_rgb = img_rgb_np * (0.6 + 0.4 * diffuse[..., None])
            top_rgb = np.clip(shaded_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([40, 40, 45, 255], dtype=np.uint8)

        elif style_clean == "bronze":
            # Antique Bronze with golden patina highlights
            base_bronze = np.array([75.0, 58.0, 45.0])
            highlight_gold = np.array([215.0, 175.0, 95.0])
            patina_dark = np.array([30.0, 22.0, 18.0])

            # High curvature / high relief depth gets golden highlights
            rel_height = (heightmap - heightmap.min()) / (heightmap.ptp() + 1e-6)
            highlight_factor = np.clip(0.6 * rel_height + 0.4 * diffuse + 0.2 * curvature, 0, 1)

            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    highlight_factor > 0.5,
                    base_bronze[c] + (highlight_gold[c] - base_bronze[c]) * (highlight_factor - 0.5) * 2.0,
                    patina_dark[c] + (base_bronze[c] - patina_dark[c]) * highlight_factor * 2.0
                )
            top_rgb = np.clip(top_rgb * (0.7 + 0.3 * diffuse[..., None]), 0, 255).astype(np.uint8)
            base_wall_color = np.array([35, 28, 22, 255], dtype=np.uint8)

        elif style_clean == "marble":
            # White Alabaster Marble with soft slate grey crevice shadows
            marble_white = np.array([242.0, 242.0, 245.0])
            slate_shadow = np.array([135.0, 140.0, 152.0])

            shade_factor = np.clip(diffuse + 0.2 * curvature, 0.2, 1.0)
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = slate_shadow[c] + (marble_white[c] - slate_shadow[c]) * shade_factor

            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([100, 105, 115, 255], dtype=np.uint8)

        elif style_clean == "gold":
            # Polished Gold Medallion
            gold_base = np.array([230.0, 180.0, 45.0])
            gold_shine = np.array([255.0, 248.0, 190.0])
            amber_shadow = np.array([115.0, 75.0, 15.0])

            shine_factor = np.clip(diffuse**1.8 + 0.3 * curvature, 0, 1)
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    shine_factor > 0.5,
                    gold_base[c] + (gold_shine[c] - gold_base[c]) * (shine_factor - 0.5) * 2.0,
                    amber_shadow[c] + (gold_base[c] - amber_shadow[c]) * shine_factor * 2.0
                )
            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([75, 50, 15, 255], dtype=np.uint8)

        elif style_clean == "detail":
            # Surface Normal Map Shading
            r = ((nx + 1.0) * 0.5 * 255.0) * diffuse
            g = ((ny + 1.0) * 0.5 * 255.0) * diffuse
            b = ((nz + 1.0) * 0.5 * 255.0) * diffuse
            top_rgb = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)
            base_wall_color = np.array([30, 30, 35, 255], dtype=np.uint8)

        else:
            # Default: Clay Sculpture (Terracotta)
            clay_base = np.array([210.0, 140.0, 110.0])
            clay_highlight = np.array([245.0, 195.0, 168.0])
            clay_shadow = np.array([125.0, 70.0, 50.0])

            shade_factor = np.clip(diffuse + 0.15 * curvature, 0, 1)
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    shade_factor > 0.5,
                    clay_base[c] + (clay_highlight[c] - clay_base[c]) * (shade_factor - 0.5) * 2.0,
                    clay_shadow[c] + (clay_base[c] - clay_shadow[c]) * shade_factor * 2.0
                )
            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([80, 50, 40, 255], dtype=np.uint8)

        # Add Alpha channel
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        top_rgba = np.concatenate([top_rgb, alpha], axis=-1)

        # Flip vertically to align with MeshExporter coordinate flipping
        top_rgba_flipped = np.flipud(top_rgba)
        top_vertex_colors = top_rgba_flipped.reshape(-1, 4)

        return top_vertex_colors, base_wall_color
