import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from PIL import Image

class ReliefRenderer:
    """
    Renders 3D bas-relief material finishes, surface normal shading,
    directional lighting, specular highlights, and crevice ambient occlusion.
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
        scale_mm: float = 1.0,
        normal_gain: float = 18.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes enhanced unit surface normals (Nx, Ny, Nz), directional diffuse,
        specular highlights, and crevice ambient occlusion.
        """
        h, w = heightmap.shape
        
        # Spatial gradients
        dz_dy, dz_dx = np.gradient(heightmap)
        
        # Multiply gradients by normal_gain (slope amplification) so subtle relief variations catch dramatic light & shadow
        nx = -dz_dx * normal_gain
        ny = -dz_dy * normal_gain
        nz = np.ones_like(heightmap)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        
        nx /= norm
        ny /= norm
        nz /= norm

        # Key directional light vector (from top-left front)
        lx, ly, lz = 0.5, 0.7, 0.5
        l_len = np.sqrt(lx**2 + ly**2 + lz**2)
        lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

        # Fill light vector (from bottom-right front)
        fx, fy, fz = -0.4, -0.5, 0.4
        f_len = np.sqrt(fx**2 + fy**2 + fz**2)
        fx, fy, fz = fx / f_len, fy / f_len, fz / f_len

        key_diffuse = np.maximum(0.0, nx * lx + ny * ly + nz * lz)
        fill_diffuse = np.maximum(0.0, nx * fx + ny * fy + nz * fz)

        diffuse = 0.7 * key_diffuse + 0.3 * fill_diffuse
        diffuse = np.clip(diffuse, 0.05, 1.0)

        # Specular Highlights (Blinn-Phong)
        hx_key, hy_key, hz_key = lx, ly, lz + 1.0
        h_len = np.sqrt(hx_key**2 + hy_key**2 + hz_key**2)
        hx_key, hy_key, hz_key = hx_key / h_len, hy_key / h_len, hz_key / h_len
        
        specular = np.maximum(0.0, nx * hx_key + ny * hy_key + nz * hz_key) ** 24

        # Robust Ambient Occlusion (Cavity / Crevice Darkening)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        heightmap_norm = (heightmap - heightmap.min()) / (np.ptp(heightmap) + 1e-6)
        cavity = cv2.morphologyEx(heightmap_norm.astype(np.float32), cv2.MORPH_BLACKHAT, kernel)
        if cavity.max() > 1e-5:
            cavity /= cavity.max()

        # Combine Laplacian for overall curvature
        lap = cv2.Laplacian(heightmap_norm.astype(np.float64), cv2.CV_64F)
        std_lap = np.std(lap) + 1e-6
        curvature = np.clip(lap / (3.0 * std_lap), -1.0, 1.0)

        # AO factor: 1.0 (no occlusion) to 0.1 (deep crevice)
        ao = np.clip(1.0 - 0.75 * cavity + 0.15 * curvature, 0.1, 1.0)

        return nx, ny, nz, diffuse, specular, ao

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

        nx, ny, nz, diffuse, specular, ao = self.compute_normals_and_shading(heightmap, scale_mm)
        style_clean = style.lower().strip()

        # Map display names if necessary
        for label, val in self.PREVIEW_STYLES.items():
            if style.lower() == label.lower():
                style_clean = val
                break

        base_wall_color = np.array([45, 45, 50, 255], dtype=np.uint8)

        if style_clean == "photo" and image_np is not None:
            # Original Photo overlay
            img_rgb = Image.fromarray(image_np).resize((w, h), Image.Resampling.LANCZOS)
            img_rgb_np = np.array(img_rgb, dtype=np.float32)
            
            # Modulate photo colors with 3D lighting & cavity occlusion
            shaded_rgb = img_rgb_np * (0.4 + 0.6 * diffuse[..., None]) * ao[..., None]
            top_rgb = np.clip(shaded_rgb + specular[..., None] * 50.0, 0, 255).astype(np.uint8)
            base_wall_color = np.array([30, 30, 35, 255], dtype=np.uint8)

        elif style_clean == "bronze":
            # Antique Bronze with golden patina highlights & dark crevice patina
            base_bronze = np.array([125.0, 88.0, 55.0])
            highlight_gold = np.array([245.0, 205.0, 115.0])
            patina_dark = np.array([25.0, 18.0, 14.0])

            lit_diffuse = diffuse * ao
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    lit_diffuse > 0.4,
                    base_bronze[c] + (highlight_gold[c] - base_bronze[c]) * (lit_diffuse - 0.4) / 0.6,
                    patina_dark[c] + (base_bronze[c] - patina_dark[c]) * (lit_diffuse / 0.4)
                )
                top_rgb[..., c] += specular * 80.0
            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([25, 18, 14, 255], dtype=np.uint8)

        elif style_clean == "marble":
            # White Alabaster Marble with soft slate grey crevice shadows
            marble_white = np.array([250.0, 250.0, 252.0])
            slate_shadow = np.array([110.0, 118.0, 135.0])

            lit_diffuse = diffuse * ao
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = slate_shadow[c] + (marble_white[c] - slate_shadow[c]) * lit_diffuse
                top_rgb[..., c] += specular * 40.0

            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([90, 95, 105, 255], dtype=np.uint8)

        elif style_clean == "gold":
            # Polished Gold Medallion
            gold_base = np.array([235.0, 185.0, 45.0])
            gold_shine = np.array([255.0, 252.0, 210.0])
            amber_shadow = np.array([90.0, 55.0, 10.0])

            lit_diffuse = (diffuse**1.5) * ao
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    lit_diffuse > 0.4,
                    gold_base[c] + (gold_shine[c] - gold_base[c]) * (lit_diffuse - 0.4) / 0.6,
                    amber_shadow[c] + (gold_base[c] - amber_shadow[c]) * (lit_diffuse / 0.4)
                )
                top_rgb[..., c] += specular * 100.0
            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([65, 40, 10, 255], dtype=np.uint8)

        elif style_clean == "detail":
            # Surface Normal Map Shading
            r = ((nx + 1.0) * 0.5 * 255.0) * (0.3 + 0.7 * diffuse) * ao
            g = ((ny + 1.0) * 0.5 * 255.0) * (0.3 + 0.7 * diffuse) * ao
            b = ((nz + 1.0) * 0.5 * 255.0) * (0.3 + 0.7 * diffuse) * ao
            top_rgb = np.clip(np.dstack([r, g, b]) + specular[..., None] * 60.0, 0, 255).astype(np.uint8)
            base_wall_color = np.array([30, 30, 35, 255], dtype=np.uint8)

        else:
            # Default: Clay Sculpture (Terracotta with deep crevice shadows and glossy highlights)
            clay_base = np.array([205.0, 135.0, 100.0])
            clay_highlight = np.array([255.0, 215.0, 190.0])
            clay_shadow = np.array([80.0, 40.0, 25.0])

            lit_diffuse = diffuse * ao
            top_rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                top_rgb[..., c] = np.where(
                    lit_diffuse > 0.4,
                    clay_base[c] + (clay_highlight[c] - clay_base[c]) * (lit_diffuse - 0.4) / 0.6,
                    clay_shadow[c] + (clay_base[c] - clay_shadow[c]) * (lit_diffuse / 0.4)
                )
                top_rgb[..., c] += specular * 60.0
            top_rgb = np.clip(top_rgb, 0, 255).astype(np.uint8)
            base_wall_color = np.array([60, 35, 25, 255], dtype=np.uint8)

        # Add Alpha channel
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        top_rgba = np.concatenate([top_rgb, alpha], axis=-1)

        # Flip vertically to align with MeshExporter coordinate flipping
        top_rgba_flipped = np.flipud(top_rgba)
        top_vertex_colors = top_rgba_flipped.reshape(-1, 4)

        return top_vertex_colors, base_wall_color
