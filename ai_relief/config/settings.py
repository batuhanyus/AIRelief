from pydantic import BaseModel, Field
from typing import Optional

class AppSettings(BaseModel):
    # Depth settings
    depth_model_path: str = Field(default="weights/depth_anything_v2", description="Local path to manually downloaded Depth Anything V2 weights")
    depth_strength: float = Field(default=1.0, description="Multiplier for the raw depth values")
    
    # Segmentation settings
    sam2_model_path: str = Field(default="weights/sam2", description="Local path to manually downloaded SAM 2 weights")
    
    # Geometry settings
    relief_max_depth_mm: float = Field(default=5.0, description="Maximum thickness of the relief in millimeters")
    base_thickness_mm: float = Field(default=2.0, description="Thickness of the solid base underneath the relief")
    resolution_scale: float = Field(default=1.0, description="Scale factor for the output mesh resolution (1.0 = native image size)")
    
    # System settings
    device: str = Field(default="cpu", description="Device to run inference on (cuda, mps, cpu)")

# Global singleton for settings
settings = AppSettings()
