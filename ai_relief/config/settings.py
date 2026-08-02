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
    device: str = Field(default="auto", description="Device to run inference on (auto, cuda, mps, cpu)")

    def get_resolved_device(self) -> str:
        """Returns the resolved torch device ('cuda', 'mps', or 'cpu') based on availability and setting."""
        import torch
        dev = self.device.lower()
        if dev in ("auto", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        elif dev == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return "cpu"

    def get_device_info(self) -> str:
        """Returns a user-friendly string describing the active hardware device."""
        import torch
        resolved = self.get_resolved_device()
        if resolved == "cuda":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            return f"CUDA GPU ({gpu_name})"
        elif resolved == "mps":
            return "Apple Silicon (MPS)"
        else:
            return "CPU"

# Global singleton for settings
settings = AppSettings()
