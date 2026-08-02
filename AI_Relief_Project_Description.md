# AI Relief

## Project Overview

Develop a production-quality Python application that converts a single
2D photograph into a high-quality, printable bas-relief 3D model
(STL/3MF/OBJ) using modern AI depth estimation, semantic segmentation,
and geometry optimization.

## Core Philosophy

Traditional pipeline:

``` text
Image
↓
Grayscale
↓
Heightmap
↓
STL
```

Target pipeline:

``` text
Photo
↓
AI depth estimation
↓
Semantic segmentation
↓
Face refinement
↓
Hair refinement
↓
Background compression
↓
Depth optimization
↓
Adaptive heightmap
↓
Mesh optimization
↓
Watertight STL
```

## Goals

-   Production-quality library and CLI
-   GPU acceleration with CPU fallback
-   AI-assisted depth estimation
-   Semantic segmentation
-   Face-aware enhancement
-   Hair-detail enhancement
-   Background compression
-   Watertight mesh generation
-   Modular architecture
-   Extensive testing and documentation

## Pipeline

1.  Image preprocessing
2.  AI depth estimation (Depth Anything V2, ZoeDepth, MiDaS)
3.  Semantic segmentation (SAM2)
4.  Face enhancement
5.  Hair enhancement
6.  Clothing optimization
7.  Background compression
8.  Relief optimization
9.  Heightmap generation
10. Watertight mesh generation

## Outputs

-   STL
-   OBJ
-   PLY
-   3MF

## GUI

-   Drag & drop
-   Live preview
-   Depth map preview
-   Heightmap preview
-   Mesh preview

## Project Structure

``` text
ai_relief/
├── cli/
├── config/
├── depth/
├── segmentation/
├── enhancement/
├── optimization/
├── geometry/
├── exporters/
├── visualization/
├── gui/
├── tests/
└── docs/
```

## Tech Stack

-   Python 3.11+
-   PyTorch
-   Depth Anything V2
-   SAM2
-   MediaPipe / InsightFace
-   OpenCV
-   NumPy
-   Pillow
-   Trimesh
-   Open3D
-   PySide6
-   Typer
-   PyTest

## Non-Goals

-   Full 3D reconstruction
-   Animation-ready meshes
-   Photogrammetry replacement

Focus exclusively on producing high-quality printable bas-relief models
from a single photograph.
