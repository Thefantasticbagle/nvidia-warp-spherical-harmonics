# Project Structure

## Overview

The project is organized into several Python modules, each focused on a specific aspect of the spherical harmonics visualization and environment map projection system.

```
nvidia-warp-stuff/
│
├── main.py                # Main application with UI and visualization logic
├── spherical_fun.py       # Spherical harmonics mathematical functions
├── raytrace_fun.py        # Ray-tracing utilities for sphere intersection
├── image_fun.py           # Image loading and skybox sampling
│
├── img/                   # Environment maps (skyboxes)
│   ├── blaubeuren_night_4k.exr    # High dynamic range environment map
│   ├── blaubeuren_night_4k.png    # LDR version of the same environment
│   ├── meadow_2_4k.exr            # Another HDR environment
│   ├── meadow_2_4k.png            # LDR version
│   ├── studio_small_09_4k.exr     # Studio lighting environment (HDR)
│   ├── studio_small_09_4k.png     # Studio lighting environment (LDR)
│   └── shviz.png                  # Screenshot for documentation
│
├── docs/                  # Documentation
│   ├── spherical_harmonics.md     # Mathematical background
│   ├── quick_start.md             # Getting started guide
│   ├── nvidia_warp_overview.md    # Overview of NVIDIA Warp
│   └── project_structure.md       # This document
│
├── requirements.txt       # Python dependencies
├── init_py.sh            # Initialization script
├── sourceme              # Environment setup script
├── .gitignore            # Git ignore configuration
└── README.md             # Project overview and documentation
```

## Core Modules

### main.py

This is the entry point for the application, containing:

- **Example Class**: Main application logic that manages rendering, UI, and state
- **Interactive UI**: Matplotlib-based UI with sliders and interactive camera control
- **Command-line Interface**: Argument parsing for customizing execution
- **Rendering Loop**: Main rendering pipeline that coordinates all other modules

### spherical_fun.py

This module implements the mathematical foundations of spherical harmonics:

- **Spherical Harmonic Functions**: Computation of real spherical harmonics for visualization
- **Associated Legendre Polynomials**: Optimized implementation of these special functions
- **Factorial Calculation**: Fast lookup table for normalization constants
- **Indexing Utilities**: Mapping from (l,m) indices to linear array indices

### raytrace_fun.py

Provides basic ray-tracing functionality:

- **Ray and Sphere Classes**: Data structures for ray-tracing
- **Intersection Logic**: Ray-sphere intersection calculation
- **Coordinate Conversion**: Cartesian to spherical coordinate transformation
- **Hit Information**: Stores intersection details like position, normal, etc.

### image_fun.py

Handles all image-related operations:

- **Image Loading**: Support for regular images and HDR/EXR formats
- **Skybox Sampling**: Mapping from directions to environment map pixels
- **Color Mapping**: Utilities for converting values to colors
- **Image Resizing**: Prepares images for use as environment maps

## Resources

### img/ Directory

Contains environment maps (skyboxes) used for visualization and lighting:

- Multiple high-dynamic-range (HDR) images in EXR format, which preserve the full lighting intensity range
- Standard PNG versions of the same environments for systems without HDR support

## Configuration Files

### requirements.txt

Lists all Python package dependencies:

```
contourpy==1.3.2
cycler==0.12.1
fonttools==4.57.0
kiwisolver==1.4.8
matplotlib==3.10.1
numpy==2.2.4
packaging==24.2
pillow==11.2.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
six==1.17.0
usd-core==25.2.post1
warp-lang==1.7.0
```

### init_py.sh and sourceme

These scripts help set up the development environment:

- **init_py.sh**: Initializes the Python environment or project
- **sourceme**: Sets environment variables needed for execution

## Documentation

### README.md

The main project documentation, containing:

- Project overview and purpose
- Key features and capabilities
- Installation and usage instructions
- Basic explanation of the underlying concepts

