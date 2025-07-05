# nvidia-warp-spherical-harmonics
Just experimenting with nvidia warp and vibecoding 8) <br>

- 📚 All documentation is AI-generated
- 🔬 100% experimental, 0% proofread guarantee
- 🤖 No files written entirely by humans

## Overview
This project demonstrates spherical harmonics visualization and image-based lighting using NVIDIA's Warp framework. It provides an interactive application to visualize spherical harmonics of different degrees and orders, and demonstrates the use of spherical harmonics for environment map encoding.

Spherical harmonics are a set of orthonormal basis functions defined on the unit sphere. They're commonly used in computer graphics for:

- Representing lighting environments
- Approximating radiance transfer
- Encoding directional functions in a compact form

This tool allows you to:

1. Visualize different spherical harmonic functions interactively
2. Encode environment maps (skyboxes) using spherical harmonics
3. Experiment with different degrees of approximation

## Features

- **Interactive Visualization**: Rotate the camera by clicking and dragging to view the spherical harmonics from different angles
- **Configurable Parameters**: 
  - Adjust the degree (l) and order (m) of the spherical harmonic function
  - Set the value range for color mapping
- **Environment Map Encoding**: Load HDR/EXR or regular image files as skyboxes and project them onto spherical harmonics
- **High-Performance Rendering**: Uses NVIDIA Warp for efficient GPU computation

## Requirements

- Python 3.12+
- NVIDIA Warp (`warp-lang==1.7.0`)
- Matplotlib
- NumPy
- Pillow
- OpenEXR

See `requirements.txt` for the complete dependencies list.

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the initialization script (if needed):
   ```bash
   ./init_py.sh
   ```

## Usage

Run the main application:

```bash
source sourceme  # If needed for environment setup
python main.py
```

Optional command line arguments:
- `--device`: Override the default Warp device
- `--width`: Output image width in pixels (default: 1024)
- `--height`: Output image height in pixels (default: 1024)
- `--headless`: Run in headless mode without opening a graphical window

## Controls

- **Click and drag**: Rotate the camera around the sphere
- **Degree (l) slider**: Changes the degree of the spherical harmonic function
- **Order (m) slider**: Changes the order of the spherical harmonic function (limited by the selected degree)
- **Value Range slider**: Adjust the color mapping range
- **Reset button**: Restore default settings

## Project Structure

- `main.py`: Primary application code with UI and rendering logic
- `spherical_fun.py`: Spherical harmonics computation functions
- `raytrace_fun.py`: Simple ray-sphere intersection for rendering
- `image_fun.py`: Image loading and skybox sampling utilities
- `img/`: Directory containing environment maps/skyboxes

## How It Works

The application uses a combination of mathematical functions and ray tracing to visualize spherical harmonics:

1. **Spherical Harmonic Functions**: Implemented as numerically stable, high-performance Warp kernels
2. **Ray Tracing**: Simple ray-sphere intersection to render the spherical function
3. **Environment Map Encoding**: Skybox images are sampled and projected onto spherical harmonic basis functions
4. **Interactive Rendering**: Camera position is updated based on user input, and the scene is re-rendered

## License

This project is provided as-is for educational and research purposes.

## Acknowledgements

- NVIDIA Warp team for the high-performance computing library
- The environment maps included in the `img` directory

