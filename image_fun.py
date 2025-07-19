import warp as wp
import numpy as np
import os

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("This example requires the Pillow package. Please install it with 'pip install Pillow'.") from err

import OpenEXR as exr
import Imath
from pathlib import Path

@wp.struct
class Skybox_attributes:
    rotation: wp.vec2

@wp.func
def map_value_to_color(value: float, min_val: float, max_val: float) -> wp.vec3:
    # Normalize value to [0,1] range for hue
    normalized = wp.clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    # Use the normalized value as hue (0-1), with full saturation and value
    # Convert HSV to RGB (we need to do this manually as warp doesn't have colorsys)
    h = normalized
    s = 1.0
    v = 1.0
    
    # HSV to RGB conversion
    c = v * s
    
    # Implementing mod operation manually since wp.fmod doesn't exist
    h_times_6 = h * 6.0
    h_times_6_mod_2 = h_times_6 - 2.0 * wp.floor(h_times_6 / 2.0)
    
    x = c * (1.0 - wp.abs(h_times_6_mod_2 - 1.0))
    m = v - c
    
    r, g, b = 0.0, 0.0, 0.0
    if h < 1.0/6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0/6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0/6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0/6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0/6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    
    return wp.vec3(r + m, g + m, b + m)

def load_exr(path, resize_width, resize_height):
    exrfile = exr.InputFile(Path(path).as_posix())
    
    header = exrfile.header()
    dw = header['displayWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    rgb_channels = ['R', 'G', 'B']
    raw_bytes = [exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)) for c in rgb_channels]
    
    rgb_vectors = [np.frombuffer(bytes, dtype=np.float32) for bytes in raw_bytes]
    rgb_maps = [vec.reshape(size[1], size[0]) for vec in rgb_vectors]
    rgb_map = np.stack(rgb_maps, axis=-1)
    
    # Some of this is probably less efficient than it could be... whatever ¯\_(ツ)_/¯
    img = Image.fromarray((rgb_map * 255).astype(np.uint8))
    resized_img = img.resize((resize_width, resize_height))
    
    target_np = np.array(resized_img)
    target_np_norm = target_np.astype(np.float32) / 255.0
    target_wp = wp.array(target_np_norm, dtype=wp.vec3)

    print(f"Skybox min: {target_np_norm.min()}, max: {target_np_norm.max()}, mean: {target_np_norm.mean()}")   
    return target_wp

def load_image(path, resize_width, resize_height):
    file_extension = os.path.splitext(path)[1].lower()
    
    if file_extension == '.exr':
        return load_exr(path, resize_width, resize_height)
    else:
        target_base = Image.open(os.path.relpath(path))
        target_resized = target_base.resize((resize_width, resize_height))
        target_np = np.array(target_resized)

        if target_np.shape[2] == 4:
            target_np = target_np[:, :, :3]
        if target_np.shape[2] != 3:
            raise ValueError("Image must be RGB or RGBA to be converted to vec3.")

        target_np_norm = target_np.astype(np.float32) / 255.0
        target_wp = wp.array(target_np_norm, dtype=wp.vec3)

        print(f"Skybox min: {target_np_norm.min()}, max: {target_np_norm.max()}, mean: {target_np_norm.mean()}")   
        return target_wp

@wp.func
def sample_skybox(spherical_coords: wp.vec2, skybox: wp.array(dtype=wp.vec3, ndim=2), skybox_attrs: Skybox_attributes) -> wp.vec3:
    spherical_coords -= skybox_attrs.rotation # inverse rotation

    # Map spherical coordinates to [0, 1] with phi adjusted by π to fix orientation
    u = (spherical_coords.y + wp.pi) / (2.0 * wp.pi)  # phi maps to u (horizontal) with offset
    v = spherical_coords.x / wp.pi                    # theta maps to v (vertical)

    width = skybox.shape[1]
    height = skybox.shape[0]

    # Ensure u is within [0,1] range (add π could push it out of range)
    u = u - wp.floor(u)

    return skybox[int(v*float(height))][int(u*float(width))]

