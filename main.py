#!/home/lars/projects/nvidia-warp-stuff/venv/bin/python3.12

SKYBOX_PATH = "img/blaubeuren_night_4k.exr"
# SKYBOX_PATH = "img/blaubeuren_night_4k.png"

import os
import numpy as np
# from pxr import UsdGeom
import warp as wp
# import warp.examples
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
# import colorsys

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("This example requires the Pillow package. Please install it with 'pip install Pillow'.") from err

import OpenEXR as exr
import Imath
from pathlib import Path

from raytrace_fun import *
from spherical_fun import *

@wp.func
def our_sample(theta: float, phi: float, l: int, m: int):
    """
    This serves as the function we are trying to sample.
    Crucially, its domain is on the surface of a 3D sphere, which means it can be encoded and approximated using spherical harmonics.
    This estimation is made by linear combinations of the spherical harmonic basis functions, which have many desired qualities like orthagonality and linear composability.
    Typical examples for functions to sample include radiance, vibration, and waves.
    (For now, we're just visualizing the spherical harmonics themself)
    """
    return spherical_harmonic_real(l, m, theta, phi)

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

@wp.func
def sample_environment(ray: Ray, skybox: wp.array(dtype=wp.vec3, ndim=2)) -> wp.vec3:
    azimuth = wp.atan2(ray.direction.y, ray.direction.x)
    inclination = wp.acos(ray.direction.z)

    # Map spherical coordinates to [0, 1]
    u = (azimuth + wp.pi) / (2.0 * wp.pi)
    v = inclination / wp.pi

    # Calculate the index in the HDRI array
    width = skybox.shape[1]
    height = skybox.shape[0]

    # Return the color from the HDRI
    return skybox[int(v*float(height))][int(u*float(width))]

@wp.kernel
def draw(cam_pos: wp.vec3, width: int, height: int, 
         pixels: wp.array(dtype=wp.vec3), color_min: float, color_max: float,
         l_param: int, m_param: int,
         skybox: wp.array(dtype=wp.vec3, ndim=2)):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    sx = 2.0 * float(x) / float(height) - 1.0
    sy = 2.0 * float(y) / float(height) - 1.0

    # compute view ray
    ro = cam_pos
    
    # Calculate forward direction based on camera position
    forward = wp.normalize(wp.vec3(0.0, 0.0, 0.0) - cam_pos)  # Look at origin
    right = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), forward))
    up = wp.cross(forward, right)
    
    # Create ray direction using camera basis vectors
    rd = wp.normalize(forward + right * sx + up * sy)
    
    ray = Ray()
    ray.origin = ro
    ray.direction = rd

    color = wp.vec3(0.0, 0.0, 0.0)

    # Check spheres
    sphere = Sphere()
    sphere.pos = wp.vec3(0.0, 0.0, 0.0)
    sphere.radius = 1.0

    hitinfo = ray_sphere(ray, sphere)
    if hitinfo.didHit:
        value = our_sample(hitinfo.localcoord[1], hitinfo.localcoord[2], l_param, m_param)
        color = map_value_to_color(value, color_min, color_max)
    else:
        color = sample_environment(ray, skybox)

    pixels[tid] = color

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

        return target_wp

class Example:
    def __init__(self, height=1024, width=1024):
        self.height = height
        self.width = width
        self.cam_pos = wp.vec3(0.0, 1.0, 2.0)
        self.cam_angle = 0.0  # Camera rotation angle around Y axis
        self.pan_speed = 0.01  # Increased for faster animation
        self.frame = 0
        
        # Color mapping range parameters
        self.color_min = -1.0
        self.color_max = 1.0
        
        # Spherical harmonic parameters
        self.l_param = 2  # Degree
        self.m_param = 1  # Order

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.skybox = load_image(SKYBOX_PATH, 1024, 1024)

    def update_camera(self):
        # Update camera angle
        self.cam_angle += self.pan_speed
        
        radius = 3.0
        self.cam_pos = wp.vec3(
            radius * np.sin(self.cam_angle),
            1.0,
            radius * np.cos(self.cam_angle)
        )

        self.frame += 1

    def render(self):
        with wp.ScopedTimer("render"):
            self.update_camera()
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.cam_pos, self.width, self.height, 
                        self.pixels, self.color_min, self.color_max,
                        self.l_param, self.m_param,
                        self.skybox],
            )

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height in pixels.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to render in animation.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(height=args.height, width=args.width)
        
        if not args.headless:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            # Set up the figure and initial frame
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(bottom=0.35)  # Make more room for all sliders
            
            img = ax.imshow(
                example.pixels.numpy().reshape((example.height, example.width, 3)),
                origin="lower",
                interpolation="antialiased"
            )
            
            ax_range = plt.axes([0.25, 0.1, 0.65, 0.03])
            
            # Add sliders for spherical harmonic parameters
            ax_l = plt.axes([0.25, 0.15, 0.65, 0.03])
            ax_m = plt.axes([0.25, 0.2, 0.65, 0.03])
            
            # Add slider for animation speed
            ax_speed = plt.axes([0.25, 0.05, 0.65, 0.03])
            
            # Create RangeSlider for color mapping
            s_range = RangeSlider(ax_range, 'Value Range', -10.0, 10.0, 
                                 valinit=(example.color_min, example.color_max))
            s_l = Slider(ax_l, 'Degree (l)', 0, 10, valinit=example.l_param, valstep=1)
            s_m = Slider(ax_m, 'Order (m)', -10, 10, valinit=example.m_param, valstep=1)
            s_speed = Slider(ax_speed, 'Speed', 0.0, 0.5, valinit=example.pan_speed)
            
            def update_sliders(val):
                # Get the range values from the RangeSlider
                example.color_min, example.color_max = s_range.val
                example.l_param = int(s_l.val)
                
                # Ensure m is within valid range for given l (-l <= m <= l)
                max_m = example.l_param
                valid_m = max(min(int(s_m.val), max_m), -max_m)
                if valid_m != s_m.val:
                    s_m.set_val(valid_m)
                example.m_param = valid_m
                example.pan_speed = s_speed.val
                
            s_range.on_changed(update_sliders)
            s_l.on_changed(update_sliders)
            s_m.on_changed(update_sliders)
            s_speed.on_changed(update_sliders)
            
            # Add a reset button
            reset_button_ax = plt.axes([0.8, 0.25, 0.1, 0.04])
            reset_button = Button(reset_button_ax, 'Reset')
            
            def reset_params(event):
                s_l.reset()
                s_m.reset()
                s_range.reset()
                s_speed.reset()
                
            reset_button.on_clicked(reset_params)
            
            def update(frame):
                # Render new frame
                example.render()
                # Update the plot data
                img.set_array(example.pixels.numpy().reshape((example.height, example.width, 3)))
                return [img]
            
            ani = FuncAnimation(fig, update, frames=args.frames, interval=10, blit=True)
            plt.show()
        else:
            # Render frames without visualization
            for i in range(args.frames):
                example.render()
