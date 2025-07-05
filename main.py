#!/home/lars/projects/nvidia-warp-stuff/venv/bin/python3.12

# SKYBOX_PATH = "img/blaubeuren_night_4k.exr"
SKYBOX_PATH = "img/meadow_2_4k.png"
# SKYBOX_PATH = "img/meadow_2_4k.exr"
# SKYBOX_PATH = "img/studio_small_09_4k.exr"
# SKYBOX_PATH = "img/studio_small_09_4k.png"
BAKE_SAMPLE_COUNT = 6000
# SKYBOX_PATH = "img/blaubeuren_night_4k.png"

import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider

from raytrace_fun import *
from spherical_fun import *
from image_fun import *

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
def cartesian_to_spherical(v: wp.vec3) -> wp.vec2:
    """
    Converts a 3D Cartesian vector to Spherical coordinates (theta, phi)
    using a Y-up coordinate system.
    theta: polar angle from the Y-axis [0, pi]
    phi: azimuthal angle in the XZ-plane [-pi, pi]
    """
    r = wp.length(v)
    if r < 1e-8:
        return wp.vec2(0.0, 0.0)
    
    v_norm = v / r
    
    # Y-up convention
    phi = wp.atan2(v_norm.z, v_norm.x)
    theta = wp.acos(wp.clamp(v_norm.y, -1.0, 1.0))
    
    return wp.vec2(theta, phi)

@wp.kernel
def draw(cam_pos: wp.vec3, width: int, height: int, 
         pixels: wp.array(dtype=wp.vec3), color_min: float, color_max: float,
         l_param: int, m_param: int,
         skybox: wp.array(dtype=wp.vec3, ndim=2),
         seed: int,
         sh_coeffs: wp.array(dtype=wp.vec3)):
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
    rd_spherical = cartesian_to_spherical(rd)

    color = wp.vec3(0.0, 0.0, 0.0)

    sphere = Sphere()
    sphere.pos = wp.vec3(0.0, 0.0, 0.0)
    sphere.radius = 1.0

    hitinfo = ray_sphere(ray, sphere)
    if hitinfo.didHit:
        norm = hitinfo.normal
        norm_spherical = cartesian_to_spherical(norm)
        # norm_spherical = rd_spherical

        for l in range(l_param):
            for m in range(-l, l + 1):
                index = sh_index(l, m)
                y_lm = spherical_harmonic_real(l, m, norm_spherical.x, norm_spherical.y)
                contribution = sh_coeffs[index] * y_lm
                color += contribution

        color = wp.min(wp.max(color, wp.vec3(0.0)), wp.vec3(1.0))
    else:
        color = sample_skybox(rd_spherical, skybox)

    pixels[tid] = color

@wp.kernel
def bake(width: int, height: int, 
         pixels: wp.array(dtype=wp.vec3),
         l_param: int,
         skybox: wp.array(dtype=wp.vec3, ndim=2),
         seed: int,
         sh_coeffs_out: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    state = wp.rand_init(seed, tid)

    dir = wp.normalize( wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state)) )
    dir_spherical = cartesian_to_spherical(dir)

    color = sample_skybox(dir_spherical, skybox)

    for l in range(l_param):
        for m in range(-l, l + 1):
            index = sh_index(l, m)
            y_lm = spherical_harmonic_real(l, m, dir_spherical.x, dir_spherical.y)
            contribution = color * y_lm * (4.0 * wp.pi) / float(BAKE_SAMPLE_COUNT)
            wp.atomic_add(sh_coeffs_out, index, contribution)

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
        self.l_param = 6  # Degree
        self.m_param = 1  # Order

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.skybox = load_image(SKYBOX_PATH, 1024, 1024)

        self.max_l = 10
        num_coeffs = (self.max_l + 1) * (self.max_l + 1)
        self.sh_coeffs = wp.zeros(num_coeffs, dtype=wp.vec3)

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

    def bake(self):
        self.sh_coeffs.zero_()

        print("SH coefficient mapping:")
        for l in range(self.l_param):
            for m in range(-l, l + 1):
                index = sh_index(l, m)
                print(f"  Index {index}: l={l}, m={m}")

        wp.launch(
            kernel=bake,
            dim=BAKE_SAMPLE_COUNT,
            inputs=[self.width, self.height, 
                    self.pixels,
                    self.l_param,
                    self.skybox,
                    164,
                    self.sh_coeffs],
        )
        wp.synchronize()

        coeffs_np = self.sh_coeffs.numpy()
        for l in range(self.l_param):
            for m in range(-l, l + 1):
                index = sh_index(l, m)
                if index < len(coeffs_np):
                    print(f"l={l}, m={m}, index={index}: {coeffs_np[index]}")

    def render(self):
        with wp.ScopedTimer("render"):
            self.update_camera()
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.cam_pos, self.width, self.height, 
                        self.pixels, self.color_min, self.color_max,
                        self.l_param, self.m_param,
                        self.skybox,
                        145,
                        self.sh_coeffs],
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
            s_l = Slider(ax_l, 'Degree (l)', 0, example.max_l, valinit=example.l_param, valstep=1)
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

                example.bake() # (should only really need to rebake if l change)
                
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
            
            example.bake()
            ani = FuncAnimation(fig, update, frames=args.frames, interval=10, blit=True)
            plt.show()
        else:
            example.bake()
            for i in range(args.frames):
                example.render()
