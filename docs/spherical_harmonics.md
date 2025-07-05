# Spherical Harmonics Implementation Details

## Mathematical Background

Spherical harmonics (SH) are a set of orthonormal basis functions defined on the unit sphere. They are analogous to the Fourier series, but for functions defined on a sphere rather than on a circle.

### Definition

The real spherical harmonics $Y_l^m(\theta, \phi)$ of degree $l$ and order $m$ are defined as:

$$Y_l^m(\theta, \phi) = \begin{cases}
\sqrt{\frac{2l+1}{4\pi} \frac{(l-|m|)!}{(l+|m|)!}} P_l^{|m|}(\cos\theta) \cos(m\phi) & \text{if } m > 0 \\
\sqrt{\frac{2l+1}{4\pi}} P_l^0(\cos\theta) & \text{if } m = 0 \\
\sqrt{\frac{2l+1}{4\pi} \frac{(l-|m|)!}{(l+|m|)!}} P_l^{|m|}(\cos\theta) \sin(|m|\phi) & \text{if } m < 0
\end{cases}$$

where $P_l^m$ are the associated Legendre polynomials, $\theta \in [0, \pi]$ is the polar angle (from the Y axis in our implementation), and $\phi \in [0, 2\pi]$ is the azimuthal angle in the XZ-plane.

### Associated Legendre Polynomials

These are special functions that appear as part of the angular solution of the Laplace equation in spherical coordinates. They are defined by the following differential equation:

$$(1-x^2) \frac{d^2 P_l^m}{dx^2} - 2x \frac{d P_l^m}{dx} + \left[l(l+1) - \frac{m^2}{1-x^2}\right] P_l^m = 0$$

In our implementation, we use hardcoded, optimized calculations for these polynomials to maximize performance in the NVIDIA Warp environment.

## Implementation in NVIDIA Warp

The implementation in `spherical_fun.py` provides high-performance computation of spherical harmonics suitable for real-time visualization and lighting calculations.

### Optimization Strategy

The key optimizations in our implementation include:

1. **Hardcoded Associated Legendre Polynomials**: Rather than using recurrence relations or series expansions, we pre-compute and hardcode the polynomial expressions for each (l, m) combination up to l=10. This approach substantially improves performance in GPU kernels.

2. **Fixed-Point Factorial Calculation**: For the normalization factors, we use a hardcoded lookup table for factorials instead of recursion or loops.

3. **Memory and Computation Efficiency**: The implementation reuses calculations (like powers of x and s) to minimize redundant operations.

### Indexing Scheme

To map from the 2D indices (l, m) to a 1D array index, we use the function:

```python
@wp.func
def sh_index(l: int, m: int) -> int:
    return l * (l + 1) + m
```

This creates a compact, zero-based indexing scheme where:
- (0,0) → 0
- (1,-1) → 1, (1,0) → 2, (1,1) → 3
- (2,-2) → 4, (2,-1) → 5, (2,0) → 6, (2,1) → 7, (2,2) → 8
- And so on...

## Environment Map Projection

The project demonstrates how spherical harmonics can be used to encode environment maps (skyboxes) by projecting the radiance function onto the SH basis:

$$L_i(\omega) \approx \sum_{l=0}^n \sum_{m=-l}^l c_l^m Y_l^m(\omega)$$

Where:
- $L_i(\omega)$ is the incoming radiance from direction $\omega$
- $c_l^m$ are the SH coefficients
- $n$ is the maximum degree used for the approximation

The coefficients are computed using Monte Carlo integration in the `bake` kernel:

```python
@wp.kernel
def bake(width: int, height: int,
         pixels: wp.array(dtype=wp.vec3),
         l_param: int,
         skybox: wp.array(dtype=wp.vec3, ndim=2),
         seed: int,
         sh_coeffs_out: wp.array(dtype=wp.vec3)):
    # ... implementation ...
    # Monte Carlo integration to compute the coefficient
    contribution = color * y_lm * (4.0 * wp.pi) / float(BAKE_SAMPLE_COUNT)
    wp.atomic_add(sh_coeffs_out, index, contribution)
```

The integration takes `BAKE_SAMPLE_COUNT` random samples over the sphere, evaluates the SH basis function at each sample point, multiplies by the radiance from the environment map, and accumulates the weighted results.

## Rendering

The visualization is done by ray tracing a sphere and evaluating the spherical harmonic function at the intersection point. For rendering with an SH-projected environment, we accumulate the contributions from all SH coefficients at each surface point.

```python
# For each spherical harmonic basis function
for l in range(l_param):
    for m in range(-l, l + 1):
        index = sh_index(l, m)
        y_lm = spherical_harmonic_real(l, m, norm_spherical.x, norm_spherical.y)
        contribution = sh_coeffs[index] * y_lm
        color += contribution
```

## Performance Considerations

Spherical harmonics calculations can be computationally expensive, especially for high degrees. Our implementation makes several trade-offs:

1. We limit the maximum degree to 10, which is sufficient for most graphics applications
2. We use hardcoded polynomials rather than generic algorithms to maximize performance
3. We leverage NVIDIA Warp for parallelization and GPU acceleration

With these optimizations, the implementation can achieve real-time performance even at high resolutions and with complex environment maps.

