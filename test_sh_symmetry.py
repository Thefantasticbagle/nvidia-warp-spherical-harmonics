#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from spherical_fun import *
import warp as wp

# Initialize Warp
wp.init()

def test_sh_symmetry():
    """Test if spherical harmonics exhibit expected symmetries"""
    
    # Test parameters
    l_test = 2
    m_test = 1
    
    # Create test points
    n_points = 100
    thetas = np.linspace(0, np.pi, n_points)
    phis = np.linspace(0, 2*np.pi, n_points)
    
    # Create a kernel to test SH evaluation
    @wp.kernel
    def test_sh_kernel(thetas: wp.array(dtype=float),
                       phis: wp.array(dtype=float),
                       results: wp.array(dtype=float),
                       l: int, m: int):
        tid = wp.tid()
        if tid < len(thetas):
            theta = thetas[tid]
            phi = phis[tid]
            results[tid] = spherical_harmonic_real(l, m, theta, phi)
    
    # Test along meridians (constant phi, varying theta)
    print("Testing symmetry along meridians...")
    theta_test = wp.array(thetas, dtype=float)
    phi_test = wp.array([0.0] * n_points, dtype=float)
    results_0 = wp.zeros(n_points, dtype=float)
    
    wp.launch(test_sh_kernel, dim=n_points, 
              inputs=[theta_test, phi_test, results_0, l_test, m_test])
    
    # Test at phi = pi
    phi_test_pi = wp.array([np.pi] * n_points, dtype=float)
    results_pi = wp.zeros(n_points, dtype=float)
    
    wp.launch(test_sh_kernel, dim=n_points,
              inputs=[theta_test, phi_test_pi, results_pi, l_test, m_test])
    
    # Test along equator (constant theta = pi/2, varying phi)
    print("Testing symmetry along equator...")
    theta_eq = wp.array([np.pi/2] * n_points, dtype=float)
    phi_eq = wp.array(phis, dtype=float)
    results_eq = wp.zeros(n_points, dtype=float)
    
    wp.launch(test_sh_kernel, dim=n_points,
              inputs=[theta_eq, phi_eq, results_eq, l_test, m_test])
    
    # Convert to numpy for analysis
    res_0 = results_0.numpy()
    res_pi = results_pi.numpy()
    res_eq = results_eq.numpy()
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Meridional cuts
    ax1.plot(thetas, res_0, label=f'φ=0', linewidth=2)
    ax1.plot(thetas, res_pi, label=f'φ=π', linewidth=2, linestyle='--')
    ax1.set_xlabel('θ (radians)')
    ax1.set_ylabel(f'Y_{l_test}^{m_test}(θ,φ)')
    ax1.set_title('Meridional Symmetry Test')
    ax1.legend()
    ax1.grid(True)
    
    # Equatorial cut
    ax2.plot(phis, res_eq, linewidth=2)
    ax2.set_xlabel('φ (radians)')
    ax2.set_ylabel(f'Y_{l_test}^{m_test}(π/2,φ)')
    ax2.set_title('Equatorial Cut')
    ax2.grid(True)
    
    # Range analysis
    all_results = np.concatenate([res_0, res_pi, res_eq])
    ax3.hist(all_results, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel(f'Y_{l_test}^{m_test} Values')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Value Distribution')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nResults for Y_{l_test}^{m_test}:")
    print(f"Meridian φ=0:  min={np.min(res_0):.3f}, max={np.max(res_0):.3f}")
    print(f"Meridian φ=π:  min={np.min(res_pi):.3f}, max={np.max(res_pi):.3f}")
    print(f"Equator:       min={np.min(res_eq):.3f}, max={np.max(res_eq):.3f}")
    print(f"Overall range: [{np.min(all_results):.3f}, {np.max(all_results):.3f}]")
    
    # Check if there are any obviously wrong patterns
    if np.all(res_eq >= 0) or np.all(res_eq <= 0):
        print("⚠️  WARNING: Equatorial cut shows only positive or only negative values!")
    
    if abs(np.min(all_results)) < 0.001 and abs(np.max(all_results)) < 0.001:
        print("⚠️  WARNING: All values are near zero - check implementation!")
        
    plt.savefig('sh_symmetry_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_results

if __name__ == "__main__":
    test_sh_symmetry()

