import warp as wp
import math

@wp.func
def factorial(n: int) -> int:
    """Calculate factorial of n"""
    result = int(1)
    # Use wp.range for dynamic loops
    for i in wp.range(2, n + 1):
        result = result * i  # Reassignment instead of mutation
    return result

@wp.func
def associated_legendre(l: int, m: int, x: float) -> float:
    """
    Compute the associated Legendre polynomial P_l^m(x)
    For spherical harmonics, we need this with x = cos(theta)
    """
    # Handle special cases first for clarity
    if l == 0 and m == 0:
        return 1.0
        
    if l == 1:
        if m == 0:
            return x
        if m == 1:
            return -wp.sqrt(1.0 - x*x)
            
    if l == 2:
        if m == 0:
            return 0.5 * (3.0 * x*x - 1.0)
        if m == 1:
            return -3.0 * x * wp.sqrt(1.0 - x*x)
        if m == 2:
            return 3.0 * (1.0 - x*x)
            
    # Add more special cases as needed for efficiency
    
    # Fallback to general case (this is not the most efficient implementation but is clear)
    pmm = float(1.0)  # P_m^m
    
    # Calculate P_m^m
    if m > 0:
        somx2 = wp.sqrt((1.0 - x) * (1.0 + x))
        fact = float(1.0)
        for i in wp.range(1, m + 1):
            pmm = pmm * (-fact * somx2)  # Reassignment
            fact = fact + 2.0  # Reassignment
    
    if l == m:
        return pmm
    
    # Calculate P_{m+1}^m
    pmmp1 = x * (2.0 * float(m) + 1.0) * pmm
    
    if l == m + 1:
        return pmmp1
    
    # Use recurrence relation to get P_l^m
    pll = pmmp1  # Initialize pll
    for ll in wp.range(m + 2, l + 1):
        pll = (x * (2.0 * float(ll) - 1.0) * pmmp1 - (float(ll + m) - 1.0) * pmm) / float(ll - m)
        pmm = pmmp1  # Reassignment
        pmmp1 = pll  # Reassignment
    
    return pll

@wp.func
def spherical_harmonic_real(l: int, m: int, theta: float, phi: float) -> float:
    """
    Compute the real spherical harmonic Y_l^m(theta, phi)
    - l: degree (0, 1, 2, ...)
    - m: order (-l, ..., 0, ..., l)
    - theta: polar angle (0 to pi)
    - phi: azimuthal angle (0 to 2pi)
    """
    # Check validity
    if l < 0 or wp.abs(m) > l:
        return 0.0
    
    # Normalization term
    if m == 0:
        norm = wp.sqrt((2.0 * float(l) + 1.0) / (4.0 * wp.pi))
    else:
        norm = wp.sqrt((2.0 * float(l) + 1.0) * float(factorial(l - wp.abs(m))) / 
                    (4.0 * wp.pi * float(factorial(l + wp.abs(m)))))
        
        if m < 0:
            # Use pow for integer exponents
            phase = 1.0
            if m % 2 != 0:  # Odd m gives negative phase
                phase = -1.0
            norm = norm * phase
    
    # Associated Legendre polynomial
    plm = associated_legendre(l, wp.abs(m), wp.cos(theta))
    
    # Angular part
    if m > 0:
        return wp.sqrt(2.0) * norm * plm * wp.cos(float(m) * phi)
    elif m < 0:
        return wp.sqrt(2.0) * norm * plm * wp.sin(float(wp.abs(m)) * phi)
    else:  # m == 0
        return norm * plm

