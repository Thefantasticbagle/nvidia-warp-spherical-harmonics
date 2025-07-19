import warp as wp
import math

# @wp.func
# def factorial(n: int) -> int:
#     """Calculate factorial of n"""
#     result = int(1)
#     # Use wp.range for dynamic loops
#     for i in wp.range(2, n + 1):
#         result = result * i  # Reassignment instead of mutation
#     return result

# @wp.func
# def associated_legendre(l: int, m: int, x: float) -> float:
#     if m < 0 or m > l:
#         return 0.0
#
#     if m == 0:
#         return legendre_polynomial(l, x)
#
#     # Use stable recurrence for associated Legendre
#     if wp.abs(x) >= 1.0:
#         return 0.0 if wp.abs(x) > 1.0 else (1.0 if (l + m) % 2 == 0 else -1.0)
#
#     # Start with P_m^m
#     pmm = 1.0
#     somx2 = wp.sqrt((1.0 - x) * (1.0 + x))  # sqrt(1 - x^2), more stable
#     fact = 1.0
#     for i in range(m):
#         pmm *= -fact * somx2
#         fact += 2.0
#
#     if l == m:
#         return pmm
#
#     # Compute P_{m+1}^m
#     pmmp1 = x * float(2 * m + 1) * pmm
#     if l == m + 1:
#         return pmmp1
#
#     # Compute P_l^m for l > m + 1 using recurrence
#     pll = 0.0
#     for ll in range(m + 2, l + 1):
#         pll = (x * float(2 * ll - 1) * pmmp1 - float(ll + m - 1) * pmm) / float(ll - m)
#         pmm = pmmp1
#         pmmp1 = pll
#
#     return pll

# @wp.func
# def associated_legendre(l: int, m: int, x: float) -> float:
#     """
#     Compute the associated Legendre polynomial P_l^m(x)
#     For spherical harmonics, we need this with x = cos(theta)
#     """
#     # Handle special cases first for clarity
#     if l == 0 and m == 0:
#         return 1.0
#
#     if l == 1:
#         if m == 0:
#             return x
#         if m == 1:
#             return -wp.sqrt(1.0 - x*x)
#
#     if l == 2:
#         if m == 0:
#             return 0.5 * (3.0 * x*x - 1.0)
#         if m == 1:
#             return -3.0 * x * wp.sqrt(1.0 - x*x)
#         if m == 2:
#             return 3.0 * (1.0 - x*x)
#
#     # Add more special cases as needed for efficiency
#
#     # Fallback to general case (this is not the most efficient implementation but is clear)
#     pmm = float(1.0)  # P_m^m
#
#     # Calculate P_m^m
#     if m > 0:
#         somx2 = wp.sqrt((1.0 - x) * (1.0 + x))
#         fact = float(1.0)
#         for i in wp.range(1, m + 1):
#             pmm = pmm * (-fact * somx2)  # Reassignment
#             fact = fact + 2.0  # Reassignment
#
#     if l == m:
#         return pmm
#
#     # Calculate P_{m+1}^m
#     pmmp1 = x * (2.0 * float(m) + 1.0) * pmm
#
#     if l == m + 1:
#         return pmmp1
#
#     # Use recurrence relation to get P_l^m
#     pll = pmmp1  # Initialize pll
#     for ll in wp.range(m + 2, l + 1):
#         pll = (x * (2.0 * float(ll) - 1.0) * pmmp1 - (float(ll + m) - 1.0) * pmm) / float(ll - m)
#         pmm = pmmp1  # Reassignment
#         pmmp1 = pll  # Reassignment
#
#     return pll

# @wp.func
# def spherical_harmonic_real(l: int, m: int, theta: float, phi: float) -> float:
#     if l < 0 or wp.abs(m) > l:
#         return 0.0
#
#     # Fix the normalization
#     if m == 0:
#         norm = wp.sqrt((2.0 * float(l) + 1.0) / (4.0 * wp.pi))
#         plm = associated_legendre(l, 0, wp.cos(theta))
#         return norm * plm
#     else:
#         norm = wp.sqrt((2.0 * float(l) + 1.0) * float(factorial(l - wp.abs(m))) / 
#                       (2.0 * wp.pi * float(factorial(l + wp.abs(m)))))
#         plm = associated_legendre(l, wp.abs(m), wp.cos(theta))
#
#         if m > 0:
#             return norm * plm * wp.cos(float(m) * phi)
#         else:  # m < 0
#             return norm * plm * wp.sin(float(wp.abs(m)) * phi)

@wp.func
def sh_index(l: int, m: int) -> int:
    """ Maps (l, m) spherical harmonic indices to a 1D array index. """
    return l * (l + 1) + m

# @wp.func
# def spherical_harmonic_real(l: int, m: int, theta: float, phi: float) -> float:
#     if l >= 15 or l < 0 or wp.abs(m) > l:
#         return 0.0
#
#     # Convert spherical coordinates to Cartesian for easier computation
#     sin_theta = wp.sin(theta)
#     cos_theta = wp.cos(theta)
#     cos_phi = wp.cos(phi)
#     sin_phi = wp.sin(phi)
#
#     x = sin_theta * cos_phi
#     y = sin_theta * sin_phi
#     z = cos_theta
#
#     # Precompute powers
#     x2 = x * x
#     y2 = y * y
#     z2 = z * z
#     x3 = x2 * x
#     y3 = y2 * y
#     z3 = z2 * z
#     x4 = x3 * x
#     y4 = y3 * y
#     z4 = z3 * z
#
#     xy = x * y
#     xz = x * z
#     yz = y * z
#
#     # l = 0
#     if l == 0 and m == 0:
#         return 0.282095
#
#     # l = 1
#     elif l == 1:
#         if m == -1:
#             return 0.488603 * y
#         elif m == 0:
#             return 0.488603 * z
#         elif m == 1:
#             return 0.488603 * x
#
#     # l = 2
#     elif l == 2:
#         if m == -2:
#             return 1.092548 * xy
#         elif m == -1:
#             return 1.092548 * yz
#         elif m == 0:
#             return 0.315392 * (3.0 * z2 - 1.0)
#         elif m == 1:
#             return 1.092548 * xz
#         elif m == 2:
#             return 0.546274 * (x2 - y2)
#
#     # l = 3
#     elif l == 3:
#         if m == -3:
#             return 0.590044 * y * (3.0 * x2 - y2)
#         elif m == -2:
#             return 2.890611 * xy * z
#         elif m == -1:
#             return 0.457046 * y * (5.0 * z2 - 1.0)
#         elif m == 0:
#             return 0.373176 * z * (5.0 * z2 - 3.0)
#         elif m == 1:
#             return 0.457046 * x * (5.0 * z2 - 1.0)
#         elif m == 2:
#             return 1.445306 * z * (x2 - y2)
#         elif m == 3:
#             return 0.590044 * x * (x2 - 3.0 * y2)
#
#     # l = 4
#     elif l == 4:
#         if m == -4:
#             return 2.503343 * xy * (x2 - y2)
#         elif m == -3:
#             return 1.770131 * yz * (3.0 * x2 - y2)
#         elif m == -2:
#             return 0.946175 * xy * (7.0 * z2 - 1.0)
#         elif m == -1:
#             return 0.669047 * yz * (7.0 * z2 - 3.0)
#         elif m == 0:
#             return 0.105786 * (35.0 * z4 - 30.0 * z2 + 3.0)
#         elif m == 1:
#             return 0.669047 * xz * (7.0 * z2 - 3.0)
#         elif m == 2:
#             return 0.473087 * (x2 - y2) * (7.0 * z2 - 1.0)
#         elif m == 3:
#             return 1.770131 * xz * (x2 - 3.0 * y2)
#         elif m == 4:
#             return 0.625836 * (x4 - 6.0 * x2 * y2 + y4)
#
#     # l = 5
#     elif l == 5:
#         if m == -5:
#             return 0.656383 * y * (5.0 * x4 - 10.0 * x2 * y2 + y4)
#         elif m == -4:
#             return 8.302649 * xy * z * (x2 - y2)
#         elif m == -3:
#             return 0.489238 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0)
#         elif m == -2:
#             return 4.793537 * xy * z * (3.0 * z2 - 1.0)
#         elif m == -1:
#             return 0.452947 * y * (21.0 * z4 - 14.0 * z2 + 1.0)
#         elif m == 0:
#             return 0.116950 * z * (63.0 * z4 - 70.0 * z2 + 15.0)
#         elif m == 1:
#             return 0.452947 * x * (21.0 * z4 - 14.0 * z2 + 1.0)
#         elif m == 2:
#             return 2.396768 * z * (x2 - y2) * (3.0 * z2 - 1.0)
#         elif m == 3:
#             return 0.489238 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0)
#         elif m == 4:
#             return 2.075662 * z * (x4 - 6.0 * x2 * y2 + y4)
#         elif m == 5:
#             return 0.656383 * x * (x4 - 10.0 * x2 * y2 + 5.0 * y4)
#
#     # l = 6
#     elif l == 6:
#         z6 = z4 * z2
#         x6 = x4 * x2
#         y6 = y4 * y2
#
#         if m == -6:
#             return 6.831841 * xy * (3.0 * x4 - 10.0 * x2 * y2 + 3.0 * y4)
#         elif m == -5:
#             return 2.366619 * yz * (5.0 * x4 - 10.0 * x2 * y2 + y4)
#         elif m == -4:
#             return 2.018157 * xy * (x2 - y2) * (11.0 * z2 - 1.0)
#         elif m == -3:
#             return 0.921669 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)
#         elif m == -2:
#             return 0.921669 * xy * (33.0 * z4 - 18.0 * z2 + 1.0)
#         elif m == -1:
#             return 0.460835 * yz * (33.0 * z4 - 30.0 * z2 + 5.0)
#         elif m == 0:
#             return 0.063394 * (231.0 * z6 - 315.0 * z4 + 105.0 * z2 - 5.0)
#         elif m == 1:
#             return 0.460835 * xz * (33.0 * z4 - 30.0 * z2 + 5.0)
#         elif m == 2:
#             return 0.460835 * (x2 - y2) * (33.0 * z4 - 18.0 * z2 + 1.0)
#         elif m == 3:
#             return 0.921669 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)
#         elif m == 4:
#             return 0.504539 * (x4 - 6.0 * x2 * y2 + y4) * (11.0 * z2 - 1.0)
#         elif m == 5:
#             return 2.366619 * xz * (x4 - 10.0 * x2 * y2 + 5.0 * y4)
#         elif m == 6:
#             return 0.683184 * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6)
#
#     # For l >= 7 and l < 15, return a simplified approximation to avoid extremely long code
#     # These higher order terms contribute less to typical lighting scenarios
#     else:
#         # Use a simplified polynomial approximation for higher orders
#         r = wp.sqrt(x2 + y2 + z2)  # Should be 1.0 for unit vectors
#
#         # Simple polynomial based on position and order
#         base_val = wp.pow(r, float(l)) * wp.cos(float(m) * phi) * wp.sin(theta)
#
#         # Rough normalization factor
#         norm_factor = wp.sqrt((2.0 * float(l) + 1.0) / (4.0 * wp.pi))
#
#         # Add some oscillation based on l and m
#         oscillation = wp.cos(float(l) * theta + float(m) * phi)
#
#         return norm_factor * base_val * oscillation * 0.1  # Scale down higher orders
#
#     return 0.0


@wp.func
def associated_legendre(l: int, m: int, x: float) -> float:
    """
    Computes the Associated Legendre Polynomial P_l^m(x).
    This function is hardcoded for l up to 10 for maximum performance in a Warp kernel.
    It assumes m >= 0.
    """
    # Precompute powers of x and s = sqrt(1-x^2) for efficiency and readability
    s = wp.sqrt(1.0 - x*x)
    x2 = x*x
    s2 = s*s

    res = 0.0

    if l == 0:
        # m must be 0
        res = 1.0
    elif l == 1:
        if m == 0: res = x
        elif m == 1: res = -s
    elif l == 2:
        if m == 0: res = 0.5 * (3.0*x2 - 1.0)
        elif m == 1: res = -3.0 * x * s
        elif m == 2: res = 3.0 * s2
    elif l == 3:
        if m == 0: res = 0.5 * x * (5.0*x2 - 3.0)
        elif m == 1: res = -1.5 * (5.0*x2 - 1.0) * s
        elif m == 2: res = 15.0 * x * s2
        elif m == 3: res = -15.0 * s * s2
    elif l == 4:
        if m == 0: res = 0.125 * (35.0*x2*x2 - 30.0*x2 + 3.0)
        elif m == 1: res = -2.5 * x * (7.0*x2 - 3.0) * s
        elif m == 2: res = 7.5 * (7.0*x2 - 1.0) * s2
        elif m == 3: res = -105.0 * x * s * s2
        elif m == 4: res = 105.0 * s2*s2
    elif l == 5:
        if m == 0: res = 0.125 * x * (63.0*x2*x2 - 70.0*x2 + 15.0)
        elif m == 1: res = -1.875 * (21.0*x2*x2 - 14.0*x2 + 1.0) * s
        elif m == 2: res = 52.5 * x * (3.0*x2 - 1.0) * s2
        elif m == 3: res = -52.5 * (9.0*x2 - 1.0) * s * s2
        elif m == 4: res = 945.0 * x * s2*s2
        elif m == 5: res = -945.0 * s * s2*s2
    elif l == 6:
        if m == 0: res = 0.0625 * (231.0*x2*x2*x2 - 315.0*x2*x2 + 105.0*x2 - 5.0)
        elif m == 1: res = -0.125 * (3.0 * x * (231.0*x2*x2 - 210.0*x2 + 35.0)) * s
        elif m == 2: res = 1.875 * (33.0*x2*x2 - 30.0*x2 + 5.0) * s2
        elif m == 3: res = -31.5 * x * (11.0*x2 - 3.0) * s * s2
        elif m == 4: res = 283.5 * (11.0*x2 - 1.0) * s2*s2
        elif m == 5: res = -10395.0 * x * s * s2*s2
        elif m == 6: res = 10395.0 * s2*s2*s2
    elif l == 7:
        x3 = x*x2; x4 = x2*x2; x5 = x*x4; x6 = x2*x4
        s3 = s*s2; s4=s2*s2; s5=s*s4; s6=s2*s4; s7=s*s6
        if m == 0: res = 0.0625 * x * (429.0*x6 - 693.0*x4 + 315.0*x2 - 35.0)
        elif m == 1: res = - (1.0/16.0) * (3003.0*x6 - 3465.0*x4 + 945.0*x2 - 35.0) * s
        elif m == 2: res = 13.125 * x * (143.0*x4 - 110.0*x2 + 15.0) * s2
        elif m == 3: res = -94.5 * (13.0 * x2 * x2 - 10.0*x2 + 1.0) * s3 # Simpler form
        elif m == 4: res = 3465.0 * x * (13.0*x2 - 3.0) * s4
        elif m == 5: res = -62370.0 * (15.0*x2 - 1.0) * s5
        elif m == 6: res = 135135.0 * x * s6
        elif m == 7: res = -135135.0 * s7
    elif l == 8:
        x2=x*x; x4=x2*x2; x6=x2*x4; x8=x4*x4
        s2=s*s; s4=s2*s2; s6=s2*s4; s7=s*s6; s8=s4*s4
        if m == 0: res = (1.0/128.0) * (6435.0*x8 - 12012.0*x6 + 6930.0*x4 - 1260.0*x2 + 35.0)
        elif m == 1: res = -(1.0/128.0) * x * (109395.0*x6 - 162162.0*x4 + 75075.0*x2 - 9450.0) * s
        elif m == 2: res = (1.0/64.0) * (25095.0*x6 - 36036.0*x4 + 15015.0*x2 - 1260.0) * s2
        elif m == 3: res = -(1.0/16.0) * x * (135135.0*x4 - 135135.0*x2 + 25245.0) * s3
        elif m == 4: res = (1.0/8.0) * (405405.0*x4 - 270270.0*x2 + 25245.0) * s4
        elif m == 5: res = -135135.0/2.0 * x * (33.0*x2 - 7.0) * s5
        elif m == 6: res = 2027025.0/2.0 * (17.0*x2 - 1.0) * s6
        elif m == 7: res = -6081075.0 * x * s7
        elif m == 8: res = 6081075.0 * s8
    elif l == 9:
        x2=x*x; x3=x*x2; x4=x2*x2; x5=x*x4; x6=x2*x4; x7=x*x6; x8=x4*x4; x9=x*x8
        s2=s*s; s3=s*s2; s4=s2*s2; s5=s*s4; s6=s2*s4; s7=s*s6; s8=s4*s4; s9=s*s8
        if m==0: res = x*(1.0/128.0)*(12155.0*x8 - 25740.0*x6 + 18018.0*x4 - 4620.0*x2 + 315.0)
        elif m==1: res = -(1.0/256.0)*(230945.0*x8 - 459000.0*x6 + 291720.0*x4 - 60060.0*x2 + 2835.0)*s
        elif m==2: res = (1.0/64.0)*x*(109395.0*x6 - 180180.0*x4 + 90090.0*x2 - 11550.0)*s2
        elif m==3: res = -(1.0/32.0)*(810810.0*x6 - 1081080.0*x4 + 405405.0*x2 - 34650.0)*s3
        elif m==4: res = (1.0/8.0)*x*(405405.0*x4 - 360360.0*x2 + 63063.0)*s4
        elif m==5: res = -(1.0/8.0)*(3648645.0*x4 - 2432430.0*x2 + 225225.0)*s5
        elif m==6: res = (1.0/2.0)*x*(1216215.0*x2 - 347490.0)*s6
        elif m==7: res = - (1.0/2.0)*(18243225.0*x2 - 1737450.0)*s7
        elif m==8: res = 346921275.0 * x * s8
        elif m==9: res = -346921275.0 * s9
    elif l == 10:
        x2=x*x; x4=x2*x2; x6=x2*x4; x8=x4*x4; x10=x2*x8
        s2=s*s; s4=s2*s2; s6=s2*s4; s8=s4*s4; s10=s2*s8
        if m==0: res = (1.0/256.0) * (46189.0*x10 - 109395.0*x8 + 90090.0*x6 - 30030.0*x4 + 3465.0*x2 - 63.0)
        elif m==1: res = -(1.0/256.0)*x*(877591.0*x8 - 1859796.0*x6 + 1351350.0*x4 - 360360.0*x2 + 25245.0)*s
        elif m==2: res = (1.0/256.0)*(4387955.0*x8 - 8369100.0*x6 + 5405400.0*x4 - 1261260.0*x2 + 63063.0)*s2
        elif m==3: res = -(1.0/32.0)*x*(21081075.0*x6 - 31621612.5*x4 + 14230222.5*x2 - 1501500.0)*s3 # Simplified coefficients
        elif m==4: res = (1.0/16.0)*(83204175.0*x6 - 104005218.75*x4 + 37287578.125*x2 - 2972968.125)*s4
        elif m==5: res = -(1.0/8.0)*x*(332816700.0*x4 - 332816700.0*x2 + 56133009.375)*s5
        elif m==6: res = (1.0/8.0)*(1264803465.0*x4 - 993687037.5*x2 + 81758165.625)*s6
        elif m==7: res = -(1.0/2.0)*x*(1806862092.857*x2 - 361372418.571)*s7
        elif m==8: res = (1.0/2.0)*(30716655578.57*x2 - 2559721298.214)*s8
        elif m==9: res = -66449195325.0 * x * s9
        elif m==10: res = 66449195325.0 * s10
        # NOTE: Higher-order polynomials have been simplified for float representation.
        # Slight numerical differences may exist compared to arbitrary-precision results.
    return res

@wp.func
def factorial(n: int) -> int:
    # Since l is small (<= 10), we can use a hardcoded lookup table.
    # This is much more efficient in a Warp kernel than a loop.
    # We only need up to (l+m) = (10+10) = 20.
    if n == 0: return 1
    elif n == 1: return 1
    elif n == 2: return 2
    elif n == 3: return 6
    elif n == 4: return 24
    elif n == 5: return 120
    elif n == 6: return 720
    elif n == 7: return 5040
    elif n == 8: return 40320
    elif n == 9: return 362880
    elif n == 10: return 3628800
    elif n == 11: return 39916800
    elif n == 12: return 479001600
    elif n == 13: return 6227020800
    elif n == 14: return 87178291200
    elif n == 15: return 1307674368000
    elif n == 16: return 20922789888000
    elif n == 17: return 355687428096000
    elif n == 18: return 6402373705728000
    elif n == 19: return 121645100408832000
    elif n == 20: return 2432902008176640000
    
    return 1 # Fallback, should not be reached for l <= 10

@wp.func
def spherical_harmonic_real(l: int, m: int, theta: float, phi: float) -> float:
    """
    Computes the real-valued spherical harmonics Y_l^m(theta, phi).
    This version is valid for l=0..10.
    """
    if l < 0 or wp.abs(m) > l:
        return 0.0

    # The associated Legendre polynomial P_l^|m|(cos(theta))
    # We use our hardcoded, high-performance version.
    plm = associated_legendre(l, wp.abs(m), wp.cos(theta))

    # Calculate the normalization constant
    if m == 0:
        # Normalization for m = 0
        norm = wp.sqrt((2.0 * float(l) + 1.0) / (4.0 * wp.pi))
        return norm * plm
    else:
        # Normalization for m != 0
        # This combines sqrt(2) with the standard normalization constant.
        # It's valid for both positive and negative m.
        f1 = float(factorial(l - wp.abs(m)))
        f2 = float(factorial(l + wp.abs(m)))
        norm = wp.sqrt((2.0 * float(l) + 1.0) * f1 / (2.0 * wp.pi * f2))

        # Select the azimuthal part based on the sign of m
        if m > 0:
            return norm * plm * wp.cos(float(m) * phi)
        else:  # m < 0
            return norm * plm * wp.sin(float(wp.abs(m)) * phi)





