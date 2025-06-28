import warp as wp

@wp.struct
class Sphere:
    pos: wp.vec3
    radius: float

@wp.struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3

@wp.struct
class HitInfo:
    didHit: bool
    dist: float
    pos: wp.vec3
    normal: wp.vec3
    localcoord: wp.vec3

@wp.func
def cartesian_to_spherical(pos: wp.vec3) -> wp.vec3:
    r = wp.len(pos)
    t = wp.atan2(wp.sqrt(pos.x * pos.x + pos.y * pos.y), pos.z)
    p = wp.atan2(pos.y, pos.x)
    return wp.vec3(float(r), float(t), float(p))

@wp.func
def ray_sphere(ray: Ray, sphere: Sphere) -> HitInfo:
    hitInfo = HitInfo(didHit=False, dist=0.0, pos=wp.vec3(0.0, 0.0, 0.0), normal=wp.vec3(0.0, 0.0, 0.0), localcoord=wp.vec3(0.0, 0.0, 0.0))
    offsetRayOrigin = ray.origin - sphere.pos

    # Solve for distance with a quadratic equation
    a = wp.dot(ray.direction, ray.direction)
    b = 2.0 * wp.dot(offsetRayOrigin, ray.direction)
    c = wp.dot(offsetRayOrigin, offsetRayOrigin) - sphere.radius * sphere.radius

    # Quadratic discriminant
    discriminant = b * b - 4.0 * a * c

    # If d > 0, the ray intersects the sphere => calculate hitinfo
    if discriminant >= 0.0:
        dist = (-b - wp.sqrt(wp.abs(discriminant))) / (2.0 * a)

        # (If the intersection happens behind the ray, ignore it)
        if dist >= 0.0:
            hitInfo.didHit = True
            hitInfo.dist = dist
            hitInfo.pos = ray.origin + ray.direction * dist
            hitInfo.normal = wp.normalize(hitInfo.pos - sphere.pos)
            hitInfo.localcoord = cartesian_to_spherical(hitInfo.normal)

    return hitInfo
