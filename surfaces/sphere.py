import numpy as np
from vector import normalize_vector

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def first_intersect(self, ray):
        # rtc_direction - direction of vector from ray.origin to sphere's position
        rtc_direction = self.position - ray.origin
        # proj_rtc_ray_len - projection length of rtc onto ray
        proj_rtc_ray_len = np.dot(rtc_direction, ray.direction)
        if proj_rtc_ray_len < 0:
            return None  # No intersection
        r_sq = self.radius ** 2
        # distance_position_ray_sq - distance squared between sphere's center and ray
        distance_position_ray_sq = np.dot(rtc_direction, rtc_direction) - proj_rtc_ray_len ** 2
        if distance_position_ray_sq < 0 or distance_position_ray_sq > r_sq:
            return None
        # ray_in_sphere_len - the length of the part of ray inside the sphere
        ray_in_sphere_len = 2 * np.sqrt(r_sq - distance_position_ray_sq)
        return proj_rtc_ray_len - 0.5 * ray_in_sphere_len

    def normal_direction(self, point_on_sphere):
        return normalize_vector(point_on_sphere - self.position)




