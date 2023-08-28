import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def first_intersect(self, ray):
        ray_normal_proj = np.dot(ray.direction, self.normal_direction(None))
        if ray_normal_proj == 0:
            return None
        intersect_point = (-1 * (np.dot(ray.origin, self.normal_direction(None)) - self.offset)) / ray_normal_proj
        return intersect_point if intersect_point > 0 else None

    def normal_direction(self, point_on_plane):
        return np.array(self.normal)
