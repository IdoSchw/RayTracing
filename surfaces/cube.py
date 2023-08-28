import numpy as np


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index
        self.top_x = self.position[0] + (self.scale / 2)
        self.bottom_x = self.position[0] - (self.scale / 2)
        self.top_y = self.position[1] + (self.scale / 2)
        self.bottom_y = self.position[1] - (self.scale / 2)
        self.top_z = self.position[2] + (self.scale / 2)
        self.bottom_z = self.position[2] - (self.scale / 2)

    def first_intersect(self, ray):
        rox, roy, roz = ray.origin[0], ray.origin[1], ray.origin[2]
        rdx, rdy, rdz = ray.direction[0], ray.direction[1], ray.direction[2]
        first_x, second_x = (self.top_x - rox) / rdx, (self.bottom_x - rox) / rdx
        if rox < 0:
            first_x, second_x = second_x, first_x
        first = first_x
        second = second_x
        first_y, second_y = (self.top_y - roy) / rdy, (self.bottom_y - roy) / rdy
        if roy < 0:
            first_y, second_y = second_y, first_y
        if first > second_y or first_y > second:
            return None
        first = first_y if first_y > first else first
        second = second_y if second_y < second else second
        first_z, second_z = (self.top_z - roz) / rdz, (self.bottom_z - roz) / rdz
        if roz < 0:
            first_z, second_z = second_z, first_z
        if first > second_z or first_z > second:
            return None
        first = first_z if first_z > first else first
        # second = second_z if second_z < second else second
        return first

    def normal_direction(self, point_on_cube):
        eps = 10 ** (-7)
        if abs(point_on_cube[0] - self.top_x) < eps:
            return np.array([1, 0, 0])
        if abs(point_on_cube[0] - self.bottom_x) < eps:
            return np.array([-1, 0, 0])
        if abs(point_on_cube[1] - self.top_y) < eps:
            return np.array([0, 1, 0])
        if abs(point_on_cube[1] - self.bottom_y) < eps:
            return np.array([0, -1, 0])
        if abs(point_on_cube[2] - self.top_z) < eps:
            return np.array([0, 0, 1])
        if abs(point_on_cube[2] - self.bottom_z) < eps:
            return np.array([0, 0, -1])



