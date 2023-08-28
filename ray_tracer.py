import argparse
import math
import time
from random import random

from PIL import Image
import numpy as np

from vector import Vector
from vector import normalize_vector, get_length
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


# Global variables
R = -1
R_x = -1
R_y = -1


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, image_path):
    image = Image.fromarray(np.uint8(image_array))
    # Save the image to a file
    image.save(image_path)


def compute_v_towards(position, v_look_at):
    # subtracting ndarrays
    v_towards = v_look_at - position
    return normalize_vector((v_towards))


def compute_xyz_axis(M):
    x_axis = normalize_vector(np.matmul(np.array([1, 0, 0]), M))
    y_axis = normalize_vector(np.matmul(np.array([0, 1, 0]), M))
    z_axis = normalize_vector(np.matmul(np.array([0, 0, 1]), M))
    return  x_axis, y_axis, z_axis


def compute_p_position_and_xy_axis(camera, width, height):
    v_towards = normalize_vector(compute_v_towards(camera.position, camera.look_at))

    rotational_M = compute_rotational_matrix((v_towards))

    x_axis, y_axis, z_axis = compute_xyz_axis(rotational_M)
    screen_middle = camera.position + z_axis * camera.screen_distance
    p_position = compute_p_position(x_axis, y_axis, width, height, screen_middle, camera.screen_width)
    return p_position, x_axis, y_axis


# screen origin point
def compute_p_position(x_axis, y_axis, width, height, screen_middle, screen_width):
    screen_height = screen_width / (width / height)
    return screen_middle - 0.5 * screen_width * x_axis - 0.5 * screen_height * y_axis

# based on the recitation
def compute_rotational_matrix(v_towards):
    dim = 3
    rotational_mat = np.zeros((dim, dim))
    a = v_towards[0]
    b = v_towards[1]
    c = v_towards[2]
    sx = -b
    cx = math.sqrt(1 - sx * sx)
    sy = -a / cx
    cy = c / cx
    rotational_mat[0][0] = cy
    rotational_mat[0][2] = sy
    rotational_mat[1][0] = -sx * sy
    rotational_mat[1][1] = cx
    rotational_mat[1][2] = sx * cy
    rotational_mat[2][0] = -cx * sy
    rotational_mat[2][1] = -sx
    rotational_mat[2][2] = cx * cy

    return rotational_mat


def create_rectangle_for_light_source(hit_point, light_source):
    ray_towards_vector = normalize_vector(np.array(hit_point) - np.array(light_source.position))
    mat_for_rect = compute_rotational_matrix(ray_towards_vector)
    x_axis = normalize_vector((np.matmul(np.array([1, 0, 0]), mat_for_rect)))
    y_axis = normalize_vector((np.matmul(np.array([0, 1, 0]), mat_for_rect)))
    rect_vertex = light_source.position - 0.5 * light_source.radius * x_axis - 0.5 * light_source.radius * y_axis
    return rect_vertex, x_axis, y_axis


def cast_ray_for_pixel_and_find_color(pixel, light_source, hit_point, min_surf_obj, ray, materials_list, surfaces_list):

    obj_normal = min_surf_obj.normal_direction(hit_point)
    hit_point = hit_point + 1e-5 * obj_normal

    light_source_direction = normalize_vector(hit_point - pixel)
    light_source_vector = Vector(pixel, light_source_direction)

    intersect_list = find_nearest_surface_wo_intersect_list(light_source_vector, surfaces_list)

    min_surf_obj, min_surface_dist = intersect_list
    if min_surface_dist is None:
        return None
    hit_point_dist = get_length(pixel, hit_point)
    if min_surface_dist < hit_point_dist:
        return None
    min_surf_material = materials_list[min_surf_obj.material_index]
    light_source_opp_dir = normalize_vector(pixel - hit_point) # light source opposite direction
    specular_color = calc_specular_color(min_surf_material, obj_normal, hit_point, light_source_opp_dir, ray)
    diffuse_color = calc_diffuse_color(min_surf_material, obj_normal, light_source_opp_dir)
    rgb_color = light_source.color * (specular_color + diffuse_color)
    return rgb_color


def compute_light_intensity(light_source, total_rays, total_rays_hit):
    shadow_intensity = light_source.shadow_intensity
    percent_of_rays_hit = total_rays_hit / total_rays
    light_intensity = (1 - shadow_intensity) + shadow_intensity * percent_of_rays_hit
    return light_intensity


def calc_specular_color(surf_material, obj_normal, hit_point, light_source_opp_dir, ray):
    Ks = surf_material.specular_color
    shininess_coeff = surf_material.shininess

    V = normalize_vector(ray.origin - hit_point)  #direction from viewer
    R = light_source_opp_dir - 2 * np.dot(light_source_opp_dir, obj_normal) * obj_normal

    R_L_dot_product = np.dot(V, R)
    specular_color = Ks * (R_L_dot_product ** shininess_coeff)
    return specular_color


def calc_diffuse_color(surf_material, obj_normal, light_source_opp_dir):
    normal_opp_light_dot_product = np.dot(obj_normal, light_source_opp_dir)
    diffuse_color = surf_material.diffuse_color * normal_opp_light_dot_product
    return np.clip(diffuse_color, 0, 1)


def compute_random_pixels_in_rectangle(light, point, scene_settings):
    shadow_rays = scene_settings.root_number_shadow_rays
    rect_vertex, x_axis, y_axis = create_rectangle_for_light_source(point, light)
    list_of_pixels = []
    for i in range(int(shadow_rays)):
        copy_vertex = np.copy(rect_vertex)
        for j in range(int(shadow_rays)):
            new_pixel = copy_vertex + random() * (x_axis / shadow_rays) + random() * (y_axis / shadow_rays)
            list_of_pixels.append(new_pixel)
            copy_vertex += x_axis/shadow_rays
        rect_vertex += y_axis/shadow_rays
    return list_of_pixels


def curr_light_source_color(light_source, hit_point, obj, ray, scene_settings, materials_list, surfaces_list):
    rgb_color = np.array([0, 0, 0], dtype='float64')
    shadow_rays_squared = int(scene_settings.root_number_shadow_rays)
    total_rays = shadow_rays_squared ** 2
    total_rays_hit = 0

    # a list of random pixels in triangle
    pixels_for_light_rays = compute_random_pixels_in_rectangle(light_source, hit_point, scene_settings)

    for pixel in pixels_for_light_rays:
        ray_color = cast_ray_for_pixel_and_find_color(pixel, light_source, hit_point, obj, ray, materials_list, surfaces_list)
        if ray_color is not None:
            total_rays_hit += 1
            rgb_color += ray_color
    light_intensity = compute_light_intensity(light_source, total_rays, total_rays_hit)
    rgb_color /= total_rays
    rgb_color *= light_intensity
    return rgb_color


def compute_combined_light_sources_color(hit_point, obj, ray, scene_settings, materials_list, surfaces_list,
                                         lights_list):
    rgb_color = np.array([0, 0, 0], dtype='float64')

    for light_source in lights_list:
        rgb_color += curr_light_source_color(light_source, hit_point, obj, ray, scene_settings, materials_list, surfaces_list)
    return rgb_color


def create_intersect_list_and_compute_color(rec_left, ray, scene_settings, materials_list, surfaces_list, lights_list):
    if rec_left == 0:  # If we reached the maximum recursion (and rec_left is 0), the returned color is the background color of the scene
        return np.array(scene_settings.background_color)
    intersect_list = create_ray_intersect_list(ray, surfaces_list)
    if len(intersect_list) == 0:  # If there is no intersection, the returned color is the background color of the scene
        return np.array(scene_settings.background_color)
    return compute_color(rec_left, intersect_list, 0, ray, scene_settings, materials_list, surfaces_list, lights_list)


def compute_color(rec_level, intersect_list, curr_intersect_ind, ray, scene_settings, materials_list, surfaces_list, lights_list):
    nearest_surface, nearest_surface_distance = intersect_list[curr_intersect_ind][0], intersect_list[curr_intersect_ind][1]
    end_of_ray = ray.origin + nearest_surface_distance * ray.direction
    ids_color = compute_combined_light_sources_color(end_of_ray, nearest_surface, ray, scene_settings, materials_list, surfaces_list, lights_list)
    # ids color means intensity * ( diffuse + specular)
    reflected_ray = create_reflected_ray(ray, nearest_surface, end_of_ray)
    ref_color = create_intersect_list_and_compute_color(rec_level - 1, reflected_ray, scene_settings, materials_list, surfaces_list, lights_list)
    # ref color means reflection color
    ref_color *= materials_list[nearest_surface.material_index].reflection_color
    surface_background_color = compute_background_color_of_surface(nearest_surface, intersect_list, curr_intersect_ind, ray, scene_settings, materials_list, surfaces_list, lights_list)
    nearest_transparency = materials_list[nearest_surface.material_index].transparency
    return np.clip(0, 1, surface_background_color * nearest_transparency + ids_color * (1-nearest_transparency) + ref_color)


def compute_background_color_of_surface(nearest_surface, intersect_list, curr_intersect_ind, ray, scene_settings, materials_list, surfaces_list, lights_list):
    nearest_surface_transparency = materials_list[nearest_surface.material_index].transparency
    background_color = np.array([1., 1., 1.])  # initial value
    if nearest_surface_transparency > 0:  # Not opaque
        if curr_intersect_ind < len(intersect_list) - 1:  # Not the last surface
            background_color *= compute_color(scene_settings.max_recursions, intersect_list, curr_intersect_ind + 1, ray, scene_settings, materials_list, surfaces_list, lights_list)
        elif curr_intersect_ind == len(intersect_list) - 1:  # The last surface (no surfaces behind)
            background_color *= np.array(scene_settings.background_color)
    return background_color


def create_ray_intersect_list(ray, surfaces_list):
    intersect_list = []
    for surface in surfaces_list:
        intersect_distance = surface.first_intersect(ray)
        if intersect_distance:  # If there is an intersection
            intersect_list.append((surface, intersect_distance))
    # We sort the list based on the distances (ascending order)
    return sorted(intersect_list, key=lambda item: item[1])


def create_reflected_ray(ray, curr_surface, point_on_surface):
    normal_direction = curr_surface.normal_direction(point_on_surface)
    reflection_direction = ray.direction - 2 * np.dot(ray.direction, normal_direction) * normal_direction
    return Vector(point_on_surface, reflection_direction)


def find_nearest_surface_wo_intersect_list(ray, surfaces_list):
    if not surfaces_list:  # No intersection
        return None, None
    nearest_surface, nearest_surface_distance = surfaces_list[0], surfaces_list[0].first_intersect(ray)
    for i in range(1, len(surfaces_list)):
        curr_surface_distance = surfaces_list[i].first_intersect(ray)
        if curr_surface_distance is None or curr_surface_distance < 0:
            continue
        if nearest_surface_distance is None or curr_surface_distance < nearest_surface_distance:
            nearest_surface, nearest_surface_distance = surfaces_list[i], curr_surface_distance
    return nearest_surface, nearest_surface_distance


def split_objects_list(objects):
    materials_list, surfaces_list, lights_list = [None], [], []
    for obj in objects:
        if isinstance(obj, Material):
            materials_list.append(obj)
        elif isinstance(obj, Light):
            lights_list.append(obj)
        else:
            surfaces_list.append(obj)
    return materials_list, surfaces_list, lights_list


def ray_cast(camera, scene_settings, materials_list, surfaces_list, lights_list, width_pixels, height_pixels):
    p_position, x_axis, y_axis = compute_p_position_and_xy_axis(camera, width_pixels, height_pixels)

    image = np.zeros((height_pixels, width_pixels, 3))

    screen_height = camera.screen_width / (width_pixels/height_pixels)
    start_time = time.time()

    for i in range(height_pixels):
        row = height_pixels - 1 - i
        point = np.copy(p_position)
        for j in range(width_pixels):
            # create a ray for each pixel
            print(f"pixel {i},{j}")
            ray = Vector(camera.position, normalize_vector((point - camera.position)))
            pixel_color = create_intersect_list_and_compute_color(scene_settings.max_recursions, ray, scene_settings, materials_list, surfaces_list, lights_list)
            image[row][j] = np.round(pixel_color*255)

            point += x_axis/(width_pixels/camera.screen_width)

        p_position += y_axis / (height_pixels / screen_height)
    # print("total runtime is: ", time.time() - start_time)
    return image


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()
    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    materials_list, surfaces_list, lights_list = split_objects_list(objects)
    width_pixels, height_pixels = args.width, args.height
    image_array = ray_cast(camera, scene_settings, materials_list, surfaces_list, lights_list, width_pixels, height_pixels)
    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
