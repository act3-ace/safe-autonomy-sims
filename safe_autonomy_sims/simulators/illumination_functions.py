"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains custom illumination functions utilized in the CWH inspection task.
"""

import math
from csv import writer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

########################################################## ILLUMINATION ##########################################################


def get_sun_position(current_time, angular_velocity, initial_theta, r_avg):
    """
    Returns current sun position based on initial sun position and elapsed time

    Parameters
    ----------
    current_time: int
        current simulation time in seconds
    angular_velocity: float
        mean motion of sun in meters per second
    initial_theta: float
        initial angle of sun with respect to chief in radians
    r_avg: float
        average distance from earth to sun in meters

    Returns
    -------
    sun_position: array
        array of sun position in meters (CWH coordinates)
    """
    d_theta = angular_velocity
    current_theta = d_theta * current_time + initial_theta
    sun_position = [r_avg * (math.cos(current_theta)), -r_avg * (math.sin(current_theta)), 0]

    return sun_position


def check_illum(point, sun_angle, r_avg, radius):
    """
    Receive a candidate point as an input
    Receive sun position as an input
    Output should be boolean (sun obstructed or not)

    Parameters
    ----------
    point: array
        point on chief
    sun_angle: float
        current sun angle
    r_avg: float
        average distance from earth to sun in meters
    radius: float
        radius of chief in meters

    Returns
    -------
    bool_val: bool
        boolean assigned for illuminated or not
    """

    # Chief position is origin [cwh dynamics]
    center = [0, 0, 0]
    normal_to_surface = normalize(point)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point + 1e-5 * normal_to_surface
    sun_position = [r_avg * (math.cos(sun_angle)), -r_avg * (math.sin(sun_angle)), 0]
    intersection_to_light = normalize(sun_position - shifted_point)

    intersect_var = sphere_intersect(center, radius, shifted_point, intersection_to_light)

    bool_val = False
    # No intersection means that the point in question is illuminated in some capacity
    # (i.e. the point on the chief is not blocked by the chief itself)
    if intersect_var is None:
        bool_val = True

    return bool_val


def num_inspected_points(points):
    """
    Returns number of inspected points

    Parameters
    ----------
    points : dict
        set of inspection points
    """
    pts = 0
    for point in points:
        if points[point]:
            pts += 1

    return pts


def save_data(points, current_time, position, sun_position, action, velocity, path):
    """Save environment data

    Parameters
    ----------
    points : dict
        inspection points
    current_time : float
        current sim time
    position : list
        position vector
    sun_position : list
        sun position vector
    action : list
        action vector
    velocity : list
        velocity vector
    path : str
        path to save data
    """
    points_bool = []
    for point in points:
        if not points[point]:
            points_bool.append(0)
        else:
            points_bool.append(1)

    # Write to csv
    temp = [current_time, position, points_bool, sun_position, action, velocity]
    with open(path, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(temp)
        f_object.close()


def evaluate_RGB(RGB):
    """
    Receive RGB array
    Return boolean based on thresholding logic
    For now, only would work for red

    Parameters
    ----------
    RGB: array
        3x1 array containing RGB value

    Returns
    -------
    RGB_bool: bool
        boolean assigned for illuminated or not
    """
    RGB_bool = True

    # Too dark
    if RGB[0] < .12:
        RGB_bool = False
    # Too bright/white
    if RGB[0] > .8 and RGB[1] > .8 and RGB[2] > .8:
        RGB_bool = False

    return RGB_bool


def compute_illum_pt(point, sun_angle, deputy_position, r_avg, radius, chief_properties, light_properties):
    """
    Receive a candidate point as an input
    Receive sun position as an input
    Returns a color (all zeros (black) for not illuminated)
    """
    # Chief position is origin [cwh dynamics]
    center = [0, 0, 0]
    normal_to_surface = normalize(point)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point + 1e-5 * normal_to_surface
    sun_position = [r_avg * (math.cos(sun_angle)), -r_avg * (math.sin(sun_angle)), 0]
    intersection_to_light = normalize(sun_position - shifted_point)
    intersect_var = sphere_intersect(center, radius, shifted_point, intersection_to_light)

    illumination = np.zeros((3))
    # No intersection means that the point in question is illuminated in some capacity
    # (i.e. the point on the chief is not blocked by the chief itself)
    if intersect_var is None:
        # Blinn-Phong Illumination Model
        # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
        illumination += np.array(chief_properties['ambient']) * np.array(light_properties['ambient'])
        illumination += np.array(chief_properties['diffuse']) * np.array(light_properties['diffuse']) * \
            np.dot(intersection_to_light, normal_to_surface)
        intersection_to_camera = normalize(deputy_position - point)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination += np.array(chief_properties['specular']) * np.array(light_properties['specular']) * \
            np.dot(normal_to_surface, H)**(chief_properties['shininess'] / 4)
        illumination = np.clip(illumination, 0, 1)
    return illumination


def save_image_test_delete(color):
    """
    Recieves RGB array and saves an image
    """
    image = np.zeros((200, 200, 3))
    for i in range(200):
        for j in range(200):
            image[i, j] = color
    string = str(round(color[0], 2)) + '_' + str(round(color[1], 2)) + '_' + str(round(color[2], 2))
    string = string.replace('.', ',')
    string = string + '.png'
    plt.imsave('figs_results/' + string, image)


def compute_illum(deputy_position, sun_position, resolution, radius, focal_length, chief_properties, light_properties, pixel_pitch):
    # pylint: disable-msg=too-many-locals
    """
    Renders the full scene using backwards ray tracing and returns a full RGB image
    """
    ratio = float(resolution[0]) / resolution[1]
    # For now, assuming deputy sensor always pointed at chief (which is origin)
    chief_position = [0, 0, 0]
    sensor_dir = normalize(chief_position - deputy_position)
    image_plane_position = deputy_position + sensor_dir * focal_length
    # There are an infinite number of vectors normal to sensor_dir -- choose one
    x = -1
    y = 1
    z = -(image_plane_position[0] * x + image_plane_position[1] * y) / image_plane_position[2]
    norm1 = normalize([x, y, z])

    # np.cross bug work-around https://github.com/microsoft/pylance-release/issues/3277
    def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a, b)

    norm2 = cross2(sensor_dir, norm1)
    # Used for x,y,z pixel locations - there will be resolution[0] * resolution[1] pixels
    x_width = np.tan((pixel_pitch / focal_length) / 2) * 2 * focal_length
    norm1_range = x_width
    norm2_range = x_width / ratio
    step_norm1 = norm1_range / (resolution[0])
    step_norm2 = norm2_range / (resolution[1])
    # 3D matrix (ie. height-by-width matrix with each entry being an array of size 3) which creates an image
    image = np.zeros((resolution[1], resolution[0], 3))
    for i in range(int(resolution[1])):  # y coords
        for j in range(int(resolution[0])):  # x coords
            # Initialize pixel
            illumination = np.zeros((3))
            # Convert to CWH coordinates
            pixel_location = image_plane_position + ((norm2_range / 2) -
                                                     (i * step_norm2)) * (norm2) + (-(norm1_range / 2) + (j * step_norm1)) * (norm1)
            ray_direction = normalize(pixel_location - deputy_position)
            dist_2_intersect = sphere_intersect(chief_position, radius, deputy_position, ray_direction)
            # Light ray hits sphere, so we continue - else get next pixel
            if dist_2_intersect is not None:
                intersection_point = deputy_position + dist_2_intersect * ray_direction
                normal_to_surface = normalize(intersection_point - chief_position)
                shifted_point = intersection_point + 1e-5 * normal_to_surface
                intersection_to_light = normalize(sun_position - shifted_point)
                intersect_var = sphere_intersect(chief_position, radius, shifted_point, intersection_to_light)
                # If the shifted point doesn't intersect with the chief on the way to the light, it is unobstructed
                if intersect_var is None:
                    # Blinn-Phong Illumination Model
                    # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

                    illumination += np.array(chief_properties['ambient']) * np.array(light_properties['ambient'])
                    illumination += np.array(chief_properties['diffuse']) * np.array(light_properties['diffuse']) * \
                        np.dot(intersection_to_light, normal_to_surface)
                    intersection_to_camera = normalize(deputy_position - intersection_point)
                    H = normalize(intersection_to_light + intersection_to_camera)
                    illumination += np.array(chief_properties['specular']) * np.array(light_properties['specular']) * \
                        np.dot(normal_to_surface, H)**(chief_properties['shininess'] / 4)
                # Shadowed
                else:
                    continue
            image[i, j] = np.clip(illumination, 0, 1)
    return image


def normalize(vector):
    """
    Normalize
    """
    return vector / np.linalg.norm(vector)


def sphere_intersect(center, radius, ray_origin, ray_direction):
    """
    Sphere intersection, returns closest distance if intersection found,
    Returns None upon no intersection
    """
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center)**2 - radius**2
    delta = b**2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    # No intersection
    return None


########################################################## ILLUMINATION ##########################################################

################################################# VISUALIZATION AND VERIFICATION #################################################


def visualizePoints(deputy_position, current_time, sun_position, points):
    """
    Plot the points on the chief and light up green when inspected
    Saves a figure locally
    """
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_title("3D Plot at time: " + str(current_time) + " (sec)")
    plt.grid()
    # Change resolution with this and changing the dpi
    fig.set_size_inches(8, 8)
    chief_position = [0, 0, 0]
    sun_vector = normalize(np.array(sun_position) - chief_position)
    line_scalar = 200
    point_inSunDir = chief_position + line_scalar * sun_vector
    ax.plot(
        [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
        color='#FFD700',
        linewidth=6,
        label='Sun vector'
    )
    temp1 = 0
    temp2 = 0
    for point in points:
        if points[point]:
            if temp1 == 0:
                ax.scatter3D(point[0], point[1], point[2], marker='.', color='green', label='Inspected Point')
                temp1 += 1
            else:
                ax.scatter3D(point[0], point[1], point[2], marker='.', color='green')
        else:
            if temp2 == 0:
                ax.scatter3D(point[0], point[1], point[2], marker='.', color='red', label='Uninspected Point')
                temp2 += 1
            else:
                ax.scatter3D(point[0], point[1], point[2], marker='.', color='red')

    ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
    ax.set_xlabel('X in CWH C.S. [meters]')
    ax.set_ylabel('Y in CWH C.S. [meters]')
    ax.set_zlabel('Z in CWH C.S. [meters]')
    ax.set_box_aspect((1, 1, 1))
    set_axes_equal(ax)
    ax.legend()
    plt.savefig('figs_results/' + 'timestep_' + str(current_time) + 'sec_PointPlot.png', dpi=100)
    plt.close(fig)


def visualize3D(deputy_position, current_time, sun_position, radius):
    """
    Plot deputy, chief and sun direction vector
    Saves a figure locally
    """
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_title("3D Plot at time: " + str(current_time) + " (sec)")
    plt.grid()
    # Change resolution with this and changing the dpi
    fig.set_size_inches(8, 8)
    chief_position = [0, 0, 0]
    sun_vector = normalize(np.array(sun_position) - chief_position)
    line_scalar = 200
    point_inSunDir = chief_position + line_scalar * sun_vector
    ax.plot(
        [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
        color='#FFD700',
        linewidth=6,
        label='Sun vector'
    )
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color='red')
    ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
    ax.set_xlabel('X in CWH C.S. [meters]')
    ax.set_ylabel('Y in CWH C.S. [meters]')
    ax.set_zlabel('Z in CWH C.S. [meters]')
    ax.set_box_aspect((1, 1, 1))
    set_axes_equal(ax)
    ax.legend()
    plt.savefig('figs_results/' + 'timestep_' + str(current_time) + 'sec_3DPlot.png', dpi=100)
    plt.close(fig)


def save_image(RGB, current_time):
    """
    Recieves RGB array and saves an image
    """

    string = 'timestep_' + str(current_time) + 'sec_camera.png'
    plt.imsave('figs_results/' + string, RGB)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.ones(shape=(max_height, total_width, 3))

    if max_height == hb:
        delta = (int(max_height / 2) - int(ha / 2))
        new_img[delta:delta + ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
    if max_height == ha:
        delta = (int(max_height / 2) - int(hb / 2))
        new_img[:ha, :wa] = imga
        new_img[delta:delta + hb, wa:wa + wb] = imgb
    return new_img


def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:, :, :3]
        if i == 0:
            output = img
        else:
            output = concat_images(output, img)
    return output


def combine_images(image1, image2, current_time):
    """
    Combines images into one image and saves result locally
    """
    im_array = [image1, image2]
    final_image = concat_n_images(im_array)
    string = 'timestep_' + str(current_time) + 'sec_COMBINED.png'
    plt.imsave('figs_results/' + string, final_image)


def render_subplots(fig, axes, deputy_position, sun_position, radius, current_time, step_rate):
    # pylint: disable-msg=too-many-statements
    """
    Real-time rendering of scene with subplots (xy, xz, yz)
    TODO: Fix autoscaling problem (need equal axes so sphere looks normal)
    """
    ax_3d = axes[0]
    ax_xy = axes[1]
    ax_xz = axes[2]
    ax_yz = axes[3]
    line_scalar = 200
    chief_position = [0, 0, 0]
    sun_vector = normalize(np.array(sun_position) - chief_position)
    point_inSunDir = chief_position + line_scalar * sun_vector
    # Only runs once at first step
    if current_time == step_rate:
        fig.suptitle('Real-time 3D Inspection Problem with Illumination', fontsize=16)
        ax_3d.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]', zlabel='Z in CWH C.S. [meters]')
        # Subplots
        ax_xy.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]')
        ax_xz.set(xlabel='X in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        ax_yz.set(xlabel='Y in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        fig.set_size_inches(10, 10)
        ax_3d.view_init(elev=20, azim=56)
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax_3d.plot_surface(x, y, z, color='red')
        ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
        ax_3d.scatter3D(chief_position[0], chief_position[1], chief_position[2], marker='o', color='red', label='Chief Spacecraft')
        ax_3d.plot(
            [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
            color='#FFD700',
            linewidth=6,
            label='Sun vector'
        )
        ax_3d.set_box_aspect((1, 1, 1))
        set_axes_equal(ax_3d)
        ax_3d.legend()
        circle1 = Circle((0, 0), radius, color='red')
        circle2 = Circle((0, 0), radius, color='red')
        circle3 = Circle((0, 0), radius, color='red')
        ax_xy.scatter(deputy_position[0], deputy_position[1], marker='o', color='blue')
        ax_xy.add_patch(circle1)
        ax_xz.scatter(deputy_position[0], deputy_position[2], marker='o', color='blue')
        ax_xz.add_patch(circle2)
        ax_yz.scatter(deputy_position[1], deputy_position[2], marker='o', color='blue')
        ax_yz.add_patch(circle3)
        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')
        plt.ion()
        plt.show()
    else:
        ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
        ax_3d.plot(
            [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
            color='#FFD700',
            linewidth=6,
            label='Sun vector'
        )
        ax_3d.autoscale()
        ax_xy.scatter(deputy_position[0], deputy_position[1], marker='o', color='blue')
        ax_xz.scatter(deputy_position[0], deputy_position[2], marker='o', color='blue')
        ax_yz.scatter(deputy_position[1], deputy_position[2], marker='o', color='blue')
        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')
    plt.draw()
    plt.pause(.0001)


def render_3d(fig, deputy_position, sun_position, radius, current_time, step_rate):
    """
    Real-time rendering of scene
    TODO: Fix autoscaling problem (need equal axes so sphere looks normal)
    """
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    line_scalar = 200
    # Initialization
    if current_time == step_rate:
        fig.set_size_inches(10, 10)
        ax.set_title("3D Plot, Inspection Problem")
        ax.set_xlabel('X in CWH C.S. [meters]')
        ax.set_ylabel('Y in CWH C.S. [meters]')
        ax.set_zlabel('Z in CWH C.S. [meters]')
        ax.view_init(elev=20, azim=56)
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax.plot_surface(x, y, z, color='red')
        ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
        chief_position = [0, 0, 0]
        ax.scatter3D(chief_position[0], chief_position[1], chief_position[2], marker='o', color='red', label='Chief Spacecraft')
        sun_vector = normalize(np.array(sun_position) - chief_position)
        point_inSunDir = chief_position + line_scalar * sun_vector
        ax.plot(
            [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
            color='#FFD700',
            linewidth=6,
            label='Sun vector'
        )
        ax.set_box_aspect((1, 1, 1))
        set_axes_equal(ax)
        ax.legend()
        plt.ion()
        plt.show()
    else:
        ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
        chief_position = [0, 0, 0]
        sun_vector = normalize(np.array(sun_position) - chief_position)
        point_inSunDir = chief_position + line_scalar * sun_vector
        ax.plot(
            [point_inSunDir[0], chief_position[0]], [point_inSunDir[1], chief_position[1]], [point_inSunDir[2], chief_position[2]],
            color='#FFD700',
            linewidth=6,
            label='Sun vector'
        )
        ax.autoscale()

    plt.draw()
    plt.pause(.0001)


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


################################################# VISUALIZATION AND VERIFICATION #################################################
