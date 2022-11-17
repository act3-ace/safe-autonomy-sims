import os
from array import array
import math
from turtle import pos
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
from numpy import random

########################################################## ILLUMINATION ##########################################################

def sun_position(current_time, dt, angular_velocity, initial_theta, r_avg):
    # Return current sun position based on initial sun position and time elapsed
    
    d_theta = angular_velocity * dt
    current_theta = d_theta * current_time + initial_theta
    sun_position = [r_avg*(math.cos(current_theta)), -r_avg*(math.sin(current_theta)), 0]  

    return sun_position

def sun_angle(current_time, dt, angular_velocity, initial_theta):
    # Return current sun angle based on initial sun position and time elapsed
    
    d_theta = angular_velocity * dt
    current_theta = d_theta * current_time + initial_theta
    return current_theta

def check_illum(point, sun_angle, r_avg, radius):
    # Receive a candidate point as an input
    # Receive sun position as an input
    # Output should be boolean (sun obstructed or not)

    # Chief position is origin [cwh dynamics]
    center = [0,0,0]
    normal_to_surface = normalize(point)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point + 1e-5 * normal_to_surface
    sun_position = [r_avg*(math.cos(sun_angle)), -r_avg*(math.sin(sun_angle)), 0]
    intersection_to_light = normalize(sun_position - shifted_point)

    intersect_var = sphere_intersect(center, radius, shifted_point, intersection_to_light)

    # No intersection means that the point in question is illuminated in some capacity
    # (i.e. the point on the chief is not blocked by the chief itself)
    if intersect_var is None:
        # print("Point is illuminated")
        return True
    else:
        # print("Point is shadowed")
        return False

def evaluate_RGB(RGB):
    # Receive RGB array 
    # Return boolean based on thresholding logic
    # For now, only would work for red
    
    RGB_bool = True

    # Too dark
    if RGB[0] < .12:
        RGB_bool = False
    # Too bright/white
    if RGB[0] > .8 and RGB[1] > .8 and RGB[2] > .8: 
        RGB_bool = False

    return RGB_bool

def compute_illum_pt(point, sun_angle, deputy_position, r_avg, radius, chief_properties, light_properties):
    # Receive a candidate point as an input
    # Receive sun position as an input
    # Output should be RGB if point is illuminated

    # Chief position is origin [cwh dynamics]
    center = [0,0,0]
    normal_to_surface = normalize(point)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point + 1e-5 * normal_to_surface
    sun_position = [r_avg*(math.cos(sun_angle)), -r_avg*(math.sin(sun_angle)), 0]
    intersection_to_light = normalize(sun_position - shifted_point)

    intersect_var = sphere_intersect(center, radius, shifted_point, intersection_to_light)

    # No intersection means that the point in question is illuminated in some capacity
    # (i.e. the point on the chief is not blocked by the chief itself)
    if intersect_var is None:
        illumination = np.zeros((3))
        
        # Blinn-Phong Illumination Model
        # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

        illumination += np.array(chief_properties['ambient']) * np.array(light_properties['ambient'])
        illumination += np.array(chief_properties['diffuse']) * np.array(light_properties['diffuse']) * np.dot(intersection_to_light, normal_to_surface)
        intersection_to_camera = normalize(deputy_position - point)
        H = normalize(intersection_to_light + intersection_to_camera)
        
        illumination += np.array(chief_properties['specular']) * np.array(light_properties['specular']) * np.dot(normal_to_surface, H)**(chief_properties['shininess']/4)

        color =  np.clip(illumination,0,1)
        return True, color
    else:
        # print("Point is shadowed")
        return False, np.zeros((3))

def save_image_test_delete(color):
    # Recieves RGB array and saves an image
    image = np.zeros((200, 200, 3))
    for i in range(200):
        for j in range(200):
            image[i,j] = color
    string = str(round(color[0],2)) + '_' + str(round(color[1],2)) + '_' + str(round(color[2],2))
    string = string.replace('.',',')
    string = string + '.png'
    plt.imsave('figs_results/' + string, image)

def compute_illum(deputy_position, sun_angle, current_time, r_avg, resolution, radius, focal_length, chief_properties, light_properties):
    # Receive a candidate point as an input
    # Receive sun position as an input
    # Output should be RGB

    visualization_flag = True
    ratio = float(resolution[0])/resolution[1]
    # For now, assuming deputy sensor always pointed at chief (which is origin)

    chief_position = [0,0,0]
    sensor_dir = normalize(chief_position - deputy_position)
    image_plane_position = deputy_position + sensor_dir * focal_length
    sun_position = [r_avg*(math.cos(sun_angle)), -r_avg*(math.sin(sun_angle)), 0]

    # There are an infinite number of vectors normal to sensor_dir -- choose one
    x = -1
    y = 1
    z = -(image_plane_position[0]*x + image_plane_position[1]*y)/image_plane_position[2]
    norm1 = normalize([x,y,z])

    # np.cross bug work-around https://github.com/microsoft/pylance-release/issues/3277
    def cross2(a:np.ndarray,b:np.ndarray)->np.ndarray:
        return np.cross(a,b)

    norm2 = cross2(sensor_dir,norm1)

    # Used for x,y,z pixel locations - there will be resolution[0] * resolution[1] pixels
    norm1_range = 1
    norm2_range = 1/ratio
    step_norm1 = norm1_range/(resolution[0])
    step_norm2 = norm2_range/(resolution[1]) 

    # 3D matrix (ie. height-by-width matrix with each entry being an array of size 3) which creates an image
    image = np.zeros((resolution[1], resolution[0], 3))

    if visualization_flag:
        visualize3D(deputy_position, current_time, sun_position)

    for i in range(int(resolution[1])): # y coords
        for j in range(int(resolution[0])): # x coords
            # Initialize pixel
            color = np.zeros((3))
            
            # Convert to CWH coordinates
            pixel_location = image_plane_position + ((norm2_range/2) - (i*step_norm2))*(norm2) + (-(norm1_range/2) + (j*step_norm1))*(norm1)

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
                    illumination = np.zeros((3))
                    
                    # Blinn-Phong Illumination Model
                    # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

                    illumination += np.array(chief_properties['ambient']) * np.array(light_properties['ambient'])
                    illumination += np.array(chief_properties['diffuse']) * np.array(light_properties['diffuse']) * np.dot(intersection_to_light, normal_to_surface)
                    intersection_to_camera = normalize(deputy_position - intersection_point)
                    H = normalize(intersection_to_light + intersection_to_camera)
                    
                    illumination += np.array(chief_properties['specular']) * np.array(light_properties['specular']) * np.dot(normal_to_surface, H)**(chief_properties['shininess']/4)

                    # Reflection
                    # color += reflection * illumination
                    # reflection *= nearest_object['reflection']

                    # origin = shifted_point
                    # direction = reflected(direction, normal_to_surface)    

                    color = illumination
                    
                # Shadowed
                else:
                    continue

            image[i,j] = np.clip(color,0,1)
            
    return image

def normalize(vector):
    return vector/np.linalg.norm(vector)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin-center)
    c = np.linalg.norm(ray_origin-center)**2 - radius**2
    delta = b**2 - 4*c
    if delta > 0:
        t1 = (-b + np.sqrt(delta))/2
        t2 = (-b - np.sqrt(delta))/2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    # No intersection
    return None

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

########################################################## ILLUMINATION ##########################################################

################################################# VISUALIZATION AND VERIFICATION #################################################

def visualizePoints(fig_obj, deputy_position, current_time, sun_position, points):
    # Plot the points on the chief and light them up when inspected

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

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_title("3D Plot at time: " + str(current_time) + " (sec)")
    plt.grid()

    # Change resolution with this and changing the dpi
    fig.set_size_inches(8,8)

    chief_position = [0,0,0]
    # sun_vector = normalize(chief_position - np.array(sun_position))
    sun_vector = normalize(np.array(sun_position) - chief_position)

    line_scalar = 200

    point_inSunDir = chief_position + line_scalar * sun_vector
    ax.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')
    
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

    ax.set_box_aspect((1,1,1)) 
    set_axes_equal(ax)
    ax.legend()

    plt.savefig('figs_results/' + 'timestep_' + str(current_time) + 'sec_PointPlot.png', dpi = 100)
    plt.close(fig)

def visualize3D(deputy_position, current_time, sun_position, radius):
    # Plot deputy, chief and sun direction vector
    
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

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_title("3D Plot at time: " + str(current_time) + " (sec)")
    plt.grid()

    # Change resolution with this and changing the dpi
    fig.set_size_inches(8,8)

    chief_position = [0,0,0]
    # sun_vector = normalize(chief_position - np.array(sun_position))
    sun_vector = normalize(np.array(sun_position) - chief_position)

    line_scalar = 200

    point_inSunDir = chief_position + line_scalar * sun_vector
    ax.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')

    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = radius*np.cos(u) * np.sin(v)
    y = radius*np.sin(u) * np.sin(v)
    z = radius*np.cos(v)
    ax.plot_surface(x, y, z, color = 'red')
    ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')
    ax.set_xlabel('X in CWH C.S. [meters]')
    ax.set_ylabel('Y in CWH C.S. [meters]')
    ax.set_zlabel('Z in CWH C.S. [meters]')

    ax.set_box_aspect((1,1,1)) 
    set_axes_equal(ax)
    ax.legend()

    plt.savefig('figs_results/' + 'timestep_' + str(current_time) + 'sec_3DPlot.png', dpi = 100)
    plt.close(fig)

def save_image(RGB, current_time):
    # Recieves RGB array and saves an image
    string = 'timestep_' + str(current_time) + 'sec_camera.png'
    plt.imsave('figs_results/' + string, RGB)

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.ones(shape=(max_height, total_width, 3))

    if max_height == hb:
        delta = (int(max_height/2) - int(ha/2))
        new_img[delta:delta+ha,:wa]=imga
        new_img[:hb,wa:wa+wb]=imgb
    if max_height == ha:
        delta = (int(max_height/2) - int(hb/2))
        new_img[:ha,:wa]=imga
        new_img[delta:delta+hb,wa:wa+wb]=imgb
        
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

def combine_images(image1, image2, current_time):
    im_array = [image1, image2]
    final_image = concat_n_images(im_array)
    string = 'timestep_' + str(current_time) + 'sec_COMBINED.png'
    plt.imsave('figs_results/' + string, final_image)

def render_subplots(fig, axes, deputy_position, sun_position, radius, current_time, step_rate):
    # Real-time rendering of scene
    # TODO: Fix autoscaling problem (need equal axes so sphere looks normal)
    # TODO: Erase previous sun vectors 

    ax_3d = axes[0]
    ax_xy = axes[1]
    ax_xz = axes[2]
    ax_yz = axes[3]
    line_scalar = 200

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

    # Only runs once at first step
    if current_time == step_rate:
        fig.suptitle('Real-time 3D Inspection Problem with Illumination', fontsize=16)
        ax_3d.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]', zlabel='Z in CWH C.S. [meters]')
        # Subplots
        ax_xy.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]')
        ax_xy.axis('square')
        ax_xz.set(xlabel='X in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        ax_xz.axis('square')
        ax_yz.set(xlabel='Y in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        ax_yz.axis('square')

        fig.set_size_inches(10,10)
        ax_3d.view_init(elev=20, azim = 56)

        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius*np.cos(u) * np.sin(v)
        y = radius*np.sin(u) * np.sin(v)
        z = radius*np.cos(v)
        ax_3d.plot_surface(x, y, z, color = 'red')
        ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')

        chief_position = [0,0,0]
        ax_3d.scatter3D(chief_position[0], chief_position[1], chief_position[2], marker='o', color='red', label='Chief Spacecraft')

        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        ax_3d.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')
        
        ax_3d.set_box_aspect((1,1,1)) 
        set_axes_equal(ax_3d)
        ax_3d.legend()

        circle1 = Circle((0,0), radius, color = 'red')
        circle2 = Circle((0,0), radius, color = 'red')
        circle3 = Circle((0,0), radius, color = 'red')

        ax_xy.scatter(deputy_position[0],deputy_position[1], marker='o', color = 'blue')
        ax_xy.add_patch(circle1)
        ax_xz.scatter(deputy_position[0],deputy_position[2], marker='o', color = 'blue')
        ax_xz.add_patch(circle2)
        ax_yz.scatter(deputy_position[1],deputy_position[2], marker='o', color = 'blue')
        ax_yz.add_patch(circle3)

        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')

        plt.ion()
        plt.show()

        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())

    else:

        ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')

        chief_position = [0,0,0]
        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        ax_3d.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')

        ax_3d.autoscale()

        ax_xy.scatter(deputy_position[0],deputy_position[1], marker='o', color = 'blue')
        ax_xz.scatter(deputy_position[0],deputy_position[2], marker='o', color = 'blue')
        ax_yz.scatter(deputy_position[1],deputy_position[2], marker='o', color = 'blue')
        # ax.set_box_aspect((1,1,1)) 
        # set_axes_equal(ax)
        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')
        
    plt.draw()
    plt.pause(.0001)

def render_subplots_points(deputy_position, sun_position, current_time, step_rate, new_pts_list, radius):
    # Real-time rendering of scene
    # TODO: Fix autoscaling problem (need equal axes so sphere looks normal)
    # TODO: Erase previous sun vectors 
    # TODO: Color points initially and then just update the color of each point so dont keep plotting same stuff 

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

    # Only runs once at first step
    if current_time == step_rate:
        fig = plt.figure(1)
        fig.suptitle('Real-time 3D Inspection Problem with Illumination', fontsize=16)
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_3d.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]', zlabel='Z in CWH C.S. [meters]')

        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xy.set(xlabel='X in CWH C.S. [meters]', ylabel='Y in CWH C.S. [meters]')
        ax_xy.axis('square')
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_xz.set(xlabel='X in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        ax_xz.axis('square')
        ax_yz = fig.add_subplot(2, 2, 4)
        ax_yz.set(xlabel='Y in CWH C.S. [meters]', ylabel='Z in CWH C.S. [meters]')
        ax_yz.axis('square')

        fig.set_size_inches(10,10)

    line_scalar = 200

    # Also runs only once
    if not hasattr('animation_objects'):

        ax_3d.view_init(elev=20, azim = 56)
        animation_objects = {}

        red_pts = _state.points

        for point in _state.points:

            if _state.points[point]:
                ax_3d.scatter3D(point[0], point[1], point[2], marker='.', color='green')

                if point[2] > 0:
                    ax_xy.scatter(point[0], point[1], color = 'green')
                if point[1] < 0:
                    ax_xz.scatter(point[0], point[2], color = 'green')
                if point[0] > 0:
                    ax_yz.scatter(point[1], point[2], color = 'green')
            else:
                ax_3d.scatter3D(point[0], point[1], point[2], marker='.', color='red')

                if point[2] > 0:
                    ax_xy.scatter(point[0], point[1], color = 'red')
                if point[1] < 0:
                    ax_xz.scatter(point[0], point[2], color = 'red')
                if point[0] > 0:
                    ax_yz.scatter(point[1], point[2], color = 'red')
        
        animation_objects["deputy"] = ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')

        chief_position = [0,0,0]
        animation_objects["fake_chief"] = ax_3d.scatter3D(chief_position[0], chief_position[1], chief_position[2], marker='o', color='red', label='Chief Spacecraft')

        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        animation_objects["sun"] = ax_3d.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')
        
        ax_3d.set_box_aspect((1,1,1)) 
        set_axes_equal(ax_3d)
        ax_3d.legend()

        animation_objects["xy"] = ax_xy.scatter(deputy_position[0],deputy_position[1], marker='o', color = 'blue')
        animation_objects["xz"] = ax_xz.scatter(deputy_position[0],deputy_position[2], marker='o', color = 'blue')
        animation_objects["yz"] = ax_yz.scatter(deputy_position[1],deputy_position[2], marker='o', color = 'blue')

        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')

        plt.ion()
        plt.show()

        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())

    else:

        ax_3d.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue')
        
        for i in range(0,len(new_pts_list)):
            point = new_pts_list[i]
            print(point)
            ax_3d.scatter3D(point[0], point[1], point[2], marker='.', color='green')
            if point[2] > 0:
                ax_xy.scatter(point[0], point[1], color = 'green')
            if point[1] < 0:
                ax_xz.scatter(point[0], point[2], color = 'green')
            if point[0] > 0:
                ax_yz.scatter(point[1], point[2], color = 'green')

        chief_position = [0,0,0]
        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        ax_3d.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6)

        ax_3d.autoscale()

        ax_xy.scatter(deputy_position[0],deputy_position[1], marker='o', color = 'blue')
        ax_xz.scatter(deputy_position[0],deputy_position[2], marker='o', color = 'blue')
        ax_yz.scatter(deputy_position[1],deputy_position[2], marker='o', color = 'blue')
        
        ax_xz.axis('square')
        ax_xy.axis('square')
        ax_yz.axis('square')

        # ax.set_box_aspect((1,1,1)) 
        # set_axes_equal(ax)

    plt.draw()
    plt.pause(.0001)

def render_3d(fig, deputy_position, sun_position, radius, current_time, step_rate):
    # Real-time rendering of scene
    # TODO: Fix autoscaling problem (need equal axes so sphere looks normal)
    # TODO: Erase previous sun vectors 

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

    ax = fig.gca(projection='3d')
    line_scalar = 200

    # Initialization
    if current_time == step_rate:
        fig.set_size_inches(10,10)

        ax.set_title("3D Plot, Inspection Problem")
        ax.set_xlabel('X in CWH C.S. [meters]')
        ax.set_ylabel('Y in CWH C.S. [meters]')
        ax.set_zlabel('Z in CWH C.S. [meters]')
        ax.view_init(elev = 20, azim = 56)

        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax.plot_surface(x, y, z, color = 'red')
        ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')

        chief_position = [0,0,0]
        ax.scatter3D(chief_position[0], chief_position[1], chief_position[2], marker='o', color='red', label='Chief Spacecraft')

        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        ax.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')
        
        ax.set_box_aspect((1,1,1)) 
        set_axes_equal(ax)
        ax.legend()

        plt.ion()
        plt.show()

    else:
        ax.scatter3D(deputy_position[0], deputy_position[1], deputy_position[2], marker='o', color='blue', label='Deputy Spacecraft')

        chief_position = [0,0,0]
        sun_vector = normalize(np.array(sun_position) - chief_position)

        point_inSunDir = chief_position + line_scalar * sun_vector
        ax.plot([point_inSunDir[0], chief_position[0]],[point_inSunDir[1], chief_position[1]],[point_inSunDir[2], chief_position[2]], color='#FFD700', linewidth = 6, label = 'Sun vector')

        ax.autoscale()
        # ax.set_box_aspect((1,1,1)) 
        # set_axes_equal(ax)

    plt.draw()
    plt.pause(.0001)

################################################# VISUALIZATION AND VERIFICATION #################################################