"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module creates an animation for the inspection environment.
"""
import abc
import math
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseAnimation(abc.ABC):
    """Base animation module"""

    def __init__(self):
        self.bool_array_locations = False

    def points_on_sphere_fibonacci(self, num_points: int, radius: float) -> list:
        """
        Generate a set of equidistant points on sphere using the
        Fibonacci Sphere algorithm: https://arxiv.org/pdf/0912.4540.pdf

        Parameters
        ----------
        num_points: int
            number of points to attempt to place on a sphere
        radius: float
            radius of the sphere

        Returns
        -------
        points: list
            Set of equidistant points on sphere in cartesian coordinates
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            r = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * r
            z = math.sin(theta) * r

            points.append(radius * np.array([x, y, z]))

        return points

    def make_animation(
        self,
        data_path: str,
        save_dir: str,
        mode: str = 'operator',
        last_step: bool = False,
        radius: float = 10.,
        num_points: int = 100,
        mass: float = 12
    ):
        """Function to make and save animation/plots

        Parameters
        ----------
        data_path: str
            String defining path to data
        save_dir: str
            String defining path to save mp4/png
        mode: str
            Mode for plots to make: 'operator' (position + operator focused metrics),
            'obs_act' (position + observation/action data), or '3d_pos' (only 3d position plot)
        last_step: bool
            True to only plot last step of episode, False to make video of all steps
        radius: float
            Radius of points sphere
        num_points: int
            Number of points on sphere
        mass: float
            Mass of spacecraft (for delta-v calculation)
        """

        tmp_dir = '/tmp/eval_results/animation_plots'

        times, positions, bool_arrays, sun_vectors, control_vectors, velocities = self.get_data(data_path)

        # Make temp directory
        if not os.path.exists(tmp_dir) and not last_step:
            os.mkdir(tmp_dir)

        # Calculate values
        points = self.points_on_sphere_fibonacci(num_points, radius)
        total_time = times[-1]
        step_size = times[1] - times[0]
        max_pos = []
        for x in positions:
            max_pos.append(max(abs(x)))
        max_pos = max(max_pos) * 1.1

        # Set plot values
        if mode == 'operator':
            plt.figure(figsize=(20, 15.04))
            dim0 = (5, 6)
            rowspan_3d = 3
            z_label_pad = 12
        elif mode == 'obs_act':
            plt.figure(figsize=(20, 12))
            dim0 = (4, 6)
            rowspan_3d = 4
            z_label_pad = 5
        elif mode == '3d_pos':
            plt.figure(figsize=(10.08, 8.64))
            dim0 = (3, 3)
            rowspan_3d = 3
            z_label_pad = 12
        plt.rcParams.update({'font.size': 16, 'text.usetex': True})

        # Setup 3d position plot
        ax_3d = plt.subplot2grid(dim0, (0, 0), colspan=3, rowspan=rowspan_3d, projection='3d')
        ax_3d.set_xlabel('In-Track [m]')
        ax_3d.set_ylabel('Cross-Track [m]')
        ax_3d.set_zlabel('Radial [m]')
        ax_3d.set_xlim([-max_pos, max_pos])
        ax_3d.set_ylim([-max_pos, max_pos])
        ax_3d.set_zlim([-max_pos, max_pos])
        ax_3d.set_aspect("equal")
        ax_3d.azim = 120
        ax_3d.xaxis.labelpad = 20
        ax_3d.yaxis.labelpad = 20
        ax_3d.zaxis.labelpad = z_label_pad
        if last_step:
            ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, color=plt.cm.cool(0), linewidth=2, label="Deputy Position")
        else:
            ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, '*', color=plt.cm.cool(0), linewidth=2, label="Deputy Position")
        ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, linewidth=5, color='y', label="Sun Vector")
        if self.bool_array_locations:
            ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, 'g*', label="Inspected Point")
            ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, 'r*', label="Uninspected Point")
        else:
            ax_3d.plot(max_pos * 2, max_pos * 2, max_pos * 2, 'r*', label="Inspection Point")
        ax_3d.legend(loc='upper left')

        # Setup plots
        if mode == 'operator':
            ax_ri = plt.subplot2grid(dim0, (3, 0), rowspan=2)
            ax_ri.grid(True)
            ax_ri.set_xlabel('In-Track [m]')
            ax_ri.set_ylabel('Radial [m]')
            ax_ri.set_xlim([-max_pos, max_pos])
            ax_ri.set_ylim([-max_pos, max_pos])
            ax_ri.set_aspect("equal")
            ax_ri.invert_xaxis()

            ax_rc = plt.subplot2grid(dim0, (3, 1), rowspan=2)
            ax_rc.grid(True)
            ax_rc.set_xlabel('Cross-Track [m]')
            ax_rc.set_ylabel('Radial [m]')
            ax_rc.set_xlim([-max_pos, max_pos])
            ax_rc.set_ylim([-max_pos, max_pos])
            ax_rc.set_aspect("equal")
            ax_rc.invert_xaxis()

            ax_ic = plt.subplot2grid(dim0, (3, 2), rowspan=2)
            ax_ic.grid(True)
            ax_ic.set_xlabel('In-Track [m]')
            ax_ic.set_ylabel('Cross-Track [m]')
            ax_ic.set_xlim([-max_pos, max_pos])
            ax_ic.set_ylim([-max_pos, max_pos])
            ax_ic.set_aspect("equal")
            ax_ic.invert_xaxis()
            ax_ic.invert_yaxis()

            max_rp = []
            for x in positions:
                max_rp.append(np.linalg.norm(x))
            max_rp = max(max_rp) * 1.1
            ax_rp = plt.subplot2grid(dim0, (1, 3), colspan=3)
            ax_rp.set_xlim([0, total_time])
            ax_rp.set_ylim([0, max_rp])
            ax_rp.set_xlabel('Time [sec]')
            ax_rp.set_ylabel('Relative Distance [m]')
            ax_rp.grid(True)

            max_rv = []
            for x in velocities:
                max_rv.append(np.linalg.norm(x))
            max_rv = max(max_rv) * 1.1
            ax_rv = plt.subplot2grid(dim0, (2, 3), colspan=3)
            ax_rv.set_xlim([0, total_time])
            ax_rv.set_ylim([0, max_rv])
            ax_rv.set_xlabel('Time [sec]')
            ax_rv.set_ylabel('Relative Speed [m/s]')
            ax_rv.grid(True)

            max_dv = 0
            for x in control_vectors:
                max_dv += np.sum(np.abs(x)) / mass * step_size
            max_dv = max_dv * 1.1
            ax_dv = plt.subplot2grid(dim0, (4, 3), colspan=3)
            ax_dv.set_xlim([0, total_time])
            ax_dv.set_ylim([0, max_dv])
            ax_dv.set_xlabel('Time [sec]')
            ax_dv.set_ylabel('Delta V [m/s]')
            ax_dv.grid(True)

        if mode in ('operator', 'obs_act'):
            ax_ip = plt.subplot2grid(dim0, (0, 3), colspan=3)
            ax_ip.set_xlim([0, total_time])
            ax_ip.set_ylim([0, 100])
            ax_ip.set_xlabel('Time [sec]')
            ax_ip.set_ylabel('Points Inspected')
            ax_ip.grid(True)

            ax_u = plt.subplot2grid(dim0, (3, 3), colspan=3)
            ax_u.set_xlim([0, total_time])
            ax_u.set_ylim([-1.1, 1.1])
            ax_u.set_xlabel('Time [sec]')
            ax_u.set_ylabel('Control [N]')
            ax_u.grid(True)
            ax_u.plot(-10, -10, 'b', label=r'$F_x$')
            ax_u.plot(-10, -10, 'r', label=r'$F_y$')
            ax_u.plot(-10, -10, 'g', label=r'$F_z$')
            ax_u.legend()

        if mode == 'obs_act':
            max_rp = []
            for x in positions:
                max_rp.append(max(abs(x)))
            max_rp = max(max_rp) * 1.1
            ax_rp = plt.subplot2grid(dim0, (1, 3), colspan=3)
            ax_rp.set_xlim([0, total_time])
            ax_rp.set_ylim([-max_rp, max_rp])
            ax_rp.set_xlabel('Time [sec]')
            ax_rp.set_ylabel('Position [m]')
            ax_rp.grid(True)
            ax_rp.plot(-10, -10, 'b', label=r'$x$')
            ax_rp.plot(-10, -10, 'r', label=r'$y$')
            ax_rp.plot(-10, -10, 'g', label=r'$z$')
            ax_rp.legend()

            max_rv = []
            for x in velocities:
                max_rv.append(max(abs(x)))
            max_rv = max(max_rv) * 1.1
            ax_rv = plt.subplot2grid(dim0, (2, 3), colspan=3)
            ax_rv.set_xlim([0, total_time])
            ax_rv.set_ylim([-max_rv, max_rv])
            ax_rv.set_xlabel('Time [sec]')
            ax_rv.set_ylabel('Velocity [m/s]')
            ax_rv.grid(True)
            ax_rv.plot(-10, -10, 'b', label=r'$\dot{x}$')
            ax_rv.plot(-10, -10, 'r', label=r'$\dot{y}$')
            ax_rv.plot(-10, -10, 'g', label=r'$\dot{z}$')
            ax_rv.legend()

        # Initialize values
        filenames = []
        delta_v = np.sum(np.abs(control_vectors[0])) / mass * step_size
        print('Animation Progress:')
        for i in range(1, len(times)):
            # Get values
            time = times[i]
            last_pos = positions[i - 1]
            pos = positions[i]
            bool_array = bool_arrays[i]
            sun = sun_vectors[i]
            control = control_vectors[i]
            velocity = velocities[i]

            if last_step:
                # 3d position plot
                ax_3d.plot(
                    [last_pos[1], pos[1]], [last_pos[2], pos[2]], [last_pos[0], pos[0]], color=plt.cm.cool(i / len(times)), linewidth=2
                )
                if mode == 'operator':
                    ax_ri.plot([last_pos[1], pos[1]], [last_pos[0], pos[0]], color=plt.cm.cool(i / len(times)), linewidth=2)
                    ax_rc.plot([last_pos[2], pos[2]], [last_pos[0], pos[0]], color=plt.cm.cool(i / len(times)), linewidth=2)
                    ax_ic.plot([last_pos[1], pos[1]], [last_pos[2], pos[2]], color=plt.cm.cool(i / len(times)), linewidth=2)

            if not last_step:
                # 3d position plot
                ax_3d.plot([pos[1]], [pos[2]], [pos[0]], '*', color=plt.cm.cool(i / len(times)), linewidth=2)
                if mode == 'operator':
                    ax_ri.plot([pos[1]], [pos[0]], '*', color=plt.cm.cool(i / len(times)), linewidth=2)
                    ax_rc.plot([pos[2]], [pos[0]], '*', color=plt.cm.cool(i / len(times)), linewidth=2)
                    ax_ic.plot([pos[1]], [pos[2]], '*', color=plt.cm.cool(i / len(times)), linewidth=2)

                # Sun vector plot
                sun_vector = sun / np.linalg.norm(sun)
                point_inSunDir = max_pos * sun_vector
                sun_line, = ax_3d.plot([point_inSunDir[1], 0], [point_inSunDir[2], 0], [point_inSunDir[0], 0], linewidth=5, color='y')

                # Time text boxes
                angle = np.arctan2(sun_vector[1], sun_vector[0])
                satellite_time = (-angle / (2 * np.pi) + 0.5) * 24
                mins, hrs = math.modf(satellite_time)
                mins *= 60
                txt = ax_3d.text2D(0.858, 0.90, f"LST: {hrs:02.0f}:{mins:02.0f}", transform=ax_3d.transAxes)
                mins, hrs = math.modf(time / 3600)
                mins *= 60
                txt2 = ax_3d.text2D(0.75, 0.95, f"Time Elapsed: {hrs:02.0f}:{mins:02.0f}", transform=ax_3d.transAxes)

            if not last_step or i == len(times) - 1:
                c = 'r'
                # Points on sphere plot
                plot_points = []
                for j, p in enumerate(bool_array):
                    if self.bool_array_locations:
                        if p:
                            c = 'g'
                        else:
                            c = 'r'

                    plot_points.extend(ax_3d.plot(points[j][1], points[j][2], points[j][0], '*', color=c))
                    if mode == 'operator':
                        plot_points.extend(ax_ri.plot(points[j][1], points[j][0], '*', color=c))
                        plot_points.extend(ax_rc.plot(points[j][2], points[j][0], '*', color=c))
                        plot_points.extend(ax_ic.plot(points[j][1], points[j][2], '*', color=c))

            last_time = times[i - 1]
            last_delta_v = delta_v
            delta_v += np.sum(np.abs(control)) / mass * step_size
            if mode == 'operator':
                ax_dv.plot([last_time, time], [last_delta_v, delta_v], 'b')

                last_rel_pos = np.linalg.norm(last_pos)
                rel_pos = np.linalg.norm(pos)
                ax_rp.plot([last_time, time], [last_rel_pos, rel_pos], 'b')

                last_rel_vel = np.linalg.norm(velocities[i - 1])
                rel_vel = np.linalg.norm(velocity)
                ax_rv.plot([last_time, time], [last_rel_vel, rel_vel], 'b')

            if mode in ('operator', 'obs_act'):
                last_control = control_vectors[i - 1]
                ax_u.plot([last_time, time], [last_control[0], control[0]], 'b')
                ax_u.plot([last_time, time], [last_control[1], control[1]], 'r')
                ax_u.plot([last_time, time], [last_control[2], control[2]], 'g')

                last_pts = sum(bool_arrays[i - 1])
                pts = sum(bool_array)
                ax_ip.plot([last_time, time], [last_pts, pts], 'b')

            if mode == 'obs_act':
                ax_rp.plot([last_time, time], [last_pos[0], pos[0]], 'b')
                ax_rp.plot([last_time, time], [last_pos[1], pos[1]], 'r')
                ax_rp.plot([last_time, time], [last_pos[2], pos[2]], 'g')

                last_vel = velocities[i - 1]
                ax_rv.plot([last_time, time], [last_vel[0], velocity[0]], 'b')
                ax_rv.plot([last_time, time], [last_vel[1], velocity[1]], 'r')
                ax_rv.plot([last_time, time], [last_vel[2], velocity[2]], 'g')

            if not last_step:
                # Save figure
                filename = tmp_dir + '/frame' + str(i) + '.png'
                filenames.append(filename)
                plt.tight_layout()
                plt.savefig(filename)

                # Remove temporary elements
                sun_line.remove()
                for p in plot_points:
                    p.remove()
                txt.remove()
                txt2.remove()

            # Print progress
            var = (i + 1) / len(times)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(var * 20), var * 100))
            sys.stdout.flush()

        if last_step:
            # Make plot
            plt.tight_layout()
            plt.savefig(save_dir + 'episode_plot.png')
        else:
            # Make video
            with imageio.get_writer(save_dir + 'episode_animation.mp4', mode='I', fps=30) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

            # Remove temp files
            for filename in set(filenames):
                os.remove(filename)
            os.rmdir(tmp_dir)

        print('\n')
        print('Metrics:')
        print(f'Inspected Points: {np.count_nonzero(bool_array):0.0f}/{len(bool_array):0.0f}')
        print(f'Time: {time:0.1f} sec')
        print(f'Delta V: {delta_v:0.2f} m/s')

    @abc.abstractmethod
    def get_data(self, data_path: str) -> tuple:
        """Function to get requried data from file

        Parameters
        ----------
        data_path: str
            String defining path to data

        Returns
        -------
        tuple
            tuple of np.ndarrays:
            times, positions, bool_arrays, sun_vectors, control_vectors, velocities
        """
        raise NotImplementedError


class AnimationFromCSV(BaseAnimation):
    """
    Animation class getting data from csv.
    Use save_data_flag in illumination parameters.
    """

    def __init__(self):
        super().__init__()
        self.bool_array_locations = True

    def get_data(self, data_path):
        self.bool_array_locations = True
        # Get data from csv
        data = pd.read_csv(data_path, header=None, converters={1: self.s2a, 2: self.s2a, 3: self.s2a, 4: self.s2a, 5: self.s2a})
        times = data[0].values
        positions = np.array(list(data[1].values))
        bool_arrays = np.array(list(data[2].values))
        sun_vectors = np.array(list(data[3].values))
        control_vectors = np.array(list(data[4].values))
        velocities = np.array(list(data[5].values))
        return times, positions, bool_arrays, sun_vectors, control_vectors, velocities

    def s2a(self, x):
        """Helper function for reading csv"""
        x = x.replace('[', '')
        x = x.replace(']', '')
        x = x.replace(',', '')
        x = np.array([float(val) for val in x.split()])
        return x


class AnimationFromCheckpoint(BaseAnimation):
    """
    Animation class getting data from saved ray checkpoint
    """

    def __init__(self, time_step: float = 10):
        super().__init__()
        self.bool_array_locations = False
        self.time_step = time_step

    def get_data(self, data_path):
        positions = np.zeros((1, 3))
        velocities = np.zeros((1, 3))
        sun_vectors = np.zeros((1, 3))
        bool_arrays = np.zeros((1, 99))
        for state in data_path["ObservationVector"][0]:
            pos = state.data['ObserveSensor_Sensor_Position'].value['direct_observation']
            positions = np.append(positions, [pos], axis=0)
            vel = state.data['ObserveSensor_Sensor_Velocity'].value['direct_observation']
            velocities = np.append(velocities, [vel], axis=0)
            try:
                th = state.data['ObserveSensor_Sensor_SunAngle'].value['direct_observation'][0]
            except KeyError:
                th = 0.
            sun = np.array([np.cos(th), np.sin(th), 0])
            sun_vectors = np.append(sun_vectors, [sun], axis=0)
            try:
                new_bool = state.data['ObserveSensor_Sensor_BoolArray'].value['direct_observation']
                self.bool_array_locations = True
            except KeyError:
                ins_pts = int(state.data['ObserveSensor_Sensor_InspectedPoints'].value['direct_observation'])
                new_bool = np.zeros(99)
                new_bool[0:ins_pts] = 1
                self.bool_array_locations = False
            bool_arrays = np.append(bool_arrays, [new_bool], axis=0)
        positions = np.delete(positions, 0, axis=0)
        velocities = np.delete(velocities, 0, axis=0)
        sun_vectors = np.delete(sun_vectors, 0, axis=0)
        bool_arrays = np.delete(bool_arrays, 0, axis=0)

        times = np.arange(len(positions)) * self.time_step

        control_vectors = np.zeros((1, 3))
        for control in data_path["ControlVector"][0]:
            Fx = control.data[list(control.data.keys())[0]].value
            Fy = control.data[list(control.data.keys())[1]].value
            Fz = control.data[list(control.data.keys())[2]].value
            control_vectors = np.append(control_vectors, [np.concatenate((Fx, Fy, Fz))], axis=0)
        control_vectors = np.delete(control_vectors, 0, axis=0)

        return times, positions, bool_arrays, sun_vectors, control_vectors, velocities
