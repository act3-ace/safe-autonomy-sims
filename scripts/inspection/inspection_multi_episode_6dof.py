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
from safe_autonomy_sims.simulators.inspection_simulator import InspectionPoints


class BaseMultiEpisodeAnimation(abc.ABC):
    """Base animation module"""

    def __init__(self):
        self.bool_array_locations = False

    def make_single_episode_animation(
        self,
        data: dict,
        save_dir: str,
        mode: str = 'operator',
        last_step: bool = False,
        radius: float = 10.,
        num_points: int = 100,
        mass: float = 12,
        points_algorithm: str = "cmu",
        fig_prefix: str = "",
    ):
        """Function to make and save animation/plots

        Parameters
        ----------
        data: dict
            Dictionary giving the plot quantities
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
        points_algorithm: str
            points algorithm to use. Either "cmu" or "fibonacci"
        fig_prefix: str
            String to prepend to the 
        """

        tmp_dir = '/tmp/eval_results/animation_plots'

        times = data['times']
        positions = data['positions']
        bool_arrays = data['bool_arrays']
        sun_vectors = data['sun_vectors']
        thrust_control_vectors = data['thrust_control_vectors']
        velocities = data['velocities']

        if points_algorithm == "cmu":
            bool_arrays = np.delete(bool_arrays, -1, 1)  # TODO: fix bool array number of points

        # Make temp directory
        if not os.path.exists(tmp_dir) and not last_step:
            os.mkdir(tmp_dir)

        # Calculate values
        if points_algorithm == "cmu":
            points = InspectionPoints.points_on_sphere_cmu(InspectionPoints, num_points, radius)
        elif points_algorithm == "fibonacci":
            points = InspectionPoints.points_on_sphere_fibonacci(InspectionPoints, num_points, radius)
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
            for x in thrust_control_vectors:
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
        delta_v = np.sum(np.abs(thrust_control_vectors[0])) / mass * step_size
        print('Animation Progress:')
        for i in range(1, len(times)):
            # Get values
            time = times[i]
            last_pos = positions[i - 1]
            pos = positions[i]
            bool_array = bool_arrays[i]
            sun = sun_vectors[i]
            control = thrust_control_vectors[i]
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
                last_control = thrust_control_vectors[i - 1]
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
            plt.savefig(os.path.join(save_dir, fig_prefix + 'episode_plot.png'))
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


# class AnimationFromCSV(BaseAnimation):
#     """
#     Animation class getting data from csv.
#     Use save_data_flag in illumination parameters.
#     """

#     def __init__(self):
#         super().__init__()
#         self.bool_array_locations = True

#     def get_data(self, data_path):
#         self.bool_array_locations = True
#         # Get data from csv
#         data = pd.read_csv(data_path, header=None, converters={1: self.s2a, 2: self.s2a, 3: self.s2a, 4: self.s2a, 5: self.s2a})
#         times = data[0].values
#         positions = np.array(list(data[1].values))
#         bool_arrays = np.array(list(data[2].values))
#         sun_vectors = np.array(list(data[3].values))
#         control_vectors = np.array(list(data[4].values))
#         velocities = np.array(list(data[5].values))
#         return times, positions, bool_arrays, sun_vectors, control_vectors, velocities

#     def s2a(self, x):
#         """Helper function for reading csv"""
#         x = x.replace('[', '')
#         x = x.replace(']', '')
#         x = x.replace(',', '')
#         x = np.array([float(val) for val in x.split()])
#         return x


class MultiEpisodeAnimationFromCheckpoint(BaseMultiEpisodeAnimation):
    """
    Animation class getting data from saved ray checkpoint
    """

    def __init__(self, time_step: float = 10):
        super().__init__()
        self.bool_array_locations = False
        self.time_step = time_step

    def get_data(self, data_path: pd.DataFrame):
        self.n_episodes = len(data_path.index)
        out_data = {
            'positions': [],
            'velocities': [],
            'sun_vectors': [],
            'bool_arrays': [],
            'times': [],
            'thrust_controls': [],
            'moment_controls': [],
        }

        for i in range(self.n_episodes):
            this_series = data_path.loc[i]
            control_vector_list = this_series['ControlVector']
            obs_vector_list = this_series['ObservationVector']

            assert len(control_vector_list) == len(obs_vector_list)
            steps_this_episode = len(obs_vector_list)

            episode_positions = np.zeros((steps_this_episode, 3))
            episode_velocities = np.zeros((steps_this_episode, 3))
            episode_sun_vectors = np.zeros((steps_this_episode, 3))
            episode_bool_arrays = np.zeros((steps_this_episode, 100))
            episode_times = np.arange(steps_this_episode) * self.time_step
            thrust_control_vectors = np.zeros((steps_this_episode, 3))
            moment_control_vectors = np.zeros((steps_this_episode, 3))
            
            for j in range(steps_this_episode):
                obs_dict = obs_vector_list[j].data
                control_dict = control_vector_list[j].data

                pos = obs_dict['ObserveSensor_Sensor_Position'].value['direct_observation']
                episode_positions[j] = pos.copy()
                vel = obs_dict['ObserveSensor_Sensor_Velocity'].value['direct_observation']
                episode_velocities[j] = vel.copy()

                try:
                    sun_angle = obs_dict['ObserveSensor_Sensor_SunAngle'].value['direct_observation']
                except KeyError:
                    sun_angle = np.array([1.0, 0.0, 0.0])
                if len(sun_angle) == 1:
                    sun = np.array([np.cos(sun_angle), -np.sin(sun_angle), 0])
                elif len(sun_angle) == 3:
                    sun = sun_angle
                episode_sun_vectors[j] = sun.copy()

                try:
                    new_bool = obs_dict['ObserveSensor_Sensor_BoolArray'].value['direct_observation']
                    self.bool_array_locations = True
                except KeyError:
                    ins_pts = obs_dict['ObserveSensor_Sensor_InspectedPoints'].value['direct_observation']
                    new_bool = np.zeros(100)
                    new_bool[0:ins_pts] = 1
                    self.bool_array_locations = False
                episode_bool_arrays[j] = new_bool.copy()

                cv_keys = list(control_dict.keys())
                fx_key = [k for k in cv_keys if 'x_thrust' in k.lower()][0]
                fy_key = [k for k in cv_keys if 'y_thrust' in k.lower()][0]
                fz_key = [k for k in cv_keys if 'z_thrust' in k.lower()][0]
                mx_key = [k for k in cv_keys if 'x_moment' in k.lower()][0]
                my_key = [k for k in cv_keys if 'y_moment' in k.lower()][0]
                mz_key = [k for k in cv_keys if 'z_moment' in k.lower()][0]
                thrust_cv = np.array([
                    control_dict[fx_key].value,
                    control_dict[fy_key].value,
                    control_dict[fz_key].value
                ])
                thrust_control_vectors[j] = thrust_cv[:, 0].copy()
                moment_cv = np.array([
                    control_dict[mx_key].value,
                    control_dict[my_key].value,
                    control_dict[mz_key].value
                ])
                moment_control_vectors[j] = moment_cv[:, 0].copy()
            
            out_data['positions'].append(episode_positions.copy())
            out_data['velocities'].append(episode_velocities.copy())
            out_data['sun_vectors'].append(episode_sun_vectors.copy())
            out_data['bool_arrays'].append(episode_bool_arrays.copy())
            out_data['times'].append(episode_times.copy())
            out_data['thrust_controls'].append(thrust_control_vectors.copy())
            out_data['moment_controls'].append(moment_control_vectors.copy())
        return out_data
    
    def calc_delta_v(self, data: dict, mass: float = 12):
        delta_v_list = []
        for i in range(self.n_episodes):
            thrust_controls = data['thrust_controls'][i]
            delta_v = np.cumsum(np.sum(np.abs(thrust_controls), axis=1)) / mass * self.time_step
            delta_v_list.append(delta_v)
        return delta_v_list
            
    def make_overall_plots(
            self, 
            data: dict,
            save_dir: str, 
            bins: int = 30,
            mode: str = 'operator',
            last_step: bool = False,
            radius: float = 10.,
            num_points: int = 100,
            mass: float = 12,
            points_algorithm: str = "cmu"
            ):
        delta_v_list = data['delta_vs']
        final_delta_vs = np.array([dv[-1] for dv in delta_v_list])
        # delta_vs_flat = np.concatenate(delta_v_list)
        
        plt.figure()
        plt.xlabel('Total Delta V (m/s)')
        plt.ylabel('Episode Count')
        plt.hist(final_delta_vs, bins=bins)
        plt.savefig(os.path.join(save_dir, 'delta_v_hist.png'))

        bool_arrays_list = data['bool_arrays']
        wins = np.array([np.sum(b[-1]) >= 95 for b in bool_arrays_list])

        # Print statistics
        win_rate = np.sum(wins) / self.n_episodes
        print("Won {} of {} episodes for a rate of {}".format(np.sum(wins), self.n_episodes, f'{win_rate:.2f}'))
        print('Mean Delta V = {} m/s'.format(f'{np.mean(final_delta_vs)}'))
        print('Standard Deviation of Delta V = {} m/s'.format(f'{np.std(final_delta_vs)}'))

        # Get best and worst successful episodes by Delta V metric
        best_win_idx = np.argmin(final_delta_vs[wins])
        worst_win_idx = np.argmax(final_delta_vs[wins])
        mid_win_idx = np.argsort(final_delta_vs[wins])[np.sum(wins)//2]

        best_win_idx = np.arange(self.n_episodes)[wins][best_win_idx]
        worst_win_idx = np.arange(self.n_episodes)[wins][worst_win_idx]
        mid_win_idx = np.arange(self.n_episodes)[wins][mid_win_idx]

        best_win_data = {
            'positions': data['positions'][best_win_idx],
            'velocities': data['velocities'][best_win_idx],
            'times': data['times'][best_win_idx],
            'bool_arrays': data['bool_arrays'][best_win_idx],
            'sun_vectors': data['sun_vectors'][best_win_idx],
            'thrust_control_vectors': data['thrust_controls'][best_win_idx]
        }
        print('Rendering Win Episode with lowest Delta V')
        self.make_single_episode_animation(best_win_data,
                                           save_dir=save_dir,
                                           mode=mode,
                                           last_step=last_step,
                                           radius=radius,
                                           num_points=num_points,
                                           mass=mass,
                                           fig_prefix='best_win_')
        
        worst_win_data = {
            'positions': data['positions'][worst_win_idx],
            'velocities': data['velocities'][worst_win_idx],
            'times': data['times'][worst_win_idx],
            'bool_arrays': data['bool_arrays'][worst_win_idx],
            'sun_vectors': data['sun_vectors'][worst_win_idx],
            'thrust_control_vectors': data['thrust_controls'][worst_win_idx]
        }
        print('Rendering Win Episode with highest Delta V')
        self.make_single_episode_animation(worst_win_data,
                                           save_dir=save_dir,
                                           mode=mode,
                                           last_step=last_step,
                                           radius=radius,
                                           num_points=num_points,
                                           mass=mass,
                                           fig_prefix='worst_win_')
        print('Rendering Win Episode with median Delta V')
        mid_win_data = {
            'positions': data['positions'][mid_win_idx],
            'velocities': data['velocities'][mid_win_idx],
            'times': data['times'][mid_win_idx],
            'bool_arrays': data['bool_arrays'][mid_win_idx],
            'sun_vectors': data['sun_vectors'][mid_win_idx],
            'thrust_control_vectors': data['thrust_controls'][mid_win_idx]
        }
        self.make_single_episode_animation(mid_win_data,
                                           save_dir=save_dir,
                                           mode=mode,
                                           last_step=last_step,
                                           radius=radius,
                                           num_points=num_points,
                                           mass=mass,
                                           fig_prefix='mid_win_')

       