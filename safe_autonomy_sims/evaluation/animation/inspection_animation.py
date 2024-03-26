"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import yaml
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer

from safe_autonomy_sims.evaluation.animation.base_animation import BaseAnimationModule
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from safe_autonomy_sims.simulators.inspection_simulator import InspectionPoints


class InspectionAnimation(BaseAnimationModule):
    """
    Module for making animations/plots from a checkpoint for the inspection environment

    Parameters
    ----------
    expr_config_path: str
        The absolute path to the experiment config used in training
    task_config_path: str
        The absolute path to the task_config used in training
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    parameters_path: str
        Path to a yml file with parameters
    radius: float
        Radius of the sphere of inspection points
    fft_time: float
        Total time for the Free Flight Trajectory (FFT) calculations in seconds, by default 2 days
    points_algorithm: str
        Points algorithm to use, either "cmu" or "fibonacci"
    """

    def __init__(
        self,
        expr_config_path: str = 'safe-autonomy-sims/configs/translational-inspection/experiment.yml',
        task_config_path: str = 'safe-autonomy-sims/configs/translational-inspection/task.yml',
        platform_serializer_class: PlatformSerializer = SerializeCWH3D,
        parameters_path: str = 'safe-autonomy-sims/configs/translational-inspection/parameters.yml',
        radius: float = 10.,
        fft_time: float = 172800.,
        points_algorithm: str = "cmu",
        **kwargs,
    ):

        with open(parameters_path, 'r', encoding='utf-8') as file:
            parameters = yaml.safe_load(file)

        if 'step_size' in parameters:
            self.step_size = parameters['step_size']
        else:
            self.step_size = 10.

        if 'num_points' in parameters:
            self.num_points = parameters['num_points']
        else:
            self.num_points = 100

        if 'mass' in parameters:
            self.mass = parameters['mass']
        else:
            self.mass = 12

        if 'mean_motion' in parameters:
            self.n = parameters['mean_motion']
        else:
            self.n = 0.001027

        self.radius = radius
        self.fft_time = fft_time
        self.points_algorithm = points_algorithm

        self.lw = 3  # linewidth
        self.bool_array_locations = False
        self.delta_v = 0.0
        self.ffts = np.array([])
        self.temp_elements: list[typing.Any] = []

        # Calculate values
        if self.points_algorithm == "cmu":
            self.points = InspectionPoints.points_on_sphere_cmu(self.num_points, radius)
        elif self.points_algorithm == "fibonacci":
            self.points = InspectionPoints.points_on_sphere_fibonacci(self.num_points, radius)
        else:
            raise ValueError('Points algorithm must be either "cmu" or "fibonacci"')

        super().__init__(
            expr_config_path=expr_config_path,
            task_config_path=task_config_path,
            platform_serializer_class=platform_serializer_class,
            start_index=1,
            **kwargs
        )

    def get_closest_fft(self, pos, vel):
        """Get closest relative position to the origin along a Free Flight Trajectory (FFT)

        Parameters
        ----------
        pos: list
            position vector
        vel: list
            velocity vector
        """
        time = np.arange(0, self.fft_time, self.step_size)
        positions = []
        for t in time:
            x = (4 - 3 * np.cos(self.n * t)) * pos[0] + np.sin(self.n * t
                                                               ) * vel[0] / self.n + 2 / self.n * (1 - np.cos(self.n * t)) * vel[1]
            y = 6 * (np.sin(self.n * t) - self.n * t) * pos[0] + pos[
                1] - 2 / self.n * (1 - np.cos(self.n * t)) * vel[0] + (4 * np.sin(self.n * t) - 3 * self.n * t) * vel[1] / self.n
            z = pos[2] * np.cos(self.n * t) + vel[2] / self.n * np.sin(self.n * t)
            positions.append(np.linalg.norm([x, y, z]))
        return min(positions)

    def get_data_from_dataframe(self, dataframe):
        positions = []
        velocities = []
        sun_vectors = []
        bool_arrays = []
        scores = []
        for state in dataframe["ObservationVector"][0]:
            positions.append(state.data['Obs_Sensor_Position'].value['direct_observation'].value)
            velocities.append(state.data['Obs_Sensor_Velocity'].value['direct_observation'].value)
            try:
                score = state.data['Obs_Sensor_InspectedPointsScore'].value['direct_observation']
            except KeyError:
                score = np.NAN
            scores.append(score)
            try:
                th = state.data['Obs_Sensor_SunAngle'].value['direct_observation'].value[0]
            except KeyError:
                th = 0.
            sun = np.array([np.cos(th), -np.sin(th), 0])
            sun_vectors.append(sun)
            try:
                new_bool = state.data['Obs_Sensor_BoolArray'].value['direct_observation']
                self.bool_array_locations = True
            except KeyError:
                ins_pts = int(state.data['Obs_Sensor_InspectedPoints'].value['direct_observation'].value)
                new_bool = np.zeros(self.num_points)
                new_bool[0:ins_pts] = 1
            bool_arrays.append(new_bool)

        times = np.arange(len(positions)) * self.step_size

        control_vectors = []
        for control in dataframe["ControlVector"][0]:
            Fx = control.data[list(control.data.keys())[0]].value
            Fy = control.data[list(control.data.keys())[1]].value
            Fz = control.data[list(control.data.keys())[2]].value
            control_vectors.append([Fx, Fy, Fz])

        data = {}

        data['times'] = times
        data['positions'] = np.array(positions)
        data['bool_arrays'] = np.array(bool_arrays)
        data['sun_vectors'] = np.array(sun_vectors)
        data['control_vectors'] = np.array(control_vectors)
        data['velocities'] = np.array(velocities)
        data['scores'] = np.array(scores)

        if self.points_algorithm == "cmu":
            data['bool_arrays'] = np.delete(data['bool_arrays'], -1, 1)  # TODO: fix bool array number of points

        return data

    def setup_ric(self, data, max_pos, return_plots):
        """Set up RIC plots.

        Parameters
        ----------
        data : dict
            data to plot
        max_pos : float
            max position value
        return_plots : dict
            plots to modify

        Returns
        -------
        dict
            modified plots in RIC frame
        """
        ax_ri = plt.subplot2grid((5, 4), (0, 0), colspan=2, rowspan=3)
        ax_ri.grid(True)
        ax_ri.set_xlabel('In-Track [m]')
        ax_ri.set_ylabel('Radial [m]')
        ax_ri.set_xlim([-max_pos, max_pos])
        ax_ri.set_ylim([-max_pos, max_pos])
        ax_ri.set_aspect("equal")
        ax_ri.invert_xaxis()
        if self.filetype == 'png':
            ax_ri.plot(max_pos * 2, max_pos * 2, color=plt.cm.cool(0), linewidth=self.lw, label="Deputy Position")
        else:
            ax_ri.plot(max_pos * 2, max_pos * 2, '*', color=plt.cm.cool(0), linewidth=self.lw, label="Deputy Position")
            ax_ri.plot(max_pos * 2, max_pos * 2, linewidth=5, color='y', label="Sun Vector")
        ax_ri.plot(data['positions'][0][1], data['positions'][0][0], 'b*', markersize=10, label="Initial Position")
        if self.bool_array_locations:
            ax_ri.plot(max_pos * 2, max_pos * 2, 'g*', label="Inspected Point")
            ax_ri.plot(max_pos * 2, max_pos * 2, 'r*', label="Uninspected Point")
        else:
            ax_ri.plot(max_pos * 2, max_pos * 2, 'r*', label="Inspection Point")
        ax_ri.legend()
        return_plots['ax_ri'] = ax_ri

        ax_rc = plt.subplot2grid((5, 4), (3, 0), rowspan=2)
        ax_rc.grid(True)
        ax_rc.set_xlabel('Cross-Track [m]')
        ax_rc.set_ylabel('Radial [m]')
        ax_rc.set_xlim([-max_pos, max_pos])
        ax_rc.set_ylim([-max_pos, max_pos])
        ax_rc.set_aspect("equal")
        ax_rc.invert_xaxis()
        ax_rc.plot(data['positions'][0][2], data['positions'][0][0], 'b*', markersize=10)
        return_plots['ax_rc'] = ax_rc

        ax_ic = plt.subplot2grid((5, 4), (3, 1), rowspan=2)
        ax_ic.grid(True)
        ax_ic.set_xlabel('In-Track [m]')
        ax_ic.set_ylabel('Cross-Track [m]')
        ax_ic.set_xlim([-max_pos, max_pos])
        ax_ic.set_ylim([-max_pos, max_pos])
        ax_ic.set_aspect("equal")
        ax_ic.invert_xaxis()
        ax_ic.invert_yaxis()
        ax_ic.plot(data['positions'][0][1], data['positions'][0][2], 'b*', markersize=10)
        return_plots['ax_ic'] = ax_ic

        return return_plots

    def setup_rp_rv(self, data, return_plots):
        """Set up relative position and velocity plots

        Parameters
        ----------
        data : dict
            data to plot
        return_plots : dict
            plots to modify

        Returns
        -------
        dict
            modified plots with relative position and velocity
        """
        max_rp = []
        for x in data['positions']:
            max_rp.append(np.linalg.norm(x))
        max_rp = max(max_rp)
        self.ffts = []
        for pos, vel in zip(data['positions'], data['velocities']):
            self.ffts.append(self.get_closest_fft(pos, vel))
        max_rp = max(max_rp, max(self.ffts)) * 1.1
        ax_rp = plt.subplot2grid((5, 4), (1, 2), colspan=3)
        ax_rp.set_xlim([0, data['times'][-1]])
        ax_rp.set_ylim([0, max_rp])
        ax_rp.set_xlabel('Time [sec]')
        ax_rp.set_ylabel('Relative Distance [m]')
        ax_rp.grid(True)
        ax_rp.plot(-10, -10, 'r', label=r'Closest FFT', linewidth=self.lw)
        ax_rp.plot(-10, -10, 'b', label=r'Trajectory', linewidth=self.lw)
        ax_rp.legend()
        return_plots['ax_rp'] = ax_rp

        max_rv = []
        for x in data['velocities']:
            max_rv.append(np.linalg.norm(x))
        max_rv = max(max_rv) * 1.1
        ax_rv = plt.subplot2grid((5, 4), (2, 2), colspan=3)
        ax_rv.set_xlim([0, data['times'][-1]])
        ax_rv.set_ylim([0, max_rv])
        ax_rv.set_xlabel('Time [sec]')
        ax_rv.set_ylabel('Relative Speed [m/s]')
        ax_rv.grid(True)
        return_plots['ax_rv'] = ax_rv

        return return_plots

    def setup_plots(self, data):
        return_plots = {}

        max_positions = []
        for x in data['positions']:
            max_positions.append(max(abs(x)))
        max_pos = float(max(max_positions) * 1.1)

        self.delta_v = np.sum(np.abs(data['control_vectors'][0])) / self.mass * self.step_size

        # Set plot values
        plt.figure(figsize=(20, 15.04))
        plt.rcParams.update({'font.size': 20, 'text.usetex': True})

        # Setup plots
        return_plots = self.setup_ric(data, max_pos, return_plots)
        return_plots = self.setup_rp_rv(data, return_plots)

        max_dv = 0
        for x in data['control_vectors']:
            max_dv += np.sum(np.abs(x)) / self.mass * self.step_size
        max_dv = max_dv * 1.1
        ax_dv = plt.subplot2grid((5, 4), (4, 2), colspan=3)
        ax_dv.set_xlim([0, data['times'][-1]])
        ax_dv.set_ylim([0, max_dv])
        ax_dv.set_xlabel('Time [sec]')
        ax_dv.set_ylabel(r'Cumulative $\Delta$V [m/s]')
        ax_dv.grid(True)
        return_plots['ax_dv'] = ax_dv

        ax_ip = plt.subplot2grid((5, 4), (0, 2), colspan=3)
        ax_ip.set_xlim([0, data['times'][-1]])
        ax_ip.set_ylim([0, 100])
        ax_ip.set_xlabel('Time [sec]')
        ax_ip.set_ylabel('Points Inspected')
        ax_ip.grid(True)
        ax_ip.plot(-10, -10, 'b', linewidth=self.lw, label='Percentage')
        if not any(np.isnan(data['scores'])):
            ax_ip.plot(-10, -10, 'r', linewidth=self.lw, label='Score')
        ax_ip.legend()
        return_plots['ax_ip'] = ax_ip

        ax_u = plt.subplot2grid((5, 4), (3, 2), colspan=3)
        ax_u.set_xlim([0, data['times'][-1]])
        y_lim = 1 / self.mass * self.step_size * 1.1
        ax_u.set_ylim([-y_lim, y_lim])
        ax_u.set_xlabel('Time [sec]')
        ax_u.set_ylabel('Thrust [m/s]')
        ax_u.grid(True)
        ax_u.plot(-10, -10, 'b', label=r'$R$', linewidth=self.lw)
        ax_u.plot(-10, -10, 'r', label=r'$I$', linewidth=self.lw)
        ax_u.plot(-10, -10, 'g', label=r'$C$', linewidth=self.lw)
        ax_u.legend()
        return_plots['ax_u'] = ax_u

        return return_plots

    def make_plots_in_loop(self, axes, data, i):
        max_rv = []
        for x in data['velocities']:
            max_rv.append(max(abs(x)))
        max_rv = max(max_rv) * 1.1

        max_positions = []
        for x in data['positions']:
            max_positions.append(max(abs(x)))
        max_pos = float(max(max_positions) * 1.1)

        vel_color = np.linalg.norm(data['velocities'][i]) / (max_rv / 1.1)

        axes['ax_ri'].plot(
            [data['positions'][i - 1][1], data['positions'][i][1]], [data['positions'][i - 1][0], data['positions'][i][0]],
            color=plt.cm.cool(vel_color),
            linewidth=self.lw
        )
        axes['ax_rc'].plot(
            [data['positions'][i - 1][2], data['positions'][i][2]], [data['positions'][i - 1][0], data['positions'][i][0]],
            color=plt.cm.cool(vel_color),
            linewidth=self.lw
        )
        axes['ax_ic'].plot(
            [data['positions'][i - 1][1], data['positions'][i][1]], [data['positions'][i - 1][2], data['positions'][i][2]],
            color=plt.cm.cool(vel_color),
            linewidth=self.lw
        )

        if self.filetype != 'png':
            self.animate_sun_and_points(axes, data, i, max_pos)

        if self.filetype != 'png' or i == len(data['times']) - 1:
            c = 'r'
            # Points on sphere plot
            plot_points = []
            for j, p in enumerate(data['bool_arrays'][i]):
                if self.bool_array_locations:
                    if p:
                        c = 'g'
                    else:
                        c = 'r'

                plot_points.extend(axes['ax_ri'].plot(self.points[j][1], self.points[j][0], '*', color=c))
                plot_points.extend(axes['ax_rc'].plot(self.points[j][2], self.points[j][0], '*', color=c))
                plot_points.extend(axes['ax_ic'].plot(self.points[j][1], self.points[j][2], '*', color=c))
            self.temp_elements.append(plot_points)

        last_delta_v = self.delta_v
        self.delta_v += np.sum(np.abs(data['control_vectors'][i])) / self.mass * self.step_size
        axes['ax_dv'].plot([data['times'][i - 1], data['times'][i]], [last_delta_v, self.delta_v], 'b', linewidth=self.lw)

        last_rel_pos = np.linalg.norm(data['positions'][i - 1])
        rel_pos = np.linalg.norm(data['positions'][i])
        axes['ax_rp'].plot([data['times'][i - 1], data['times'][i]], [self.ffts[i - 1], self.ffts[i]], 'r', linewidth=self.lw)
        axes['ax_rp'].plot([data['times'][i - 1], data['times'][i]], [last_rel_pos, rel_pos], 'b', linewidth=self.lw)

        last_rel_vel = np.linalg.norm(data['velocities'][i - 1])
        rel_vel = np.linalg.norm(data['velocities'][i])
        axes['ax_rv'].plot(
            [data['times'][i - 1], data['times'][i]], [last_rel_vel, rel_vel], color=plt.cm.cool(vel_color), linewidth=self.lw
        )

        last_control_dv = data['control_vectors'][i - 1] / self.mass * self.step_size
        control_dv = data['control_vectors'][i] / self.mass * self.step_size
        axes['ax_u'].plot([data['times'][i - 1], data['times'][i]], [last_control_dv[0], control_dv[0]], 'b', linewidth=self.lw)
        axes['ax_u'].plot([data['times'][i - 1], data['times'][i]], [last_control_dv[1], control_dv[1]], 'r', linewidth=self.lw)
        axes['ax_u'].plot([data['times'][i - 1], data['times'][i]], [last_control_dv[2], control_dv[2]], 'g', linewidth=self.lw)

        last_pts = sum(data['bool_arrays'][i - 1]) / self.num_points * 100
        pts = sum(data['bool_arrays'][i]) / self.num_points * 100
        axes['ax_ip'].plot([data['times'][i - 1], data['times'][i]], [last_pts, pts], 'b', linewidth=self.lw)
        if not any(np.isnan(data['scores'])):
            axes['ax_ip'].plot(
                [data['times'][i - 1], data['times'][i]], [data['scores'][i - 1] * 100, data['scores'][i] * 100], 'r', linewidth=self.lw
            )

    def animate_sun_and_points(self, axes, data, i, max_pos):
        """Animate the sun vector and points

        Parameters
        ----------
        axes : pyplot.Axes
            Plot axes
        data : dict
            data to plot
        i : int
            data index
        max_pos : float
            maximum position value
        """
        # Sun vector plot
        sun_vector = data['sun_vectors'][i] / np.linalg.norm(data['sun_vectors'][i])
        point_inSunDir = max_pos / 1.1 * sun_vector
        sun_line1, = axes['ax_ri'].plot([point_inSunDir[1], 0], [point_inSunDir[0], 0], linewidth=5, color='y')
        sun_line2, = axes['ax_rc'].plot([point_inSunDir[2], 0], [point_inSunDir[0], 0], linewidth=5, color='y')
        sun_line3, = axes['ax_ic'].plot([point_inSunDir[1], 0], [point_inSunDir[2], 0], linewidth=5, color='y')
        self.temp_elements.append(sun_line1)
        self.temp_elements.append(sun_line2)
        self.temp_elements.append(sun_line3)

        # Time text boxes
        angle = np.arctan2(sun_vector[1], sun_vector[0])
        satellite_time = (-angle / (2 * np.pi) + 0.5) * 24
        mins, hrs = math.modf(satellite_time)
        mins *= 60
        txt = axes['ax_ri'].text(0.75, -0.1, f"LST: {hrs:02.0f}:{mins:02.0f}", transform=axes['ax_ri'].transAxes)
        mins, hrs = math.modf(data['times'][i] / 3600)
        mins *= 60
        txt2 = axes['ax_ri'].text(0.0, -0.1, f"Time Elapsed: {hrs:02.0f}:{mins:02.0f}", transform=axes['ax_ri'].transAxes)

        self.temp_elements.append(txt)
        self.temp_elements.append(txt2)

    def remove_temp_elements(self, axes):
        for tmp in self.temp_elements:
            if isinstance(tmp, list):
                for p in tmp:
                    p.remove()
            else:
                tmp.remove()

        self.temp_elements = []

    def print_metrics(self, data):
        print('Metrics:')
        print(f"Inspected Points: {np.count_nonzero(data['bool_arrays'][-1]):0.0f}/{len(data['bool_arrays'][-1]):0.0f}")
        print(f"Time: {data['times'][-1]:0.1f} sec")
        print(f"Delta V: {self.delta_v:0.2f} m/s")
