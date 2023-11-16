"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.spatial.transform import Rotation

from safe_autonomy_sims.evaluation.animation.base_animation import BaseAnimationModule
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from safe_autonomy_sims.simulators.inspection_simulator import InspectionPoints


class Arrow3D(FancyArrowPatch):
    """Make a 3d arrow
    Credit: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, _ = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):  # pylint:disable=unused-argument
        """3d projection"""
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    art = ax.add_artist(arrow)
    return art


setattr(Axes3D, 'arrow3D', _arrow3D)


class SixDofAnimation(BaseAnimationModule):
    """
    Module for making animations from a checkpoint for the 6 DoF inspection environment

    Parameters
    ----------
    expr_config_path: str
        The absolute path to the experiment config used in training
    task_config_path: str
        The absolute path to the task_config used in training
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    radius: float
        Radius of the sphere of inspection points
    num_points: int
        Number of inspection points
    """

    def __init__(
        self,
        expr_config_path: str = 'safe-autonomy-sims/configs/weighted-six-dof-inspection-v2/experiment.yml',
        task_config_path: str = 'safe-autonomy-sims/configs/weighted-six-dof-inspection-v2/task.yml',
        platform_serializer_class: PlatformSerializer = SerializeCWH3D,
        radius: float = 10.,
        num_points: int = 100,
        **kwargs
    ):
        super().__init__(
            expr_config_path=expr_config_path,
            task_config_path=task_config_path,
            platform_serializer_class=platform_serializer_class,
            **kwargs
        )
        self.init_sensor_vec = np.array([1., 0., 0.])
        self.temp_elements = []
        self.radius = radius
        self.num_points = num_points

    def get_data_from_dataframe(self, dataframe):
        positions = []
        velocities = []
        inspected_points = []
        sun_angles = []
        quaternions = []
        angular_velocities = []
        reaction_wheel_velocities = []
        temperatures = []
        energy = []

        for state in dataframe["ObservationVector"][0]:
            positions.append(state.data['ObserveSensor_Sensor_Position'].value['direct_observation'])
            velocities.append(state.data['ObserveSensor_Sensor_Velocity'].value['direct_observation'])
            inspected_points.append(state.data['ObserveSensor_Sensor_InspectedPoints'].value['direct_observation'])
            sun_angles.append(state.data['ObserveSensor_Sensor_SunAngle'].value['direct_observation'])
            quaternions.append(state.data['ObserveSensor_Sensor_Quaternion'].value['direct_observation'])
            angular_velocities.append(state.data['ObserveSensor_Sensor_AngularVelocity'].value['direct_observation'])
            try:
                reaction_wheel_velocities.append(state.data['ObserveSensor_Sensor_ReactionWheelVelocity'].value['direct_observation'])
                temperatures.append(state.data['ObserveSensor_Sensor_Temperature'].value['direct_observation'])
                energy.append(state.data['ObserveSensor_Sensor_Energy'].value['direct_observation'])
            except KeyError:
                pass

        data = {}

        data['positions'] = np.array(positions)
        data['velocities'] = np.array(velocities)
        data['inspected_points'] = np.array(inspected_points)
        data['sun_angles'] = np.array(sun_angles)
        data['quaternions'] = np.array(quaternions)
        data['angular_velocities'] = np.array(angular_velocities)
        data['reaction_wheel_velocities'] = np.array(reaction_wheel_velocities)
        data['temperatures'] = np.array(temperatures)
        data['energy'] = np.array(energy)

        control_vectors = []
        for control in dataframe["ControlVector"][0]:
            step_control = []
            for v in control.data.values():
                step_control.append(v.value[0])
            control_vectors.append(step_control)
        data['control_vectors'] = np.array(control_vectors)

        return data

    def setup_plots(self, data):
        return_plots = {}

        max_position = np.max(abs(data['positions']))
        max_xy_pos = np.max(abs(data['positions'][:, 0:2]))

        fig1 = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 16, 'text.usetex': True})
        ax1 = fig1.add_subplot(121, projection='3d')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$z$')
        ax1.set_xlim([-max_position * 1.1, max_position * 1.1])
        ax1.set_ylim([-max_position * 1.1, max_position * 1.1])
        ax1.set_zlim([-max_position * 1.1, max_position * 1.1])
        ax1.set_aspect("equal")

        ax2 = fig1.add_subplot(122)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_xlim([-max_xy_pos * 1.1, max_xy_pos * 1.1])
        ax2.set_ylim([-max_xy_pos * 1.1, max_xy_pos * 1.1])
        ax2.set_aspect("equal")
        ax2.grid(True)

        points = InspectionPoints.points_on_sphere_fibonacci(InspectionPoints, self.num_points, self.radius)
        for p in points:
            ax1.plot(p[0], p[1], p[2], 'g*', markersize=5)
            ax2.plot(p[0], p[1], 'g*', markersize=5)

        return_plots['ax1'] = ax1
        return_plots['ax2'] = ax2
        return return_plots

    def make_plots_in_loop(self, axes, data, i):
        max_position = np.max(abs(data['positions']))
        max_xy_pos = np.max(abs(data['positions'][:, 0:2]))

        if self.filetype != 'png':
            self.temp_elements = []

            rot = Rotation.from_quat(data['quaternions'][i])
            sensor_vec = rot.apply(self.init_sensor_vec)
            arrow = axes['ax1'].arrow3D(
                data['positions'][i][0],
                data['positions'][i][1],
                data['positions'][i][2],
                data['positions'][i][0] + sensor_vec[0] * max_position / 2,
                data['positions'][i][1] + sensor_vec[1] * max_position / 2,
                data['positions'][i][2] + sensor_vec[2] * max_position / 2,
                fc='c',
                ec='k',
                mutation_scale=20
            )
            arrow_xy = axes['ax2'].annotate(
                '',
                xy=(data['positions'][i][0] + sensor_vec[0] * max_xy_pos / 3, data['positions'][i][1] + sensor_vec[1] * max_xy_pos / 3),
                xytext=(data['positions'][i][0], data['positions'][i][1]),
                arrowprops={
                    'fc': 'c', 'ec': 'k'
                }
            )

            self.temp_elements.append(arrow)
            self.temp_elements.append(arrow_xy)

            plot_position = []
            plot_position.extend(
                axes['ax1'].plot(data['positions'][i][0], data['positions'][i][1], data['positions'][i][2], 'b*', markersize=15)
            )
            plot_position.extend(axes['ax2'].plot(data['positions'][i][0], data['positions'][i][1], 'b*', markersize=15))
            sun_vec = np.array([np.cos(data['sun_angles'][i]), -np.sin(data['sun_angles'][i]), 0.])
            point_inSunDir = max_position * sun_vec
            point_inSunDir_xy = max_xy_pos * sun_vec
            sun_line, = axes['ax1'].plot(
                [point_inSunDir[0], 0], [point_inSunDir[1], 0], [point_inSunDir[2], 0], linewidth=5, color='y', zorder=0
            )
            sun_line2, = axes['ax2'].plot([point_inSunDir_xy[0], 0], [point_inSunDir_xy[1], 0], linewidth=5, color='y', zorder=0)

            self.temp_elements.append(sun_line)
            self.temp_elements.append(sun_line2)
            self.temp_elements.append(plot_position)

        else:
            if i != 0:
                axes['ax1'].plot(
                    [data['positions'][i - 1][0], data['positions'][i][0]], [data['positions'][i - 1][1], data['positions'][i][1]],
                    [data['positions'][i - 1][2], data['positions'][i][2]],
                    'b',
                    linewidth=3
                )
                axes['ax2'].plot(
                    [data['positions'][i - 1][0], data['positions'][i][0]], [data['positions'][i - 1][1], data['positions'][i][1]],
                    'b',
                    linewidth=3
                )

    def remove_temp_elements(self, axes):
        for tmp in self.temp_elements:
            if isinstance(tmp, list):
                for p in tmp:
                    p.remove()
            else:
                tmp.remove()
        self.temp_elements = []

    def print_metrics(self, data):
        pass


if __name__ == '__main__':
    animation = SixDofAnimation(checkpoint_path='example_checkpoint')
    animation.make_animation(filetype='gif')
