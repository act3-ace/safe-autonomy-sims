"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains the InspectionSimulator for interacting with the CWH inspection task simulator
"""
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import ValueWithUnits
from corl.simulators.base_simulator_state import BaseSimulatorState
from pydantic import BaseModel, validator
from safe_autonomy_dynamics.cwh import CWHSpacecraft
from sklearn.cluster import KMeans

import saferl.simulators.illumination_functions as illum
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator, SafeRLSimulatorResetValidator, SafeRLSimulatorValidator


class IlluminationValidator(BaseModel):
    """
    mean_motion: float
        A float representing the mean motion of the spacecraft in Low Earth Orbit (LEO) in [RADIANS/SECOND].
    avg_rad_Earth2Sun: float
        A float representing the average distance between the Earth and the Sun in [METERS].
    sun_angle: float
        A float representing the initial relative angle of sun wrt chief in [RADIANS] assuming sun travels in xy plane.
    light_properties: dict
        A dict containing the ambient, specular and diffuse light properties.
    chief_properties: dict
        A dict containing the ambient, specular, diffuse, shininess and reflective properties of the chief spacecraft.
    resolution: list
        A list containing the resolution of the sensor, represented by x and y pixel density respectively.
    focal_length: float
        A float representing the focal length of the sensor in [METERS]. The virtual image is created a
        distance of focal length away from the sensor origin.
    bin_ray_flag: bool
        A bool flag for utilization of "binary ray" vs. illumination features.
    """

    mean_motion: float = 0.001027
    avg_rad_Earth2Sun: float = 150000000000
    sun_angle: typing.Union[ValueWithUnits, float] = 0.0
    light_properties: dict
    chief_properties: dict
    resolution: list
    focal_length: float
    bin_ray_flag: bool
    render_flag_3d: bool = False
    render_flag_subplots: bool = False
    save_data_flag: bool = False


class InspectionSimulatorState(BaseSimulatorState):
    """
    The basemodel for the state of the InspectionSimulator.

    points: dict
        The dictionary containing the points the agent needs to inspect.
        Keys: (x,y,z) tuple. Values: True if inspected, False otherwise.
    """
    points: typing.Dict


def points_on_sphere_fibonacci(num_points: int, radius: float) -> list:
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
        r = radius * math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * r
        z = math.sin(theta) * r

        points.append((x, y, z))

    return points


def points_on_sphere_cmu(num_points: int, radius: float) -> list:
    """
    Generate a set of equidistant points on a sphere using the algorithm
    in https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf. Number
    of points may not be exact.

    Mostly the same as CMU algorithm, most important tweak is that the constant "a" should not depend on r
    (Paper assumed r = 1)

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

    a = 4.0 * math.pi * (1 / num_points)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(0, m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0, m_phi):
            phi = 2.0 * math.pi * n / m_phi
            point = (radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi), radius * math.cos(theta))
            points.append(point)

    return points


class InspectionSimulatorValidator(SafeRLSimulatorValidator):
    """
    A validator for the InspectionSimulator config.

    step_size: float
        A float representing how many simulated seconds pass each time the simulator updates.
    """
    step_size: float
    num_points: int
    radius: float
    points_algorithm: str = "cmu"
    illumination_params: typing.Union[IlluminationValidator, None] = None

    @validator("points_algorithm")
    def valid_algorithm(cls, v):
        """
        Check if provided algorithm is a valid choice.
        """
        valid_algs = ["cmu", "fibonacci"]
        if v not in valid_algs:
            raise ValueError(f"field points_algorithm must be one of {valid_algs}")
        return v


class InspectionSimulatorResetValidator(SafeRLSimulatorResetValidator):
    """
    A validator for the InspectionSimulator reset.
    """
    sun_angle: typing.Union[ValueWithUnits, float] = 0.0


class InspectionSimulator(SafeRLSimulator):
    """
    Simulator for CWH Inspection Task. Interfaces CWH platforms with underlying CWH entities in inspection simulation.
    """

    @property
    def get_simulator_validator(self):
        return InspectionSimulatorValidator

    @property
    def get_reset_validator(self) -> typing.Type[InspectionSimulatorResetValidator]:
        return InspectionSimulatorResetValidator

    def _construct_platform_map(self) -> dict:
        return {
            'default': (CWHSpacecraft, CWHPlatform),
            'cwh': (CWHSpacecraft, CWHPlatform),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = self._add_points()
        self.last_points_inspected = 0
        self.last_cluster = None
        self.illum_flag = False
        if self.config.illumination_params is not None:
            self.illum_flag = True
            if self.config.illumination_params.render_flag_3d:
                self.fig = plt.figure(1)
            if self.config.illumination_params.render_flag_subplots:
                self.fig = plt.figure(1)
                self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
                self.ax_xy = self.fig.add_subplot(2, 2, 2)
                self.ax_xz = self.fig.add_subplot(2, 2, 3)
                self.ax_yz = self.fig.add_subplot(2, 2, 4)
            if self.config.illumination_params.save_data_flag:
                self.path_to_data = "C:/Users/david/Documents/Documents/Texas_AM/AFRL/safe-autonomy-sims/matlab_animations/temp.csv"

    def reset(self, config):
        super().reset(config)
        if self.config.illumination_params is not None:
            self._update_initial_sun_angle(config)
        self.points = self._add_points()
        self.last_points_inspected = 0
        self.last_cluster = None
        self._state = InspectionSimulatorState(sim_platforms=self._state.sim_platforms, points=self.points, sim_time=self.clock)
        return self._state

    def _update_initial_sun_angle(self, config):
        assert self.config.illumination_params is not None, "Cannot assign a sun angle without defined illumination parameters"
        sun_angle = config['sun_angle']
        if isinstance(sun_angle, ValueWithUnits):
            sun_angle = sun_angle.value
        self.config.illumination_params.sun_angle = sun_angle

    def _step_update_sim_statuses(self, step_size: float):
        # update points
        for platform in self._state.sim_platforms.values():
            platform_id = platform.name
            entity = self.sim_entities[platform_id]
            self._update_points(entity.position)

            # update the observation space with number of inspected points
            platform.num_inspected_points = illum.num_inspected_points(self._state.points)
            if self.illum_flag:
                platform.sun_angle = np.array(
                    [
                        illum.get_sun_angle(
                            self.clock, self.config.illumination_params.mean_motion, self.config.illumination_params.sun_angle
                        )
                    ]
                )
            if platform.num_inspected_points[0] != self.last_points_inspected:
                platform.cluster_location = self._kmeans_find_nearest(entity.position)
            self.last_points_inspected = platform.num_inspected_points[0]

            if self.illum_flag:
                # render scene every m simulation seconds
                if (
                    self.config.illumination_params.render_flag_3d or self.config.illumination_params.render_flag_subplots
                    or self.config.illumination_params.save_data_flag
                ):
                    current_time = self.clock
                    sun_position = illum.get_sun_position(
                        current_time,
                        self.config.illumination_params.mean_motion,
                        self.config.illumination_params.sun_angle,
                        self.config.illumination_params.avg_rad_Earth2Sun
                    )
                    m = 10
                    if (current_time % (m)) == 0:
                        if self.config.illumination_params.render_flag_3d:
                            illum.render_3d(self.fig, entity.position, sun_position, self.config.radius, current_time, m)
                        if self.config.illumination_params.render_flag_subplots:
                            axes = [self.ax_3d, self.ax_xy, self.ax_xz, self.ax_yz]
                            illum.render_subplots(self.fig, axes, entity.position, sun_position, self.config.radius, current_time, m)
                    if self.config.illumination_params.save_data_flag:
                        action = np.array(platform.get_applied_action(), dtype=np.float32)
                        illum.save_data(
                            self._state.points, current_time, entity.position, sun_position, action, entity.velocity, self.path_to_data
                        )

        # return same as parent
        return self._state

    def _add_points(self) -> dict:
        """
        Generate a map of inspection point coordinates to inspected state.

        Returns
        -------
        points_dict
            dict of points_dict[cartesian_point] = initial_inspected_state
        """
        if self.config.points_algorithm == "cmu":
            points_alg = points_on_sphere_cmu
        else:
            points_alg = points_on_sphere_fibonacci
        points = points_alg(num_points=self.config.num_points, radius=self.config.radius)
        points_dict = {point: False for point in points}
        return points_dict

    def _update_points(self, position):
        """
        Update the inspected state of all inspection points given an inspector's position.

        Parameters
        ----------
        position: tuple or array
            inspector's position in cartesian coords

        Returns
        -------
        None
        """
        # calculate h of the spherical cap (inspection zone)
        r = self.config.radius
        rt = np.linalg.norm(position)
        h = 2 * r * ((rt - r) / (2 * rt))

        p_hat = position / np.linalg.norm(position)  # position unit vector (inspection zone cone axis)
        for point, inspected in self._state.points.items():
            # check that point hasn't already been inspected
            if not inspected:
                # if no illumination params detected
                if not self.illum_flag:
                    # project point onto inspection zone axis and check if in inspection zone
                    self._state.points[point] = np.dot(point, p_hat) >= r - h
                else:
                    mag = np.dot(point, p_hat)
                    if mag >= r - h:
                        r_avg = self.config.illumination_params.avg_rad_Earth2Sun
                        chief_properties = self.config.illumination_params.chief_properties
                        light_properties = self.config.illumination_params.light_properties
                        current_theta = illum.get_sun_angle(
                            self.clock, self.config.illumination_params.mean_motion, self.config.illumination_params.sun_angle
                        )
                        if self.config.illumination_params.bin_ray_flag:
                            self._state.points[point] = illum.check_illum(point, current_theta, r_avg, r)
                        else:
                            RGB = illum.compute_illum_pt(point, current_theta, position, r_avg, r, chief_properties, light_properties)
                            self._state.points[point] = illum.evaluate_RGB(RGB)

    def _kmeans_find_nearest(self, position):
        """Finds nearest cluster of uninspected points using kmeans clustering"""
        uninspected = []
        for point, inspected in self._state.points.items():
            if not inspected:
                if self.illum_flag:
                    if self.check_if_illuminated(point, position):
                        uninspected.append(point)
                else:
                    uninspected.append(point)
        if len(uninspected) == 0:
            out = np.array([0., 0., 0.])
        else:
            n = math.ceil(len(uninspected) / 10)
            data = np.array(uninspected)
            if self.last_cluster is None:
                init = "random"
            else:
                if n > self.last_cluster.shape[0]:
                    idxs = np.random.choice(self.last_cluster.shape[0], size=n - self.last_cluster.shape[0])
                    new = np.array(uninspected)[idxs, :]
                    init = np.vstack((self.last_cluster, new))
                else:
                    init = self.last_cluster[0:n, :]
            kmeans = KMeans(
                init=init,
                n_clusters=n,
                n_init=10,
                max_iter=50,
            )
            kmeans.fit(data)
            self.last_cluster = kmeans.cluster_centers_
            dist = []
            for center in self.last_cluster:
                dist.append(np.linalg.norm(position - center))
            out = kmeans.cluster_centers_[np.argmin(dist)]
            out = out / np.linalg.norm(out)
        return out

    def check_if_illuminated(self, point, position):
        """Check if points is illuminated"""
        r = self.config.radius
        r_avg = self.config.illumination_params.avg_rad_Earth2Sun
        chief_properties = self.config.illumination_params.chief_properties
        light_properties = self.config.illumination_params.light_properties
        current_theta = illum.get_sun_angle(
            self.clock, self.config.illumination_params.mean_motion, self.config.illumination_params.sun_angle
        )
        if self.config.illumination_params.bin_ray_flag:
            illuminated = illum.check_illum(point, current_theta, r_avg, r)
        else:
            RGB = illum.compute_illum_pt(point, current_theta, position, r_avg, r, chief_properties, light_properties)
            illuminated = illum.evaluate_RGB(RGB)
        return illuminated


PluginLibrary.AddClassToGroup(InspectionSimulator, "InspectionSimulator", {})
