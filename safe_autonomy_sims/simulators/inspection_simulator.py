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
import copy
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import Quantity
from pydantic import BaseModel, validator
from ray.rllib import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from safe_autonomy_simulation import Entity
from safe_autonomy_simulation.sims.spacecraft import CWHSpacecraft, SixDOFSpacecraft
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans

import safe_autonomy_sims.simulators.illumination_functions as illum
from safe_autonomy_sims.platforms.cwh.cwh_platform import CWHPlatform, CWHSixDOFPlatform
from safe_autonomy_sims.simulators.saferl_simulator import (
    SafeRLSimulator,
    SafeRLSimulatorResetValidator,
    SafeRLSimulatorState,
    SafeRLSimulatorValidator,
)


class IlluminationValidator(BaseModel):
    """
    A configuration validator for illumination parameters

    Attributes
    ----------
    mean_motion: float
        A float representing the mean motion of the spacecraft in Low Earth Orbit (LEO) in [RADIANS/SECOND].
    avg_rad_Earth2Sun: float
        A float representing the average distance between the Earth and the Sun in [METERS].
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
    light_properties: dict
    chief_properties: dict
    resolution: list
    focal_length: float
    bin_ray_flag: bool
    render_flag_3d: bool = False
    render_flag_subplots: bool = False
    pixel_pitch: float


class InspectionPointsValidator(BaseModel):
    """
    A configuration validator for an InspectionPoints object.

    Attributes
    ----------
    num_points: int
        The number of inspectable points maintained.
    radius: float
        The radius of the sphere on which the points will be generated.
    points_algorithm: str
        The name of the algorithm used to generate initial point positions.
    sensor_fov: float
        The field of view of the inspector's camera sensor in radians.
        Default is 2 * pi (everything is in view).
    initial_sensor_unit_vec: list
        The initial direction the inspector's camera sensor is pointing.
    illumination_params: typing.Union[IlluminationValidator, None]
        The parameters defining lighting of the environment.
    """
    num_points: int
    radius: float
    points_algorithm: str = "cmu"
    sensor_fov: float = 2 * np.pi
    initial_sensor_unit_vec: list = [1., 0., 0.]
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


class InspectionPoints:
    """
    A class maintaining the inspection status of an entity.
    """

    def __init__(self, parent_entity: CWHSpacecraft, priority_vector: np.ndarray, **kwargs):
        self.config: InspectionPointsValidator = self.get_validator()(**kwargs)
        self.sun_angle = 0.0
        self.clock = 0.0
        self.parent_entity = parent_entity
        self.priority_vector = priority_vector
        self.init_priority_vector = copy.deepcopy(self.priority_vector)
        (self._default_points_position_dict, self.points_position_dict, self.points_inspected_dict,
         self.points_weights_dict) = self._add_points()
        self.last_points_inspected = 0
        self.last_cluster = None

    @staticmethod
    def get_validator() -> typing.Type[InspectionPointsValidator]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return InspectionPointsValidator

    def _add_points(self):
        """
        Generate a map of inspection point coordinates to inspected state.

        Returns
        -------
        points_dict
            dict of points_dict[cartesian_point] = initial_inspected_state
        """
        if self.config.points_algorithm == "cmu":
            points_alg = InspectionPoints.points_on_sphere_cmu
        else:
            points_alg = InspectionPoints.points_on_sphere_fibonacci
        points = points_alg(self.config.num_points, self.config.radius)  # TODO: HANDLE POSITION UNITS*
        points_position_dict = {}
        points_inspected_dict = {}
        points_weights_dict = {}
        for i, point in enumerate(points):
            points_position_dict[i] = point
            points_inspected_dict[i] = False
            points_weights_dict[i] = np.arccos(
                np.dot(-self.priority_vector, point) / (np.linalg.norm(-self.priority_vector) * np.linalg.norm(point))
            ) / np.pi

        # Normalize weighting
        total_weight = sum(list(points_weights_dict.values()))
        points_weights_dict = {k: w / total_weight for k, w in points_weights_dict.items()}

        default_points_position = copy.deepcopy(points_position_dict)

        return default_points_position, points_position_dict, points_inspected_dict, points_weights_dict

    # inspected or not
    def update_points_inspection_status(self, inspector_entity):
        """
        Update the inspected state of all inspection points given an inspector's position.

        Parameters
        ----------
        inspector_entity : PyObject
            sim entity of inspector platform

        Returns
        -------
        None
        """
        # calculate h of the spherical cap (inspection zone)
        position = inspector_entity.position
        if isinstance(inspector_entity, SixDOFSpacecraft):
            rotation = Rotation.from_quat(inspector_entity.orientation)
            r_c = rotation.apply(self.config.initial_sensor_unit_vec)
        else:
            r_c = -position
        r_c = r_c / np.linalg.norm(r_c)

        r = self.config.radius
        rt = np.linalg.norm(position)
        h = 2 * r * ((rt - r) / (2 * rt))

        p_hat = position / np.linalg.norm(position)  # position unit vector (inspection zone cone axis)

        for point_id, point_position in self.points_position_dict.items():  # pylint: disable=too-many-nested-blocks
            # check that point hasn't already been inspected
            if not self.points_inspected_dict[point_id]:
                p = point_position - position
                cos_theta = np.dot(p / np.linalg.norm(p), r_c)
                angle_to_point = np.arccos(cos_theta)
                # If the point can be inspected (within FOV)
                if angle_to_point <= self.config.sensor_fov / 2:
                    # if no illumination params detected
                    if not self.config.illumination_params:
                        # project point onto inspection zone axis and check if in inspection zone
                        if np.dot(point_position, p_hat) >= r - h:
                            self.points_inspected_dict[point_id] = inspector_entity.name
                    else:
                        mag = np.dot(point_position, p_hat)
                        if mag >= r - h:
                            r_avg = self.config.illumination_params.avg_rad_Earth2Sun
                            chief_properties = self.config.illumination_params.chief_properties
                            light_properties = self.config.illumination_params.light_properties
                            current_theta = self.sun_angle
                            if self.config.illumination_params.bin_ray_flag:
                                if illum.check_illum(point_position, current_theta, r_avg, r):
                                    self.points_inspected_dict[point_id] = inspector_entity.name
                            else:
                                RGB = illum.compute_illum_pt(
                                    point_position, current_theta, position, r_avg, r, chief_properties, light_properties
                                )
                                if illum.evaluate_RGB(RGB):
                                    self.points_inspected_dict[point_id] = inspector_entity.name

    def kmeans_find_nearest_cluster(self, position):
        """Finds nearest cluster of uninspected points using kmeans clustering

        Parameters
        ----------
        position : list
            position vector

        Returns
        -------
        list
            unit vector pointing to nearest cluster
        """
        uninspected = []
        for point_id, inspected in self.points_inspected_dict.items():
            point_position = self.points_position_dict[point_id]
            if not inspected:
                if self.config.illumination_params:
                    if self.check_if_illuminated(point_position, position):
                        uninspected.append(point_position)
                else:
                    uninspected.append(point_position)
        if len(uninspected) == 0:
            out = np.array([0., 0., 0.])
        else:
            n = math.ceil(len(uninspected) / 10)
            data = np.array(uninspected)
            if self.last_cluster is None:
                init = np.zeros((n, 3))
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
        """Check if point is illuminated

        Parameters
        ----------
        point : list
            point position vector
        position : list
            agent position vector

        Returns
        -------
        bool
            True if point is illuminated, False otherwise
        """
        r = self.config.radius
        r_avg = self.config.illumination_params.avg_rad_Earth2Sun
        chief_properties = self.config.illumination_params.chief_properties
        light_properties = self.config.illumination_params.light_properties
        current_theta = self.sun_angle
        if self.config.illumination_params.bin_ray_flag:
            illuminated = illum.check_illum(point, current_theta, r_avg, r)
        else:
            RGB = illum.compute_illum_pt(point, current_theta, position, r_avg, r, chief_properties, light_properties)
            illuminated = illum.evaluate_RGB(RGB)
        return illuminated

    @staticmethod
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
            r = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * r
            z = math.sin(theta) * r

            points.append(radius * np.array([x, y, z]))

        return points

    @staticmethod
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

                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)

                points.append(np.array([x, y, z]))

        return points

    def update_points_position(self):
        """
        Update the locations of the points on the chief after rotation
        """

        # get parent entity info
        parent_position = self.parent_entity.position
        parent_orientation = self.parent_entity.orientation
        parent_orientation = Rotation.from_quat(parent_orientation)

        self.priority_vector = parent_orientation.apply(self.init_priority_vector)

        for point_id, default_position in self._default_points_position_dict.items():
            # rotate about origin
            new_position = parent_orientation.apply(default_position)
            # translate from origin
            new_position = new_position + parent_position
            self.points_position_dict[point_id] = new_position

    # getters / setters
    def get_num_points_inspected(self, inspector_entity: Entity = None):
        """Get total number of points inspected"""
        num_points = 0
        if inspector_entity:
            # count number of points inspected by the provided entity
            for _, point_inspector_entity in self.points_inspected_dict.items():
                num_points += 1 if point_inspector_entity == inspector_entity.name else 0
        else:
            # count the total number of points inspected
            for _, point_inspector_entity in self.points_inspected_dict.items():
                num_points += 1 if point_inspector_entity else 0

        return num_points

    def get_percentage_of_points_inspected(self, inspector_entity: Entity = None):
        """Get the percentage of points inspected"""
        total_num_points = len(self.points_inspected_dict.keys())

        if inspector_entity:
            percent = self.get_num_points_inspected(inspector_entity=inspector_entity) / total_num_points
        else:
            percent = self.get_num_points_inspected() / total_num_points
        return percent

    def get_cluster_location(self, inspector_position):
        """Get the location of the nearest cluster of uninspected points"""
        return self._kmeans_find_nearest(inspector_position)

    def get_total_weight_inspected(self, inspector_entity: Entity = None):
        """Get total weight of points inspected"""
        weights = 0
        if inspector_entity:
            for point_inspector_entity, weight in zip(self.points_inspected_dict.values(), self.points_weights_dict.values()):
                weights += weight if point_inspector_entity == inspector_entity.name else 0.
        else:
            for point_inspector_entity, weight in zip(self.points_inspected_dict.values(), self.points_weights_dict.values()):
                weights += weight if point_inspector_entity else 0.
        return weights

    def set_sun_angle(self, sun_angle: np.ndarray):
        """Get the current sun angle"""
        self.sun_angle = float(sun_angle)


class InspectionSimulatorState(SafeRLSimulatorState):
    """
    The basemodel for the state of the InspectionSimulator.

    Attributes
    ----------
    inspection_points_map: dict
        The dictionary containing the points the agent needs to inspect.
        Keys: point_id int. Values: InspectionPoints objects.
    total_steps: int
        The total number of steps simulated since the simulator was initialized
    delta_v_scale: float
        The scale of the delta_v reward. This value is updated over time, based on the value of inspected_points_value.
        See safe_autonomy_sims/rewards/cwh/inspection_rewards for more details.
        Note that this feature is experimental, and currently does not synchronize between workers.
    sun_angle: typing.Union[Quantity, float]
        Angle of the Sun in the x-y plane
    priority_vector: np.ndarray
        Vector indicating priority of points to be inspected
    """
    inspection_points_map: typing.Dict[str, InspectionPoints]
    total_steps: int
    delta_v_scale: float
    sun_angle: typing.Union[Quantity, float] = 0.0
    priority_vector: np.ndarray

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True


class InspectionSimulatorValidator(SafeRLSimulatorValidator):
    """
    A configuration validator for the InspectionSimulator config.

    Attributes
    ----------
    illumination_params: IlluminationValidator
        dict of illumination parameters, described above
    steps_until_update: int
        Number of steps between updates to the delta_v_scale
    delta_v_updater_criteria: str
        Criteria for updating the delta-v scale.
        Either 'score' for total points score, or 'count' for total number of points
    delta_v_scale_bounds: list
        lower and upper bounds for the value of delta_v_scale
    delta_v_scale_step: float
        The amount to advance/retract the delta_v_scale by at each update
    inspected_points_update_bounds: list
        bounds for when to update the delta_v_scale.
        if inspected_points_value >= inspected_points_update_bounds[1], delta_v_scale is advanced by delta_v_scale_step
        if inspected_points_value <= inspected_points_update_bounds[0], delta_v_scale is retracted by delta_v_scale_step
    sensor_fov: float
        field of view of the sensor (radians).
        Default is 2 * pi (everything is in view).
    initial_sensor_unit_vec: list
        If using the 6DOF spacecraft model, initial unit vector along sensor boresight.
        By default [1., 0., 0.]
    inspection_points_map: dict
        A map of entity name strings to InspectionPoints objects, which track the inspection progress of each entity.
    """

    illumination_params: typing.Union[IlluminationValidator, None] = None
    steps_until_update: int
    delta_v_updater_criteria: str = 'count'
    delta_v_scale_bounds: list
    delta_v_scale_step: float
    inspected_points_update_bounds: list
    sensor_fov: float = 2 * np.pi
    initial_sensor_unit_vec: list = [1., 0., 0.]
    inspection_points_map: typing.Dict[str, InspectionPointsValidator]

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True


class InspectionSimulatorResetValidator(SafeRLSimulatorResetValidator):
    """
    A configuration validator for the InspectionSimulator reset method.

    priority_vector_azimuth_angle: Quantity, float
        Azimuth angle of the priority vector for weighting points
    priority_vector_elevation_angle: Quantity, float
        Elevation angle of the priority vector for weighting points
    """
    priority_vector_azimuth_angle: typing.Union[Quantity, float] = 0.0
    priority_vector_elevation_angle: typing.Union[Quantity, float] = 0.0

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True


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
            'six-dof': (SixDOFSpacecraft, CWHSixDOFPlatform),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inspection_points_map = {}

        if self.config.illumination_params is not None:
            if self.config.illumination_params.render_flag_3d:
                self.fig = plt.figure(1)
            if self.config.illumination_params.render_flag_subplots:
                self.fig = plt.figure(1)
                self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
                self.ax_xy = self.fig.add_subplot(2, 2, 2)
                self.ax_xz = self.fig.add_subplot(2, 2, 3)
                self.ax_yz = self.fig.add_subplot(2, 2, 4)

        self.total_steps = 0
        self.inspected_points_value_list = []
        self.update_step_current = 0
        self.delta_v_scale = 0
        self.delta_v_updater_criteria = self.config.delta_v_updater_criteria
        self.steps_until_update = self.config.steps_until_update
        self.delta_v_scale_bounds = self.config.delta_v_scale_bounds
        self.delta_v_scale_step = self.config.delta_v_scale_step
        self.inspected_points_update_bounds = self.config.inspected_points_update_bounds
        self.sun_angle = 0.
        self.priority_vector = np.zeros(3)

    def create_inspection_points_map(self):
        """
        Create map of inspection points for each entity
        """
        points_map = {}
        for entity_name, inspection_points_validator in self.config.inspection_points_map.items():
            # TODO: there must be a better way to use validators
            parent_entity = self.sim_entities[entity_name]
            points_map[entity_name] = InspectionPoints(
                parent_entity=parent_entity,
                radius=inspection_points_validator.radius,
                num_points=inspection_points_validator.num_points,
                points_algorithm=inspection_points_validator.points_algorithm,
                sensor_fov=self.config.sensor_fov,
                initial_sensor_unit_vec=self.config.initial_sensor_unit_vec,
                illumination_params=self.config.illumination_params,
                priority_vector=self.priority_vector,
            )

        return points_map

    def _construct_simulator_state(self) -> dict:
        return InspectionSimulatorState(
            sim_platforms=self.sim_platforms,
            inspection_points_map=self.inspection_points_map,
            sim_time=self.clock,
            sim_entities=self.sim_entities,
            total_steps=self.total_steps,
            delta_v_scale=self.delta_v_scale,
            sun_angle=self.sun_angle,
            priority_vector=self.priority_vector,
        )

    def reset(self, config):
        super().reset(config)
        if self.config.illumination_params is not None:
            self.sun_angle = self.sim_entities['sun'].theta

        self._get_initial_priority_vector(config)

        if self.inspection_points_map:
            # calculate delta_v_scale
            # TODO: assumes first agent
            agent = list(self.inspection_points_map.keys())[0]
            final_num_points_inspected = self.inspection_points_map[agent].get_num_points_inspected()
            if final_num_points_inspected != 0:
                if self.delta_v_updater_criteria == 'score':
                    self.inspected_points_value_list.append(self.inspection_points_map[agent].get_total_weight_inspected())
                elif self.delta_v_updater_criteria == 'count':
                    self.inspected_points_value_list.append(self.inspection_points_map[agent].get_percentage_of_points_inspected())
                else:
                    raise ValueError('delta_v_updater_criteria must be either "score" or "count"')
            if self.total_steps >= self.update_step_current:
                self.update_step_current += self.steps_until_update
                mean_inspected_points_value = np.mean(self.inspected_points_value_list)
                self.inspected_points_value_list = []
                if mean_inspected_points_value >= self.inspected_points_update_bounds[1]:
                    self.delta_v_scale += self.delta_v_scale_step
                if mean_inspected_points_value <= self.inspected_points_update_bounds[0]:
                    self.delta_v_scale -= self.delta_v_scale_step

        self.delta_v_scale = np.clip(self.delta_v_scale, self.delta_v_scale_bounds[0], self.delta_v_scale_bounds[1])

        # reset points map
        self.inspection_points_map = self.create_inspection_points_map()
        for points in self.inspection_points_map.values():
            points.update_points_position()
            # TODO: assign priority vector to specific agent
            self.priority_vector = points.priority_vector
        # illuminate
        if self.config.illumination_params:
            # pass sun_angle to InspectionPoints objs
            for points in self.inspection_points_map.values():
                points.set_sun_angle(self.sun_angle)

        self._update_inspection_points_statuses()

        self._state = self._construct_simulator_state()

        self.update_sensor_measurements()
        return self._state

    def _get_initial_priority_vector(self, config):
        """Get the initial priority vector for weighting points"""
        azi = config["priority_vector_azimuth_angle"]
        if isinstance(azi, Quantity):
            azi = azi.m
        ele = config["priority_vector_elevation_angle"]
        if isinstance(ele, Quantity):
            ele = ele.m

        self.priority_vector[0] = np.cos(azi) * np.cos(ele)
        self.priority_vector[1] = np.sin(azi) * np.cos(ele)
        self.priority_vector[2] = np.sin(ele)

    def _step_update_sim_statuses(self, step_size: float):
        self.total_steps += 1
        self._state.total_steps = self.total_steps

        # update inspection points positions
        for points in self.inspection_points_map.values():
            points.update_points_position()
            # TODO: assign priority vector to specific agent
            self.priority_vector = points.priority_vector
            self._state.priority_vector = points.priority_vector

        # illuminate
        if self.config.illumination_params:
            self.sun_angle = self.sim_entities['sun'].theta
            # pass sun_angle to InspectionPoints objs
            for points in self.inspection_points_map.values():
                points.set_sun_angle(self.sun_angle)
            self._state.sun_angle = float(self.sun_angle)

        # update inspection points statuses
        self._update_inspection_points_statuses()

        if self.config.illumination_params:
            # render scene every m simulation seconds
            if (self.config.illumination_params.render_flag_3d or self.config.illumination_params.render_flag_subplots):
                current_time = self.clock
                sun_position = illum.get_sun_position(
                    current_time,
                    self.config.illumination_params.mean_motion,
                    self.sun_angle,
                    self.config.illumination_params.avg_rad_Earth2Sun
                )
                m = 10
                if (current_time % (m)) == 0:
                    if self.config.illumination_params.render_flag_3d:
                        # TODO: remove "chief" assumption
                        illum.render_3d(
                            self.fig,
                            self.agent_sim_entities["blue0"].position,
                            sun_position,
                            self.inspection_points_map["chief"].config.radius,
                            current_time,
                            m
                        )
                    if self.config.illumination_params.render_flag_subplots:
                        axes = [self.ax_3d, self.ax_xy, self.ax_xz, self.ax_yz]
                        illum.render_subplots(
                            self.fig,
                            axes,
                            self.agent_sim_entities["blue0"].position,
                            sun_position,
                            self.inspection_points_map["chief"].config.radius,
                            current_time,
                            m
                        )

    def _update_inspection_points_statuses(self):
        for inspection_entity_name, points in self.inspection_points_map.items():
            for inspector_entity_name, inspector_entity in self.agent_sim_entities.items():
                if inspection_entity_name != inspector_entity_name:
                    points.update_points_inspection_status(inspector_entity)


PluginLibrary.AddClassToGroup(InspectionSimulator, "InspectionSimulator", {})


class InspectionCallbacks(DefaultCallbacks):
    """
    Custom callbacks for the Inspection Simulator
    Log value for the delta_v_scale
    """

    def on_episode_end(self, *, worker, base_env: BaseEnv, policies: typing.Dict[PolicyID, Policy], episode, **kwargs) -> None:
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        env = base_env.get_sub_environments()[episode.env_id]
        episode.custom_metrics['info/delta_v_scale'] = env.simulator.delta_v_scale
        episode.custom_metrics['info/total_steps'] = env.simulator.total_steps
