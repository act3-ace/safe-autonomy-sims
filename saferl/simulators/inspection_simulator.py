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

import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from pydantic import validator
from safe_autonomy_dynamics.cwh import CWHSpacecraft

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator, SafeRLSimulatorValidator


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
    # work around to get algorithm to generate enough points
    n = num_points * radius**2

    # generate points using CMU algorithm
    n_count = 0
    a = (4 * math.pi * (radius**2)) / n
    d = math.sqrt(a)
    m_theta = round(math.pi / d) + 1
    d_theta = math.pi / m_theta
    d_phi = a / d_theta
    for m in range(0, m_theta):
        theta = math.pi * ((m + 0.5) / m_theta)
        m_phi = round((2 * math.pi * math.sin(theta)) / d_phi)
        for n in range(0, m_phi):
            phi = (2 * math.pi * n) / m_phi
            point = (radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi), radius * math.cos(theta))
            points.append(point)
            n_count += 1
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

    @validator("points_algorithm")
    def valid_algorithm(cls, v):
        """
        Check if provided algorithm is a valid choice.
        """
        valid_algs = ["cmu", "fibonacci"]
        if v not in valid_algs:
            raise ValueError(f"field points_algorithm must be one of {valid_algs}")
        return v


class InspectionSimulator(SafeRLSimulator):
    """
    Simulator for CWH Inspection Task. Interfaces CWH platforms with underlying CWH entities in Inspection simulation.
    """

    @property
    def get_simulator_validator(self):
        return InspectionSimulatorValidator

    def _construct_platform_map(self) -> dict:
        return {
            'default': (CWHSpacecraft, CWHPlatform),
            'cwh': (CWHSpacecraft, CWHPlatform),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: concern, we just added a new variable to StateDict...
        #  do we need to extend it or does it not know there is a points variable until
        self._state.points = self._add_points()

    def reset(self, config):
        super().reset(config)
        # self._state.clear()
        self._state.points = self._add_points()
        return self._state

    def _step_update_sim_statuses(self, step_size: float):
        # update points
        for _, entity in self.agent_sim_entities.items():
            self._update_points(entity.position)
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
                # project point onto inspection zone axis and check if in inspection zone
                self._state.points[point] = np.dot(point, p_hat) >= r - h


PluginLibrary.AddClassToGroup(InspectionSimulator, "InspectionSimulator", {})
