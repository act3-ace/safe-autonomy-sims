"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements Run Time Assurance for the inspection task.
"""
import math
from typing import Union

import numpy as np
from run_time_assurance.utils import to_jnp_array_jit
from run_time_assurance.zoo.cwh.inspection_1v1 import (
    CHIEF_RADIUS_DEFAULT,
    DEPUTY_RADIUS_DEFAULT,
    FOV_DEFAULT,
    R_MAX_DEFAULT,
    SUN_VEL_DEFAULT,
    U_MAX_DEFAULT,
    V0_DEFAULT,
    V0_DISTANCE_DEFAULT,
    V1_COEF_DEFAULT,
    VEL_LIMIT_DEFAULT,
    Inspection1v1RTA,
)
from safe_autonomy_simulation.sims.spacecraft.defaults import M_DEFAULT, N_DEFAULT

from safe_autonomy_sims.glues.rta_glue import RTAGlue, RTAGlueValidator, RTASingleton


class CWHInspection1v1RTAGlueValidator(RTAGlueValidator):
    """
    A configuration validator for CWHInspection1v1RTAGlue.

    Attributes
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    chief_radius : float, optional
        radius of collision for chief spacecraft, by default CHIEF_RADIUS_DEFAULT
    deputy_radius : float, optional
        radius of collision for each deputy spacecraft, by default DEPUTY_RADIUS_DEFAULT
    collision_radius : Union[float, None], optional
        Total collision radius - alternative to defining chief radius and deputy radius separately. By default None
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
    r_max : float, optional
        maximum relative distance from chief, by default R_MAX_DEFAULT
    fov : float, optional
        sensor field of view, by default FOV_DEFAULT
    vel_limit : float, optional
        max velocity magnitude, by default VEL_LIMIT_DEFAULT
    sun_vel : float, optional
        velocity of sun in x-y plane (rad/sec), by default SUN_VEL_DEFAULT
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    constraints : list, optional
        list of constraint keys to include
    """
    m: float = M_DEFAULT
    n: float = N_DEFAULT
    chief_radius: float = CHIEF_RADIUS_DEFAULT
    deputy_radius: float = DEPUTY_RADIUS_DEFAULT
    collision_radius: Union[float, None] = None
    v0: float = V0_DEFAULT
    v0_distance: float = V0_DISTANCE_DEFAULT
    v1_coef: float = V1_COEF_DEFAULT
    r_max: float = R_MAX_DEFAULT
    fov: float = FOV_DEFAULT
    vel_limit: float = VEL_LIMIT_DEFAULT
    sun_vel: float = SUN_VEL_DEFAULT
    control_bounds_high: float = U_MAX_DEFAULT
    control_bounds_low: float = -U_MAX_DEFAULT
    constraints: list = []


class RTAGlueCWHInspection1v1(RTAGlue):
    """
    RTA Glue to wrap CWH Inspection 1v1 RTA from the run-time-assurance package.
    """

    def __init__(self, **kwargs):
        self.config: CWHInspection1v1RTAGlueValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return CWHInspection1v1RTAGlueValidator

    def _get_singleton(self):
        return InspectionRTASingleton

    def _get_rta_args(self) -> dict:
        if self.config.collision_radius is not None:
            self.config.chief_radius = self.config.collision_radius / 2
            self.config.deputy_radius = self.config.collision_radius / 2

        return {
            "m": self.config.m,
            "n": self.config.n,
            "chief_radius": self.config.chief_radius,
            "deputy_radius": self.config.deputy_radius,
            "v0": self.config.v0,
            "v0_distance": self.config.v0_distance,
            "v1_coef": self.config.v1_coef,
            "r_max": self.config.r_max,
            "fov": self.config.fov,
            "vel_limit": self.config.vel_limit,
            "sun_vel": self.config.sun_vel,
            "control_bounds_high": self.config.control_bounds_high,
            "control_bounds_low": self.config.control_bounds_low,
            "constraints_to_use": self.config.constraints,
        }


class InspectionRTASingleton(RTASingleton):
    """Inspection Singleton
    """

    def _create_rta_module(self, **rta_args):
        return UpdatedInspection1v1RTA(**rta_args)


class UpdatedInspection1v1RTA(Inspection1v1RTA):
    """Updated RTA module with _get_state"""

    def _get_state(self, input_state):
        if len(input_state) == 7:
            input_state = np.array(input_state)
            input_state = np.concatenate((input_state, np.array([0.])))
            theta = input_state[6]
            m = math.floor(abs(theta / (2 * np.pi)))
            theta -= np.sign(theta) * m * 2 * np.pi
            input_state[6] = 2 * np.pi - theta
        else:
            input_state = np.concatenate((input_state, np.array([0., 0.])))
        return to_jnp_array_jit(input_state)
