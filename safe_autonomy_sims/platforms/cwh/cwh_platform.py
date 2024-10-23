"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines the platforms used with safe_autonomy_sims CWHSimulator class.
It represents a spacecraft operating under the Clohessy-Wiltshire dynamics model, with and without rotation.
"""

import typing

import numpy as np
from corl.libraries.units import corl_get_ureg
from corl.simulators.base_platform import BasePlatformValidator
from safe_autonomy_simulation.sims.spacecraft import CWHSpacecraft, SixDOFSpacecraft
from safe_autonomy_simulation.entities import PhysicalEntity

from safe_autonomy_sims.platforms.common.platform import BaseSafeRLPlatform


class CWHPlatformValidator(BasePlatformValidator):
    """
    A configuration validator for CWHPlatform

    Attributes
    ----------
    platform : CWHSpacecraft
        underlying dynamics platform
    """

    platform: CWHSpacecraft


class CWHPlatform(BaseSafeRLPlatform):
    """
    A platform representing a spacecraft operating under CWH dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function.
    """

    def __init__(self, platform_name, platform, parts_list, sim_time=0.0):
        """
        Parameters
        ----------
        platform_name : str
            Name of the platform.
        platform : sim_entity
            Backend simulation entity associated with the platform.
        parts_list : dict
            Platform-specific parts configuration dictionary.
        sim_time : float
            simulation time at platform creation
        """
        self.config: CWHPlatformValidator
        super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list, sim_time=sim_time)
        self._platform = self.config.platform
        self._last_applied_action = corl_get_ureg().Quantity(np.array([0, 0, 0], dtype=np.float32), "newtons")

    @staticmethod
    def get_validator() -> typing.Type[CWHPlatformValidator]:
        """
        get validator for this CWHPlatform

        Returns
        -------
        CWHPlatformValidator -- validator the platform will use to generate a configuration
        """
        return CWHPlatformValidator

    def get_applied_action(self):
        """
        Returns the action stored in this platform.

        Returns
        -------
        typing.Any -- Any sort of stored action.
        """
        return self._last_applied_action

    def __eq__(self, other):
        if isinstance(other, CWHPlatform):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and self.sim_time == other.sim_time
            return eq
        return False

    def save_action_to_platform(self, action, axis):
        """
        Saves an action to the platform if it matches the action space.

        Parameters
        ----------
        action: typing.Any
            The action to store in the platform
        axis: int
            The index of the action space where the action shall be saved.
        """
        if isinstance(action.m, np.ndarray) and len(action.m) == 1:
            # TODO: is there a way to ensure quantity coming in is of correct units?
            #       currently, incoming action is 'dimensionless'.
            # action = action.to('newton')
            self._last_applied_action.m[axis] = action.m[0]
        else:
            raise TypeError(
                f"Action saved to platform is of incompatible type:\
                             expected Quantity with numpy.ndarray of length 1, but got {type(action.m)}"
            )

    @property
    def position(self) -> np.ndarray:
        """
        The position of the platform.

        Returns
        -------
        np.ndarray
            The position vector of the platform.
        """
        return self._platform.position

    @property
    def velocity(self):
        """
        The velocity of the platform.

        Returns
        -------
        np.ndarray
            The velocity vector of the platform.
        """
        return self._platform.velocity

    @property
    def operable(self):
        return True

    def entity_relative_position(self, target_entity: PhysicalEntity):
        """
        The position of target_entity relative to self (without rotation)

        Parameters
        ----------
        target_entity: Entity
            Entity to compute relative position from

        Returns
        -------
        np.ndarray
            The relative position of target_entity
        """
        return target_entity.position - self._platform.position

    def entity_relative_velocity(self, target_entity: PhysicalEntity):
        """
        The velocity of target_entity relative to self (without rotation)

        Parameters
        ----------
        target_entity: Entity
            Entity to compute relative velocity from

        Returns
        -------
        np.ndarray
            The relative velocity of target_entity
        """
        return target_entity.velocity - self._platform.velocity


class CWHSixDOFPlatformValidator(BasePlatformValidator):
    """
    A configuration validator for CWHSixDOFPlatform

    Attributes
    ----------
    platform : SixDOFSpacecraft
        underlying dynamics platform
    """
    platform: SixDOFSpacecraft


class CWHSixDOFPlatform(CWHPlatform):
    """
    A platform representing a spacecraft operating under CWH dynamics with rotation.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function.
    """

    def __init__(self, platform_name, platform, parts_list, sim_time=0.0):
        """
        Parameters
        ----------
        platform_name : str
            Name of the platform.
        platform : sim_entity
            Backend simulation entity associated with the platform.
        parts_list : dict
            Platform-specific parts configuration dictionary.
        sim_time : float
            simulation time at platform creation
        """
        self.config: CWHSixDOFPlatformValidator
        super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list, sim_time=sim_time)
        self._last_applied_action = corl_get_ureg().Quantity(np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), "newtons")

    @staticmethod
    def get_validator() -> typing.Type[CWHSixDOFPlatformValidator]:
        """
        get validator for this CWHPlatform

        Returns:
            CWHPlatformValidator -- validator the platform will use to generate a configuration
        """
        return CWHSixDOFPlatformValidator

    def __eq__(self, other):
        if isinstance(other, CWHSixDOFPlatform):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and self.sim_time == other.sim_time
            return eq
        return False

    @property
    def quaternion(self):
        """
        The quaternion of the platform.

        Returns
        -------
        np.ndarray
            The quaternion vector of the platform.
        """
        return self._platform.orientation

    @property
    def angular_velocity(self):
        """
        The angular_velocity of the platform.

        Returns
        -------
        np.ndarray
            The angular_velocity vector of the platform.
        """
        return self._platform.angular_velocity
