"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains controllers for the CWH platform.
"""

import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BaseController, BasePlatformPartValidator

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh_simulator import CWHSimulator
from saferl.simulators.inspection_simulator import InspectionSimulator


class CWHController(BaseController):
    """
    A controller created to interface with the CWH platform.
    """

    def apply_control(self, control: np.ndarray) -> None:
        """
        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        """
        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError


class ThrustControllerValidator(BasePlatformPartValidator):
    """
    Controller config validator for the ThrustController.

    axis : int
        The index of the action space element corresponding to this controller's actions.
    """
    axis: int


class ThrustController(CWHController):
    """
    A controller to control thrust on the CWH platform.

    Parameters
    ----------
    parent_platform : cwh_platform
        The platform to which the controller belongs.
    config : dict
        Contains configuration properties.
    """

    def __init__(
        self,
        parent_platform,
        config,
        property_class=cwh_props.ThrustProp,
    ):  # pylint: disable=W0102
        self.config: ThrustControllerValidator
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        ThrustControllerValidator
            Config validator for the ThrustController.
        """

        return ThrustControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        """
        Applies control to the parent platform.

        Parameters
        ----------
        control
            ndarray describing the control to the platform.
        """
        self.parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        """
        Retrieve the applied control to the parent platform.

        Returns
        -------
        np.ndarray
            Previously applied action.
        """
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    ThrustController, "Controller_Thrust", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)

PluginLibrary.AddClassToGroup(
    ThrustController, "Controller_Thrust", {
        "simulator": InspectionSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
