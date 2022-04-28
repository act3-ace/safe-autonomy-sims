"""
This module contains controllers for the CWH platform.
"""

import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseController, BasePlatformPartValidator

import saferl.core.platforms.cwh.cwh_properties as cwh_props
from saferl.core.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.core.simulators.cwh_simulator import CWHSimulator


class CWHController(BaseController):
    """
    A controller created to interface with the CWH platform.
    """

    # @property
    # def name(self):
    #     """
    #     Returns
    #     -------
    #     String
    #         name of the controller
    #     """
    #     return self.config.name + self.__class__.__name__

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
    Controller config validator for the ThrustController
    """

    axis: int


class ThrustController(CWHController):
    """
    A controller to control thrust on the CWH platrorm.

    Params
    -------
    parent_platform : cwh_platform
        the platform to which the controller belongs
    config : dict
        contains configuration proprties

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
        Params
        ------
        cls : constructor function

        Returns
        -------
        ThrustControllerValidator
            config validator for the ThrustController

        """

        return ThrustControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        """
        Applies control to the parent platform

        Params
        ------
        control
            ndarray describing the control to the platform
        """
        self.parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        """
        Retreive the applied control to the parent platform

        Returns
        -------
        np.ndarray
            Previously applied action

        """
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    ThrustController, "Controller_Thrust", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
