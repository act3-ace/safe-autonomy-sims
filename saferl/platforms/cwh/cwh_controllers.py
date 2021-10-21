"""
This module contains controllers for the CWH platform.
"""

import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp, Prop
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh.cwh_simulator import CWHSimulator


class CWHController(BaseController):
    """
    A controller created to interface with the CWH platform.
    """

    @property
    def name(self):
        """
        Returns
        -------
        String
            name of the controller
        """
        return self.config.name + self.__class__.__name__

    @property
    def control_properties(self) -> Prop:
        """
        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError

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


class ThrustControllerValidator(BaseControllerValidator):
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
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):

        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
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

    @property
    def control_properties(self) -> Prop:
        """
        Returns
        -------
        control_props: MultiBoxProp
            describes the range of values and units of the thrust possible from this controller
        """
        control_props = MultiBoxProp(name=f"{self.name} Thrust", low=[-1], high=[1], unit=["newtons"], description="Thrust")
        return control_props

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
