"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains common controllers for the platforms.
"""

import numpy as np
from corl.libraries.plugin_library import PluginLibrary  # pylint: disable=E0401
from corl.simulators.base_parts import BaseController, BasePlatformPartValidator  # pylint: disable=E0401
from pydantic import BaseModel, PyObject

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.cwh_simulator import CWHSimulator
from saferl.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator
from saferl.simulators.inspection_simulator import InspectionSimulator


class CommonController(BaseController):
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


class RateControllerValidator(BasePlatformPartValidator):
    """
    Generic rate controller validator.

    axis: int
        Index in combined control vector for this controller's output action.
    """
    axis: int


class ControllerPropValidator(BaseModel):
    """
    Controller Prop validator.

    property_class: PyObject
        The Prop class defining the controller's bounds and units
    """
    property_class: PyObject


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector
class RateController(CommonController):
    """
    Generic rate controller. Writes control action to platform's control vector and reads applied action from platform.
    """

    def __init__(
        self,
        parent_platform,
        config,
    ):  # pylint: disable=W0102
        self.config: RateControllerValidator
        self.prop_config: ControllerPropValidator = self.get_prop_validator(**config)  # get property class
        super().__init__(property_class=self.prop_config.property_class, parent_platform=parent_platform, config=config)

    @property
    def get_validator(self):
        """
        Property to return the generic RateController validator.
        """
        return RateControllerValidator

    @property
    def get_prop_validator(self):
        """
        Property to return the generic RateController validator.
        """
        return ControllerPropValidator

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    RateController, "RateController", {
        "simulator": InspectionSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)

PluginLibrary.AddClassToGroup(RateController, "RateController", {"simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH})

PluginLibrary.AddClassToGroup(
    RateController,
    "RateController",
    {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    },
)

PluginLibrary.AddClassToGroup(
    RateController,
    "RateController",
    {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    },
)
