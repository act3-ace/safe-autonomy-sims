"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains controllers for the Dubins platform.
"""
import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BaseController, BasePlatformPartValidator

import saferl.platforms.dubins.dubins_properties as dubins_props
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsController(BaseController):
    """
    Generic dubins controller.
    """

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector


class RateControllerValidator(BasePlatformPartValidator):
    """
    Generic rate controller validator.

    axis: int
        Index in combined control vector for this controller's output action.
    """

    axis: int


class RateController(DubinsController):
    """
    Generic rate controller. Writes control action to platform's control vector and reads applied action from platform.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RateControllerValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        return RateControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    RateController,
    "Controller_GenericRate",
    {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    },
)
PluginLibrary.AddClassToGroup(
    RateController,
    "Controller_GenericRate",
    {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    },
)


class AccelerationController(RateController):
    """
    Applies acceleration control to dubins platform.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.AccelerationProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)


PluginLibrary.AddClassToGroup(
    AccelerationController,
    "Controller_Acceleration",
    {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    },
)
PluginLibrary.AddClassToGroup(
    AccelerationController,
    "Controller_Acceleration",
    {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    },
)


class YawRateController(RateController):
    """
    Applies Yaw control to dubins platform.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.YawRateProp):  # pylint: disable=W0102
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)


PluginLibrary.AddClassToGroup(
    YawRateController, "Controller_YawRate", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    YawRateController, "Controller_YawRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)

# ------ 2D Only --------


class CombinedTurnRateAccelerationController(DubinsController):
    """
    Applies heading rate and acceleration control to dubins platform.

    Parameters
    ----------
    parent_platform : DubinsPlatform
        The platform to which the controller belongs.
    config : dict
        Contains configuration properties.
    """

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
        property_class=dubins_props.YawAndAccelerationProp,
    ):  # pylint: disable=W0102

        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedTurnRateAccelerationController,
    "Controller_TurnAcc",
    {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    },
)

# --------- 3D Only ------------


class CombinedPitchRollAccelerationController(DubinsController):
    """
    Applies pitch rate, roll rate, and acceleration control to dubins platform.

    Parameters
    ----------
    parent_platform : DubinsPlatform
        The platform to which the controller belongs.
    config : dict
        Contains configuration properties.
    """

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
        property_class=dubins_props.PitchRollAndAccelerationProp,
    ):  # pylint: disable=W0102
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedPitchRollAccelerationController,
    "Controller_PitchRollAcc",
    {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    },
)


class PitchRateController(RateController):
    """
    Applies pitch rate control to dubins platform.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.PitchRateProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)


PluginLibrary.AddClassToGroup(
    PitchRateController, "Controller_PitchRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RollRateController(RateController):
    """
    Applies roll rate control to dubins platform.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.RollRateProp):  # pylint: disable=W0102
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)


PluginLibrary.AddClassToGroup(
    RollRateController, "Controller_RollRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
