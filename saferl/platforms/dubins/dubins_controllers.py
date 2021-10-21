"""
This module contains controllers for the Dubins platform.
"""
import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp, Prop
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator

from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsController(BaseController):
    """Generic dubins controller
    """

    @property
    def name(self):
        return self.config.name + self.__class__.__name__

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector


class RateControllerValidator(BaseControllerValidator):
    """Generic rate controller validator.

    axis: index in combined control vector for this controller's output action
    """
    axis: int


class RateController(DubinsController):
    """Generic rate controller. Writes control action to platform's control vector and reads applied action from platform.

    Parameters
    ----------
    parent_platform : DubinsPlatform
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
        return RateControllerValidator

    @property
    def control_properties(self) -> Prop:
        raise NotImplementedError

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


class AccelerationController(RateController):
    """Applies acceleration control to dubins platform
    """

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="Acceleration", low=[-10], high=[10], unit=["m/s/s"], description="Acceleration")
        return control_props


PluginLibrary.AddClassToGroup(
    AccelerationController,
    "Controller_Acceleration", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class YawRateController(RateController):
    """Applies Yaw control to dubins platform
    """

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="YawRate", low=[np.deg2rad(-6)], high=[np.deg2rad(6)], unit=["rad/s"], description="Yaw Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    YawRateController, "Controller_YawRate", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)

# ------ 2D Only --------


class CombinedTurnRateAccelerationController(DubinsController):
    """Applies heading rate and acceleration control to dubins plaform

    Parameters
    ----------
    parent_platform : DubinsPlatform
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

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(
            name="TurnAcceleration",
            low=[np.deg2rad(-6), -10],
            high=[np.deg2rad(6), 10],
            unit=["rad/s, m/s/s"],
            description="Combined Turn Rate and Acceleration"
        )
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedTurnRateAccelerationController,
    "Controller_TurnAcc", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)

# --------- 3D Only ------------


class CombinedPitchRollAccelerationController(DubinsController):
    """Applies pitch rate, roll rate, and acceleration control to dubins platform

    Parameters
    ----------
    parent_platform : DubinsPlatform
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

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(
            name="PitchRollAcc",
            low=[np.deg2rad(-6), np.deg2rad(-6), -10],
            high=[np.deg2rad(6), np.deg2rad(6), 10],
            unit=["rad/s, rad/s, m/s/s"],
            description="Combined Pitch Rate, Roll Rate, and Acceleration"
        )
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedPitchRollAccelerationController,
    "Controller_PitchRollAcc", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PitchRateController(RateController):
    """Applies pitch rate control to dubins platform
    """

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="PitchRate", low=[np.deg2rad(-6)], high=[np.deg2rad(6)], unit=["rad/s"], description="Pitch Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    PitchRateController, "Controller_PitchRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RollRateController(RateController):
    """Applies roll rate control to dubins platform
    """

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="RollRate", low=[np.deg2rad(-6)], high=[np.deg2rad(6)], unit=["rad/s"], description="Roll Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    RollRateController, "Controller_RollRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
