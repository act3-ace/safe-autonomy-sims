import typing

import numpy as np
from act3_rl_core.libraries.property import MultiBoxProp, Prop
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator
from act3_rl_core.libraries.plugin_library import PluginLibrary

from pydantic import validator

from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes


class DubinsController(BaseController):

    @property
    def name(self):
        return self.config.name + self.__class__.__name__

    def control_properties(self) -> Prop:
        raise NotImplementedError

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector

class RateControllerValidator(BaseControllerValidator):
    axis: int
    bounds: typing.List[float]

    @validator("bounds")
    def check_len(cls, v):
        check_len = 2
        if len(v) != check_len:
            raise ValueError(f"Bounds provided to validator is not length {check_len}")
        return v


class RateController(DubinsController):
    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):
        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
        return RateControllerValidator

    def control_properties(self) -> Prop:
        raise NotImplementedError

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


class AccelerationControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[float]] = [-96.5, 96.5]


class AccelerationController(RateController):

    @classmethod
    def get_validator(cls):
        return AccelerationControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="Acceleration", low=[self.config.bounds[0]], high=[self.config.bounds[1]],
                                     unit=["ft/s/s"], description="Acceleration")
        return control_props


PluginLibrary.AddClassToGroup(
    AccelerationController, "Controller_Acceleration", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class YawRateControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[float]] = [-10, 10]


class YawRateController(RateController):

    @classmethod
    def get_validator(cls):
        return YawRateControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="YawRate", low=[self.config.bounds[0]], high=[self.config.bounds[1]],
                                     unit=["deg/s"], description="Yaw Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    YawRateController, "Controller_YawRate", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


# ------ 2D Only --------


class CombinedTurnRateAccelerationControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[typing.List]] = [[-10, -96.5], [10, 96.5]]


class CombinedTurnRateAccelerationController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):

        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
        return CombinedTurnRateAccelerationControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name=f"TurnAcceleration", low=[bound for bound in self.config.bounds[0]],
                                     high=[bound for bound in self.config.bounds[1]],
                                     unit=["deg/s, ft/s/s"], description="Combined Turn Rate and Acceleration")
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedTurnRateAccelerationController, "Controller_TurnAcc", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


# --------- 3D Only ------------


class CombinedPitchRollAccelerationControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[typing.List]] = [[-5, -10, -96.5], [5, 10, 96.5]]


class CombinedPitchRollAccelerationController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):

        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
        return CombinedPitchRollAccelerationControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name=f"PitchRollAcc", low=[bound for bound in self.config.bounds[0]],
                                     high=[bound for bound in self.config.bounds[1]],
                                     unit=["deg/s, deg/s, ft/s/s"],
                                     description="Combined Pitch Rate, Roll Rate, and Acceleration")
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedPitchRollAccelerationController, "Controller_PitchRollAcc", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PitchRateControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[float]] = [-5, 5]


class PitchRateController(RateController):

    @classmethod
    def get_validator(cls):
        return PitchRateControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="PitchRate", low=[self.config.bounds[0]], high=[self.config.bounds[1]],
                                     unit=["deg/s"], description="Pitch Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    PitchRateController, "Controller_PitchRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RollRateControllerValidator(RateControllerValidator):
    bounds: typing.Optional[typing.List[float]] = [-10, 10]


class RollRateController(RateController):

    @classmethod
    def get_validator(cls):
        return RollRateControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="RollRate", low=[self.config.bounds[0]], high=[self.config.bounds[1]],
                                     unit=["deg/s"], description="Roll Rate")
        return control_props


PluginLibrary.AddClassToGroup(
    RollRateController, "Controller_RollRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
