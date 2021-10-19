import numpy as np
from act3_rl_core.libraries.property import MultiBoxProp, Prop
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator
from act3_rl_core.libraries.plugin_library import PluginLibrary

from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes


class DubinsController(BaseController):

    @property
    def name(self):
        return self.config.name + self.__class__.__name__

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


class CombinedTurnVelocityRateController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):

        super().__init__(parent_platform=parent_platform, config=config)

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name=f"TurnAcceleration", low=[np.deg2rad(-6), -10], high=[np.deg2rad(6), 10],
                                     unit=["rad/s, m/s/s"], description="Combined Turn Rate and Acceleration")
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    CombinedTurnVelocityRateController, "Controller_TurnAcc", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector

class AxisControllerValidator(BaseControllerValidator):
    axis: int


class AccelerationController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):
        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
        return AxisControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="Acceleration", low=[-10], high=[10],
                                     unit=["m/s/s"], description="Acceleration")
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    AccelerationController, "Controller_Acceleration", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class TurnRateController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
    ):
        super().__init__(parent_platform=parent_platform, config=config)

    @classmethod
    def get_validator(cls):
        return AxisControllerValidator

    @property
    def control_properties(self) -> Prop:
        control_props = MultiBoxProp(name="TurnRate", low=[np.deg2rad(-6)], high=[np.deg2rad(6)],
                                     unit=["rad/s"], description="Turn Rate")
        return control_props

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    TurnRateController, "Controller_TurnRate", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
