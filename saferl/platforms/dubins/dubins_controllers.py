import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator

import saferl.platforms.dubins.dubins_properties as dubins_props
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsController(BaseController):

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


# TODO: Find a better way to save actions to platform instead of storing an axis in the saved vector


class RateControllerValidator(BaseControllerValidator):
    axis: int


class RateController(DubinsController):

    @classmethod
    def get_validator(cls):
        return RateControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


class AccelerationController(RateController):

    def __init__(self, parent_platform, config, control_properties=dubins_props.AccelerationProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)


PluginLibrary.AddClassToGroup(
    AccelerationController,
    "Controller_Acceleration", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class YawRateController(RateController):

    def __init__(self, parent_platform, config, control_properties=dubins_props.YawRateProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)


PluginLibrary.AddClassToGroup(
    YawRateController, "Controller_YawRate", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)

# ------ 2D Only --------


class CombinedTurnRateAccelerationController(DubinsController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
        control_properties=dubins_props.YawAndAccelerationProp,
        exclusiveness=set()
    ):  # pylint: disable=W0102

        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)

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

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
        control_properties=dubins_props.PitchRollAndAccelerationProp,
        exclusiveness=set()
    ):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)

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

    def __init__(self, parent_platform, config, control_properties=dubins_props.PitchRateProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)


PluginLibrary.AddClassToGroup(
    PitchRateController, "Controller_PitchRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RollRateController(RateController):

    def __init__(self, parent_platform, config, control_properties=dubins_props.RollRateProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)


PluginLibrary.AddClassToGroup(
    RollRateController, "Controller_RollRate", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
