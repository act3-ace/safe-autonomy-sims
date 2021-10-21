import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh.cwh_simulator import CWHSimulator


class CWHController(BaseController):

    @property
    def name(self):
        return self.config.name + self.__class__.__name__

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


class ThrustControllerValidator(BaseControllerValidator):
    axis: int


class ThrustController(CWHController):

    def __init__(
        self,
        parent_platform,  # type: ignore # noqa: F821
        config,
        control_properties=cwh_props.ThrustProp,
        exclusiveness=set()
    ):  # pylint: disable=W0102
        super().__init__(control_properties=control_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness)

    @classmethod
    def get_validator(cls):
        return ThrustControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(action=control, axis=self.config.axis)

    def get_applied_control(self) -> np.ndarray:
        return np.array([self.parent_platform.get_applied_action()[self.config.axis]], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    ThrustController, "Controller_Thrust", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
