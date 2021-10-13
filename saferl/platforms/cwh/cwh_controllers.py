import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import (BaseController,
                                                BaseControllerValidator)
from saferl.platforms.cwh.cwh_available_platforms import \
    CWHAvailablePlatformTypes
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
    ):
        control_props = MultiBoxProp(name="",
                                     low=[-1],
                                     high=[1],
                                     unit=["newtons"],
                                     description="Thrust")
        super().__init__(control_properties=control_props,
                         parent_platform=parent_platform,
                         config=config)
        self.control_properties.name = self.config.name

    @classmethod
    def get_validator(cls):
        return ThrustControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.next_action[self.config.axis] = control

    def get_applied_control(self) -> np.ndarray:
        return np.array([self._parent_platform.next_action[self.config.axis]],
                        dtype=np.float32)


PluginLibrary.AddClassToGroup(ThrustController, "Controller_Thrust", {
    "simulator": CWHSimulator,
    "platform_type": CWHAvailablePlatformTypes.CWH
})
