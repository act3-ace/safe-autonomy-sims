import numpy as np

from act3_rl_core.simulators.base_parts import BaseController
from act3_rl_core.libraries.property import MultiBoxProp, Prop
from act3_rl_core.simulators.base_platform import BasePlatform
from act3.core.plugins.plugin_library import PluginLibrary
from simulators.cwh_available_platforms import CWHAvailablePlatformTypes
from simulators.cwh_simulator import CWHSimulator


class CWHController(BaseController):
    def __init__(
            self,
            parent_platform,  # type: ignore # noqa: F821
            config,
    ):
        self.config = config
        self._parent_platform = parent_platform

    @property
    def name(self):
        return self.config["name"] + self.__class__.__name__

    def parent_platform(self) -> 'BasePlatform':
        return self._parent_platform

    def control_properties(self) -> Prop:
        raise NotImplementedError

    def apply_control(self, control: np.ndarray) -> None:
        raise NotImplementedError

    def get_applied_control(self) -> np.ndarray:
        raise NotImplementedError


class ThrustController(CWHController):
    def __init__(
            self,
            parent_platform,  # type: ignore # noqa: F821
            config,
    ):
        super().__init__(parent_platform, config)

    @property
    def control_properties(self) -> Prop:
        return MultiBoxProp(
            name=self.config["name"],
            low=[-1],
            high=[1],
            unit=["newtons"],
            description="Thrust"
        )

    def apply_control(self, control: np.ndarray) -> None:
        self._parent_platform.next_action[self.config["axis"]] = control

    def get_applied_control(self) -> np.ndarray:
        return np.array([self._parent_platform.next_action[self.config["axis"]]], dtype=np.float32)

PluginLibrary.AddClassToGroup(
    ThrustController, "Controller_Thrust", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
