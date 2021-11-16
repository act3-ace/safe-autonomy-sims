"""
Glue containing RTA module for filtering actions.
"""
import copy
import numbers
import typing

import gym
import numpy as np
from act3_rl_core.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from act3_rl_core.glues.base_wrapper import BaseWrapperGlue
from act3_rl_core.glues.common.controller_glue import ControllerGlue
from act3_rl_core.simulators.base_parts import BaseController, BaseControllerValidator
from act3_rl_core.simulators.base_platform import BasePlatform
from pydantic import PyObject


def flip_rta(control):
    """
    Simple filter which flips the sign of the input control.

    Parameters
    ----------
    control: dictionary of input controls

    Returns
    -------
        dict
            The filtered control dictionary
    """
    filtered_control = {}
    for k, v in control.items():
        filtered_control[k] = -v
    return filtered_control


class RTAGlueValidator(BaseMultiWrapperGlueValidator):
    """
    Validator for RTAGlue class.

    rta: RTA module which filters actions based on a safety function
    """
    rta: PyObject = flip_rta


class RTAGlue(BaseMultiWrapperGlue):
    """
    Glue containing RTA module for filtering actions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.controller_glues = []
        self.storage_platform = StoragePlatform(platform_name="rta_storage", platform=None, parts_list=[])
        storage_controller_glue_list = self._get_replacement_storage_glues(self, parent=None)
        self.storage_platform._controllers = [glue._controller for glue in storage_controller_glue_list]

    @classmethod
    def get_validator(cls):
        return RTAGlueValidator

    def action_space(self) -> gym.spaces.Space:
        action_space_dict = {}
        action_spaces = [glue.action_space() for glue in self.glues()]
        action_space_dict[self.config.name] = gym.spaces.tuple.Tuple(tuple(action_spaces))
        return gym.spaces.Dict(action_space_dict)

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict]) -> None:
        assert isinstance(action, dict)
        action = next(iter(action.values()))
        for i in range(len(self.glues())):
            self.glues()[i].apply_action(action[i])
        filtered_action = self.rta(self.storage_platform.stored_action)
        for controller_glue in self.controller_glues:
            controller_glue.apply_action(filtered_action)

    def observation_space(self) -> gym.spaces.Space:
        return None

    def get_observation(self) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        return None

    def _get_replacement_storage_glues(self, glue, parent):
        storage_glues = []
        if isinstance(glue, ControllerGlue):
            storage_glues.append(self._replace_controller_glue(glue, parent=parent))
        else:
            wrapped_list = list(glue.config.wrapped)
            for wrapped_glue in wrapped_list:
                storage_glues.extend(self._get_replacement_storage_glues(glue=wrapped_glue, parent=glue))
        return storage_glues

    def _replace_controller_glue(self, controller_glue, parent):
        self.controller_glues.append(controller_glue)
        replacement = copy.deepcopy(controller_glue)
        replacement_properties = controller_glue._controller.control_properties  # pylint: disable=W0212
        replacement._controller = StorageController(  # pylint: disable=W0212
            properties=replacement_properties, parent_platform=self.storage_platform, config={"key": controller_glue.key}
        )
        if isinstance(parent, BaseWrapperGlue):
            parent.config.wrapped = replacement
        else:
            parent.config.wrapped = [
                wrapped_glue if wrapped_glue is not controller_glue else replacement for wrapped_glue in parent.config.wrapped
            ]
        return replacement

    @property
    def rta(self):
        """
        RTA function attached to glue

        Returns
        -------
            RTA function attached to glue.
        """
        return self.config.rta


class StorageControllerValidator(BaseControllerValidator):
    """
    Validator for StorageController class

    key: string key value corresponding to stored action location in StoragePlatform
    """
    key: str


class StorageController(BaseController):
    """
    Basic storage controller class which mimics a platform controller and stores an action on a storage platform.
    """

    def __init__(self, properties, parent_platform, config, exclusiveness=set()):  # pylint: disable=W0102,W0231
        self.config = self.get_validator()(**config)
        self._properties = properties
        self._parent_platform = parent_platform
        self.exclusiveness = exclusiveness

    @classmethod
    def get_validator(cls):
        return StorageControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.stored_action[self.key] = control

    def get_applied_control(self) -> typing.Union[np.ndarray, numbers.Number]:
        return self.parent_platform.stored_action[self.key]

    @property
    def key(self):
        """
        Storage controller key.

        Returns
        -------
            str
                Storage controller key.
        """
        return self.config.key


class StoragePlatform(BasePlatform):
    """
    Platform which stores a (compound) action which can later be read.
    """

    def __init__(self, platform_name, platform, parts_list):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list)
        self.stored_action = {}

    @property
    def operable(self) -> bool:
        return False

    def _get_part_list(self, part_class_list, part_base_class):
        part_list = [part_class for part_class in part_class_list if isinstance(part_class, part_base_class)]
        return part_list
