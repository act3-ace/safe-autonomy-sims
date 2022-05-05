"""
This module implements Run Time Assurance for Clohessy-Wiltshire spacecraft
"""
from collections import OrderedDict
import typing

import numpy as np

from run_time_assurance.rta import RTAModule
from run_time_assurance.zoo.cwh.docking_3d import Docking3dExplicitSwitchingRTA, Docking3dImplicitOptimizationRTA
from saferl.core.glues.rta_glue import RTAGlue, RTAGlueValidator


# class RTAGlueCWHDocking2dValidator(RTAGlueValidator):
#     rta: 

class RTAGlueCWHDocking3d(RTAGlue):

    def _get_rta_state_vector(self, observation: typing.Dict) -> np.ndarray:
        position = observation['ObserveSensor_Sensor_Position']['direct_observation']
        velocity = observation['ObserveSensor_Sensor_Velocity']['direct_observation']
        state_vec = np.concatenate((position, velocity))
        return state_vec

    def _get_action_vector_from_action(self, action: tuple) -> np.ndarray:
        actions_ordered = []
        for controller_action in action:
            actions_ordered += list(controller_action.values())
        control_vector = np.concatenate(actions_ordered)
        return control_vector

    def _get_action_from_action_vector(self, combined_action_vector: np.ndarray) -> tuple:
        combined_action_left = combined_action_vector
        controller_action_spaces = self.controller_glue_action_space()
        action_list = []

        for controller_action_space in controller_action_spaces:
            controller_action = OrderedDict()
            for action_key, action_space in controller_action_space.items():
                action_length = np.prod(action_space.shape)
                action_value = combined_action_left[:action_length]
                combined_action_left = combined_action_left[action_length:]
                controller_action[action_key] = action_value
            action_list.append(controller_action)
        return tuple(action_list)

    def _instantiate_rta_module(self) -> RTAModule:
        # return Docking3dExplicitSwitchingRTA()
        return Docking3dImplicitOptimizationRTA()
