"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements functions that define the terminal conditions
for the inspection environment.
"""
import typing
from collections import OrderedDict

import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes, SharedDoneFuncBase, SharedDoneFuncBaseValidator
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict

from safe_autonomy_sims.utils import get_closest_fft_distance


class SuccessfulInspectionDoneValidator(DoneFuncBaseValidator):
    """
    A configuration validator for the SuccessfulInspectionDoneValidator.

    Attributes
    ----------
    inspection_entity_name: str
        The name of the entity under inspection.
    weight_threshold : float
        Points score value indicating success.
        By default None, so success occurs when all points are inspected
    """
    inspection_entity_name: str = "chief"
    weight_threshold: typing.Union[float, None] = None


class SuccessfulInspectionDoneFunction(DoneFuncBase):
    """
    A done function that determines if the deputy has successfully
    inspected the chief.

    Attributes
    ----------
    config: SuccessfulInspectionDoneValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulInspectionDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

        Returns
        -------
        SuccessfulInspectionDoneValidator
            Config validator for the SuccessfulInspectionDoneFunction.
        """
        return SuccessfulInspectionDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> bool:
        """
        Parameters
        ----------
        observation : OrderedDict
            the current observation
        action : OrderedDict
            the current action
        next_observation : OrderedDict
            the incoming observation
        next_state : StateDict
            the incoming state
        observation_space: StateDict
            the observation space
        observation_units: StateDict
            the observation units

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        if self.config.weight_threshold is not None:
            weight = next_state.inspection_points_map[self.config.inspection_entity_name].get_total_weight_inspected()
            done_check = weight >= self.config.weight_threshold
        else:
            inspection_points = next_state.inspection_points_map[self.config.inspection_entity_name]
            done_check = all(inspection_points.points_inspected_dict.values())

        done = bool(done_check)
        if done:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN

        return done


class SafeSuccessfulInspectionDoneValidator(SuccessfulInspectionDoneValidator):
    """
    A configuration validator for the SafeSuccessfulInspectionDoneFunction.

    Attributes
    ----------
    mean_motion : float
        orbital mean motion in rad/s of current Hill's reference frame
    crash_region_radius : float
        The radius of the crashing region in meters.
    fft_time_step : float
        Time step to compute the FFT trajectory. FFT is computed for 1 orbit.
    """
    mean_motion: float
    crash_region_radius: float
    fft_time_step: float = 1


class SafeSuccessfulInspectionDoneFunction(SuccessfulInspectionDoneFunction):
    """
    A done function that determines if the deputy has successfully
    inspected the chief.

    Considers if a Free Flight Trajectory once the episode ends
    **would not** result in a collision.

    Attributes
    ----------
    config: SafeSuccessfulInspectionDoneValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: SafeSuccessfulInspectionDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

        Returns
        -------
        SafeSuccessfulInspectionDoneValidator
            Config validator for the SafeSuccessfulInspectionDoneFunction."""
        return SafeSuccessfulInspectionDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> bool:
        """
        Parameters
        ----------
        observation : OrderedDict
            the current observation
        action : OrderedDict
            the current action
        next_observation : OrderedDict
            the incoming observation
        next_state : StateDict
            the incoming state
        observation_space: StateDict
            the observation space
        observation_units: StateDict
            the observation units

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        done = super().__call__(observation, action, next_observation, next_state, observation_space, observation_units)

        if done:
            pos = next_state.sim_platforms[self.config.platform_name].position
            vel = next_state.sim_platforms[self.config.platform_name].velocity
            state = np.concatenate((pos, vel))
            n = self.config.mean_motion
            times = np.arange(0, 2 * np.pi / n, self.config.fft_time_step)
            dist = get_closest_fft_distance(state, self.config.mean_motion, times)
            if dist >= self.config.crash_region_radius:
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN
            else:
                # TODO: why is done set to False if deputy is within crash radius? Would crash not end episode?
                done = False

        return done


class CrashAfterSuccessfulInspectionDoneFunction(SuccessfulInspectionDoneFunction):
    """
    A done function that determines if the deputy has successfully
    inspected the chief.

    Considers if a Free Flight Trajectory once the episode ends
    **would** result in a collision.

    Attributes
    ----------
    config: SafeSuccessfulInspectionDoneValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: SafeSuccessfulInspectionDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

        Returns
        -------
        SafeSuccessfulInspectionDoneValidator
            Config validator for the CrashAfterSuccessfulInspectionDoneFunction.
        """
        return SafeSuccessfulInspectionDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> bool:
        """
        Parameters
        ----------
        observation : OrderedDict
            the current observation
        action : OrderedDict
            the current action
        next_observation : OrderedDict
            the incoming observation
        next_state : StateDict
            the incoming state
        observation_space: StateDict
            the observation space
        observation_units: StateDict
            the observation units

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        done = super().__call__(observation, action, next_observation, next_state, observation_space, observation_units)

        if done:
            pos = next_state.sim_platforms[self.config.platform_name].position
            vel = next_state.sim_platforms[self.config.platform_name].velocity
            state = np.concatenate((pos, vel))
            n = self.config.mean_motion
            times = np.arange(0, 2 * np.pi / n, self.config.fft_time_step)
            dist = get_closest_fft_distance(state, self.config.mean_motion, times)
            if dist < self.config.crash_region_radius:
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
            else:
                done = False

        return done


class MultiagentSuccessfulInspectionDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    A configuration validator for the MultiagentSuccessfulDockingDoneFunction.

    Attributes
    ----------
    inspection_entity_name: str
        The name of the entity under inspection.
    weight_threshold : float
        Points score value indicating success.
        By default None, so success occurs when all points are inspected
    """
    inspection_entity_name: str = "chief"
    weight_threshold: typing.Union[float, None] = None


class MultiagentSuccessfulInspectionDoneFunction(SharedDoneFuncBase):
    """
    A done function which determines whether every agent in the
    environment has reached a specified successful done condition.

    Attributes
    ----------
    config: MultiagentSuccessfulInspectionDoneFunctionValidator
        The function's validated configuration parameters
    """

    @staticmethod
    def get_validator() -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Returns
        -------
        MultiagentSuccessfulDockingDoneFunctionValidator
            Config validator for the MultiagentSuccessfulInspectionDoneFunction.
        """
        return MultiagentSuccessfulInspectionDoneFunctionValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
        local_dones: DoneDict,
        local_done_info: OrderedDict
    ) -> DoneDict:
        """
        Parameters
        ----------
        observation : OrderedDict
            the current observation
        action : OrderedDict
            the current action
        next_observation : OrderedDict
            the incoming observation
        next_state : StateDict
            the incoming state
        observation_space: StateDict
            the observation space
        observation_units: StateDict
            the observation units

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        done = DoneDict()

        if self.config.weight_threshold is not None:
            weight = next_state.inspection_points_map[self.config.inspection_entity_name].get_total_weight_inspected()
            done_check = weight >= self.config.weight_threshold
        else:
            inspection_points = next_state.inspection_points_map[self.config.inspection_entity_name]
            done_check = all(inspection_points.points_inspected_dict.values())

        if done_check:
            for k in local_dones.keys():
                done[k] = True

        return done
