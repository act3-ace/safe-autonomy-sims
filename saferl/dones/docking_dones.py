"""
Functions that define the terminal conditions for the Docking Environment.
This in turn defines whether the end is episode is reached or not.
"""

import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the max_distance data needed for
    computations in the MaxDistanceDoneFucntion.
    """

    max_distance: float


class MaxDistanceDoneFunction(DoneFuncBase):
    """
    A done function that determines if the max distance has been traveled or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: MaxDistanceDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        MaxDistanceDoneValidator
            config validator for the MaxDistanceDoneFucntion

        """
        return MaxDistanceDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
            done : DoneDict
                dictionary containing the condition condition for the current agent

        """

        done = DoneDict()

        # compute distance to origin
        platform = get_platform_by_name(next_state, self.agent)
        position = platform.position

        # compute to origin
        origin = np.array([0, 0, 0])
        dist = np.linalg.norm(origin - np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: float
    velocity_limit: float


class SuccessfulDockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulDockingDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return SuccessfulDockingDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """
        # eventually will include velocity constraint
        done = DoneDict()
        deputy = get_platform_by_name(next_state, self.agent)

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(deputy.position) - origin)

        # add constraint for velocity
        in_docking_region = radial_distance <= docking_region_radius
        within_limit = np.linalg.norm(np.array(deputy.velocity)) <= self.config.velocity_limit

        if in_docking_region and within_limit:
            done[self.agent] = True
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN
        elif in_docking_region and not within_limit:
            done[self.agent] = True
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        else:
            done[self.agent] = False
        return done


class DockingVelocityLimitDoneFunctionValidator(DoneFuncBaseValidator):
    """
    Validator for the DockingVelocityLimitDoneFunction

    Attributes
    ----------
    velocity_limit : float
        the velocity limit constraint that the deputy cannot exceed
    """
    velocity_limit: float


class DockingVelocityLimitDoneFunction(DoneFuncBase):
    """
    This done fucntion determines whether the velocity limit has been exceeded or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingVelocityLimitDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        DockingVelocityLimitDoneFunctionValidator : Done Function
            done function for the DockingVelocityLimitDoneFunction
        """
        return DockingVelocityLimitDoneFunctionValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent
        """

        done = DoneDict()

        deputy = get_platform_by_name(next_state, self.agent)

        curr_vel_mag = np.linalg.norm(deputy.velocity)

        done[self.agent] = curr_vel_mag > self.config.velocity_limit

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class DockingRelativeVelocityConstraintDoneFunctionValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains essential peices of data for the done function
    """
    constraint_velocity: float
    target: str


# needs a reference object
class DockingRelativeVelocityConstraintDoneFunction(DoneFuncBase):
    """
    A done function that checks if the docking velocity relative to a target object has exceeded a certain specified threshold velocity.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingRelativeVelocityConstraintDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        DockingRelativeVelocityConstraintDoneFunctionValidator : DoneFunctionValidator
        """

        return DockingRelativeVelocityConstraintDoneFunctionValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent
        """
        # eventually will include velocity constraint
        done = DoneDict()
        # platform = get_platform_name(next_state,self.agent)
        deputy = get_platform_by_name(next_state, self.agent)
        target = get_platform_by_name(next_state, self.config.target)
        # pos = platform.position

        curr_vel_mag = np.linalg.norm(np.array(deputy.velocity) - np.array(target.velocity))

        done[self.agent] = curr_vel_mag > self.config.constraint_velocity

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


# if __name__ == "__main__":
#     from act3_rl_core.libraries.state_dict import StateDict
#
#     import saferl.platforms.cwh.cwh_controllers as c
#     import saferl.platforms.cwh.cwh_sensors as s
#     from saferl.platforms.cwh.cwh_platform import CWHPlatform
#     from saferl_sim.cwh.cwh import CWHSpacecraft
#
#     agent_name = "blue0"
#     lead_name = "lead"
#     cut_name = "MaxDistanceDone"
#     max_distance = 10000
#
#     observation = np.array([0, 0, 0])
#     action = np.array([0, 0, 0])
#     next_observation = np.array([0, 0, 0])
#
#     aircraft = CWHSpacecraft
#     aircraft_config = [
#         (c.ThrustController, {
#             'axis': 0
#         }), (c.ThrustController, {
#             'axis': 1
#         }), (c.ThrustController, {
#             'axis': 2
#         }), (s.PositionSensor, {}), (s.VelocitySensor, {})
#     ]
#     platform = CWHPlatform(platform_name=agent_name, platform=aircraft(name=agent_name, x=10001), platform_config=aircraft_config)
#
#     state = StateDict({"episode_state": {agent_name: {cut_name: None}}, "sim_platforms": [platform]})
#
#     max_dist_done = MaxDistanceDoneFunction(agent_name=agent_name, name=cut_name, max_distance=max_distance)
#     done_dict = max_dist_done(observation=observation, action=action, next_observation=next_observation, next_state=state)
#
#     print(done_dict)
