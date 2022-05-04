"""
This module contains functions that define common terminal conditions across environments.
"""

from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name


class TimeoutDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the TimeoutDoneFunction config contains the max_sim_time value.
    """
    max_sim_time: float


class TimeoutDoneFunction(DoneFuncBase):
    """
    A done function that determines if the max episode time has been reached.
    """

    def __init__(self, **kwargs) -> None:
        self.config: TimeoutDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        TimeoutDoneValidator
            config validator for the TimeoutDoneValidator.
        """
        return TimeoutDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Parameters
        ----------
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
            dictionary containing the done condition for the current agent.
        """

        done = DoneDict()

        # get sim time
        platform = get_platform_by_name(next_state, self.agent)
        sim_time = platform.sim_time

        done[self.agent] = sim_time >= self.config.max_sim_time

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        self._set_all_done(done)
        return done
