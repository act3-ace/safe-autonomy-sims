"""
This module contains the base Simulator class used by the saferl team's CWH and Dubins simulators.
"""

import abc

import numpy as np
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.simulators.base_simulator import BaseSimulator, BaseSimulatorValidator


class SafeRLSimulatorValidator(BaseSimulatorValidator):
    """
    step_size: A float representing how many simulated seconds pass each time the simulator updates
    """
    step_size: float


class SafeRLSimulator(BaseSimulator):
    """
    The base simulator class used by the CWH and Dubins simulators. SafeRLSimulator is responsible for
    initializing the platform objects for a simulation
    and knowing how to set up episodes based on input parameters from a parameter provider.
    It is also responsible for reporting the simulation state at each timestep.
    """

    @classmethod
    def get_simulator_validator(cls):
        return SafeRLSimulatorValidator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim_entities = self.get_sim_entities()
        self._state = StateDict()
        self.clock = 0.0

    def reset(self, config):
        self._state.clear()
        self.clock = 0.0
        self.reset_sim_entities(config)
        self._state.sim_platforms = self.get_platforms()
        self.update_sensor_measurements()
        return self._state

    @abc.abstractmethod
    def get_sim_entities(self) -> dict:
        """
        Gets the correct backend simulation entity for each agent.

        Returns
        -------
        dict[str: sim_entity]
            Dictionary mapping agent id to simulation backend entity.
        """
        ...

    @abc.abstractmethod
    def get_platforms(self) -> tuple:
        """
        Gets the platform object associated with each simulation entity.

        Returns
        -------
        tuple
            Collection of platforms associated with each simulation entity.
        """
        ...

    @abc.abstractmethod
    def reset_sim_entities(self, config):
        """
        Reset simulation entities to an initial state.
        """
        ...

    def update_sensor_measurements(self):
        """
        Update and cache all the measurements of all the sensors on each platform
        """
        for plat in self._state.sim_platforms:
            for sensor in plat.sensors:
                sensor.calculate_and_cache_measurement(state=self._state.sim_platforms)

    def mark_episode_done(self, done_string: str):
        pass

    def save_episode_information(self, dones, rewards, observations):
        pass

    def step(self):
        for platform in self._state.sim_platforms:
            agent_id = platform.name
            action = np.array(platform.get_applied_action(), dtype=np.float32)
            entity = self.sim_entities[agent_id]
            entity.step_compute(sim_state=None, action=action, step_size=self.config.step_size)
            entity.step_apply()
            platform.sim_time = self.clock
        self.update_sensor_measurements()
        self.clock += self.config.step_size
        return self._state
