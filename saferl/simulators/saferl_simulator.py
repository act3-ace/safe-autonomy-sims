import abc

import numpy as np
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.simulators.base_simulator import BaseSimulator, BaseSimulatorValidator


class SafeRLSimulatorValidator(BaseSimulatorValidator):
    step_size: float


class SafeRLSimulator(BaseSimulator):

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
    def get_sim_entities(self):
        ...

    @abc.abstractmethod
    def get_platforms(self):
        ...

    @abc.abstractmethod
    def reset_sim_entities(self, config):
        ...

    def update_sensor_measurements(self):
        """
        Update and caches all the measurements of all the sensors on each platform
        """
        for plat in self._state.sim_platforms:
            for sensor in plat.sensors:
                sensor.calculate_and_cache_measurement(state=self._state.sim_platforms)

    def mark_episode_done(self, done_string: str):
        pass

    def save_episode_information(self, **kwargs):
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
