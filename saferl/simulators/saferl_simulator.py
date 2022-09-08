"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains the base Simulator class used by the saferl team's CWH and Dubins simulators.
"""
# pylint: disable=W0123,W0611

import abc
import typing

import numpy as np
from corl.libraries.state_dict import StateDict
from corl.libraries.units import ValueWithUnits
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorResetValidator, BaseSimulatorValidator
from pydantic import BaseModel, PyObject

import saferl  # noqa: F401


class SafeRLSimulatorValidator(
    BaseSimulatorValidator,
):
    """
    A validator for the SafeRLSimulator config.

    step_size: float
        A float representing how many simulated seconds pass each time the simulator updates.
    """
    step_size: float


class InitializerResetValidator(BaseModel):
    """
    A validator for the Initializaer config.

    functor: str
        The class module of the Initializer to be instantiated.
    config: dict
        The dict of values to be passed to the Initializer's construtor.
    """
    functor: PyObject
    config: typing.Dict = {}


class SafeRLSimulatorResetValidator(BaseSimulatorResetValidator):
    """
    Validator for SafeRLSimulator reset configs.

    Parameters
    ----------
    platforms: dict
        Contains individual initialization dicts for each agent.
        Key is platform name, value is platform's initialization dict.
    initializer: InitializerResetValidator
        Optionally, the user can define the functor and config for a BaseInitializer class, which modifies
        the agent-specific initialization dicts found in platforms.
    additional_entities: dict
        Contains individual initialization dicts for additional, non-agent simulation entities
        Key is entity name, value is entity's initialization dict.
    """
    platforms: typing.Optional[typing.Dict[str, typing.Dict]] = {}
    initializer: typing.Optional[typing.Union[InitializerResetValidator, None]] = None
    additional_entities: typing.Dict[str, typing.Dict] = {}


class SafeRLSimulator(BaseSimulator):
    """
    The base simulator class used by the CWH and Dubins simulators. SafeRLSimulator is responsible for
    initializing the platform objects for a simulation and knowing how to set up episodes based on input
    parameters from a parameter provider. It is also responsible for reporting the simulation state at
    each timestep.
    """

    @property
    def get_simulator_validator(self):
        return SafeRLSimulatorValidator

    @property
    def get_reset_validator(self):
        return SafeRLSimulatorResetValidator

    @property
    def sim_time(self) -> float:
        return self.clock

    @property
    def platforms(self) -> typing.List:
        return list(self._state.sim_platforms)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform_map = self._construct_platform_map()
        self.sim_entities = self._construct_sim_entities()
        self._state = StateDict()
        self.clock = 0.0

    def reset(self, config):
        config = self.get_reset_validator(**config)
        if config.initializer is not None:
            # if an initializer defined, pass agent reset configs through it
            initializer = config.initializer.functor(config=config.initializer.config)
            config.platforms = initializer(config.platforms)
        self._state.clear()
        self.clock = 0.0
        self.sim_entities = self._construct_sim_entities(config)
        self._state.sim_platforms = self.construct_platforms()
        self.update_sensor_measurements()
        return self._state

    @abc.abstractmethod
    def _construct_platform_map(self) -> dict:
        ...

    def _construct_sim_entities(self, reset_config: SafeRLSimulatorResetValidator = None):
        """Constructs the simulator

        Parameters
        ----------
        reset_config : SafeRLSimulatorResetValidator
            reset config validator

        Returns
        -------
        dict[str: sim_entity]
            Dictionary mapping entity id to simulation backend entity.
        """
        if reset_config is None:
            reset_config = SafeRLSimulatorResetValidator()

        agent_sim_entities = self._construct_agent_sim_entities(reset_config.platforms)

        additional_sim_entities = self._construct_additional_sim_entities(reset_config)

        sim_entities = agent_sim_entities

        for entity_name, entity in additional_sim_entities.items():
            assert entity_name not in sim_entities, \
                f"entity name collision: '{entity_name}' is used for an agent entity and an additional entity"
            sim_entities[entity_name] = entity

        return sim_entities

    def _construct_agent_sim_entities(self, platforms: dict = None) -> dict:
        """
        Gets the correct backend simulation entity for each agent.

        Parameters
        ----------
        platforms: dict
            Platforms initialization entry from reset config containing initialization parameters for backend sim
            entities.

        Returns
        -------
        dict[str: sim_entity]
            Dictionary mapping agent id to simulation backend entity.
        """

        sim_entities = {}
        for agent_id, agent_config in self.config.agent_configs.items():
            sim_config = agent_config.sim_config
            sim_config_kwargs = sim_config.get("kwargs", {})

            if platforms is None:
                agent_reset_config = {}
            else:
                agent_reset_config = platforms.get(agent_id, {})

            entity_kwargs = {**sim_config_kwargs, **agent_reset_config}

            for key, val in entity_kwargs.items():
                if isinstance(val, ValueWithUnits):
                    entity_kwargs[key] = val.value

            entity_class = self.platform_map[sim_config.get('platform', 'default')][0]
            sim_entities[agent_id] = entity_class(name=agent_id, **entity_kwargs)

        return sim_entities

    def _construct_additional_sim_entities(self, reset_config: SafeRLSimulatorResetValidator):
        """Constructs the simulator

        Parameters
        ----------
        reset_config : SafeRLSimulatorResetValidator
            reset config validator

        Returns
        -------
        dict[str: sim_entity]
            Dictionary mapping entity id to simulation backend entity.
        """
        entities = {}

        for entity_name, entity_config in reset_config.additional_entities.items():
            assert entity_name not in entities, f"entity name collision: '{entity_name}' is used twice"
            entity_class = self.platform_map[entity_config.get('entity')][0]
            entities[entity_name] = entity_class(entity_config['config'])

        return entities

    def construct_platforms(self) -> tuple:
        """
        Gets the platform object associated with each simulation entity.

        Returns
        -------
        tuple
            Collection of platforms associated with each simulation entity.
        """
        sim_platforms = []
        for agent_id, entity in self.sim_entities.items():
            agent_config = self.config.agent_configs[agent_id]
            sim_config = agent_config.sim_config
            platform_config = agent_config.platform_config
            platform_class = self.platform_map[sim_config.get('platform', 'default')][1]
            sim_platforms.append(platform_class(platform_name=agent_id, platform=entity, platform_config=platform_config))
        return tuple(sim_platforms)

    def update_sensor_measurements(self):
        """
        Update and cache all the measurements of all the sensors on each platform.
        """
        for plat in self._state.sim_platforms:
            for sensor in plat.sensors:
                sensor.calculate_and_cache_measurement(state=self._state.sim_platforms)

    def mark_episode_done(self, done_info: typing.OrderedDict, episode_state: typing.OrderedDict):
        """
        Takes in the done_info specifying how the episode completed
        and does any book keeping around ending an episode

        Arguments:
            done_info {OrderedDict} -- The Dict describing which Done conditions ended an episode
            episode_state {OrderedDict} -- The episode state at the end of the simulation
        """

    def save_episode_information(self, dones, rewards, observations):
        pass

    def step(self):
        for platform in self._state.sim_platforms:
            agent_id = platform.name
            action = np.array(platform.get_applied_action(), dtype=np.float32)
            entity = self.sim_entities[agent_id]
            entity.step(action=action, step_size=self.config.step_size)
            platform.sim_time = self.clock
        self.update_sensor_measurements()
        self.clock += self.config.step_size
        return self._state
