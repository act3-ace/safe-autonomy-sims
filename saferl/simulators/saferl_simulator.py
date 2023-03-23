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
from corl.libraries.units import ValueWithUnits
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorResetValidator, BaseSimulatorValidator
from corl.simulators.base_simulator_state import BaseSimulatorState
from pydantic import BaseModel, PyObject
from safe_autonomy_dynamics.base_models import BaseEntity

from saferl.utils import KeyCollisionError, shallow_dict_merge


class SafeRLSimulatorValidator(
    BaseSimulatorValidator,
):
    """
    A validator for the SafeRLSimulator config.
    """
    ...


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


class SafeRLSimulatorState(BaseSimulatorState):
    """
    The basemodel for the state of the InspectionSimulator.

    points: dict
        The dictionary containing the points the agent needs to inspect.
        Keys: (x,y,z) tuple. Values: True if inspected, False otherwise.
    """
    sim_entities: typing.Dict


class SafeRLSimulator(BaseSimulator):
    """
    The base simulator class used by the CWH and Dubins simulators. SafeRLSimulator is responsible for
    initializing the platform objects for a simulation and knowing how to set up episodes based on input
    parameters from a parameter provider. It is also responsible for reporting the simulation state at
    each timestep.
    """

    @property
    def get_simulator_validator(self) -> typing.Type[SafeRLSimulatorValidator]:
        return SafeRLSimulatorValidator

    @property
    def get_reset_validator(self) -> typing.Type[SafeRLSimulatorResetValidator]:
        return SafeRLSimulatorResetValidator

    @property
    def step_size(self) -> float:
        """Simulator step size in seconds"""
        return 1 / self.frame_rate

    @property
    def sim_time(self) -> float:
        return self.clock

    @property
    def platforms(self) -> typing.List:
        return list(self._state.sim_platforms)

    def __init__(self, **kwargs):
        self.config: SafeRLSimulatorValidator
        super().__init__(**kwargs)
        self.agent_sim_entities: typing.Dict[str, BaseEntity] = {}
        self.additional_sim_entities: typing.Dict[str, BaseEntity] = {}
        self.platform_map = self._construct_platform_map()
        self.sim_entities = self._construct_sim_entities()
        self.clock = 0.0
        self.last_entity_actions = {}

        self._state: SafeRLSimulatorState = None

    def reset(self, config):
        config = self.get_reset_validator(**config)
        if config.initializer is not None:
            # if an initializer defined, pass agent reset configs through it
            initializer = config.initializer.functor(config=config.initializer.config)
            config.platforms = initializer(config.platforms)

        self.clock = 0.0
        self.last_entity_actions = {}
        self.sim_entities = self._construct_sim_entities(config)
        sim_platforms = self.construct_platforms()
        self._state = SafeRLSimulatorState(sim_platforms=sim_platforms, sim_time=self.clock, sim_entities=self.sim_entities)
        self.update_sensor_measurements()
        return self._state

    @abc.abstractmethod
    def _construct_platform_map(self) -> dict:
        ...

    def _construct_sim_entities(self, reset_config: SafeRLSimulatorResetValidator = None) -> typing.Dict[str, BaseEntity]:
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

        self.agent_sim_entities = self._construct_agent_sim_entities(reset_config.platforms)

        self.additional_sim_entities = self._construct_additional_sim_entities(reset_config)

        sim_entities = shallow_dict_merge(self.agent_sim_entities, self.additional_sim_entities, in_place=False, allow_collisions=False)

        return sim_entities

    def _construct_agent_sim_entities(self, platforms: dict = None) -> typing.Dict[str, BaseEntity]:
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
            platform_config = agent_config.platform_config
            sim_config_kwargs = platform_config.get("kwargs", {})

            if platforms is None:
                agent_reset_config = {}
            else:
                agent_reset_config = platforms.get(agent_id, {})

            entity_kwargs = {**sim_config_kwargs, **agent_reset_config}

            for key, val in entity_kwargs.items():
                if isinstance(val, ValueWithUnits):
                    entity_kwargs[key] = val.value

            entity_class = self.platform_map[platform_config.get('platform', 'default')][0]
            sim_entities[agent_id] = entity_class(name=agent_id, **entity_kwargs)

        return sim_entities

    def _construct_additional_sim_entities(self, reset_config: SafeRLSimulatorResetValidator) -> typing.Dict[str, BaseEntity]:
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
            if entity_name in entities:
                KeyCollisionError(entity_name, f"additional entity name collision: '{entity_name}' is used twice")
            entity_class = self.platform_map[entity_config.get('platform', 'default')][0]
            entities[entity_name] = entity_class(name=entity_name, **entity_config['config'])

        return entities

    def construct_platforms(self) -> dict:
        """
        Gets the platform object associated with each simulation entity.

        Returns
        -------
        tuple
            Collection of platforms associated with each simulation entity.
        """
        sim_platforms = {}
        for platform_id, entity in self.agent_sim_entities.items():
            agent_config = self.config.agent_configs[platform_id]
            platform_config = agent_config.platform_config
            platform_class = self.platform_map[platform_config.get('platform', 'default')][1]
            sim_platforms[platform_id] = platform_class(platform_name=platform_id, platform=entity, parts_list=agent_config.parts_list)
        return sim_platforms

    def update_sensor_measurements(self):
        """
        Update and cache all the measurements of all the sensors on each platform.
        """
        for plat in self._state.sim_platforms.values():
            for sensor in plat.sensors.values():
                sensor.calculate_and_cache_measurement(state=self._state)

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

    def step(self, platforms_to_action):
        step_size = self.step_size
        self._step_entity_state(step_size=step_size, platforms_to_action=platforms_to_action)
        self._step_update_time(step_size=step_size)
        self._step_update_sim_statuses(step_size=step_size)
        self.update_sensor_measurements()
        return self._state

    def _step_entity_state(self, step_size: float, platforms_to_action: typing.Set[str]):
        entity_actions = self._step_get_entity_actions(step_size=step_size, platforms_to_action=platforms_to_action)

        for entity_name, entity in self.sim_entities.items():
            action = entity_actions.get(entity_name, None)
            entity.step(action=action, step_size=step_size)

    def _step_get_entity_actions(
        self,
        step_size: float,  # pylint: disable = unused-argument
        platforms_to_action: typing.Set[str]
    ) -> typing.Dict:
        entity_actions = {}
        for platform in self._state.sim_platforms.values():
            platform_id = platform.name
            if platform_id in platforms_to_action:
                action = np.array(platform.get_applied_action(), dtype=np.float32)
                entity_actions[platform_id] = action
                self.last_entity_actions[platform_id] = action
            else:
                entity_actions[platform_id] = self.last_entity_actions[platform_id]
        return entity_actions

    def _step_update_time(self, step_size: float):
        self.clock += step_size
        self._state.sim_time = self.clock  # pylint: disable=W0201
        for platform in self._state.sim_platforms.values():
            platform.sim_time = self.clock

    def _step_update_sim_statuses(self, step_size: float):
        """perform custom updates on derived simulation status properties"""
