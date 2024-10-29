"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains the base Simulator class used by the safe_autonomy_sims team's CWH and Dubins simulators.
"""
# pylint: disable=W0123,W0611,W0221

import abc
import typing
from graphlib import CycleError, TopologicalSorter

from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorResetValidator, BaseSimulatorValidator
from corl.simulators.base_simulator_state import BaseSimulatorState
from pydantic import BaseModel, validator
from pydantic.types import PyObject
from safe_autonomy_simulation.entities import Entity

from safe_autonomy_sims.simulators.initializers.initializer import (
    Accessor,
    EntityAttributeAccessor,
    SimAttributeAccessor,
    StripUnitsInitializer,
)
from safe_autonomy_sims.utils import KeyCollisionError


class SafeRLSimulatorValidator(
    BaseSimulatorValidator,
):
    """
    A validator for the SafeRLSimulator config.
    """


class InitializerResetValidator(BaseModel):
    """
    A configuration validator for the Initializer reset config.

    Attributes
    ----------
    functor: str
        The class module of the Initializer to be instantiated.
    config: dict
        The dict of values to be passed to the Initializer's construtor.
    """
    functor: PyObject
    config: typing.Dict = {}


class AgentResetParamsValidator(BaseModel):
    """A configuration validator for Agent platform reset parameters

    Attributes
    ----------
    initializer: InitializerResetValidator
        params for bulding an initializer object
    config: Dict
        Dictionary of initialization parameters to pass to the entity's initializer/constructor
    """
    initializer: typing.Optional[InitializerResetValidator] = None
    config: typing.Dict = {}


class AdditionalEntityValidator(BaseModel):
    """A configuration validator for additional sim entities

    Attributes
    ----------
    platform: str
        Name of platform type that can be found in sim's platform map
        Uses this value to extract the entity class associated with this platform
        Do no use with entity_class parameter. Use only one.
    entity_class: PyObject or str
        Python class of entity object. Can be specified using python identifier string
        Do not use with platform parameter. Use only one.
    initializer: InitializerResetValidator
        params for bulding an initializer object
    config: Dict
        Dictionary of initialization parameters to pass to the entity's initializer/constructor
    """
    platform: typing.Optional[str] = None
    entity_class: typing.Optional[PyObject] = None
    initializer: typing.Optional[InitializerResetValidator] = None
    config: typing.Dict[str, typing.Any] = {}

    @validator("entity_class")
    def entity_class_platform_xor(cls, entity_class, values):
        """validator to make sure one and exactly one of platform or entity_class is used"""
        platform = values['platform']
        if entity_class is None and platform is None:
            raise ValueError("Must specify platform or entity_class")
        if entity_class is not None and platform is not None:
            raise ValueError("Do not specify both platform and entity_class. Use one parameter only.")
        return entity_class


class SafeRLSimulatorResetValidator(BaseSimulatorResetValidator):
    """
    A configuration validator for the SafeRLSimulator reset method.

    Attributes
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
    # StripUnitsInitializer is the default as package dependencies could be using Pint and CoRL doesn't
    # use Pint
    default_initializer: InitializerResetValidator = InitializerResetValidator(functor=StripUnitsInitializer)
    additional_entities: typing.Dict[str, AdditionalEntityValidator] = {}
    init_state: typing.Optional[BaseModel] = None


class SafeRLSimulatorState(BaseSimulatorState):
    """
    The basemodel for the state of the SafeRLSimulator.

    Attributes
    ----------
    dict
        simulation entities
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

    def __init__(self, **kwargs: dict[str, typing.Any]):
        self.config: SafeRLSimulatorValidator
        super().__init__(**kwargs)
        self.agent_sim_entities: typing.Dict[str, Entity] = {}
        self.additional_sim_entities: typing.Dict[str, Entity] = {}
        self.platform_map = self._construct_platform_map()
        self.sim_entities: typing.Dict[str, Entity] = {}
        self.clock = 0.0
        self.last_entity_actions: typing.Dict[typing.Any, typing.Any] = {}
        self.sim_platforms = self.construct_platforms()
        self.init_state = None

        self._state: SafeRLSimulatorState

    def reset(self, config):
        reset_config = self.get_reset_validator(**config)
        self.init_state = reset_config.init_state

        self.clock = 0.0
        self.last_entity_actions = {}

        entity_init_map = self._construct_entity_init_map(reset_config)
        self.sim_entities = self._construct_sim_entities(reset_config, entity_init_map)
        self.sim_platforms = self.construct_platforms()

        self._state = self._construct_simulator_state()
        self.update_sensor_measurements()
        return self._state

    def _construct_entity_init_map(self, reset_config: SafeRLSimulatorResetValidator):
        entity_init_map = {}
        for agent_id, agent_config in self.config.agent_configs.items():
            platform_config = agent_config.platform_config
            sim_config_kwargs = platform_config.get("kwargs", {})

            if reset_config.platforms is None:
                agent_reset_config = AgentResetParamsValidator()
            else:
                agent_reset_config = AgentResetParamsValidator(**reset_config.platforms.get(agent_id, {}))

            agent_init_params = {**sim_config_kwargs, **agent_reset_config.config}

            entity_class = self.platform_map[platform_config.get('platform', 'default')][0]

            if agent_reset_config.initializer is None:
                entity_initializer = reset_config.default_initializer.functor(reset_config.default_initializer.config)
            else:
                entity_initializer = agent_reset_config.initializer.functor(agent_reset_config.initializer.config)

            entity_init_map[agent_id] = {
                'class': entity_class,
                'initializer': entity_initializer,
                'params': agent_init_params,
            }

        for entity_name, entity_config in reset_config.additional_entities.items():
            if entity_name in entity_init_map:
                KeyCollisionError(entity_name, f"additional entity name collision: '{entity_name}' is used twice")

            if entity_config.platform is not None:
                entity_class = self.platform_map[entity_config.platform][0]
            else:
                entity_class = entity_config.entity_class

            if entity_config.initializer is None:
                entity_initializer = reset_config.default_initializer.functor(reset_config.default_initializer.config)
            else:
                entity_initializer = entity_config.initializer.functor(entity_config.initializer.config)

            entity_init_map[entity_name] = {
                'class': entity_class,
                'initializer': entity_initializer,
                'params': entity_config.config,
            }

        entity_init_map = self._construct_entity_accessors(entity_init_map)

        return entity_init_map

    def _construct_entity_accessors(self, entity_init_map):
        for _, entity_map_items in entity_init_map.items():
            for param_name, param in entity_map_items['params'].items():
                if isinstance(param, dict) and "accessor" in param:
                    accessor_key = param["accessor"]
                    if accessor_key == "entity_attribute":
                        accessor_obj = EntityAttributeAccessor(**param.get('config', {}))
                    elif accessor_key == "sim_attribute":
                        accessor_obj = SimAttributeAccessor(**param.get('config', {}))
                    else:
                        raise ValueError(f"Invalid accessor {accessor_key}")
                    entity_map_items['params'][param_name] = accessor_obj

        return entity_init_map

    def _generate_entity_init_order(self, entity_init_map) -> typing.List[str]:

        # key is node name, value is set of predecessors (i.e. dependencies)
        dependency_graph: typing.Dict[str, typing.Set] = {entity_name: set() for entity_name in entity_init_map}

        for entity_name, entity_map_items in entity_init_map.items():
            for _, param in entity_map_items['params'].items():
                if isinstance(param, Accessor):
                    dependency_graph[entity_name] = dependency_graph[entity_name].union(param.dependencies)

        try:
            sorter = TopologicalSorter(graph=dependency_graph)
            init_order = list(sorter.static_order())
        except CycleError as e:
            raise ValueError(
                "Cycle detected among entity initializer accessors."
                "Make sure that entities with accessors don't depend on each other"
            ) from e

        return init_order

    def _resolve_accessors(self, params, sim_entities):
        for param_name, param in params.items():
            if isinstance(param, Accessor):
                params[param_name] = param.access(self, sim_entities)

        return params

    @abc.abstractmethod
    def _construct_platform_map(self) -> dict:
        ...

    def _construct_simulator_state(self) -> dict:
        return SafeRLSimulatorState(sim_platforms=self.sim_platforms, sim_time=self.clock, sim_entities=self.sim_entities)

    def _construct_sim_entities(
        self,
        reset_config: SafeRLSimulatorResetValidator,  # pylint: disable=unused-argument
        entity_init_map: typing.Dict[str, typing.Dict]
    ) -> typing.Dict[str, Entity]:
        """Constructs the simulator

        Parameters
        ----------
        reset_config : SafeRLSimulatorResetValidator
            reset config validator
        entity_init_map: dict
            map of per entity initialization items. Each entity name is a key with a 'class', 'initializer', and 'params' nested dict items

        Returns
        -------
        dict[str: sim_entity]
            Dictionary mapping entity id to simulation backend entity.
        """

        init_order = self._generate_entity_init_order(entity_init_map)

        sim_entities: typing.Dict[str, typing.Any] = {}
        for entity_name in init_order:
            entity_init_items = entity_init_map[entity_name]
            params = entity_init_items['params']
            params = self._resolve_accessors(params, sim_entities)
            transformed_params = entity_init_items['initializer'](**params)
            entity = entity_init_items['class'](name=entity_name, **transformed_params)
            sim_entities[entity_name] = entity

        self.agent_sim_entities = {}
        self.additional_sim_entities = {}
        for entity_name, entity in sim_entities.items():
            if entity_name in self.config.agent_configs:
                self.agent_sim_entities[entity_name] = entity
            else:
                self.additional_sim_entities[entity_name] = entity

        return sim_entities

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

    def mark_episode_done(self, done_info: typing.OrderedDict, episode_state: typing.OrderedDict, metadata: dict | None = None):
        """
        Takes in the done_info specifying how the episode completed
        and does any book keeping around ending an episode

        Parameters
        ----------
        done_info : OrderedDict
            The Dict describing which Done conditions ended an episode
        episode_state : OrderedDict
            The episode state at the end of the simulation
        """

    def save_episode_information(self, dones, rewards, observations, observation_units):
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

            if action is not None:
                entity.add_control(action)

            entity.step(step_size=step_size)

    def _step_get_entity_actions(
        self,
        step_size: float,  # pylint: disable = unused-argument
        platforms_to_action: typing.Set[str]
    ) -> typing.Dict:
        entity_actions = {}
        for platform in self._state.sim_platforms.values():
            platform_id = platform.name
            if platform_id in platforms_to_action:
                action = platform.get_applied_action().m
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
