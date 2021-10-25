# Module saferl.simulators.cwh.cwh_simulator


## Classes

### CWHPlatformConfigValidator {: #CWHPlatformConfigValidator }

```python
class CWHPlatformConfigValidator(*, position: List[float], velocity: List[float])
```


------

#### Base classes {: #CWHPlatformConfigValidator-bases }

* `pydantic.BaseModel`


------

#### Methods {: #CWHPlatformConfigValidator-methods }

[**check_position_len**](#CWHPlatformConfigValidator.check_position_len){: #CWHPlatformConfigValidator.check_position_len }

```python
def check_position_len(cls, v, field)
```


------

### CWHSimulator {: #CWHSimulator }

```python
class CWHSimulator(self, **kwargs)
```

BaseSimulator is responsible for initializing the platform objects for a simulation
and knowing how to setup episodes based on input parameters from a parameter provider
it is also responsible for reporting the simulation state at each timestep

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHSimulator-bases }

* `act3_rl_core.BaseSimulator`


------

#### Methods {: #CWHSimulator-methods }

[**get_platforms**](#CWHSimulator.get_platforms){: #CWHSimulator.get_platforms }

```python
def get_platforms(self)
```


------

[**get_reset_validator**](#CWHSimulator.get_reset_validator){: #CWHSimulator.get_reset_validator }

```python
def get_reset_validator(cls)
```

returns the validator that can be used to validate episode parameters
coming into the reset function from the environment class

Returns:
    BaseSimulatorResetValidator -- The validator to use during resets

------

[**get_simulator_validator**](#CWHSimulator.get_simulator_validator){: #CWHSimulator.get_simulator_validator }

```python
def get_simulator_validator(cls)
```

returns the validator for the configuration options to the simulator
the kwargs to this class are validated and put into a defined struct
potentially raising based on invalid configurations

Returns:
    BaseSimulatorValidator -- The validator to use for this simulation class

------

[**mark_episode_done**](#CWHSimulator.mark_episode_done){: #CWHSimulator.mark_episode_done }

```python
def mark_episode_done(self)
```

Takes in the string specifying how the episode completed
and does any book keeping around ending an episode

Arguments:
    done_string {str} -- The string describing which Done condition ended an episode

------

[**reset**](#CWHSimulator.reset){: #CWHSimulator.reset }

```python
def reset(self, config)
```

reset resets the simulation and sets up a new episode

Arguments:
    config {typing.Dict[str, typing.Any]} -- The parameters to
            validate and use to setup this episode

Returns:
    StateDict -- The simulation state, has a .sim_platforms attr
                to access the platforms made by the simulation

------

[**save_episode_information**](#CWHSimulator.save_episode_information){: #CWHSimulator.save_episode_information }

```python
def save_episode_information(self, **kwargs)
```

provides a way to save information about the current episode
based on the environment

Arguments:
    dones {[type]} -- the current done info of the step
    rewards {[type]} -- the reward info for this step
    observations {[type]} -- the observations for this step

------

[**step**](#CWHSimulator.step){: #CWHSimulator.step }

```python
def step(self)
```

advances the simulation platforms and returns the state

Returns:
    StateDict -- The state after the simulation updates, has a
                .sim_platforms attr to access the platforms made by the simulation

------

[**update_sensor_measurements**](#CWHSimulator.update_sensor_measurements){: #CWHSimulator.update_sensor_measurements }

```python
def update_sensor_measurements(self)
```

Update and caches all the measurements of all the sensors on each platform

------

### CWHSimulatorResetValidator {: #CWHSimulatorResetValidator }

```python
class CWHSimulatorResetValidator(*, agent_initialization: Dict[str, saferl.simulators.cwh.cwh_simulator.CWHPlatformConfigValidator] = {'blue0': CWHPlatformConfigValidator(position=[0.0, 1.0, 2.0], velocity=[0.0, 0.0, 0.0])})
```

Validator to use to validate the reset input to a simulator class
allows the simulator class to take EPP params and structure/validate them


------

#### Base classes {: #CWHSimulatorResetValidator-bases }

* `act3_rl_core.BaseSimulatorResetValidator`


------

### CWHSimulatorValidator {: #CWHSimulatorValidator }

```python
class CWHSimulatorValidator(*, worker_index: int = 0, vector_index: int = 0, agent_configs: Dict[str, act3_rl_core.simulators.base_simulator.SimulatorPlatformValidator], step_size: int)
```

worker_index: what worker this simulator class is running on < used for render
vector_index: what vector index this simulator class is running on < used for render
agent_configs: the mapping of agent names to a dict describing the platform


------

#### Base classes {: #CWHSimulatorValidator-bases }

* `act3_rl_core.BaseSimulatorValidator`
