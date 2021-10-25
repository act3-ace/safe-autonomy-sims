# Module saferl.simulators.base_models.platforms


## Classes

### ActionPreprocessor {: #ActionPreprocessor }

```python
class ActionPreprocessor(self, name)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #ActionPreprocessor-bases }

* `abc.ABC`


------

#### Methods {: #ActionPreprocessor-methods }

[**preprocess**](#ActionPreprocessor.preprocess){: #ActionPreprocessor.preprocess }

```python
def preprocess(self, action)
```


------

### ActionPreprocessorContinuousRescale {: #ActionPreprocessorContinuousRescale }

```python
class ActionPreprocessorContinuousRescale(self, name, bounds)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #ActionPreprocessorContinuousRescale-bases }

* [`ActionPreprocessor `](./#ActionPreprocessor)


------

#### Methods {: #ActionPreprocessorContinuousRescale-methods }

[**preprocess**](#ActionPreprocessorContinuousRescale.preprocess){: #ActionPreprocessorContinuousRescale.preprocess }

```python
def preprocess(self, action)
```


------

### ActionPreprocessorDiscreteMap {: #ActionPreprocessorDiscreteMap }

```python
class ActionPreprocessorDiscreteMap(self, name, vals)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #ActionPreprocessorDiscreteMap-bases }

* [`ActionPreprocessor `](./#ActionPreprocessor)


------

#### Methods {: #ActionPreprocessorDiscreteMap-methods }

[**preprocess**](#ActionPreprocessorDiscreteMap.preprocess){: #ActionPreprocessorDiscreteMap.preprocess }

```python
def preprocess(self, action)
```


------

### ActionPreprocessorPassThrough {: #ActionPreprocessorPassThrough }

```python
class ActionPreprocessorPassThrough(name)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #ActionPreprocessorPassThrough-bases }

* [`ActionPreprocessor `](./#ActionPreprocessor)


------

#### Methods {: #ActionPreprocessorPassThrough-methods }

[**preprocess**](#ActionPreprocessorPassThrough.preprocess){: #ActionPreprocessorPassThrough.preprocess }

```python
def preprocess(self, action)
```


------

### AgentController {: #AgentController }

```python
class AgentController(self, actuator_set, config)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #AgentController-bases }

* [`BaseController `](./#BaseController)


------

#### Methods {: #AgentController-methods }

[**gen_actuation**](#AgentController.gen_actuation){: #AgentController.gen_actuation }

```python
def gen_actuation(self, state, action=None)
```


------

[**setup_action_space**](#AgentController.setup_action_space){: #AgentController.setup_action_space }

```python
def setup_action_space(self)
```


------

### BaseActuator {: #BaseActuator }

```python
class BaseActuator()
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseActuator-bases }

* `abc.ABC`


------

#### Instance attributes {: #BaseActuator-attrs }

* **default**{: #BaseActuator.default } 

* **name**{: #BaseActuator.name } 

* **space**{: #BaseActuator.space } 


------

### BaseActuatorSet {: #BaseActuatorSet }

```python
class BaseActuatorSet(self, actuators)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Methods {: #BaseActuatorSet-methods }

[**gen_control**](#BaseActuatorSet.gen_control){: #BaseActuatorSet.gen_control }

```python
def gen_control(self, actuation=None)
```


------

### BaseController {: #BaseController }

```python
class BaseController()
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseController-bases }

* `abc.ABC`


------

#### Methods {: #BaseController-methods }

[**gen_actuation**](#BaseController.gen_actuation){: #BaseController.gen_actuation }

```python
def gen_actuation(self, state, action=None)
```


------

### BaseDynamics {: #BaseDynamics }

```python
class BaseDynamics()
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseDynamics-bases }

* `abc.ABC`


------

#### Methods {: #BaseDynamics-methods }

[**step**](#BaseDynamics.step){: #BaseDynamics.step }

```python
def step(self, step_size, state, control)
```


------

### BaseEnvObj {: #BaseEnvObj }

```python
class BaseEnvObj(self, name)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #BaseEnvObj-bases }

* `abc.ABC`


------

#### Instance attributes {: #BaseEnvObj-attrs }

* **orientation**{: #BaseEnvObj.orientation } 

* **position**{: #BaseEnvObj.position } 

* **velocity**{: #BaseEnvObj.velocity } 

* **x**{: #BaseEnvObj.x } 

* **y**{: #BaseEnvObj.y } 

* **z**{: #BaseEnvObj.z } 


------

### BaseLinearODESolverDynamics {: #BaseLinearODESolverDynamics }

```python
class BaseLinearODESolverDynamics(self, integration_method='Euler')
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #BaseLinearODESolverDynamics-bases }

* [`BaseODESolverDynamics `](./#BaseODESolverDynamics)


------

#### Methods {: #BaseLinearODESolverDynamics-methods }

[**dx**](#BaseLinearODESolverDynamics.dx){: #BaseLinearODESolverDynamics.dx }

```python
def dx(self, t, state_vec, control)
```


------

[**gen_dynamics_matrices**](#BaseLinearODESolverDynamics.gen_dynamics_matrices){: #BaseLinearODESolverDynamics.gen_dynamics_matrices }

```python
def gen_dynamics_matrices(self)
```


------

[**step**](#BaseLinearODESolverDynamics.step){: #BaseLinearODESolverDynamics.step }

```python
def step(self, step_size, state, control)
```


------

[**update_dynamics_matrices**](#BaseLinearODESolverDynamics.update_dynamics_matrices){: #BaseLinearODESolverDynamics.update_dynamics_matrices }

```python
def update_dynamics_matrices(self, state_vec)
```


------

### BaseODESolverDynamics {: #BaseODESolverDynamics }

```python
class BaseODESolverDynamics(self, integration_method='Euler')
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #BaseODESolverDynamics-bases }

* [`BaseDynamics `](./#BaseDynamics)


------

#### Methods {: #BaseODESolverDynamics-methods }

[**dx**](#BaseODESolverDynamics.dx){: #BaseODESolverDynamics.dx }

```python
def dx(self, t, state_vec, control)
```


------

[**step**](#BaseODESolverDynamics.step){: #BaseODESolverDynamics.step }

```python
def step(self, step_size, state, control)
```


------

### BasePlatform {: #BasePlatform }

```python
class BasePlatform(self, name, dynamics, actuator_set, state, controller)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #BasePlatform-bases }

* [`BaseEnvObj `](./#BaseEnvObj)


------

#### Instance attributes {: #BasePlatform-attrs }

* **orientation**{: #BasePlatform.orientation } 

* **position**{: #BasePlatform.position } 

* **velocity**{: #BasePlatform.velocity } 

* **x**{: #BasePlatform.x } 

* **y**{: #BasePlatform.y } 

* **z**{: #BasePlatform.z } 


------

#### Methods {: #BasePlatform-methods }

[**generate_info**](#BasePlatform.generate_info){: #BasePlatform.generate_info }

```python
def generate_info(self)
```


------

[**register_dependent_obj**](#BasePlatform.register_dependent_obj){: #BasePlatform.register_dependent_obj }

```python
def register_dependent_obj(self, obj)
```


------

[**reset**](#BasePlatform.reset){: #BasePlatform.reset }

```python
def reset(self, **kwargs)
```


------

[**step**](#BasePlatform.step){: #BasePlatform.step }

```python
def step(self, sim_state, step_size, action=None)
```


------

[**step_apply**](#BasePlatform.step_apply){: #BasePlatform.step_apply }

```python
def step_apply(self)
```


------

[**step_compute**](#BasePlatform.step_compute){: #BasePlatform.step_compute }

```python
def step_compute(self, sim_state, step_size, action=None)
```


------

### BasePlatformState {: #BasePlatformState }

```python
class BasePlatformState(self, **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #BasePlatformState-bases }

* [`BaseEnvObj `](./#BaseEnvObj)


------

#### Instance attributes {: #BasePlatformState-attrs }

* **orientation**{: #BasePlatformState.orientation } 

* **position**{: #BasePlatformState.position } 

* **velocity**{: #BasePlatformState.velocity } 

* **x**{: #BasePlatformState.x } 

* **y**{: #BasePlatformState.y } 

* **z**{: #BasePlatformState.z } 


------

#### Methods {: #BasePlatformState-methods }

[**reset**](#BasePlatformState.reset){: #BasePlatformState.reset }

```python
def reset(self, **kwargs)
```


------

### BasePlatformStateVectorized {: #BasePlatformStateVectorized }

```python
class BasePlatformStateVectorized(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BasePlatformStateVectorized-bases }

* [`BasePlatformState `](./#BasePlatformState)


------

#### Instance attributes {: #BasePlatformStateVectorized-attrs }

* **orientation**{: #BasePlatformStateVectorized.orientation } 

* **position**{: #BasePlatformStateVectorized.position } 

* **vector**{: #BasePlatformStateVectorized.vector } 

* **vector_shape**{: #BasePlatformStateVectorized.vector_shape } 

* **velocity**{: #BasePlatformStateVectorized.velocity } 

* **x**{: #BasePlatformStateVectorized.x } 

* **y**{: #BasePlatformStateVectorized.y } 

* **z**{: #BasePlatformStateVectorized.z } 


------

#### Methods {: #BasePlatformStateVectorized-methods }

[**build_vector**](#BasePlatformStateVectorized.build_vector){: #BasePlatformStateVectorized.build_vector }

```python
def build_vector(self)
```


------

[**reset**](#BasePlatformStateVectorized.reset){: #BasePlatformStateVectorized.reset }

```python
def reset(self, vector=None, vector_deep_copy=True, **kwargs)
```


------

### ContinuousActuator {: #ContinuousActuator }

```python
class ContinuousActuator(self, name, bounds, default)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #ContinuousActuator-bases }

* [`BaseActuator `](./#BaseActuator)


------

#### Instance attributes {: #ContinuousActuator-attrs }

* **bounds**{: #ContinuousActuator.bounds } 

* **default**{: #ContinuousActuator.default } 

* **name**{: #ContinuousActuator.name } 

* **space**{: #ContinuousActuator.space } 


------

### PassThroughController {: #PassThroughController }

```python
class PassThroughController(self)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #PassThroughController-bases }

* [`BaseController `](./#BaseController)


------

#### Methods {: #PassThroughController-methods }

[**gen_actuation**](#PassThroughController.gen_actuation){: #PassThroughController.gen_actuation }

```python
def gen_actuation(self, state, action=None)
```
