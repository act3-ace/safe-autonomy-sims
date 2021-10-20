# Module saferl.simulators.dubins.backend.platforms


## Classes

### BaseDubinsPlatform {: #BaseDubinsPlatform }

```python
class BaseDubinsPlatform(name, dynamics, actuator_set, state, controller)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseDubinsPlatform-bases }

* [`BasePlatform `](../../../base_models/platforms-py#BasePlatform)


------

#### Instance attributes {: #BaseDubinsPlatform-attrs }

* **gamma**{: #BaseDubinsPlatform.gamma } 

* **heading**{: #BaseDubinsPlatform.heading } 

* **orientation**{: #BaseDubinsPlatform.orientation } 

* **pitch**{: #BaseDubinsPlatform.pitch } 

* **position**{: #BaseDubinsPlatform.position } 

* **roll**{: #BaseDubinsPlatform.roll } 

* **v**{: #BaseDubinsPlatform.v } 

* **velocity**{: #BaseDubinsPlatform.velocity } 

* **x**{: #BaseDubinsPlatform.x } 

* **y**{: #BaseDubinsPlatform.y } 

* **yaw**{: #BaseDubinsPlatform.yaw } 

* **z**{: #BaseDubinsPlatform.z } 


------

#### Methods {: #BaseDubinsPlatform-methods }

[**generate_info**](#BaseDubinsPlatform.generate_info){: #BaseDubinsPlatform.generate_info }

```python
def generate_info(self)
```


------

### BaseDubinsState {: #BaseDubinsState }

```python
class BaseDubinsState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseDubinsState-bases }

* [`BasePlatformStateVectorized `](../../../base_models/platforms-py#BasePlatformStateVectorized)


------

#### Instance attributes {: #BaseDubinsState-attrs }

* **gamma**{: #BaseDubinsState.gamma } 

* **heading**{: #BaseDubinsState.heading } 

* **orientation**{: #BaseDubinsState.orientation } 

* **pitch**{: #BaseDubinsState.pitch } 

* **position**{: #BaseDubinsState.position } 

* **roll**{: #BaseDubinsState.roll } 

* **v**{: #BaseDubinsState.v } 

* **vector**{: #BaseDubinsState.vector } 

* **vector_shape**{: #BaseDubinsState.vector_shape } 

* **velocity**{: #BaseDubinsState.velocity } 

* **x**{: #BaseDubinsState.x } 

* **y**{: #BaseDubinsState.y } 

* **yaw**{: #BaseDubinsState.yaw } 

* **z**{: #BaseDubinsState.z } 


------

### Dubins2dActuatorSet {: #Dubins2dActuatorSet }

```python
class Dubins2dActuatorSet(self)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins2dActuatorSet-bases }

* [`BaseActuatorSet `](../../../base_models/platforms-py#BaseActuatorSet)


------

### Dubins2dDynamics {: #Dubins2dDynamics }

```python
class Dubins2dDynamics(self, v_min=10, v_max=100, *args, **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins2dDynamics-bases }

* [`BaseODESolverDynamics `](../../../base_models/platforms-py#BaseODESolverDynamics)


------

#### Methods {: #Dubins2dDynamics-methods }

[**dx**](#Dubins2dDynamics.dx){: #Dubins2dDynamics.dx }

```python
def dx(self, t, state_vec, control)
```


------

[**step**](#Dubins2dDynamics.step){: #Dubins2dDynamics.step }

```python
def step(self, step_size, state, control)
```


------

### Dubins2dPlatform {: #Dubins2dPlatform }

```python
class Dubins2dPlatform(self, name, controller=None, rta=None, v_min=10, v_max=100)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins2dPlatform-bases }

* [`BaseDubinsPlatform `](./#BaseDubinsPlatform)


------

#### Instance attributes {: #Dubins2dPlatform-attrs }

* **gamma**{: #Dubins2dPlatform.gamma } 

* **heading**{: #Dubins2dPlatform.heading } 

* **orientation**{: #Dubins2dPlatform.orientation } 

* **pitch**{: #Dubins2dPlatform.pitch } 

* **position**{: #Dubins2dPlatform.position } 

* **roll**{: #Dubins2dPlatform.roll } 

* **v**{: #Dubins2dPlatform.v } 

* **velocity**{: #Dubins2dPlatform.velocity } 

* **x**{: #Dubins2dPlatform.x } 

* **y**{: #Dubins2dPlatform.y } 

* **yaw**{: #Dubins2dPlatform.yaw } 

* **z**{: #Dubins2dPlatform.z } 


------

### Dubins2dState {: #Dubins2dState }

```python
class Dubins2dState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #Dubins2dState-bases }

* [`BaseDubinsState `](./#BaseDubinsState)


------

#### Instance attributes {: #Dubins2dState-attrs }

* **gamma**{: #Dubins2dState.gamma } 

* **heading**{: #Dubins2dState.heading } 

* **orientation**{: #Dubins2dState.orientation } 

* **pitch**{: #Dubins2dState.pitch } 

* **position**{: #Dubins2dState.position } 

* **roll**{: #Dubins2dState.roll } 

* **v**{: #Dubins2dState.v } 

* **vector**{: #Dubins2dState.vector } 

* **vector_shape**{: #Dubins2dState.vector_shape } 

* **velocity**{: #Dubins2dState.velocity } 

* **x**{: #Dubins2dState.x } 

* **y**{: #Dubins2dState.y } 

* **yaw**{: #Dubins2dState.yaw } 

* **z**{: #Dubins2dState.z } 


------

#### Methods {: #Dubins2dState-methods }

[**build_vector**](#Dubins2dState.build_vector){: #Dubins2dState.build_vector }

```python
def build_vector(self, x=0, y=0, heading=0, v=50, **kwargs)
```


------

### Dubins3dActuatorSet {: #Dubins3dActuatorSet }

```python
class Dubins3dActuatorSet(self)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins3dActuatorSet-bases }

* [`BaseActuatorSet `](../../../base_models/platforms-py#BaseActuatorSet)


------

### Dubins3dDynamics {: #Dubins3dDynamics }

```python
class Dubins3dDynamics(
    self,
     v_min=10,
     v_max=100,
     roll_min=-math.pi / 2,
     roll_max=math.pi / 2,
     g=32.17,
     *args,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins3dDynamics-bases }

* [`BaseODESolverDynamics `](../../../base_models/platforms-py#BaseODESolverDynamics)


------

#### Methods {: #Dubins3dDynamics-methods }

[**dx**](#Dubins3dDynamics.dx){: #Dubins3dDynamics.dx }

```python
def dx(self, t, state_vec, control)
```


------

[**step**](#Dubins3dDynamics.step){: #Dubins3dDynamics.step }

```python
def step(self, step_size, state, control)
```


------

### Dubins3dPlatform {: #Dubins3dPlatform }

```python
class Dubins3dPlatform(self, name, controller=None, v_min=10, v_max=100)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Dubins3dPlatform-bases }

* [`BaseDubinsPlatform `](./#BaseDubinsPlatform)


------

#### Instance attributes {: #Dubins3dPlatform-attrs }

* **gamma**{: #Dubins3dPlatform.gamma } 

* **heading**{: #Dubins3dPlatform.heading } 

* **orientation**{: #Dubins3dPlatform.orientation } 

* **pitch**{: #Dubins3dPlatform.pitch } 

* **position**{: #Dubins3dPlatform.position } 

* **roll**{: #Dubins3dPlatform.roll } 

* **v**{: #Dubins3dPlatform.v } 

* **velocity**{: #Dubins3dPlatform.velocity } 

* **x**{: #Dubins3dPlatform.x } 

* **y**{: #Dubins3dPlatform.y } 

* **yaw**{: #Dubins3dPlatform.yaw } 

* **z**{: #Dubins3dPlatform.z } 


------

#### Methods {: #Dubins3dPlatform-methods }

[**generate_info**](#Dubins3dPlatform.generate_info){: #Dubins3dPlatform.generate_info }

```python
def generate_info(self)
```


------

### Dubins3dState {: #Dubins3dState }

```python
class Dubins3dState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #Dubins3dState-bases }

* [`BaseDubinsState `](./#BaseDubinsState)


------

#### Instance attributes {: #Dubins3dState-attrs }

* **gamma**{: #Dubins3dState.gamma } 

* **heading**{: #Dubins3dState.heading } 

* **orientation**{: #Dubins3dState.orientation } 

* **pitch**{: #Dubins3dState.pitch } 

* **position**{: #Dubins3dState.position } 

* **roll**{: #Dubins3dState.roll } 

* **v**{: #Dubins3dState.v } 

* **vector**{: #Dubins3dState.vector } 

* **vector_shape**{: #Dubins3dState.vector_shape } 

* **velocity**{: #Dubins3dState.velocity } 

* **x**{: #Dubins3dState.x } 

* **y**{: #Dubins3dState.y } 

* **yaw**{: #Dubins3dState.yaw } 

* **z**{: #Dubins3dState.z } 


------

#### Methods {: #Dubins3dState-methods }

[**build_vector**](#Dubins3dState.build_vector){: #Dubins3dState.build_vector }

```python
def build_vector(self,
     x=0,
     y=0,
     z=0,
     heading=0,
     gamma=0,
     roll=0,
     v=100,
     **kwargs)
```
