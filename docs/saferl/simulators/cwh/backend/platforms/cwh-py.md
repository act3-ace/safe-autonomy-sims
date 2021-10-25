# Module saferl.simulators.cwh.backend.platforms.cwh


## Classes

### BaseCWHSpacecraft {: #BaseCWHSpacecraft }

```python
class BaseCWHSpacecraft(name, dynamics, actuator_set, state, controller)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseCWHSpacecraft-bases }

* [`BasePlatform `](../../../../base_models/platforms-py#BasePlatform)


------

#### Instance attributes {: #BaseCWHSpacecraft-attrs }

* **orientation**{: #BaseCWHSpacecraft.orientation } 

* **position**{: #BaseCWHSpacecraft.position } 

* **velocity**{: #BaseCWHSpacecraft.velocity } 

* **x**{: #BaseCWHSpacecraft.x } 

* **x_dot**{: #BaseCWHSpacecraft.x_dot } 

* **y**{: #BaseCWHSpacecraft.y } 

* **y_dot**{: #BaseCWHSpacecraft.y_dot } 

* **z**{: #BaseCWHSpacecraft.z } 


------

#### Methods {: #BaseCWHSpacecraft-methods }

[**generate_info**](#BaseCWHSpacecraft.generate_info){: #BaseCWHSpacecraft.generate_info }

```python
def generate_info(self)
```


------

### CWH2dActuatorSet {: #CWH2dActuatorSet }

```python
class CWH2dActuatorSet(self)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWH2dActuatorSet-bases }

* [`BaseActuatorSet `](../../../../base_models/platforms-py#BaseActuatorSet)


------

### CWH2dDynamics {: #CWH2dDynamics }

```python
class CWH2dDynamics(self, m=12, n=0.001027, integration_method='Euler')
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWH2dDynamics-bases }

* [`BaseLinearODESolverDynamics `](../../../../base_models/platforms-py#BaseLinearODESolverDynamics)


------

#### Methods {: #CWH2dDynamics-methods }

[**gen_dynamics_matrices**](#CWH2dDynamics.gen_dynamics_matrices){: #CWH2dDynamics.gen_dynamics_matrices }

```python
def gen_dynamics_matrices(self)
```


------

### CWH2dState {: #CWH2dState }

```python
class CWH2dState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #CWH2dState-bases }

* [`BasePlatformStateVectorized `](../../../../base_models/platforms-py#BasePlatformStateVectorized)


------

#### Instance attributes {: #CWH2dState-attrs }

* **orientation**{: #CWH2dState.orientation } 

* **position**{: #CWH2dState.position } 

* **vector**{: #CWH2dState.vector } 

* **vector_shape**{: #CWH2dState.vector_shape } 

* **velocity**{: #CWH2dState.velocity } 

* **x**{: #CWH2dState.x } 

* **x_dot**{: #CWH2dState.x_dot } 

* **y**{: #CWH2dState.y } 

* **y_dot**{: #CWH2dState.y_dot } 

* **z**{: #CWH2dState.z } 


------

#### Methods {: #CWH2dState-methods }

[**build_vector**](#CWH2dState.build_vector){: #CWH2dState.build_vector }

```python
def build_vector(self, x=0, y=0, x_dot=0, y_dot=0, **kwargs)
```


------

### CWH3dActuatorSet {: #CWH3dActuatorSet }

```python
class CWH3dActuatorSet(self)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWH3dActuatorSet-bases }

* [`BaseActuatorSet `](../../../../base_models/platforms-py#BaseActuatorSet)


------

### CWH3dDynamics {: #CWH3dDynamics }

```python
class CWH3dDynamics(self, m=12, n=0.001027, integration_method='Euler')
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWH3dDynamics-bases }

* [`BaseLinearODESolverDynamics `](../../../../base_models/platforms-py#BaseLinearODESolverDynamics)


------

#### Methods {: #CWH3dDynamics-methods }

[**gen_dynamics_matrices**](#CWH3dDynamics.gen_dynamics_matrices){: #CWH3dDynamics.gen_dynamics_matrices }

```python
def gen_dynamics_matrices(self)
```


------

### CWH3dState {: #CWH3dState }

```python
class CWH3dState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #CWH3dState-bases }

* [`BasePlatformStateVectorized `](../../../../base_models/platforms-py#BasePlatformStateVectorized)


------

#### Instance attributes {: #CWH3dState-attrs }

* **orientation**{: #CWH3dState.orientation } 

* **position**{: #CWH3dState.position } 

* **vector**{: #CWH3dState.vector } 

* **vector_shape**{: #CWH3dState.vector_shape } 

* **velocity**{: #CWH3dState.velocity } 

* **x**{: #CWH3dState.x } 

* **x_dot**{: #CWH3dState.x_dot } 

* **y**{: #CWH3dState.y } 

* **y_dot**{: #CWH3dState.y_dot } 

* **z**{: #CWH3dState.z } 

* **z_dot**{: #CWH3dState.z_dot } 


------

#### Methods {: #CWH3dState-methods }

[**build_vector**](#CWH3dState.build_vector){: #CWH3dState.build_vector }

```python
def build_vector(self, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0, **kwargs)
```


------

### CWHSpacecraft2d {: #CWHSpacecraft2d }

```python
class CWHSpacecraft2d(self, name, controller=None)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHSpacecraft2d-bases }

* [`BaseCWHSpacecraft `](./#BaseCWHSpacecraft)


------

#### Instance attributes {: #CWHSpacecraft2d-attrs }

* **orientation**{: #CWHSpacecraft2d.orientation } 

* **position**{: #CWHSpacecraft2d.position } 

* **velocity**{: #CWHSpacecraft2d.velocity } 

* **x**{: #CWHSpacecraft2d.x } 

* **x_dot**{: #CWHSpacecraft2d.x_dot } 

* **y**{: #CWHSpacecraft2d.y } 

* **y_dot**{: #CWHSpacecraft2d.y_dot } 

* **z**{: #CWHSpacecraft2d.z } 


------

### CWHSpacecraft3d {: #CWHSpacecraft3d }

```python
class CWHSpacecraft3d(self, name, controller=None)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHSpacecraft3d-bases }

* [`BaseCWHSpacecraft `](./#BaseCWHSpacecraft)


------

#### Instance attributes {: #CWHSpacecraft3d-attrs }

* **orientation**{: #CWHSpacecraft3d.orientation } 

* **position**{: #CWHSpacecraft3d.position } 

* **velocity**{: #CWHSpacecraft3d.velocity } 

* **x**{: #CWHSpacecraft3d.x } 

* **x_dot**{: #CWHSpacecraft3d.x_dot } 

* **y**{: #CWHSpacecraft3d.y } 

* **y_dot**{: #CWHSpacecraft3d.y_dot } 

* **z**{: #CWHSpacecraft3d.z } 

* **z_dot**{: #CWHSpacecraft3d.z_dot } 


------

#### Methods {: #CWHSpacecraft3d-methods }

[**generate_info**](#CWHSpacecraft3d.generate_info){: #CWHSpacecraft3d.generate_info }

```python
def generate_info(self)
```
