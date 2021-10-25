# Module saferl.simulators.cwh.backend.platforms.oriented


## Classes

### CWHOriented2dActuatorSet {: #CWHOriented2dActuatorSet }

```python
class CWHOriented2dActuatorSet(self)
```


Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHOriented2dActuatorSet-bases }

* [`BaseActuatorSet `](../../../../base_models/platforms-py#BaseActuatorSet)


------

### CWHOriented2dDynamics {: #CWHOriented2dDynamics }

```python
class CWHOriented2dDynamics(self, platform, integration_method='Euler')
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHOriented2dDynamics-bases }

* [`BaseLinearODESolverDynamics `](../../../../base_models/platforms-py#BaseLinearODESolverDynamics)


------

#### Methods {: #CWHOriented2dDynamics-methods }

[**dx**](#CWHOriented2dDynamics.dx){: #CWHOriented2dDynamics.dx }

```python
def dx(self, t, state_vec, control)
```


------

[**gen_dynamics_matrices**](#CWHOriented2dDynamics.gen_dynamics_matrices){: #CWHOriented2dDynamics.gen_dynamics_matrices }

```python
def gen_dynamics_matrices(self)
```


------

### CWHOriented2dState {: #CWHOriented2dState }

```python
class CWHOriented2dState(**kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #CWHOriented2dState-bases }

* [`BasePlatformStateVectorized `](../../../../base_models/platforms-py#BasePlatformStateVectorized)


------

#### Instance attributes {: #CWHOriented2dState-attrs }

* **orientation**{: #CWHOriented2dState.orientation } 

* **position**{: #CWHOriented2dState.position } 

* **react_wheel_ang_vel**{: #CWHOriented2dState.react_wheel_ang_vel } 

* **theta**{: #CWHOriented2dState.theta } 

* **theta_dot**{: #CWHOriented2dState.theta_dot } 

* **vector**{: #CWHOriented2dState.vector } 

* **vector_shape**{: #CWHOriented2dState.vector_shape } 

* **velocity**{: #CWHOriented2dState.velocity } 

* **x**{: #CWHOriented2dState.x } 

* **x_dot**{: #CWHOriented2dState.x_dot } 

* **y**{: #CWHOriented2dState.y } 

* **y_dot**{: #CWHOriented2dState.y_dot } 

* **z**{: #CWHOriented2dState.z } 


------

#### Methods {: #CWHOriented2dState-methods }

[**build_vector**](#CWHOriented2dState.build_vector){: #CWHOriented2dState.build_vector }

```python
def build_vector(self,
     x=0,
     y=0,
     theta=0,
     x_dot=0,
     y_dot=0,
     theta_dot=0,
     react_wheel_ang_vel=0,
     **kwargs)
```


------

### CWHSpacecraftOriented2d {: #CWHSpacecraftOriented2d }

```python
class CWHSpacecraftOriented2d(self, name, controller=None, **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHSpacecraftOriented2d-bases }

* [`BasePlatform `](../../../../base_models/platforms-py#BasePlatform)


------

#### Instance attributes {: #CWHSpacecraftOriented2d-attrs }

* **orientation**{: #CWHSpacecraftOriented2d.orientation } 

* **position**{: #CWHSpacecraftOriented2d.position } 

* **react_wheel_ang_vel**{: #CWHSpacecraftOriented2d.react_wheel_ang_vel } 

* **theta**{: #CWHSpacecraftOriented2d.theta } 

* **theta_dot**{: #CWHSpacecraftOriented2d.theta_dot } 

* **velocity**{: #CWHSpacecraftOriented2d.velocity } 

* **x**{: #CWHSpacecraftOriented2d.x } 

* **x_dot**{: #CWHSpacecraftOriented2d.x_dot } 

* **y**{: #CWHSpacecraftOriented2d.y } 

* **y_dot**{: #CWHSpacecraftOriented2d.y_dot } 

* **z**{: #CWHSpacecraftOriented2d.z } 


------

#### Methods {: #CWHSpacecraftOriented2d-methods }

[**generate_info**](#CWHSpacecraftOriented2d.generate_info){: #CWHSpacecraftOriented2d.generate_info }

```python
def generate_info(self)
```
