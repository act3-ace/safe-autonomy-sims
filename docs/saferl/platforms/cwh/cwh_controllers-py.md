# Module saferl.platforms.cwh.cwh_controllers


## Classes

### CWHController {: #CWHController }

```python
class CWHController(control_properties, config, parent_platform, exclusiveness=<act3_rl_core.simulators.base_parts.MutuallyExclusiveParts object at 0x7f5ab545bca0>)
```

BaseController base abstraction for a controller. A controller is used to move a platform with action commands.
The actions are usually changing the desired rates or applied forces to the platform.


------

#### Base classes {: #CWHController-bases }

* `act3_rl_core.BaseController`


------

#### Instance attributes {: #CWHController-attrs }

* control_properties The properties of the control given to the apply_control function

Returns
-------
Prop
    The properties of the control given tot he apply_control function

* name The name for this platform part

Returns
-------
str
    The name for this platform part

* parent_platform The parent platform this platform part is attached to

Returns
-------
BasePlatform
    The parent platform this platform part is attached to


------

#### Methods {: #CWHController-methods }

[**apply_control**](#CWHController.apply_control){: #CWHController.apply_control }

```python
def apply_control(self, control: np.ndarray) -> None
```

The generic method to apply the control for this controller.

Parameters
----------
control
    The control to be executed by the controller

------

[**get_applied_control**](#CWHController.get_applied_control){: #CWHController.get_applied_control }

```python
def get_applied_control(self) -> np.ndarray
```

Get the previously applied control that was given to the apply_control function
Returns
-------
previously applied control that was given to the apply_control function

------

### ThrustController {: #ThrustController }

```python
class ThrustController(
    self,
    parent_platform,  # type: ignore # noqa: F821
    config,)
```

BaseController base abstraction for a controller. A controller is used to move a platform with action commands.
The actions are usually changing the desired rates or applied forces to the platform.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #ThrustController-bases }

* [`CWHController `](./#CWHController)


------

#### Instance attributes {: #ThrustController-attrs }

* control_properties The properties of the control given to the apply_control function

Returns
-------
Prop
    The properties of the control given tot he apply_control function

* name The name for this platform part

Returns
-------
str
    The name for this platform part

* parent_platform The parent platform this platform part is attached to

Returns
-------
BasePlatform
    The parent platform this platform part is attached to


------

#### Methods {: #ThrustController-methods }

[**apply_control**](#ThrustController.apply_control){: #ThrustController.apply_control }

```python
def apply_control(self, control: np.ndarray) -> None
```

The generic method to apply the control for this controller.

Parameters
----------
control
    The control to be executed by the controller

------

[**get_applied_control**](#ThrustController.get_applied_control){: #ThrustController.get_applied_control }

```python
def get_applied_control(self) -> np.ndarray
```

Get the previously applied control that was given to the apply_control function
Returns
-------
previously applied control that was given to the apply_control function

------

[**get_validator**](#ThrustController.get_validator){: #ThrustController.get_validator }

```python
def get_validator(cls)
```

return the validator that will be used on the configuration
of this part

------

### ThrustControllerValidator {: #ThrustControllerValidator }

```python
class ThrustControllerValidator(*, name: str = None, axis: int)
```

Base validator for controller parts


------

#### Base classes {: #ThrustControllerValidator-bases }

* `act3_rl_core.BaseControllerValidator`
