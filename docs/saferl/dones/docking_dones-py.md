# Module saferl.dones.docking_dones


## Classes

### MaxDistanceDoneFunction {: #MaxDistanceDoneFunction }

```python
class MaxDistanceDoneFunction(self, **kwargs)
```

Base implementation for done functors
    

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #MaxDistanceDoneFunction-bases }

* `act3_rl_core.DoneFuncBase`


------

#### Instance attributes {: #MaxDistanceDoneFunction-attrs }

* agent The agent to which this done is applied

* name gets the name fo the functor

Returns
-------
str
    The name of the functor


------

#### Methods {: #MaxDistanceDoneFunction-methods }

[**get_validator**](#MaxDistanceDoneFunction.get_validator){: #MaxDistanceDoneFunction.get_validator }

```python
def get_validator(cls)
```

get validator for this Done Functor

Returns:
    DoneFuncBaseValidator -- validator the done functor will use to generate a configuration

------

### MaxDistanceDoneValidator {: #MaxDistanceDoneValidator }

```python
class MaxDistanceDoneValidator(*, name: str = None, agent_name: str, early_stop: bool = True, max_distance: float)
```

Initialize the done condition

All done conditions have three common parameters, with any others being handled by subclass validators

Parameters
----------
agent : str
    Name of the agent to which this done condition applies
early_stop : bool, optional
    If True, set the done condition on "__all__" once any done condition is True.  This is by default True.
name : str
    A name applied to this done condition, by default the name of the class.


------

#### Base classes {: #MaxDistanceDoneValidator-bases }

* `act3_rl_core.DoneFuncBaseValidator`


------

### SuccessfulDockingDoneFunction {: #SuccessfulDockingDoneFunction }

```python
class SuccessfulDockingDoneFunction(self, **kwargs)
```

Base implementation for done functors
    

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #SuccessfulDockingDoneFunction-bases }

* `act3_rl_core.DoneFuncBase`


------

#### Instance attributes {: #SuccessfulDockingDoneFunction-attrs }

* agent The agent to which this done is applied

* name gets the name fo the functor

Returns
-------
str
    The name of the functor


------

#### Methods {: #SuccessfulDockingDoneFunction-methods }

[**get_validator**](#SuccessfulDockingDoneFunction.get_validator){: #SuccessfulDockingDoneFunction.get_validator }

```python
def get_validator(cls)
```

get validator for this Done Functor

Returns:
    DoneFuncBaseValidator -- validator the done functor will use to generate a configuration

------

### SuccessfulDockingDoneValidator {: #SuccessfulDockingDoneValidator }

```python
class SuccessfulDockingDoneValidator(*, name: str = None, agent_name: str, early_stop: bool = True, docking_region_radius: float)
```

Initialize the done condition

All done conditions have three common parameters, with any others being handled by subclass validators

Parameters
----------
agent : str
    Name of the agent to which this done condition applies
early_stop : bool, optional
    If True, set the done condition on "__all__" once any done condition is True.  This is by default True.
name : str
    A name applied to this done condition, by default the name of the class.


------

#### Base classes {: #SuccessfulDockingDoneValidator-bases }

* `act3_rl_core.DoneFuncBaseValidator`
