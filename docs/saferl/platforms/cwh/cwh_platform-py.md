# Module saferl.platforms.cwh.cwh_platform


## Classes

### CWHPlatform {: #CWHPlatform }

```python
class CWHPlatform(self, platform, platform_config, seed=None)
```

The __________________ as it's platform and
allows for saving an action to the platform for when the platform needs
to give an action to the environment during the environment step function

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #CWHPlatform-bases }

* `act3_rl_core.BasePlatform`


------

#### Instance attributes {: #CWHPlatform-attrs }

* controllers Controllers attached to this platform

Returns
------
List
    list of all controllers attached to this platform

* name name the name of this object

Returns
-------
str
    The name of this object

* operable Is the platform operable?

Returns
-------
bool
    Is the platform operable?

* **position**{: #CWHPlatform.position } 

* sensors Sensors attached to this platform

Returns
------
List
    list of all sensors attached to this platform

* **velocity**{: #CWHPlatform.velocity } 


------

#### Methods {: #CWHPlatform-methods }

[**get_applied_action**](#CWHPlatform.get_applied_action){: #CWHPlatform.get_applied_action }

```python
def get_applied_action(self)
```
