# Module saferl.platforms.cwh.cwh_sensors


## Classes

### CWHSensor {: #CWHSensor }

```python
class CWHSensor(parent_platform, config)
```

BaseSensor base abstraction for a sensor. A sensor is a attached to a platform
and provides information about the environment.


------

#### Base classes {: #CWHSensor-bases }

* `act3_rl_core.BaseSensor`


------

#### Instance attributes {: #CWHSensor-attrs }

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

#### Methods {: #CWHSensor-methods }

[**measurement_properties**](#CWHSensor.measurement_properties){: #CWHSensor.measurement_properties }

```python
def measurement_properties(self)
```

The properties of the object returned by the get_measurement function

Returns
-------
Prop
    The properties of the measurement returned by the get_measurement function

------

### PositionSensor {: #PositionSensor }

```python
class PositionSensor(parent_platform, config)
```

BaseSensor base abstraction for a sensor. A sensor is a attached to a platform
and provides information about the environment.


------

#### Base classes {: #PositionSensor-bases }

* [`CWHSensor `](./#CWHSensor)


------

#### Instance attributes {: #PositionSensor-attrs }

* measurement_properties The properties of the object returned by the get_measurement function

Returns
-------
Prop
    The properties of the measurement returned by the get_measurement function

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

### VelocitySensor {: #VelocitySensor }

```python
class VelocitySensor(parent_platform, config)
```

BaseSensor base abstraction for a sensor. A sensor is a attached to a platform
and provides information about the environment.


------

#### Base classes {: #VelocitySensor-bases }

* [`CWHSensor `](./#CWHSensor)


------

#### Instance attributes {: #VelocitySensor-attrs }

* measurement_properties The properties of the object returned by the get_measurement function

Returns
-------
Prop
    The properties of the measurement returned by the get_measurement function

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
