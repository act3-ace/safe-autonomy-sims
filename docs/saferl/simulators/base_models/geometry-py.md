# Module saferl.simulators.base_models.geometry


## Classes

### BaseGeometry {: #BaseGeometry }

```python
class BaseGeometry(name)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #BaseGeometry-bases }

* [`BaseEnvObj `](../platforms-py#BaseEnvObj)


------

#### Instance attributes {: #BaseGeometry-attrs }

* **orientation**{: #BaseGeometry.orientation } 

* **position**{: #BaseGeometry.position } 

* **velocity**{: #BaseGeometry.velocity } 

* **x**{: #BaseGeometry.x } 

* **y**{: #BaseGeometry.y } 

* **z**{: #BaseGeometry.z } 


------

#### Methods {: #BaseGeometry-methods }

[**contains**](#BaseGeometry.contains){: #BaseGeometry.contains }

```python
def contains(self, other)
```


------

[**generate_info**](#BaseGeometry.generate_info){: #BaseGeometry.generate_info }

```python
def generate_info(self)
```


------

### Circle {: #Circle }

```python
class Circle(self, name, x=0, y=0, z=0, radius=1)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Circle-bases }

* [`Point `](./#Point)


------

#### Instance attributes {: #Circle-attrs }

* **orientation**{: #Circle.orientation } 

* **position**{: #Circle.position } 

* **velocity**{: #Circle.velocity } 

* **x**{: #Circle.x } 

* **y**{: #Circle.y } 

* **z**{: #Circle.z } 


------

#### Methods {: #Circle-methods }

[**contains**](#Circle.contains){: #Circle.contains }

```python
def contains(self, other)
```


------

[**generate_info**](#Circle.generate_info){: #Circle.generate_info }

```python
def generate_info(self)
```


------

### Cylinder {: #Cylinder }

```python
class Cylinder(self, name, x=0, y=0, z=0, radius=1, height=1)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Cylinder-bases }

* [`Circle `](./#Circle)


------

#### Instance attributes {: #Cylinder-attrs }

* **orientation**{: #Cylinder.orientation } 

* **position**{: #Cylinder.position } 

* **velocity**{: #Cylinder.velocity } 

* **x**{: #Cylinder.x } 

* **y**{: #Cylinder.y } 

* **z**{: #Cylinder.z } 


------

#### Methods {: #Cylinder-methods }

[**contains**](#Cylinder.contains){: #Cylinder.contains }

```python
def contains(self, other)
```


------

[**generate_info**](#Cylinder.generate_info){: #Cylinder.generate_info }

```python
def generate_info(self)
```


------

### Point {: #Point }

```python
class Point(self, name, x=0, y=0, z=0)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Base classes {: #Point-bases }

* [`BaseGeometry `](./#BaseGeometry)


------

#### Instance attributes {: #Point-attrs }

* **orientation**{: #Point.orientation } 

* **position**{: #Point.position } 

* **velocity**{: #Point.velocity } 

* **x**{: #Point.x } 

* **y**{: #Point.y } 

* **z**{: #Point.z } 


------

#### Methods {: #Point-methods }

[**contains**](#Point.contains){: #Point.contains }

```python
def contains(self, other)
```


------

[**generate_info**](#Point.generate_info){: #Point.generate_info }

```python
def generate_info(self)
```


------

[**reset**](#Point.reset){: #Point.reset }

```python
def reset(self, **kwargs)
```


------

### RelativeCircle {: #RelativeCircle }

```python
class RelativeCircle(
    self,
     ref,
     track_orientation=False,
     x_offset=None,
     y_offset=None,
     z_offset=None,
     r_offset=None,
     theta_offset=None,
     aspect_angle=None,
     init=None,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Constructs RelativeGeometry Object
Must specify either a Cartesian offset or Polar offset from the ref object

Parameters
----------
ref
    Reference EnvObj that positions and orientations are relative to
shape
    Underlying Geometry object
track_orientation
    Whether rotate around its ref object as the ref's orientation rotates
    If True, behaves as if attached to ref with a rigid rod (rotates around ref).
    If False, behaves as if attached to ref with a gimble.
x_offset
    Cartesian offset component.
y_offset
    Cartesian offset component.
z_offset
    Cartesian offset component. Can mix with Polar offset to add a Z offset
r_offset
    Polar offset component. Distance from ref.
theta_offset
    Polar offset component. Radians. Azimuth angle offset of relative vector.
aspect_angle
    Polar offset component. Degrees. Can use instead of theta_offset
euler_decomp_axis
    Euler decomposition of rotation tracking into a subset of the Euler angles
    Allows tracking of planar rotations only (such as xy plane rotations only)
    NotImplemented
init
    Initialization Dictionary

Returns
-------
None


------

#### Base classes {: #RelativeCircle-bases }

* [`RelativeGeometry `](./#RelativeGeometry)


------

#### Instance attributes {: #RelativeCircle-attrs }

* **orientation**{: #RelativeCircle.orientation } 

* **position**{: #RelativeCircle.position } 

* **radius**{: #RelativeCircle.radius } 

* **velocity**{: #RelativeCircle.velocity } 

* **x**{: #RelativeCircle.x } 

* **y**{: #RelativeCircle.y } 

* **z**{: #RelativeCircle.z } 


------

### RelativeCylinder {: #RelativeCylinder }

```python
class RelativeCylinder(
    self,
     ref,
     track_orientation=False,
     x_offset=None,
     y_offset=None,
     z_offset=None,
     r_offset=None,
     theta_offset=None,
     aspect_angle=None,
     init=None,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Constructs RelativeGeometry Object
Must specify either a Cartesian offset or Polar offset from the ref object

Parameters
----------
ref
    Reference EnvObj that positions and orientations are relative to
shape
    Underlying Geometry object
track_orientation
    Whether rotate around its ref object as the ref's orientation rotates
    If True, behaves as if attached to ref with a rigid rod (rotates around ref).
    If False, behaves as if attached to ref with a gimble.
x_offset
    Cartesian offset component.
y_offset
    Cartesian offset component.
z_offset
    Cartesian offset component. Can mix with Polar offset to add a Z offset
r_offset
    Polar offset component. Distance from ref.
theta_offset
    Polar offset component. Radians. Azimuth angle offset of relative vector.
aspect_angle
    Polar offset component. Degrees. Can use instead of theta_offset
euler_decomp_axis
    Euler decomposition of rotation tracking into a subset of the Euler angles
    Allows tracking of planar rotations only (such as xy plane rotations only)
    NotImplemented
init
    Initialization Dictionary

Returns
-------
None


------

#### Base classes {: #RelativeCylinder-bases }

* [`RelativeGeometry `](./#RelativeGeometry)


------

#### Instance attributes {: #RelativeCylinder-attrs }

* **height**{: #RelativeCylinder.height } 

* **orientation**{: #RelativeCylinder.orientation } 

* **position**{: #RelativeCylinder.position } 

* **radius**{: #RelativeCylinder.radius } 

* **velocity**{: #RelativeCylinder.velocity } 

* **x**{: #RelativeCylinder.x } 

* **y**{: #RelativeCylinder.y } 

* **z**{: #RelativeCylinder.z } 


------

### RelativeGeometry {: #RelativeGeometry }

```python
class RelativeGeometry(
    self,
     ref,
     shape,
     track_orientation=False,
     x_offset=None,
     y_offset=None,
     z_offset=None,
     r_offset=None,
     theta_offset=None,
     aspect_angle=None,
     euler_decomp_axis=None,
     init=None,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Constructs RelativeGeometry Object
Must specify either a Cartesian offset or Polar offset from the ref object

Parameters
----------
ref
    Reference EnvObj that positions and orientations are relative to
shape
    Underlying Geometry object
track_orientation
    Whether rotate around its ref object as the ref's orientation rotates
    If True, behaves as if attached to ref with a rigid rod (rotates around ref).
    If False, behaves as if attached to ref with a gimble.
x_offset
    Cartesian offset component.
y_offset
    Cartesian offset component.
z_offset
    Cartesian offset component. Can mix with Polar offset to add a Z offset
r_offset
    Polar offset component. Distance from ref.
theta_offset
    Polar offset component. Radians. Azimuth angle offset of relative vector.
aspect_angle
    Polar offset component. Degrees. Can use instead of theta_offset
euler_decomp_axis
    Euler decomposition of rotation tracking into a subset of the Euler angles
    Allows tracking of planar rotations only (such as xy plane rotations only)
    NotImplemented
init
    Initialization Dictionary

Returns
-------
None


------

#### Base classes {: #RelativeGeometry-bases }

* [`BaseEnvObj `](../platforms-py#BaseEnvObj)


------

#### Instance attributes {: #RelativeGeometry-attrs }

* **orientation**{: #RelativeGeometry.orientation } 

* **position**{: #RelativeGeometry.position } 

* **velocity**{: #RelativeGeometry.velocity } 

* **x**{: #RelativeGeometry.x } 

* **y**{: #RelativeGeometry.y } 

* **z**{: #RelativeGeometry.z } 


------

#### Methods {: #RelativeGeometry-methods }

[**contains**](#RelativeGeometry.contains){: #RelativeGeometry.contains }

```python
def contains(self, other)
```


------

[**generate_info**](#RelativeGeometry.generate_info){: #RelativeGeometry.generate_info }

```python
def generate_info(self)
```


------

[**reset**](#RelativeGeometry.reset){: #RelativeGeometry.reset }

```python
def reset(self, **kwargs)
```


------

[**step**](#RelativeGeometry.step){: #RelativeGeometry.step }

```python
def step(self, *args, **kwargs)
```


------

[**step_apply**](#RelativeGeometry.step_apply){: #RelativeGeometry.step_apply }

```python
def step_apply(self, *args, **kwargs)
```


------

[**step_compute**](#RelativeGeometry.step_compute){: #RelativeGeometry.step_compute }

```python
def step_compute(self, *args, **kwargs)
```


------

[**update**](#RelativeGeometry.update){: #RelativeGeometry.update }

```python
def update(self)
```


------

### RelativePoint {: #RelativePoint }

```python
class RelativePoint(
    self,
     ref,
     track_orientation=False,
     x_offset=None,
     y_offset=None,
     z_offset=None,
     r_offset=None,
     theta_offset=None,
     aspect_angle=None,
     init=None,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Constructs RelativeGeometry Object
Must specify either a Cartesian offset or Polar offset from the ref object

Parameters
----------
ref
    Reference EnvObj that positions and orientations are relative to
shape
    Underlying Geometry object
track_orientation
    Whether rotate around its ref object as the ref's orientation rotates
    If True, behaves as if attached to ref with a rigid rod (rotates around ref).
    If False, behaves as if attached to ref with a gimble.
x_offset
    Cartesian offset component.
y_offset
    Cartesian offset component.
z_offset
    Cartesian offset component. Can mix with Polar offset to add a Z offset
r_offset
    Polar offset component. Distance from ref.
theta_offset
    Polar offset component. Radians. Azimuth angle offset of relative vector.
aspect_angle
    Polar offset component. Degrees. Can use instead of theta_offset
euler_decomp_axis
    Euler decomposition of rotation tracking into a subset of the Euler angles
    Allows tracking of planar rotations only (such as xy plane rotations only)
    NotImplemented
init
    Initialization Dictionary

Returns
-------
None


------

#### Base classes {: #RelativePoint-bases }

* [`RelativeGeometry `](./#RelativeGeometry)


------

#### Instance attributes {: #RelativePoint-attrs }

* **orientation**{: #RelativePoint.orientation } 

* **position**{: #RelativePoint.position } 

* **velocity**{: #RelativePoint.velocity } 

* **x**{: #RelativePoint.x } 

* **y**{: #RelativePoint.y } 

* **z**{: #RelativePoint.z } 


------

### RelativeSphere {: #RelativeSphere }

```python
class RelativeSphere(
    self,
     ref,
     track_orientation=False,
     x_offset=None,
     y_offset=None,
     z_offset=None,
     r_offset=None,
     theta_offset=None,
     aspect_angle=None,
     init=None,
     **kwargs)
```

Helper class that provides a standard way to create an ABC using
inheritance.

Constructs RelativeGeometry Object
Must specify either a Cartesian offset or Polar offset from the ref object

Parameters
----------
ref
    Reference EnvObj that positions and orientations are relative to
shape
    Underlying Geometry object
track_orientation
    Whether rotate around its ref object as the ref's orientation rotates
    If True, behaves as if attached to ref with a rigid rod (rotates around ref).
    If False, behaves as if attached to ref with a gimble.
x_offset
    Cartesian offset component.
y_offset
    Cartesian offset component.
z_offset
    Cartesian offset component. Can mix with Polar offset to add a Z offset
r_offset
    Polar offset component. Distance from ref.
theta_offset
    Polar offset component. Radians. Azimuth angle offset of relative vector.
aspect_angle
    Polar offset component. Degrees. Can use instead of theta_offset
euler_decomp_axis
    Euler decomposition of rotation tracking into a subset of the Euler angles
    Allows tracking of planar rotations only (such as xy plane rotations only)
    NotImplemented
init
    Initialization Dictionary

Returns
-------
None


------

#### Base classes {: #RelativeSphere-bases }

* [`RelativeGeometry `](./#RelativeGeometry)


------

#### Instance attributes {: #RelativeSphere-attrs }

* **orientation**{: #RelativeSphere.orientation } 

* **position**{: #RelativeSphere.position } 

* **radius**{: #RelativeSphere.radius } 

* **velocity**{: #RelativeSphere.velocity } 

* **x**{: #RelativeSphere.x } 

* **y**{: #RelativeSphere.y } 

* **z**{: #RelativeSphere.z } 


------

### Sphere {: #Sphere }

```python
class Sphere(name, x=0, y=0, z=0, radius=1)
```

Helper class that provides a standard way to create an ABC using
inheritance.


------

#### Base classes {: #Sphere-bases }

* [`Circle `](./#Circle)


------

#### Instance attributes {: #Sphere-attrs }

* **orientation**{: #Sphere.orientation } 

* **position**{: #Sphere.position } 

* **velocity**{: #Sphere.velocity } 

* **x**{: #Sphere.x } 

* **y**{: #Sphere.y } 

* **z**{: #Sphere.z } 


------

#### Methods {: #Sphere-methods }

[**contains**](#Sphere.contains){: #Sphere.contains }

```python
def contains(self, other)
```


## Functions

### angle_wrap {: #angle_wrap }

```python
def angle_wrap(angle, mode='pi')
```


------

### distance {: #distance }

```python
def distance(a, b)
```
