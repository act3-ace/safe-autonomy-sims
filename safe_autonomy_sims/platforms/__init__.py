"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements various CoRL platforms for use with the docking
and inspection environments.

Platforms represent objects within the environment
(vehicles, robots, etc.). Platforms can contain multiple custom
`PlatformPart`s which define platform behavior. The most common
types of parts are `Controller`s and `Sensor`s. `Controller`s
apply controls to thrusters and control surfaces to interact with
environment dynamics. `Sensor`s measure various aspects of the
environment. Platform parts are often analogous to the parts used
in a physical system.
"""
