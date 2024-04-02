"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements the simulators used in the safe
autonomy environments.

Simulators are responsible for coordinating interactions between
the various objects involved in a simulation.

CoRL simulators act as a communication layer between CoRL and any environment
simulator you want to use.

`SafeRLSimulator` and its subclasses pair
[platforms](../platforms/index.md) with their backend simulation
entities to enable accurate environment interactions with respect to
implemented dynamics models. They also coordinate execution of
[done functions](../dones/index.md) to terminate training
episodes appropriately.
"""
