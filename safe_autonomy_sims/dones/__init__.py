"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements done functions for safe autonomy reinforcement
learning environments.

Done functions are called by [simulators](../simulators/index.md)
at each timestep of the simulation. These functions are tasked with
determining when an agent has reached a done condition for their
current training episode on a given task. They take in the current
and previous observation and action spaces and output a boolean value.
"""
