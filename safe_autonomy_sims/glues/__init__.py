"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements CoRL glues for the docking and inspection
environments.

Glues connect an agent [platform](../platforms/index.md) to an RL
training framework by providing an endpoint for observations and
actions.

A glue may transform and send data from a platform part to an RL
training framework as an observation. The full set of glues which
provide observations over all agents in the environment defines the
environment's observation space.

A glue may also transform and send data from a training framework to
a platform part as an action. The full set of glues which provide
actions over all agents in the environment defines the environment's
action space.

Multiple glues may be used in a decorator pattern to reuse common
transformation logic.
"""
