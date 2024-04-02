"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements the reward functions used across the
safe autonomy environments.

Rewards are used to incentivize certain [agent](../agents/index.md)
behavior during training. Over the course of a training episode,
the following reward functions take in current and previous
observations and actions and return a `RewardDict`,
allocating an appropriate reward (or penalty) to an agent based on
its behavior and the [simulation](../simulators/index.md) state.
"""
