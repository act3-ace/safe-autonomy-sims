"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements Runtime Assurance modules for use
within the CoRL framework.

Runtime Assurance (RTA) is a safety technique which intercepts
a control or action before it is executed and checks it against
a well-defined safety constraint to see if the action is safe.
If the action is safe, the RTA module allows it to execute.
If the action is deemed **unsafe** the RTA module provides an
alternative action which is guaranteed to be safe under the assumption
that the system is already in a safe state.
"""
