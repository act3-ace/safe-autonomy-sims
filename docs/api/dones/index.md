# dones

DoneFunctions are called by Simulators at each timestep of the simulation.
These functions are tasked with determining when an agent has reached a done
condition for their current training episode on a given task.
They take in the current and previous observation and action spaces
and output a DoneDict. The DoneDict maps agent names to DoneStatuses.

- [common dones](../../reference/dones/common_dones.md)
- [docking dones](../../reference/dones/docking_dones.md)
