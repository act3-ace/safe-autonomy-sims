# simulators

Simulators are responsible for coordinating interactions between the various
objects involved in a Simulation. They extend [BaseSimulator]() from CoRL. SafeRLSimulators
and its subclasses pair Platforms with their backend Entities to enable accurate
environment interactions with respect to implemented dynamics models. They also coordinate
execution of DoneFunctions to terminate training episodes appropriately.

- [cwh simulator](../../../reference/core/simulators/cwh_simulator.md)
- [dubins simulator](../../../reference/core/simulators/dubins_simulator.md)
- [saferl simulator](../../../reference/core/simulators/saferl_simulator.md)
