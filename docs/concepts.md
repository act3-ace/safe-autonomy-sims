# Concepts

The safe-autonomy-sims repo was programmed to depend on the ACT3 CoRL
framework. An overview of the concepts extended in 'saferl/core' can be
found [here](https://act3-rl.github.com/act3-ace/corl/concepts/).

Concepts unique to safe-autonomy-sims, defined in saferl/backend:

##  Entities

Entities represent an Aircraft or Spacecraft that exists 
within the environment (agent controlled or otherwise). Entities
maintain a state, which encapsulates position, velocity, 
orientation, and control values. These values are used 
with a given task's dynamics model to calculate state transitions
within a step() method.

## Dynamics

Dynamics represent mathematical models which approximate entity-environment
interactions to facilitate state transitions.  
