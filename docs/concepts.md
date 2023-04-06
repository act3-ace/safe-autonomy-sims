# Concepts

The safe-autonomy-sims repo was programmed to depend on the ACT3 CoRL
framework. An overview of the concepts extended in 'safe_autonomy_sims/core' can be
found [here]({{corl_docs_url}}/concepts/).

Concepts unique to safe-autonomy-sims, defined in safe_autonomy_sims/backend:

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
