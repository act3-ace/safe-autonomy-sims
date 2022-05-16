# glues

Glues connect an agent platform to a RL training framework
by providing an endpoint for observations and actions.

A glue may transform and send data from a platform part
to a training framework as an observation. The full set of
glues which provide observations over all agents in the
environment defines the environment's observation space.

A glue may transform and send data from a training
framework to a platform part as an action. The full set
of glues which provide actions over all agents in the
environment defines the environment's action space.

Multiple glues may be used in a decorator pattern to 
reuse common transformation logic. See [CoRL
documentation](https://act3-rl.github.com/act3-ace/corl/reference/glues/__init__/) for more details!

- [normal](normal/index.md)
- [Magnitude glue](../../../reference/core/glues/magnitude_glue.md)
- [Rejoin region glue](../../../reference/core/glues/rejoin_region_glue.md)
- [Rta glue](../../../reference/core/glues/rta_glue.md)
- [Unit vector glue](../../../reference/core/glues/unit_vector_glue.md)
- [Vel limit glue](../../../reference/core/glues/vel_limit_glue.md)
